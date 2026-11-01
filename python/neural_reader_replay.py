"""Bounded, complete-bank replay of existing synthetic fit interactions.

This training helper reads no files and does not select checkpoints. Query
bindings are copied from the frozen writer; evidence never supplies a new
subject. Every selected question retains its complete original fact bank.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import random

from neural_fit_support import sample_relation_queries
from neural_reader_evidence_curriculum import augment_evidence


FIT_REPLAY_SEEDS = {
    "fit": 5321,
    "authored_fit": 5321,
    "episode_fit": 5341,
    "lifecycle_fit": 5342,
    "relation_variation_fit": 5343,
}
EVIDENCE_COUNTS = (0, 1, 4)


def _values(value):
    return {value} if isinstance(value, str) else set(value or ())


def _supports(question, fact):
    if fact.get("status") == 2:
        return False
    if "support_texts" in question:
        return fact["text"] in question["support_texts"]
    return (fact.get("entity") in _values(question.get("entities", question.get("entity")))
            and fact.get("prop") in _values(question.get("prop"))
            and ("kind" not in question or question["kind"] == fact.get("kind")))


def _support_count(question, facts):
    return sum(_supports(question, fact) for fact in facts)


def _teacher_input_key(row):
    """Effective current/original texts and identities consumed by query()."""
    text = row["text"]
    initial_text = row.get("initial_text", text)
    reference = row.get("reference", text)
    initial_reference = row.get("initial_reference", row.get("reference")) or initial_text
    return text, reference, initial_text, initial_reference


def _target_signature(row, facts):
    """Targets visible to reader losses, independent of observation provenance."""
    values = dict(
        entities=sorted(_values(row.get("entities", row.get("entity")))),
        prop=sorted(_values(row.get("prop"))),
        kind=row.get("kind", "absent"),
        support_mask=[_supports(row, fact) for fact in facts],
        semantic_specific=bool(row.get("semantic_specific")),
    )
    for key in ("allowed_kinds", "source_kinds", "binding_contrast_texts"):
        values[key] = sorted(_values(row[key])) if key in row else "absent"
    return json.dumps(values, sort_keys=True)


def _deduplicate_originals(rows, facts):
    grouped = defaultdict(list)
    for row in rows:
        grouped[_teacher_input_key(row)].append(row)
    originals = []
    for key in sorted(grouped):
        aliases = grouped[key]
        if len({_target_signature(row, facts) for row in aliases}) != 1:
            identifiers = sorted({value for row in aliases for value in _observation_ids(row)})
            raise ValueError("conflicting fit replay targets for identical teacher input: "
                             + json.dumps(dict(text=key[0], observations=identifiers)))
        # Original observation IDs/contexts are provenance after learned query
        # binding. Equivalent aliases must not overweight the same interaction.
        original = deepcopy(min(aliases, key=lambda row: json.dumps(row, sort_keys=True)))
        original["replay_equivalent_observations"] = sorted(
            {value for row in aliases for value in _observation_ids(row)}
        )
        original["replay_equivalent_rows"] = len(aliases)
        originals.append(original)
    return originals


def _validate_teacher_targets(rows, facts):
    signatures = {}
    for row in rows:
        key = _teacher_input_key(row)
        signature = _target_signature(row, facts)
        if key in signatures and signatures[key] != signature:
            raise ValueError("conflicting fit replay targets for identical teacher input: " + key[0])
        signatures[key] = signature


def _observation_ids(row):
    return {value for value in (
        row.get("observation_id"), row.get("original_observation_id"),
        row.get("source_id"), row.get("observation", {}).get("source_id"),
    ) if value}


def _validate_fit_bank(name, group, excluded):
    if name not in FIT_REPLAY_SEEDS:
        raise ValueError("old fit replay needs a registered fit bank, excluding new bindings")
    if not isinstance(group, dict) or not {"facts", "queries"} <= group.keys():
        raise ValueError("old fit replay needs facts and queries")
    if group.get("evaluation_only") or group.get("split", "fit") != "fit":
        raise ValueError("old fit replay is fit-only")
    for key in ("facts", "queries"):
        if not isinstance(group[key], list) or not group[key]:
            raise ValueError("old fit replay needs a nonempty complete bank and questions")
        for row in group[key]:
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"]:
                raise ValueError("old fit replay needs nonempty row texts")
            if (row.get("evaluation_only") or row.get("split", "fit") != "fit"
                    or "evidence_challenge_view" in row):
                raise ValueError("evaluation/development rows cannot enter fit replay")
            if row.get("binding_fit"):
                raise ValueError("new binding facts cannot enter old fit replay")
            if _observation_ids(row) & excluded:
                raise ValueError("qualification observation cannot enter fit replay")
    if len({fact["text"] for fact in group["facts"]}) != len(group["facts"]):
        raise ValueError("old fit replay needs unique complete fact texts")
    if len({fact["text"] for fact in group["facts"] if fact.get("status") != 2}) < 4:
        raise ValueError("four-cue fit replay needs four distinct live facts")


def _balanced_originals(rows, facts, rng, count):
    """Round-robin null/single/multiple supports, relation and temporal views."""
    buckets = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=_teacher_input_key):
        support = min(_support_count(row, facts), 2)
        relation = tuple(sorted(_values(row.get("prop"))))
        key = (relation, str(row.get("kind", -1)), str(row.get("query_class", "")))
        buckets[support][key].append(row)
    for groups in buckets.values():
        for values in groups.values():
            rng.shuffle(values)
    categories = sorted(buckets)
    rng.shuffle(categories)
    counts = Counter()
    chosen = []
    while categories and len(chosen) < count:
        remaining = []
        for category in categories:
            if len(chosen) == count:
                break
            keys = [key for key, values in buckets[category].items() if values]
            minimum = min(counts[category, key] for key in keys)
            key = rng.choice(sorted(key for key in keys if counts[category, key] == minimum))
            chosen.append(buckets[category][key].pop())
            counts[category, key] += 1
            if any(buckets[category].values()):
                remaining.append(category)
        categories = remaining
    return chosen


def build_replay_groups(old_groups, *, split="fit", query_limit=24,
                        excluded_observation_ids=(), seeds=None):
    """Return named zero/one/four-cue fit views of unchanged complete banks.

    Supplied groups must be existing fit banks, with queries already bound by
    the learned writer where observations exist. Existing fit cue texts are
    preferred to reuse frozen C features. Only missing selected views require
    newly generated features; ``replay_feature_texts`` returns their exact
    text requirements. No gold reference or subject prefix is constructed.
    """
    if split != "fit":
        raise ValueError("old fit replay is fit-only")
    if type(query_limit) is not int or not 1 <= query_limit <= 32:
        raise ValueError("fit replay question limit must be an integer in 1..32")
    if not isinstance(old_groups, dict) or not old_groups:
        raise ValueError("old fit replay needs named fit banks")
    seed_values = dict(FIT_REPLAY_SEEDS)
    if seeds is not None:
        if (not isinstance(seeds, dict) or not seeds.keys() <= old_groups.keys()
                or any(type(value) is not int for value in seeds.values())):
            raise ValueError("fit replay seeds must be named integers")
        seed_values.update(seeds)
    excluded = set(excluded_observation_ids)
    result = {}
    for name in sorted(old_groups):
        group = old_groups[name]
        _validate_fit_bank(name, group, excluded)
        _validate_teacher_targets(group["queries"], group["facts"])
        originals = [row for row in group["queries"] if not row.get("augmented")
                     and "\nActivated evidence:\n" not in row["text"]]
        if not originals:
            raise ValueError("fit replay requires unconditioned original questions")
        for row in originals:
            if "observation" in row and not row.get("binding_predicted"):
                raise ValueError("fit replay observation queries need learned writer binding")
        originals = _deduplicate_originals(originals, group["facts"])
        selected = _balanced_originals(originals, group["facts"],
                                      random.Random(seed_values[name]), query_limit)
        existing = defaultdict(list)
        for row in group["queries"]:
            if row.get("augmented") and row.get("evidence_count") in (1, 4):
                text = row.get("initial_text")
                reference = row.get("initial_reference") or text
                # Existing augment_evidence rows represent an ordinary
                # original input. A different NN reference is a different
                # teacher interaction, even if its question text is an alias.
                key = (text, reference, text, reference)
                existing[key, row["evidence_count"]].append(row)
        fresh = defaultdict(list)
        for original in selected:
            # The augmentation helper itself de-duplicates by text. Generate
            # each distinct teacher input separately so NN-reference aliases
            # never drop a question or share another interaction's cues.
            generated = augment_evidence(
                dict(facts=[fact for fact in group["facts"] if fact.get("status") != 2],
                     queries=[original]), seed_values[name], 600, split="fit",
            )
            for row in generated["queries"]:
                if row.get("augmented") and row.get("evidence_count") in (1, 4):
                    fresh[_teacher_input_key(original), row["evidence_count"]].append(row)
        views = {count: [] for count in EVIDENCE_COUNTS}
        for original in selected:
            for count in EVIDENCE_COUNTS:
                if count:
                    key = _teacher_input_key(original)
                    candidates = existing[key, count] or fresh[key, count]
                    if not candidates:
                        raise ValueError("fit replay cannot construct every selected cue view")
                    if any(_target_signature(candidate, group["facts"])
                           != _target_signature(original, group["facts"]) for candidate in candidates):
                        raise ValueError("fit replay evidence targets differ from original interaction")
                    cue_rng = random.Random(f"fit-replay:{seed_values[name]}:{key!r}:{count}")
                    candidate = cue_rng.choice(sorted(candidates, key=lambda row: row["text"]))
                    cues = candidate.get("evidence_sources", ())
                    live = {fact["text"] for fact in group["facts"] if fact.get("status") != 2}
                    if len(cues) != count or len(set(cues)) != count or not set(cues) <= live:
                        raise ValueError("fit replay cues must be distinct complete live facts")
                    row = deepcopy(original)
                    row.update({key: deepcopy(candidate[key]) for key in (
                        "text", "evidence_sources", "evidence_type", "evidence_mix",
                    ) if key in candidate})
                    row["augmented"] = True
                    row.pop("reference", None)
                else:
                    row = deepcopy(original)
                    row["evidence_sources"] = []
                row.update(initial_text=original.get("initial_text", original["text"]), evidence_count=count,
                           replay_evidence_view=count, replay_bank=name)
                initial_reference = original.get("initial_reference", original.get("reference"))
                row.pop("initial_reference", None)
                if initial_reference:
                    row["initial_reference"] = initial_reference
                views[count].append(row)
        for count, queries in views.items():
            result[f"{name}_cue_{count}"] = dict(facts=deepcopy(group["facts"]), queries=queries)
    return result


def replay_feature_texts(groups):
    """Exact frozen feature inputs used by DynamicFeatures for replay only."""
    result = set()
    for group in groups.values():
        for rows in (group["facts"], group["queries"]):
            for row in rows:
                result.add(row["text"])
                result.update(row[key] for key in ("reference", "initial_reference", "initial_text")
                              if row.get(key))
    return sorted(result)


def replay_manifest(groups, existing_texts=()):
    """Counts and the exact additional C text list, without accessing caches."""
    texts = replay_feature_texts(groups)
    existing = set(existing_texts)
    return dict(
        split="fit", selection_data=False, groups={
            name: dict(facts=len(group["facts"]), queries=len(group["queries"]),
                       support_counts=dict(sorted(Counter(_support_count(q, group["facts"])
                                                          for q in group["queries"]).items())))
            for name, group in sorted(groups.items())
        }, feature_texts=len(texts),
        texts_sha256=hashlib.sha256(json.dumps(texts).encode()).hexdigest(),
        additional_feature_texts=[text for text in texts if text not in existing],
    )


def sample_replay_group(groups, rng):
    """Sample complete banks and cue views uniformly, independent of bank size."""
    if not groups:
        raise ValueError("replay sampling needs complete banks")
    name = rng.choice(sorted(groups))
    return name, groups[name]


def sample_replay_queries(group, rng, count=32):
    """Balance null/single/multiple complete supports, then relation families."""
    if type(count) is not int or count < 0:
        raise ValueError("replay batch size must be a nonnegative integer")
    rows, facts = group["queries"], group["facts"]
    if any(row.get("evaluation_only") for row in rows):
        raise ValueError("evaluation queries cannot enter fit replay sampling")
    buckets = defaultdict(list)
    for row in rows:
        buckets[min(_support_count(row, facts), 2)].append(row)
    sizes = {key: 0 for key in buckets}
    pending = sorted(buckets)
    rng.shuffle(pending)
    remaining = min(count, len(rows))
    while pending and remaining:
        available = []
        for key in pending:
            if not remaining:
                break
            sizes[key] += 1
            remaining -= 1
            if sizes[key] < len(buckets[key]):
                available.append(key)
        pending = available
    selected = [row for key in sorted(buckets)
                for row in sample_relation_queries(buckets[key], rng, sizes[key])]
    rng.shuffle(selected)
    return selected
