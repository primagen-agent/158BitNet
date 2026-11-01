"""Balanced zero-evidence fit batches for current memory time channels.

Query subjects remain the learned original bindings. Recall uses complete
physical banks; source-kind minibatches are classification supervision only.
This helper reads no files and never supplies serving rules or gold prefixes.
"""

from copy import deepcopy
import hashlib
import json

from neural_reader_replay import (
    _deduplicate_originals, _observation_ids, _supports, _validate_teacher_targets,
)

TEMPORAL_MAIN_SEED = 5362
TEMPORAL_KIND_SEED = 5363
LIFECYCLE_VIEWS = ("current", "history", "tentative", "all_versions")


def _validate_group(group, excluded):
    if (not isinstance(group, dict) or not {"facts", "queries"} <= group.keys()
            or group.get("split", "fit") != "fit" or group.get("evaluation_only")):
        raise ValueError("temporal curriculum requires fit-only banks")
    for key in ("facts", "queries"):
        if not isinstance(group[key], list) or not group[key]:
            raise ValueError("temporal curriculum needs complete facts and questions")
        for row in group[key]:
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"]:
                raise ValueError("temporal curriculum needs nonempty row texts")
            observation = row.get("observation", {})
            if (row.get("split", "fit") != "fit" or row.get("evaluation_only")
                    or "evidence_challenge_view" in row or row.get("binding_fit")
                    or observation.get("split", "fit") != "fit"
                    or observation.get("evaluation_only")):
                raise ValueError("development, evaluation and new bindings cannot enter temporal fit")
            if _observation_ids(row) & excluded:
                raise ValueError("qualification observations cannot enter temporal fit")


def _originals(group):
    rows = [row for row in group["queries"] if not row.get("augmented")
            and "\nActivated evidence:\n" not in row["text"]
            and row.get("evidence_count", 0) == 0]
    for row in rows:
        if "observation" in row and not row.get("binding_predicted"):
            raise ValueError("temporal observations require learned query binding")
        if row.get("unresolved"):
            raise ValueError("temporal queries require a resolved learned binding")
    _validate_teacher_targets(rows, group["facts"])
    return _deduplicate_originals(rows, group["facts"])


def _active_facts(facts):
    total = max(len(facts), max((f.get("insert_order", i) + 1
                               for i, f in enumerate(facts)), default=0))
    return [dict(deepcopy(f), insert_order=f.get("insert_order", i),
                 bank_totalcount=max(total, f.get("bank_totalcount", total)))
            for i, f in enumerate(facts) if f.get("status") != 2]


def _source_kind_pools(facts):
    labels, rows = {}, {}
    for fact in facts:
        kind = fact.get("source_kind", fact.get("kind"))
        if type(kind) is not int or kind not in (0, 1, 2):
            raise ValueError("temporal source kind must be the original text kind in 0..2")
        key = (fact["text"], fact.get("reference", fact["text"]))
        if key in labels and labels[key] != kind:
            raise ValueError("conflicting original source-kind labels for one fact input")
        labels[key] = kind
        # Physical versions can share one original source text. Classification
        # de-duplicates the input; recall keeps every live physical position.
        encoded = json.dumps(fact, sort_keys=True)
        if key not in rows or encoded < json.dumps(rows[key], sort_keys=True):
            rows[key] = deepcopy(fact)
    pools = {kind: [] for kind in (0, 1, 2)}
    for key in sorted(rows):
        pools[labels[key]].append(rows[key])
    return pools


def prepare_temporal_fit(fit_group, lifecycle_group, *, split="fit",
                         excluded_observation_ids=()):
    """Return independent pools and a manifest using existing fit text only."""
    if split != "fit":
        raise ValueError("temporal curriculum is fit-only")
    excluded = set(excluded_observation_ids)
    for group in (fit_group, lifecycle_group):
        _validate_group(group, excluded)
    fit_rows, life_rows = _originals(fit_group), _originals(lifecycle_group)
    fit_facts, life_facts = _active_facts(fit_group["facts"]), _active_facts(lifecycle_group["facts"])
    positive = {kind: [] for kind in (0, 1, 2)}
    null = []
    for row in fit_rows:
        if any(_supports(row, fact) for fact in fit_facts):
            kind = row.get("kind")
            if type(kind) is not int or kind not in positive:
                raise ValueError("old fit temporal positives require a kind in 0..2")
            positive[kind].append(row)
        else:
            null.append(row)
    life = {view: [] for view in LIFECYCLE_VIEWS}
    for row in life_rows:
        if row.get("lifecycle_view") not in life:
            raise ValueError("lifecycle queries need current/history/tentative/all_versions views")
        life[row["lifecycle_view"]].append(row)
    if any(len(pool) < 8 for pool in [*positive.values(), null, *life.values()]):
        raise ValueError("temporal main strata need at least eight distinct original questions")
    kinds = _source_kind_pools(fit_group["facts"] + lifecycle_group["facts"])
    if any(len(pool) < 12 for pool in kinds.values()):
        raise ValueError("temporal source-kind strata need twelve distinct original fact inputs")
    prepared = dict(fit_facts=fit_facts, lifecycle_facts=life_facts,
                    fit_positive=positive, fit_null=null, lifecycle_queries=life,
                    source_kind_facts=kinds)
    texts = sorted({row["text"] for row in fit_rows + life_rows
                    + fit_group["facts"] + lifecycle_group["facts"]}
                   | {row[key] for row in fit_rows + life_rows
                      + fit_group["facts"] + lifecycle_group["facts"]
                      for key in ("reference", "initial_text", "initial_reference") if row.get(key)})
    manifest = dict(
        split="fit", selection_data=False, main_seed=TEMPORAL_MAIN_SEED,
        kind_seed=TEMPORAL_KIND_SEED,
        main_schedule="four lifecycle zero-cue steps, one old-fit zero-cue step",
        main_query_batch=32, lifecycle_queries=len(life_rows),
        fit_original_rows=sum(not q.get("augmented") and "\nActivated evidence:\n" not in q["text"]
                              and q.get("evidence_count", 0) == 0 for q in fit_group["queries"]),
        fit_original_inputs=len(fit_rows), fit_positive_counts={k: len(v) for k, v in positive.items()},
        fit_null_queries=len(null), lifecycle_view_counts={k: len(v) for k, v in life.items()},
        fit_recall_bank=len(fit_facts), lifecycle_recall_bank=len(life_facts),
        source_kind_facts=sum(map(len, kinds.values())),
        source_kind_counts={k: len(v) for k, v in kinds.items()}, source_kind_batch=36,
        source_kind_label="source_kind if authored, otherwise original kind; never physical status",
        required_texts=len(texts), texts_sha256=hashlib.sha256(json.dumps(texts).encode()).hexdigest(),
        no_texts_created=True,
    )
    return prepared, manifest


def sample_temporal_main(prepared, step, rng):
    """One full-bank main batch; independent RNG from replay and kind sampling."""
    if type(step) is not int or step < 1:
        raise ValueError("temporal main step must be a positive integer")
    if step % 5:
        name, facts = "lifecycle_zero", prepared["lifecycle_facts"]
        rows = [row for view in LIFECYCLE_VIEWS
                for row in rng.sample(prepared["lifecycle_queries"][view], 8)]
    else:
        name, facts = "old_fit_zero", prepared["fit_facts"]
        rows = [row for kind in (0, 1, 2) for row in rng.sample(prepared["fit_positive"][kind], 8)]
        rows += rng.sample(prepared["fit_null"], 8)
    rng.shuffle(rows)
    return name, deepcopy(rows), deepcopy(facts)


def sample_temporal_kind(prepared, rng):
    """36 source examples for classification only; never a smaller recall bank."""
    rows = [row for kind in (0, 1, 2)
            for row in rng.sample(prepared["source_kind_facts"][kind], 12)]
    rng.shuffle(rows)
    return deepcopy(rows)
