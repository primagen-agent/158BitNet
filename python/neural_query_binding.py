"""Fit-only objectives and guards for the existing learned query binder."""

import math

import torch
from torch.nn import functional as F

from neural_slot_model import QUERY_KEYS


QUERY_BINDING_MODES = ("query", "query-gate", "query-source", "query-context")
QUERY_BINDING_CLASSES = ("self_contained", "resolved", "ambiguous")


def query_binding_class(row):
    need, refs = row["whole_need"], row["whole_refs"]
    if type(need) is not bool or not isinstance(refs, list) or any(
        not isinstance(ref, str) or not ref for ref in refs
    ):
        raise ValueError("query binding requires boolean need and literal reference labels")
    if not need and refs:
        raise ValueError("self-contained query binding cannot require reference sources")
    return "self_contained" if not need else "resolved" if refs else "ambiguous"


def validate_query_binding_args(args):
    mode = args.train_heads
    weight = getattr(args, "query_binding_anchor_weight", 0.0)
    margin = getattr(args, "query_binding_anchor_margin", 1.0)
    generalization = getattr(args, "query_binding_generalization", False)
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(margin) or margin <= 0:
        raise ValueError("query-binding anchor weight/margin must be finite and nonnegative/positive")
    if mode in ("query-gate", "query-source", "query-context") and not getattr(args, "resume", None):
        raise ValueError(f"{mode} requires a trained current-format --resume")
    if (weight or margin != 1.0 or generalization) and (
        mode not in QUERY_BINDING_MODES or not getattr(args, "resume", None)
    ):
        raise ValueError("query-binding anchors/generalization require resumed query binding training")


def require_query_binding_fit(row):
    if (row.get("evaluation_only") or row.get("split", "fit") != "fit"
            or row.get("query_binding_split", "fit") != "fit"
            or row.get("evaluation_role", "fit-teacher") != "fit-teacher"):
        raise ValueError("query binding supervision and sampling accept authored fit rows only")


def validate_query_binding_fit_rows(mode, rows):
    """Reject unusable fit data before extracting backbone features."""
    if not rows:
        raise ValueError("query binding training requires nonempty authored fit rows")
    classes = []
    for row in rows:
        require_query_binding_fit(row)
        classes.append(query_binding_class(row))
    if mode == "query-source" and all(name == "self_contained" for name in classes):
        raise ValueError("query-source requires fit queries that use reference-source supervision")


def sample_query_binding_batch(rows, rng, per_class=8):
    """Give each nonempty binding class the same bounded sampling quota."""
    if type(per_class) is not int or per_class < 1:
        raise ValueError("query binding class quota must be a positive integer")
    groups = {name: [] for name in QUERY_BINDING_CLASSES}
    for row in rows:
        require_query_binding_fit(row)
        groups[query_binding_class(row)].append(row)
    batch = []
    for values in groups.values():
        novel = [i for i, row in enumerate(values) if query_binding_is_novel(row)]
        replay = [i for i, row in enumerate(values) if not query_binding_is_novel(row)]
        selected = rng.sample(novel, min(per_class // 2, len(novel)))
        selected += rng.sample(replay, min(per_class // 2, len(replay)))
        remaining = [i for i in range(len(values)) if i not in selected]
        selected += rng.sample(remaining, min(per_class - len(selected), len(remaining)))
        batch.extend(values[i] for i in selected)
    rng.shuffle(batch)
    return batch


def query_binding_is_novel(row):
    """Authored curriculum provenance only; this never chooses a serving subject."""
    return row.get("query_binding_split") == "fit" or bool(row.get("query_binding_family"))


def query_binding_metrics(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("query binding predictions must match observation rows")
    counts = {name: dict(rows=0, need_correct=0, full_correct=0,
                         ambiguous_unsafe=0, wrong_subjects=0)
              for name in QUERY_BINDING_CLASSES}
    for row, prediction in zip(rows, predictions):
        name = query_binding_class(row)
        need, refs = prediction["need"], prediction["references"]
        if type(need) is not bool or not isinstance(refs, list) or (not need and refs):
            raise ValueError("query binding prediction must follow the serving need gate")
        values = counts[name]
        values["rows"] += 1
        values["need_correct"] += need == row["whole_need"]
        values["full_correct"] += need == row["whole_need"] and refs == row["whole_refs"]
        values["ambiguous_unsafe"] += name == "ambiguous" and (not need or bool(refs))
        values["wrong_subjects"] += bool(refs) and refs != row["whole_refs"]
    total = len(rows)
    classes = {}
    for name, values in counts.items():
        count = values["rows"]
        classes[name] = dict(values,
            need_exact=values["need_correct"] / count if count else None,
            full_binding_exact=values["full_correct"] / count if count else None)
    return dict(
        query_binding_classes=classes,
        query_binding_full_exact=sum(v["full_correct"] for v in counts.values()) / total if total else None,
        query_binding_need_exact=sum(v["need_correct"] for v in counts.values()) / total if total else None,
        query_ambiguous_unsafe=sum(v["ambiguous_unsafe"] for v in counts.values()),
        query_binding_wrong_subjects=sum(v["wrong_subjects"] for v in counts.values()),
    )


def query_binding_score(metrics):
    """Worst class, class mean and row mean over the named selection groups."""
    groups = [metrics] if isinstance(metrics, dict) else list(metrics)
    counts = {name: [0, 0] for name in QUERY_BINDING_CLASSES}
    for group in groups:
        classes = _checked_binding_classes(group)
        for name in QUERY_BINDING_CLASSES:
            values = classes[name]
            counts[name][0] += values["rows"]
            counts[name][1] += values["full_correct"]
    active = [correct / rows for rows, correct in counts.values() if rows]
    total = sum(v[0] for v in counts.values())
    if not active or not total:
        raise ValueError("query binding selection requires annotated development queries")
    return (min(active), sum(active) / len(active), sum(v[1] for v in counts.values()) / total)


def _checked_binding_classes(metrics):
    classes = metrics["query_binding_classes"]
    if set(classes) != set(QUERY_BINDING_CLASSES):
        raise ValueError("query binding metrics must retain every class")
    for name, values in classes.items():
        count = values["rows"]
        if type(count) is not int or count < 0:
            raise ValueError("invalid query binding class count")
        for key in ("need_correct", "full_correct", "ambiguous_unsafe", "wrong_subjects"):
            value = values[key]
            if type(value) is not int or not 0 <= value <= count:
                raise ValueError("invalid query binding outcome count")
        if values["full_correct"] > values["need_correct"]:
            raise ValueError("full query binding requires a correct need gate")
        for ratio, key in (("need_exact", "need_correct"), ("full_binding_exact", "full_correct")):
            value = values[ratio]
            if count:
                if (not isinstance(value, (int, float)) or not math.isfinite(value)
                    or abs(value - values[key] / count) > 1e-12):
                    raise ValueError("invalid query binding class accuracy")
            elif value is not None:
                raise ValueError("empty query binding classes have no accuracy")
    return classes


def query_binding_safe(current, initial):
    """No class or ambiguity safety loss can be bought by another class."""
    try:
        current_classes = _checked_binding_classes(current)
        initial_classes = _checked_binding_classes(initial)
        for name in QUERY_BINDING_CLASSES:
            now, old = current_classes[name], initial_classes[name]
            if now["rows"] != old["rows"]:
                return False
            if now["rows"] and (now["need_exact"] < old["need_exact"]
                               or now["full_binding_exact"] < old["full_binding_exact"]):
                return False
            if (now["ambiguous_unsafe"] > old["ambiguous_unsafe"]
                or now["wrong_subjects"] > old["wrong_subjects"]):
                return False
        return (current["query_ambiguous_unsafe"] <= initial["query_ambiguous_unsafe"]
                and current["query_binding_wrong_subjects"] <= initial["query_binding_wrong_subjects"])
    except (KeyError, TypeError, ValueError):
        return False


def snapshot_query_binding_parent(model):
    """An unregistered, independent functional snapshot of all binder tensors."""
    state = model.state_dict()
    return {name: state[name].detach().clone() for name in QUERY_KEYS}


@torch.no_grad()
def query_binding_parent_logits(model, candidate, source_words, source_types, parent, *, raw_context):
    inputs = torch.cat((model.augment(source_words, hidden_only=True), source_types), -1)
    candidate = candidate + F.linear(raw_context, parent["query_context.weight"]).tanh()

    def linear(name, value):
        return F.linear(value, parent[name + ".weight"], parent[name + ".bias"])

    encoded = linear("query_ref_encode", inputs).tanh()
    cue = linear("query_ref_query", candidate).tanh()[None].expand_as(encoded)
    hidden = linear("query_ref_pair.0", torch.cat((cue, encoded), -1)).tanh()
    tags = linear("query_ref_pair.2", hidden)
    origin = source_types @ linear("query_origin", candidate)
    tags = tags + torch.stack((torch.zeros_like(origin), origin, origin), -1)
    return linear("query_need", candidate).squeeze(), tags


def make_query_binding_teacher(parent_need, parent_tags, gold_need, gold_tags, margin=1.0):
    """Freeze component correctness and capped parent margins on authored fit."""
    if type(gold_need) is not bool or not math.isfinite(margin) or margin <= 0:
        raise ValueError("query binding teacher requires a boolean need and positive finite margin")
    if parent_need.numel() != 1 or parent_tags.ndim != 2 or parent_tags.shape[1] != 3:
        raise ValueError("query binding teacher requires a scalar need and three BIO classes")
    if gold_tags.shape != parent_tags.shape[:1] or gold_tags.dtype != torch.long:
        raise ValueError("query binding BIO labels must match the source sequence")
    if not bool(torch.isfinite(parent_need).all() and torch.isfinite(parent_tags).all()):
        raise ValueError("query binding parent must have finite logits")
    with torch.no_grad():
        sign = parent_need.new_tensor(1.0 if gold_need else -1.0)
        need_mask = (parent_need > 0) == gold_need
        need_margin = (sign * parent_need).clamp(min=0.0, max=margin)
        targets = gold_tags.detach().clone()
        tag_mask = (parent_tags.argmax(-1) == targets) & gold_need
        chosen = parent_tags.gather(-1, targets[:, None]).squeeze(-1)
        alternatives = parent_tags.masked_fill(F.one_hot(targets, 3).bool(), -torch.inf).amax(-1)
        tag_margin = (chosen - alternatives).clamp(min=0.0, max=margin)
        return dict(need_mask=need_mask.detach().clone(), need_sign=sign,
                    need_margin=need_margin.detach().clone(), tag_mask=tag_mask,
                    tag_targets=targets, tag_margin=tag_margin.detach().clone())


def query_binding_anchor_parts(need, tags, teacher):
    need_loss = F.relu(teacher["need_margin"] - teacher["need_sign"] * need).square()
    need_loss = torch.where(teacher["need_mask"], need_loss, need * 0)
    targets = teacher["tag_targets"]
    chosen = tags.gather(-1, targets[:, None]).squeeze(-1)
    alternatives = tags.masked_fill(F.one_hot(targets, 3).bool(), -torch.inf).amax(-1)
    violations = F.relu(teacher["tag_margin"] - (chosen - alternatives)).square()
    violations = torch.where(teacher["tag_mask"], violations, tags[:, 0] * 0)
    return need_loss, violations.amax() if len(violations) else tags.sum() * 0
