#!/usr/bin/env python3
"""Train a neural identity channel on a frozen resident-memory baseline.

Role annotations are training-only. Inference consumes contextual/lexical token
vectors and resident tensors, never names, token strings, gold spans or IDs.
"""
import argparse
import json
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from train_resident_memory_set import (FactorizedResidentMemorySet, CTokenizer, GGUFWeights,
    load_worlds, collate, evaluate, compile_states, balanced_binary_loss,
    version_log_gate, file_fingerprint, clone_state_dict)


def log_probability_to_logit(value):
    value = value.clamp_max(-1e-6)
    return value - torch.log(-torch.expm1(value))


class IdentityResidentMemorySet(nn.Module):
    has_identity_channel = True

    def __init__(self, hidden, width=128, initial_order_scale=.025):
        super().__init__()
        self.hidden, self.width = hidden, width
        self.baseline = FactorizedResidentMemorySet(hidden, width, initial_order_scale)
        self.baseline.requires_grad_(False).eval()
        self.entity_role = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1))
        self.raw_scale = nn.Parameter(torch.tensor(10.))
        self.identity_bias = nn.Parameter(torch.tensor(-5.))

    def train(self, mode=True):
        super().train(mode); self.baseline.eval()
        return self

    def write(self, tokens, token_mask):
        contextual, lexical = tokens.split(self.hidden, dim=-1)
        with torch.no_grad(): semantic = self.baseline.write(contextual, token_mask)
        role = self.entity_role(contextual).squeeze(-1)
        valid = token_mask[..., None].to(tokens.dtype)
        return torch.cat((semantic[..., :-1], lexical * valid, role[..., None] * valid, valid), -1)

    def pack_state(self, ex, em, counts, mask):
        addresses = self.write(ex, em)
        state = addresses.new_zeros(len(counts), mask.shape[1], *addresses.shape[1:])
        offset = 0
        for i, count in enumerate(counts):
            state[i, :count] = addresses[offset:offset + count]; offset += count
        return state

    def read_components(self, query, query_mask, state, slot_mask, positions):
        contextual, lexical_query = query.split(self.hidden, dim=-1)
        valid = (state[..., -1] > .5) & slot_mask[..., None]
        semantic = torch.cat((state[..., :self.width], state[..., -1:]), -1).detach()
        with torch.no_grad():
            _, counts, old = self.baseline.read_components(contextual, query_mask, semantic, slot_mask, positions)
        lexical_memory = state[..., self.width:self.width + self.hidden].detach()
        role_memory = state[..., -2]
        safe = valid.clone(); safe[..., 0] |= ~safe.any(-1)
        weights = role_memory.masked_fill(~safe, -torch.inf).softmax(-1)
        identity = F.normalize((weights[..., None] * lexical_memory).sum(-2), dim=-1)
        role_query = self.entity_role(contextual).squeeze(-1)
        similarity = torch.einsum("bnd,btd->bnt", identity, lexical_query)
        alignment = (8. * similarity + F.logsigmoid(role_query)[:, None]).masked_fill(~query_mask[:, None], -torch.inf).softmax(-1)
        attended = F.normalize(torch.einsum("bnt,btd->bnd", alignment, lexical_query), dim=-1)
        entity_read = F.softplus(self.raw_scale) * (identity * attended).sum(-1) + self.identity_bias
        entity_links = F.softplus(self.raw_scale) * torch.einsum("bid,bjd->bij", identity, identity) + self.identity_bias
        address = log_probability_to_logit(F.logsigmoid(old["address"]) + F.logsigmoid(entity_read))
        links = log_probability_to_logit(F.logsigmoid(old["links"]) + F.logsigmoid(entity_links))
        gate = version_log_gate(links, old["history"], slot_mask, positions)
        scores = log_probability_to_logit(F.logsigmoid(address) + gate).masked_fill(~slot_mask, -1e4)
        return scores, counts, {"address": address, "links": links, "history": old["history"],
                                "entity_read": entity_read, "entity_links": entity_links,
                                "role_query": role_query, "role_memory": role_memory}

    def read(self, query, query_mask, state, slot_mask, positions):
        scores, counts, _ = self.read_components(query, query_mask, state, slot_mask, positions)
        return scores, counts

    def forward(self, ex, em, qx, qm, counts, mask, positions):
        return self.read(qx, qm, self.pack_state(ex, em, counts, mask), mask, positions)


def append_lexical_features(worlds, features, weights, tokenizer):
    """Exact C tokenization plus the identity-bound GGUF input embedding table."""
    if len(worlds) != len(features): raise ValueError("lexical cache world count mismatch")
    table = torch.from_numpy(weights.get_f32("token_embd.weight", (weights.vocab, weights.hidden)).copy())
    cache, result = {}, []
    for world, feature in zip(worlds, features):
        item = {}
        for key in ("events", "queries"):
            if len(world[key]) != len(feature[key]): raise ValueError("lexical cache row count mismatch")
            item[key] = []
            for row, hidden in zip(world[key], feature[key]):
                text = row["text"]
                if text not in cache:
                    ids = tokenizer.encode(text, True)
                    if len(ids) != len(hidden) or hidden.shape[-1] != weights.hidden:
                        raise ValueError("context/lexical tokenizer or geometry mismatch")
                    lexical = F.normalize(table[torch.tensor(ids)].float(), dim=-1).half()
                    cache[text] = torch.cat((hidden, lexical), -1)
                item[key].append(cache[text])
        result.append(item)
    return result


def entity_role_labels(text, entities, tokenizer):
    """Align synthetic entity mentions to exact token bytes, for training only."""
    ids = tokenizer.encode(text, True)
    pieces = list(tokenizer.decode_pieces(ids)); pieces[0] = b""
    decoded, expected = b"".join(pieces), text.encode()
    shift = 0 if decoded == expected else 1 if decoded == b" " + expected else None
    if shift is None: raise ValueError("token bytes do not reproduce identity supervision text")
    spans, mentioned = [], set()
    for entity in entities:
        for match in re.finditer(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", text):
            spans.append((shift + len(text[:match.start()].encode()), shift + len(text[:match.end()].encode())))
            mentioned.add(entity)
    if not spans: raise ValueError("synthetic entity mention missing")
    labels, offset = [], 0
    for piece in pieces:
        end = offset + len(piece)
        labels.append(float(end > offset and any(offset < stop and end > start for start, stop in spans)))
        offset = end
    return torch.tensor(labels), mentioned


def prepare_identity_labels(worlds, features, sidecar, tokenizer):
    output = []
    for world, feature in zip(worlds, features):
        if world["split"] != "train" or world.get("evaluation_only") or world.get("locomo_used"):
            raise ValueError("identity supervision is training-only")
        annotation = sidecar["worlds"][world["world_id"]]
        entities = [key[0] for key in annotation["event_keys"]]
        if len(entities) != len(world["events"]): raise ValueError("entity annotation mismatch")
        event_roles = [entity_role_labels(e["text"], [entity], tokenizer)[0] for e, entity in zip(world["events"], entities)]
        query_roles, query_entities = [], []
        for query in world["queries"]:
            role, mentioned = entity_role_labels(query["text"], annotation["entities"], tokenizer)
            query_roles.append(role); query_entities.append(mentioned)
        if any(len(a) != len(b) for a, b in zip(event_roles + query_roles, feature["events"] + feature["queries"])):
            raise ValueError("role/token geometry mismatch")
        output.append({"events": event_roles, "queries": query_roles, "event_entities": entities, "query_entities": query_entities})
    return output


def identity_loss(model, data, labels, indices):
    ex, em, qx, qm, counts, mask, positions, target = data
    state = model.pack_state(ex, em, counts, mask)
    scores, _, component = model.read_components(qx, qm, state, mask, positions)
    read_target = torch.zeros_like(mask, dtype=torch.float32)
    link_target = torch.zeros_like(component["entity_links"])
    memory_roles = torch.zeros_like(component["role_memory"])
    query_roles = torch.zeros_like(component["role_query"])
    for row, (wi, qi) in enumerate(indices):
        annotation = labels[wi]; entities = annotation["event_entities"]
        read_target[row, :len(entities)] = torch.tensor([e in annotation["query_entities"][qi] for e in entities], device=mask.device)
        link_target[row, :len(entities), :len(entities)] = torch.tensor([[a == b for b in entities] for a in entities], device=mask.device)
        for i, role in enumerate(annotation["events"]): memory_roles[row, i, :len(role)] = role.to(mask.device)
        role = annotation["queries"][qi]; query_roles[row, :len(role)] = role.to(mask.device)
    pairs = mask[:, :, None] & mask[:, None, :]
    pairs &= ~torch.eye(mask.shape[1], device=mask.device, dtype=torch.bool)[None]
    return (balanced_binary_loss(scores, target, mask)
            + balanced_binary_loss(component["entity_read"], read_target, mask)
            + balanced_binary_loss(component["entity_links"].flatten(1), link_target.flatten(1), pairs.flatten(1))
            + balanced_binary_loss(component["role_memory"].flatten(1), memory_roles.flatten(1), (state[..., -1] > .5).flatten(1))
            + balanced_binary_loss(component["role_query"], query_roles, qm))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gguf"); p.add_argument("data"); p.add_argument("feature_cache"); p.add_argument("baseline"); p.add_argument("output")
    p.add_argument("--lib", required=True); p.add_argument("--tok-probe", required=True)
    p.add_argument("--seed", type=int, default=2810916); p.add_argument("--steps", type=int, default=1000); p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.steps < 1: p.error("steps must be positive")
    if not 0 <= a.seed < 2 ** 63: p.error("invalid training seed")
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    source = Path(a.data); train = load_worlds(source / "train.jsonl", "train"); valid = load_worlds(source / "valid.jsonl", "valid")
    old = torch.load(a.baseline, map_location="cpu", weights_only=True)
    cache = torch.load(a.feature_cache, map_location="cpu", weights_only=True)
    if cache.get("format") != "RESIDENT_MEMORY_SET_RESEARCH_V1": raise ValueError("training feature cache required")
    binding = {"backbone_sha256": file_fingerprint(a.gguf), "train_sha256": file_fingerprint(source / "train.jsonl"),
               "valid_sha256": file_fingerprint(source / "valid.jsonl")}
    for key, value in binding.items():
        if old[key] != value or cache[key] != value: raise ValueError("identity training provenance mismatch: " + key)
    if old["configuration"]["architecture"] != "factorized": raise ValueError("factorized baseline required")
    sidecar = json.loads((source / "train_supervision.json").read_text())
    if sidecar["train_sha256"] != binding["train_sha256"] or not sidecar["training_only"]: raise ValueError("supervision identity mismatch")
    tokenizer = CTokenizer(a.tok_probe, a.gguf); weights = GGUFWeights(a.gguf, a.lib)
    try:
        tx = append_lexical_features(train, cache["train"], weights, tokenizer)
        vx = append_lexical_features(valid, cache["valid"], weights, tokenizer)
        labels = prepare_identity_labels(train, tx, sidecar, tokenizer)
    finally:
        weights.close(); tokenizer._proc.terminate(); tokenizer._proc.wait()
    torch.manual_seed(a.seed)
    model = IdentityResidentMemorySet(tx[0]["events"][0].shape[-1] // 2).to(a.device)
    model.baseline.load_state_dict(old["state_dict"])
    configuration = {**vars(a), "architecture": "identity", "initial_order_scale": .025}
    metadata = {**binding, "format": "RESIDENT_IDENTITY_RESEARCH_V1", "configuration": configuration,
                "baseline_sha256": file_fingerprint(a.baseline), "identity_role_supervision_train_only": True,
                "context_feature_cache_sha256": file_fingerprint(a.feature_cache),
                "feature_layout": "normalized_final_hidden|normalized_input_embedding",
                "oracle_write_boundaries": True, "automatic_memory_accuracy": False, "test_opened": False,
                "kv_reuse": False, "lora": False, "locomo_used": False, "deployment_changed": False,
                "baseline_frozen": True, "not_deployable": True}
    optimizer = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=.001, weight_decay=.01)
    pools = [[(wi, qi) for wi, w in enumerate(train) for qi, q in enumerate(w["queries"]) if q["kind"] == kind]
             for kind in ("current", "multi", "history", "null")]
    best, _ = evaluate(model, valid, vx, a.device); state = clone_state_dict(model); selected = 0
    for step in range(1, a.steps + 1):
        model.train(); indices = [pool[i] for pool in pools for i in torch.randint(len(pool), (8,)).tolist()]
        loss = identity_loss(model, collate(train, tx, indices, a.device), labels, indices)
        if not torch.isfinite(loss): raise ValueError("nonfinite identity loss")
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad], 1.); optimizer.step()
        if step % 100 == 0 or step == a.steps:
            metrics, _ = evaluate(model, valid, vx, a.device)
            if (metrics["worst_group_accuracy"], metrics["correct"]) > (best["worst_group_accuracy"], best["correct"]):
                best, state, selected = metrics, clone_state_dict(model), step
            print(json.dumps({"phase": "identity_valid", "step": step, **metrics}), flush=True)
    final, _ = evaluate(model, valid, vx, a.device)
    torch.save({**metadata, "step": a.steps, "state_dict": clone_state_dict(model)}, root / "final_research.pt")
    model.load_state_dict(state)
    resident = compile_states(model, vx, a.device); torch.save(resident, root / "valid_resident_state.pt")
    restored = torch.load(root / "valid_resident_state.pt", map_location="cpu", weights_only=True)
    if any(not torch.equal(a, b) for a, b in zip(resident, restored)): raise AssertionError("identity state reload changed tensors")
    replay, cases = evaluate(model, valid, vx, a.device, restored)
    if replay != best: raise AssertionError("identity state reload changed results")
    baseline_metrics, baseline_cases = evaluate(model.baseline, valid, cache["valid"], a.device)
    if [len(row["selected"]) for row in cases] != [len(row["selected"]) for row in baseline_cases]:
        raise AssertionError("identity-only change altered the frozen count output")
    if any(not torch.equal(value.cpu(), old["state_dict"][key]) for key, value in model.baseline.state_dict().items()): raise AssertionError("frozen baseline changed")
    torch.save({**metadata, "selected_step": selected, "state_dict": state}, root / "selected_research.pt")
    (root / "selected_cases.json").write_text(json.dumps(cases) + "\n")
    summary = {**metadata, "selected_step": selected, "selected": best, "final": final,
               "baseline_weights_unchanged": True, "resident_state_reload_equal": True,
               "baseline_counts_unchanged": True, "baseline_reference": baseline_metrics,
               "eligible_for_writer_integration": selected > 0 and best["null_false_activations"] == 0 and best["worst_group_accuracy"] >= .8}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("IDENTITY_TRAINING_DONE", flush=True)


if __name__ == "__main__": main()
