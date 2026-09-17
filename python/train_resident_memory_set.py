#!/usr/bin/env python3
"""Jointly train resident event addresses and multi-event neural activation.

This is an oracle-write-boundary component experiment, NOT automatic memory
or end-to-end QA. Query-time forward receives tensors, never event text, gold
fields, active flags, answer strings, or gold target counts.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from c_tokenizer import CTokenizer
from ggw import GGUFWeights
from torch_backbone import TorchBackbone
from typed_memory_training import clone_state_dict, file_fingerprint, reject_evaluation_row

MAX_TARGETS = 4
DEFAULT_TRAINING_SEED = 2810916


def load_worlds(path, split):
    worlds = []
    ids = set()
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        reject_evaluation_row(row)
        if row.get("locomo_used"): raise ValueError("LoCoMo cannot enter training")
        if row["split"] != split or row["world_id"] in ids:
            raise ValueError("wrong split or duplicate world")
        ids.add(row["world_id"])
        if not 1 <= len(row["events"]) <= 32: raise ValueError("invalid memory size")
        for e in row["events"]:
            if not 0 <= e["value_start"] < e["value_end"] <= len(e["text"].encode()):
                raise ValueError("invalid diagnostic pointer bounds")
            if e["text"].encode()[e["value_start"]:e["value_end"]].decode() != e["value"]:
                raise ValueError("invalid diagnostic pointer boundary")
        for q in row["queries"]:
            target = q["targets"]
            if len(target) > MAX_TARGETS or len(set(target)) != len(target) or any(i < 0 or i >= len(row["events"]) for i in target):
                raise ValueError("invalid support set")
            if q["answers"] != [row["events"][i]["value"] for i in sorted(target)]:
                raise ValueError("answer does not match support pointers")
        worlds.append(row)
    if not worlds: raise ValueError("empty curriculum")
    return worlds


@torch.inference_mode()
def encode_worlds(worlds, backbone, tokenizer):
    cache = {}
    def encode(text):
        if text not in cache:
            ids = tokenizer.encode(text, True)
            if not 1 < len(ids) <= 128: raise ValueError("token limit; no silent truncation")
            hidden = backbone(torch.tensor(ids, device=backbone.device), return_hidden=True)
            cache[text] = F.normalize(hidden.float(), dim=-1).half().cpu()
        return cache[text]
    output = []
    for i, w in enumerate(worlds):
        output.append({"events": [encode(e["text"]) for e in w["events"]],
                       "queries": [encode(q["text"]) for q in w["queries"]]})
        if (i + 1) % 16 == 0:
            print(json.dumps({"phase": "resident_features", "worlds": i + 1, "total": len(worlds)}), flush=True)
    return output


def pad_tokens(values, device):
    length = max(len(v) for v in values)
    x = torch.zeros(len(values), length, values[0].shape[-1], device=device)
    mask = torch.zeros(len(values), length, device=device, dtype=torch.bool)
    for i, value in enumerate(values):
        x[i, :len(value)] = value.to(device)
        mask[i, 1:len(value)] = True  # BOS is not a source fact.
    return x, mask


def collate(worlds, features, indices, device):
    event_values, query_values, counts = [], [], []
    for wi, qi in indices:
        event_values.extend(features[wi]["events"])
        counts.append(len(features[wi]["events"]))
        query_values.append(features[wi]["queries"][qi])
    ex, em = pad_tokens(event_values, device); qx, qm = pad_tokens(query_values, device)
    maximum = max(counts)
    mask = torch.arange(maximum, device=device)[None, :] < torch.tensor(counts, device=device)[:, None]
    positions = torch.arange(maximum, device=device)[None, :].expand(len(indices), -1).float()
    positions = positions / (torch.tensor(counts, device=device)[:, None] - 1).clamp_min(1)
    targets = torch.zeros_like(mask, dtype=torch.float32)
    for i, (wi, qi) in enumerate(indices): targets[i, worlds[wi]["queries"][qi]["targets"]] = 1
    return ex, em, qx, qm, counts, mask, positions, targets


class ResidentMemorySet(nn.Module):
    def __init__(self, hidden, width=128, initial_order_scale=1.):
        super().__init__()
        self.project = nn.Linear(hidden, width)
        self.write_pool = nn.Parameter(torch.randn(width) * .02)
        self.read_pool = nn.Parameter(torch.randn(4, width) * .02)
        self.order = nn.Linear(2, width, bias=False)
        if not 0 <= initial_order_scale <= 1: raise ValueError("invalid order initialization scale")
        with torch.no_grad(): self.order.weight.mul_(initial_order_scale)
        self.context = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=.1,
                                                 batch_first=True, norm_first=True)
        self.pair = nn.Sequential(nn.Linear(4 * width, width), nn.GELU())
        self.score = nn.Linear(width, 1)
        self.count = nn.Sequential(nn.Linear(3 * width + 1, width), nn.GELU(), nn.Linear(width, MAX_TARGETS + 1))
        nn.init.zeros_(self.count[-1].weight)
        with torch.no_grad():
            self.count[-1].bias.fill_(-2); self.count[-1].bias[0] = 2

    def write(self, tokens, token_mask):
        x = F.silu(self.project(tokens))
        weights = (x @ self.write_pool).masked_fill(~token_mask, -torch.inf).softmax(-1)
        return F.normalize(torch.einsum("bt,btd->bd", weights, x), dim=-1)

    def read(self, query, query_mask, state, slot_mask, positions):
        x = F.silu(self.project(query))
        weights = torch.einsum("btd,hd->bht", x, self.read_pool)
        weights = weights.masked_fill(~query_mask[:, None, :], -torch.inf).softmax(-1)
        q = F.normalize(torch.einsum("bht,btd->bhd", weights, x), dim=-1)
        # Slots remain linked to immutable evidence. This contextualizes resident
        # addresses without rereading raw events or placing text in the prompt.
        safe_mask = slot_mask.clone()
        safe_mask[~safe_mask.any(-1), 0] = True
        memory = state + self.order(torch.stack((positions, 1 - positions), dim=-1))
        memory = self.context(memory, src_key_padding_mask=~safe_mask)
        memory = F.normalize(memory, dim=-1)
        a = q[:, :, None, :].expand(-1, -1, state.shape[1], -1)
        b = memory[:, None, :, :].expand(-1, q.shape[1], -1, -1)
        features = self.pair(torch.cat((a, b, a * b, (a - b).abs()), dim=-1))
        return self.activation_outputs(features, q, slot_mask)

    def activation_outputs(self, features, q, slot_mask):
        logits = torch.logsumexp(self.score(features).squeeze(-1), dim=1) - math.log(q.shape[1])
        logits = logits.masked_fill(~slot_mask, -1e4)
        slot_features = features.max(1).values
        mean = (slot_features * slot_mask[..., None]).sum(1) / slot_mask.sum(1).clamp_min(1)[:, None]
        maximum = slot_features.masked_fill(~slot_mask[..., None], -1e4).max(1).values
        maximum = torch.where(slot_mask.any(1)[:, None], maximum, torch.zeros_like(maximum))
        counts = self.count(torch.cat((mean, maximum, q.mean(1), slot_mask.sum(1)[:, None] / 32.), dim=-1))
        return logits, counts

    def pack_state(self, ex, em, counts, slot_mask):
        addresses = self.write(ex, em)
        state = torch.zeros((len(counts), slot_mask.shape[1], *addresses.shape[1:]), device=addresses.device)
        offset = 0
        for i, count in enumerate(counts):
            state[i, :count] = addresses[offset:offset + count]; offset += count
        return state

    def forward(self, ex, em, qx, qm, counts, slot_mask, positions):
        return self.read(qx, qm, self.pack_state(ex, em, counts, slot_mask), slot_mask, positions)


class TokenResidentMemorySet(ResidentMemorySet):
    """Store learned token addresses, not raw text or backbone attention K/V."""
    def __init__(self, hidden, width=128, initial_order_scale=1.):
        super().__init__(hidden, width, initial_order_scale)
        del self.write_pool

    def write(self, tokens, token_mask):
        vectors = F.normalize(F.silu(self.project(tokens)), dim=-1)
        valid = token_mask[..., None].to(vectors.dtype)
        # Last channel is storage geometry, not a learned or gold fact label.
        return torch.cat((vectors * valid, valid), dim=-1)

    def match_features(self, query, query_mask, state, slot_mask, positions, contextualize=True):
        vectors, stored_mask = state[..., :-1], state[..., -1] > .5
        stored_mask = stored_mask & slot_mask[..., None]
        means = (vectors * stored_mask[..., None]).sum(2) / stored_mask.sum(2).clamp_min(1)[..., None]
        safe_slots = slot_mask.clone(); safe_slots[~slot_mask.any(-1), 0] = True
        if contextualize:
            contextual = means + self.order(torch.stack((positions, 1 - positions), dim=-1))
            contextual = F.normalize(self.context(contextual, src_key_padding_mask=~safe_slots), dim=-1)
            keys = F.normalize(vectors + .1 * contextual[:, :, None, :], dim=-1)
        else:
            keys = vectors
        qtokens = F.normalize(F.silu(self.project(query)), dim=-1)
        affinity = torch.einsum("bqd,bntd->bnqt", qtokens, keys) * 8.
        safe_tokens = stored_mask.clone()
        safe_tokens[..., 0] |= ~safe_tokens.any(-1)
        attention = affinity.masked_fill(~safe_tokens[:, :, None, :], -torch.inf).softmax(-1)
        attended = torch.einsum("bnqt,bntd->bnqd", attention, keys)
        qexpanded = qtokens[:, None, :, :].expand(-1, state.shape[1], -1, -1)
        pair = self.pair(torch.cat((qexpanded, attended, qexpanded * attended,
                                    (qexpanded - attended).abs()), dim=-1))
        weights = torch.einsum("bqd,hd->bhq", qtokens, self.read_pool)
        weights = weights.masked_fill(~query_mask[:, None, :], -torch.inf).softmax(-1)
        features = torch.einsum("bhq,bnqd->bhnd", weights, pair)
        q = F.normalize(torch.einsum("bhq,bqd->bhd", weights, qtokens), dim=-1)
        return features, q, qtokens

    def read(self, query, query_mask, state, slot_mask, positions):
        features, q, _ = self.match_features(query, query_mask, state, slot_mask, positions)
        return self.activation_outputs(features, q, slot_mask)


def version_log_gate(link_logits, history_logits, slot_mask, positions):
    """Keep all versions for history, otherwise suppress learned older matches.

    Chronology is ordinary write metadata. Which events share an address and
    whether the query asks for history are predicted, never supplied as labels.
    """
    later = (positions[:, None, :] > positions[:, :, None])
    later = later & slot_mask[:, :, None] & slot_mask[:, None, :]
    current = (F.logsigmoid(-link_logits) * later).sum(-1)
    return torch.logaddexp(F.logsigmoid(history_logits)[:, None],
                           F.logsigmoid(-history_logits)[:, None] + current)


class FactorizedResidentMemorySet(TokenResidentMemorySet):
    """Learn fact membership separately from version links and query scope."""
    def __init__(self, hidden, width=128, initial_order_scale=1.):
        super().__init__(hidden, width, initial_order_scale)
        # Content matching must not silently prefer a position. Chronology is
        # used only by the explicit, learned-link version gate below.
        del self.order, self.context
        self.link_pair = nn.Sequential(nn.Linear(4 * width, width), nn.GELU())
        self.link_score = nn.Linear(2 * width, 1)
        self.scope = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))

    def links(self, state, slot_mask):
        vectors = state[..., :-1]
        mask = (state[..., -1] > .5) & slot_mask[..., None]
        safe = mask.clone(); safe[..., 0] |= ~safe.any(-1)
        affinity = torch.einsum("bitd,bjud->bijtu", vectors, vectors) * 8.
        weights = affinity.masked_fill(~safe[:, None, :, None, :], -torch.inf).softmax(-1)
        attended = torch.einsum("bijtu,bjud->bijtd", weights, vectors)
        source = vectors[:, :, None, :, :].expand_as(attended)
        pairs = self.link_pair(torch.cat((source, attended, source * attended,
                                         (source - attended).abs()), dim=-1))
        mean = (pairs * mask[:, :, None, :, None]).sum(3) / mask.sum(-1).clamp_min(1)[:, :, None, None]
        maximum = pairs.masked_fill(~mask[:, :, None, :, None], -1e4).max(3).values
        maximum = torch.where(mask.any(-1)[:, :, None, None], maximum, torch.zeros_like(maximum))
        logits = self.link_score(torch.cat((mean, maximum), dim=-1)).squeeze(-1)
        return (logits + logits.transpose(1, 2)) * .5

    def read_components(self, query, query_mask, state, slot_mask, positions):
        features, q, tokens = self.match_features(query, query_mask, state, slot_mask, positions, False)
        address, counts = self.activation_outputs(features, q, slot_mask)
        mean = (tokens * query_mask[..., None]).sum(1) / query_mask.sum(1).clamp_min(1)[:, None]
        maximum = tokens.masked_fill(~query_mask[..., None], -1e4).max(1).values
        history = self.scope(torch.cat((mean, maximum), -1)).squeeze(-1)
        links = self.links(state, slot_mask)
        log_gate = version_log_gate(links, history, slot_mask, positions)
        log_probability = (F.logsigmoid(address) + log_gate).clamp_max(-1e-6)
        scores = log_probability - torch.log(-torch.expm1(log_probability))
        return scores.masked_fill(~slot_mask, -1e4), counts, {
            "address": address, "links": links, "history": history}

    def read(self, query, query_mask, state, slot_mask, positions):
        scores, counts, _ = self.read_components(query, query_mask, state, slot_mask, positions)
        return scores, counts


def activation_count_scores(scores, slot_mask, cardinality_prior):
    """Score the best activated set of each size; no independent query classifier.

    E(S) = sum(event activation logits in S) + learned shared prior[|S|].
    For a given size, its maximum-energy set is the highest-scoring valid slots.
    Count supervision therefore also trains activation boundaries, rather than
    asking a separate classifier to guess an answer count from query wording.
    """
    maximum = cardinality_prior.numel() - 1
    ordered = scores.masked_fill(~slot_mask, -1e4).sort(dim=-1, descending=True).values
    ordered = F.pad(ordered, (0, max(0, maximum - scores.shape[1])), value=-1e4)[:, :maximum]
    prefix = torch.cat((scores.new_zeros(len(scores), 1), ordered.cumsum(-1)), -1)
    energies = prefix + cardinality_prior[None]
    valid = torch.arange(maximum + 1, device=scores.device)[None] <= slot_mask.sum(-1)[:, None]
    return energies.masked_fill(~valid, -1e4)


class ActivationCountResidentMemorySet(FactorizedResidentMemorySet):
    def __init__(self, hidden, width=128, initial_order_scale=1.):
        super().__init__(hidden, width, initial_order_scale)
        self.cardinality_prior = nn.Parameter(self.count[-1].bias.detach().clone())
        del self.count

    def activation_outputs(self, features, q, slot_mask):
        scores = torch.logsumexp(self.score(features).squeeze(-1), dim=1) - math.log(q.shape[1])
        # Final counts are derived after version gating, not from fact address
        # scores before stale versions have been suppressed.
        return scores.masked_fill(~slot_mask, -1e4), scores.new_zeros(len(scores), MAX_TARGETS + 1)

    def read_components(self, query, query_mask, state, slot_mask, positions):
        scores, _, components = super().read_components(query, query_mask, state, slot_mask, positions)
        return scores, activation_count_scores(scores, slot_mask, self.cardinality_prior), components


def training_version_groups(world):
    """Recover complete synthetic annotation groups, never inference inputs."""
    groups = [{i} for i in range(len(world["events"]))]
    for question in world["queries"]:
        if question["kind"] != "history": continue
        group = set(question["targets"])
        if len(group) != 2: raise ValueError("version supervision requires two-version history")
        for i in group:
            if groups[i] != {i} and groups[i] != group:
                raise ValueError("overlapping synthetic version groups")
            groups[i] = group
    return groups


def counterfactual_memory_batch(data, worlds, indices, first_augmented_row):
    """Training-only alternative observed memories, not delete/forget actions.

    Remove whole fact groups with probability 1/2, independently of the query.
    Both versions of a fact are removed together: surviving update text cannot
    expose an otherwise removed historical answer. Query features never change.
    The slot mask describes available storage, not which stored facts are true.
    Gold annotations determine training labels only, not a reader decision.
    """
    if not 0 <= first_augmented_row <= len(indices): raise ValueError("invalid paired batch boundary")
    mask, targets = data[-3].clone(), data[-1].clone()
    for row in range(first_augmented_row, len(indices)):
        world = worlds[indices[row][0]]
        reject_evaluation_row(world)
        if world["split"] != "train" or world.get("locomo_used"):
            raise ValueError("counterfactual augmentation is training-only")
        groups = sorted({tuple(sorted(group)) for group in training_version_groups(world)})
        keep = torch.rand(len(groups), device=mask.device) >= .5
        for group, retained in zip(groups, keep.tolist()):
            if not retained: mask[row, list(group)] = False
        targets[row] *= mask[row]
    return (*data[:-3], mask, data[-2], targets)


def distractor_donors(worlds, supervision_path, train_sha256):
    """Build training-only donor pools, excluding every recipient-world entity.

    Excluding the absent query entity as well is essential: a NULL question must
    not acquire a valid answer through augmentation. No annotation is a model
    input, and no development/test world can donate data.
    """
    annotation = json.loads(Path(supervision_path).read_text())
    if (annotation.get("format") != "RESIDENT_DISTRACTOR_SUPERVISION_V1"
            or annotation.get("training_only") is not True or annotation.get("train_sha256") != train_sha256):
        raise ValueError("distractor supervision identity mismatch")
    metadata = []
    for world in worlds:
        reject_evaluation_row(world)
        if world["split"] != "train" or world.get("locomo_used"):
            raise ValueError("distractor augmentation is training-only")
        meta = annotation["worlds"][world["world_id"]]
        if len(meta["entities"]) != 4 or len(set(meta["entities"])) != 4 or len(meta["event_keys"]) != len(world["events"]):
            raise ValueError("invalid distractor annotation geometry")
        if any(len(key) != 2 or key[0] not in meta["entities"] or not key[1] for key in meta["event_keys"]):
            raise ValueError("invalid annotated event key")
        for group in training_version_groups(world):
            if len({tuple(meta["event_keys"][i]) for i in group}) != 1:
                raise ValueError("inconsistent version/entity annotation")
        metadata.append(meta)
    donors = []
    for world, recipient in zip(worlds, metadata):
        candidates = []
        recipient_values = {e["value"] for e in world["events"]}
        recipient_relations = {key[1] for key in recipient["event_keys"]}
        for di, donor in enumerate(metadata):
            keep = [i for i, key in enumerate(donor["event_keys"]) if key[0] not in recipient["entities"]]
            related = recipient_relations.intersection(donor["event_keys"][i][1] for i in keep)
            if keep and related and len(keep) + len(world["events"]) <= 32 and not recipient_values.intersection(worlds[di]["events"][i]["value"] for i in keep):
                candidates.append((di, keep))
        if not candidates: raise ValueError("no unrelated donor available")
        donors.append(candidates)
    return donors


def distractor_pair_batch(worlds, features, indices, donors, device):
    """Pair each question with an identical question plus unrelated resident facts.

    Existing events and support indices retain their order. Appended donor
    versions stay together and receive annotation-only link supervision.
    """
    augmented_worlds, augmented_features, augmented_indices = [], [], []
    for wi, qi in indices:
        world = worlds[wi]
        reject_evaluation_row(world)
        if world["split"] != "train" or world.get("locomo_used"):
            raise ValueError("distractor augmentation is training-only")
        di, keep = donors[wi][int(torch.randint(len(donors[wi]), (1,)))]
        donor = worlds[di]
        reject_evaluation_row(donor)
        if donor["split"] != "train" or donor.get("locomo_used"):
            raise ValueError("distractor donor is not training data")
        offset = len(world["events"]); remap = {old: offset + j for j, old in enumerate(keep)}
        labels = []
        for query in donor["queries"]:
            if query["kind"] == "history" and any(i in remap for i in query["targets"]):
                if not all(i in remap for i in query["targets"]): raise ValueError("partial donor version group")
                # Never sampled as a question: these rows supervise version links only.
                labels.append({"kind": "history", "targets": [remap[i] for i in query["targets"]], "supervision_only": True})
        augmented_indices.append((len(worlds) + len(augmented_worlds), qi))
        extra_events = [{**donor["events"][i], "source_id": donor["world_id"] + ":" + donor["events"][i]["source_id"]} for i in keep]
        augmented_worlds.append({**world, "world_id": world["world_id"] + ":distractor:" + donor["world_id"],
                                 "events": world["events"] + extra_events,
                                 "queries": world["queries"] + labels})
        augmented_features.append({"events": features[wi]["events"] + [features[di]["events"][i] for i in keep],
                                   "queries": features[wi]["queries"]})
    batch_worlds = worlds + augmented_worlds; batch_features = features + augmented_features
    batch_indices = indices + augmented_indices
    return collate(batch_worlds, batch_features, batch_indices, device), batch_worlds, batch_indices


def paired_activation_consistency(scores, counts, mask, pairs):
    if len(scores) != 2 * pairs or pairs < 1: raise ValueError("invalid invariance pair count")
    common = mask[:pairs]
    if not (mask[pairs:] | ~common).all(): raise ValueError("a paired store removed original facts")
    probability_delta = (scores[:pairs].sigmoid() - scores[pairs:].sigmoid()).square()
    event_loss = (probability_delta * common).sum() / common.sum().clamp_min(1)
    log_a, log_b = counts[:pairs].log_softmax(-1), counts[pairs:].log_softmax(-1)
    log_mean = torch.logaddexp(log_a, log_b) - math.log(2.)
    count_loss = .5 * ((log_a.exp() * (log_a - log_mean)).sum(-1)
                       + (log_b.exp() * (log_b - log_mean)).sum(-1)).mean()
    return event_loss + count_loss


def factorized_labels(worlds, indices, maximum, device):
    """Synthetic supervision only; never called by write/read or evaluation.

    This curriculum enumerates every updated fact in a two-version history
    question. Its support annotations therefore define complete version groups.
    Do not apply this label derivation to arbitrary dialogue datasets.
    """
    address = torch.zeros(len(indices), maximum, device=device)
    links = torch.zeros(len(indices), maximum, maximum, device=device)
    history = torch.zeros(len(indices), device=device)
    for row, (wi, qi) in enumerate(indices):
        world = worlds[wi]
        groups = training_version_groups(world)
        for i, group in enumerate(groups): links[row, i, list(group)] = 1
        query = world["queries"][qi]
        for i in query["targets"]: address[row, list(groups[i])] = 1
        history[row] = query["kind"] == "history"
    return address, links, history


def balanced_binary_loss(logits, target, mask):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pos = target.bool() & mask; neg = ~target.bool() & mask
    return ((loss * pos).sum(-1) / pos.sum(-1).clamp_min(1)
            + (loss * neg).sum(-1) / neg.sum(-1).clamp_min(1)).mean()


def training_loss(model, data, worlds, indices, ranking_weight=0., invariance_pairs=0, invariance_weight=1.):
    if not isinstance(model, FactorizedResidentMemorySet):
        scores, counts = model(*data[:-1])
        return support_loss(scores, counts, data[-1], data[-3], ranking_weight)
    ex, em, qx, qm, counts, mask, positions, targets = data
    state = model.pack_state(ex, em, counts, mask)
    scores, counts, components = model.read_components(qx, qm, state, mask, positions)
    address, links, history = factorized_labels(worlds, indices, mask.shape[1], mask.device)
    pair_mask = mask[:, :, None] & mask[:, None, :]
    pair_mask &= ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
    loss = (support_loss(scores, counts, targets, mask, ranking_weight)
            + balanced_binary_loss(components["address"], address, mask)
            + balanced_binary_loss(components["links"].flatten(1), links.flatten(1), pair_mask.flatten(1))
            + F.binary_cross_entropy_with_logits(components["history"], history))
    if invariance_pairs:
        loss = loss + invariance_weight * paired_activation_consistency(scores, counts, mask, invariance_pairs)
    return loss


def create_model(hidden, width=128, initial_order_scale=1., architecture="pooled"):
    if architecture == "identity":
        from train_resident_identity import IdentityResidentMemorySet
        if hidden % 2: raise ValueError("identity features require equal contextual and lexical widths")
        return IdentityResidentMemorySet(hidden // 2, width, initial_order_scale)
    classes = {"pooled": ResidentMemorySet, "token": TokenResidentMemorySet,
               "factorized": FactorizedResidentMemorySet, "activation_count": ActivationCountResidentMemorySet}
    if architecture not in classes:
        raise ValueError("unsupported resident-memory architecture")
    return classes[architecture](hidden, width, initial_order_scale)


def initialize_training_model(hidden, architecture, initial_order_scale, seed=DEFAULT_TRAINING_SEED, device="cpu"):
    """Seed model initialization and training sampling, never the frozen corpus."""
    if not 0 <= seed < 2 ** 63: raise ValueError("training seed must be in [0, 2**63)")
    torch.manual_seed(seed)
    return create_model(hidden, initial_order_scale=initial_order_scale, architecture=architecture).to(device)


def select_sets(logits, counts, mask):
    result = torch.zeros_like(mask)
    for i, k in enumerate(counts.argmax(-1).tolist()):
        k = min(k, int(mask[i].sum()))
        if k: result[i, logits[i].masked_fill(~mask[i], -torch.inf).topk(k).indices] = True
    return result


def pairwise_binding_loss(logits, target, mask):
    """Within-question positive/negative contrast, independent of target count."""
    positive = target.bool() & mask
    negative = ~target.bool() & mask
    pairs = positive[:, :, None] & negative[:, None, :]
    # Axis 1 is the correct event, axis 2 is an incorrect event.
    cost = F.softplus(1. + logits[:, None, :] - logits[:, :, None])
    per_row = (cost * pairs).sum((1, 2)) / pairs.sum((1, 2)).clamp_min(1)
    return per_row.mean()


def support_loss(logits, counts, target, mask, ranking_weight=0.):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pos = target.bool() & mask; neg = ~target.bool() & mask
    balanced = (loss * pos).sum(1) / pos.sum(1).clamp_min(1) + (loss * neg).sum(1) / neg.sum(1).clamp_min(1)
    return (balanced.mean() + F.cross_entropy(counts, target.sum(1).long())
            + ranking_weight * pairwise_binding_loss(logits, target, mask))


@torch.inference_mode()
def compile_states(model, features, device):
    """Write once per world; future queries only receive these resident slots."""
    model.eval()
    states = []
    for world in features:
        tokens, mask = pad_tokens(world["events"], device)
        states.append(model.write(tokens, mask).cpu())
    return states


def read_batch(features, states, indices, device):
    qx, qm = pad_tokens([features[wi]["queries"][qi] for wi, qi in indices], device)
    counts = torch.tensor([len(states[wi]) for wi, _ in indices], device=device)
    maximum = max(1, int(counts.max()))
    mask = torch.arange(maximum, device=device)[None, :] < counts[:, None]
    positions = torch.arange(maximum, device=device)[None, :].expand(len(indices), -1).float()
    positions = positions / (counts - 1).clamp_min(1)[:, None]
    if states[0].ndim == 2:
        state = torch.zeros(len(indices), maximum, states[0].shape[-1], device=device)
        for i, (wi, _) in enumerate(indices): state[i, :len(states[wi])] = states[wi].to(device)
    else:
        token_count = max(states[wi].shape[1] for wi, _ in indices)
        state = torch.zeros(len(indices), maximum, token_count, states[0].shape[-1], device=device)
        for i, (wi, _) in enumerate(indices):
            state[i, :len(states[wi]), :states[wi].shape[1]] = states[wi].to(device)
    return qx, qm, state, mask, positions


@torch.inference_mode()
def evaluate(model, worlds, features, device, states=None):
    model.eval(); groups = {}; rows = []
    if states is None: states = compile_states(model, features, device)
    indices = [(wi, qi) for wi, w in enumerate(worlds) for qi in range(len(w["queries"]))]
    for start in range(0, len(indices), 32):
        items = indices[start:start + 32]
        data = read_batch(features, states, items, device)
        logits, counts = model.read(*data); pred = select_sets(logits, counts, data[-2])
        for i, (wi, qi) in enumerate(items):
            q = worlds[wi]["queries"][qi]; chosen = pred[i].nonzero().flatten().cpu().tolist()
            correct = chosen == sorted(q["targets"])
            for group in (q["kind"], f"cardinality_{len(q['targets'])}"):
                stat = groups.setdefault(group, {"correct": 0, "total": 0})
                stat["total"] += 1; stat["correct"] += correct
            # Values are copied from diagnostic oracle spans, never generated.
            answers = [worlds[wi]["events"][j]["value"] for j in chosen]
            rows.append({"world_id": worlds[wi]["world_id"], "kind": q["kind"], "query": q["text"],
                         "targets": q["targets"], "selected": chosen, "correct": correct,
                         "pointer_answers": answers, "oracle_pointer_exact": answers == q["answers"]})
    for s in groups.values(): s["accuracy"] = s["correct"] / s["total"]
    return {"groups": groups, "correct": sum(r["correct"] for r in rows), "total": len(rows),
            "worst_group_accuracy": min(s["accuracy"] for s in groups.values()),
            "null_false_activations": groups["null"]["total"] - groups["null"]["correct"]}, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("gguf", "data", "output"): p.add_argument(name)
    p.add_argument("--lib", required=True); p.add_argument("--tok-probe", required=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=DEFAULT_TRAINING_SEED,
                   help="model and sampling seed; does not change the corpus")
    p.add_argument("--initial-order-scale", type=float, default=1.)
    p.add_argument("--feature-cache")
    p.add_argument("--ranking-loss-weight", type=float, default=0.)
    p.add_argument("--counterfactual-memory", action="store_true")
    p.add_argument("--distractor-pairs", action="store_true")
    p.add_argument("--invariance-weight", type=float, default=1.)
    p.add_argument("--architecture", choices=("pooled", "token", "factorized", "activation_count"), default="pooled")
    a = p.parse_args()
    if a.steps < 1: p.error("steps must be positive")
    if not 0 <= a.seed < 2 ** 63: p.error("seed must be in [0, 2**63)")
    if a.counterfactual_memory and a.distractor_pairs: p.error("select one augmentation at a time")
    if not math.isfinite(a.invariance_weight) or a.invariance_weight < 0: p.error("invalid invariance weight")
    if (a.counterfactual_memory or a.distractor_pairs) and a.architecture not in ("factorized", "activation_count"):
        p.error("counterfactual memory requires the factorized reader")
    if not 0 <= a.initial_order_scale <= 1: p.error("order scale must be in [0, 1]")
    if not math.isfinite(a.ranking_loss_weight) or a.ranking_loss_weight < 0:
        p.error("ranking loss weight must be finite and nonnegative")
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    train = load_worlds(Path(a.data) / "train.jsonl", "train")
    valid = load_worlds(Path(a.data) / "valid.jsonl", "valid")
    if {w["world_id"] for w in train} & {w["world_id"] for w in valid}: raise ValueError("world overlap")
    if {e["text"] for w in train for e in w["events"]} & {e["text"] for w in valid for e in w["events"]}:
        raise ValueError("source overlap")
    binding = {"format": "RESIDENT_MEMORY_SET_RESEARCH_V1", "backbone_sha256": file_fingerprint(a.gguf),
               "train_sha256": file_fingerprint(Path(a.data) / "train.jsonl"),
               "valid_sha256": file_fingerprint(Path(a.data) / "valid.jsonl"), "configuration": vars(a),
               "oracle_write_boundaries": True, "automatic_memory_accuracy": False,
               "training_only_version_supervision": a.architecture in ("factorized", "activation_count"),
               "gold_version_links_at_read": False,
               "kv_reuse": False, "lora": False, "locomo_used": False, "test_opened": False}
    donors = None
    if a.distractor_pairs:
        supervision_path = Path(a.data) / "train_supervision.json"
        donors = distractor_donors(train, supervision_path, binding["train_sha256"])
        binding["training_distractor_supervision_sha256"] = file_fingerprint(supervision_path)
    print(json.dumps({"phase": "resident_data", "train_worlds": len(train), "valid_worlds": len(valid), **binding}), flush=True)
    if a.feature_cache:
        cached = torch.load(a.feature_cache, map_location="cpu", weights_only=True)
        for key in ("format", "backbone_sha256", "train_sha256", "valid_sha256"):
            if cached.get(key) != binding[key]: raise ValueError("feature cache identity mismatch: " + key)
        tx, vx = cached["train"], cached["valid"]
        for worlds, features in ((train, tx), (valid, vx)):
            if len(worlds) != len(features): raise ValueError("cached world count mismatch")
            for w, f in zip(worlds, features):
                if len(w["events"]) != len(f["events"]) or len(w["queries"]) != len(f["queries"]):
                    raise ValueError("cached world geometry mismatch")
        print(json.dumps({"phase": "resident_cache_reused", "train_worlds": len(tx), "valid_worlds": len(vx)}), flush=True)
    else:
        tokenizer = CTokenizer(a.tok_probe, a.gguf); weights = GGUFWeights(a.gguf, a.lib)
        backbone = TorchBackbone(weights, device=a.device, dtype=torch.float32).eval()
        try:
            tx = encode_worlds(train, backbone, tokenizer); vx = encode_worlds(valid, backbone, tokenizer)
        finally:
            del backbone; weights.close(); tokenizer._proc.terminate(); tokenizer._proc.wait()
    torch.save({**binding, "train": tx, "valid": vx}, root / "features.pt")
    if a.device == "cuda": torch.cuda.empty_cache()
    model = initialize_training_model(tx[0]["events"][0].shape[-1], a.architecture,
                                      a.initial_order_scale, a.seed, a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    pools = [[(wi, qi) for wi, w in enumerate(train) for qi, q in enumerate(w["queries"]) if q["kind"] == kind]
             for kind in ("current", "multi", "history", "null")]
    best, _ = evaluate(model, valid, vx, a.device); state = clone_state_dict(model); selected = 0
    for step in range(1, a.steps + 1):
        model.train()
        rows_per_pool = 4 if a.counterfactual_memory or a.distractor_pairs else 8
        indices = [pool[i] for pool in pools for i in torch.randint(len(pool), (rows_per_pool,)).tolist()]
        if a.counterfactual_memory: indices = indices + indices
        batch_worlds, batch_indices = train, indices
        if a.distractor_pairs:
            data, batch_worlds, batch_indices = distractor_pair_batch(train, tx, indices, donors, a.device)
        else:
            data = collate(train, tx, indices, a.device)
        if a.counterfactual_memory:
            data = counterfactual_memory_batch(data, train, indices, len(indices) // 2)
        loss = training_loss(model, data, batch_worlds, batch_indices, a.ranking_loss_weight,
                             len(indices) if a.distractor_pairs else 0, a.invariance_weight)
        if not torch.isfinite(loss): raise ValueError("nonfinite training loss")
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if step % 100 == 0 or step == a.steps:
            metrics, _ = evaluate(model, valid, vx, a.device)
            if (metrics["worst_group_accuracy"], metrics["correct"]) > (best["worst_group_accuracy"], best["correct"]):
                best, state, selected = metrics, clone_state_dict(model), step
            print(json.dumps({"phase": "resident_valid", "step": step, **metrics}), flush=True)
    final, final_rows = evaluate(model, valid, vx, a.device)
    train_metrics, _ = evaluate(model, train, tx, a.device)
    torch.save({**binding, "state_dict": clone_state_dict(model), "step": a.steps, "not_deployable": True}, root / "final_research.pt")
    model.load_state_dict(state)
    resident = compile_states(model, vx, a.device)
    torch.save({"format": "RESIDENT_EVENT_STATE_DIAGNOSTIC_V1", "backbone_sha256": binding["backbone_sha256"],
                "architecture": a.architecture,
                "states": resident, "world_ids": [w["world_id"] for w in valid]}, root / "valid_resident_state.pt")
    restored = torch.load(root / "valid_resident_state.pt", map_location="cpu", weights_only=True)["states"]
    if len(restored) != len(resident) or any(not torch.equal(a, b) for a, b in zip(resident, restored)):
        raise AssertionError("resident-state reload changed stored address tensors")
    restored_metrics, selected_rows = evaluate(model, valid, vx, a.device, restored)
    if restored_metrics != best: raise AssertionError("resident-state reload changed activation results")
    empty = [s[:0] for s in restored]
    empty_metrics, _ = evaluate(model, valid, vx, a.device, empty)
    torch.save({**binding, "state_dict": state, "selected_step": selected, "not_deployable": True}, root / "selected_research.pt")
    (root / "diagnostics.json").write_text(json.dumps({"selected": selected_rows, "final": final_rows}, indent=2) + "\n")
    summary = {**binding, "selected_step": selected, "selected": best, "final": final, "train": train_metrics,
               "resident_state_reload_equal": True, "empty_state_control": empty_metrics,
               "eligible_for_writer_integration": selected > 0 and best["null_false_activations"] == 0 and best["worst_group_accuracy"] >= .8,
               "deployment_changed": False}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"phase": "resident_done", **summary}), flush=True)


if __name__ == "__main__": main()
