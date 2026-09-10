#!/usr/bin/env python3
"""Formula-level regression tests for the Metis training implementation."""

import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from train_memory import (  # noqa: E402
    MemoryTrainer,
    MetisMemory,
    answer_token_slot_labels,
    autoregressive_supervision_length,
    balanced_token_selection_loss,
    balanced_truncated_svd,
    evidence_attention_loss,
    gated_delta_update,
    generation_query_evidence_map,
    memory_address_contrastive_loss,
    memory_address_transform,
    memory_key_diversity_loss,
    memory_read_preservation_loss,
    memory_target_slot_labels,
    memory_task_id,
    matching_prefix_length,
    ordered_episode_continuation,
    metis_rms_norm,
    orthogonalize_write_keys,
    parse_task_filter,
    retention_floor_loss,
    scheduled_task_weights,
    stale_answer_margin_loss,
    straight_through_binary_gate,
    straight_through_alpha_top_p,
    straight_through_top1,
    straight_through_topk,
    target_ids_by_evidence,
)
from prepare_memory_curriculum import long_memory_sample  # noqa: E402
from train_data import load_dataset  # noqa: E402


class TinyBackbone:
    def __init__(self):
        self.cfg = SimpleNamespace(
            hidden=8,
            kv_dim=4,
            q_dim=8,
            head_dim=4,
            n_heads=2,
            rms_eps=1e-5,
        )
        generator = torch.Generator().manual_seed(23)
        self.layers = [{
            "q": torch.randn(8, 8, generator=generator),
            "k": torch.randn(4, 8, generator=generator),
            "v": torch.randn(4, 8, generator=generator),
            "o": torch.eye(8),
            "attn_norm": torch.ones(8),
        }]

    def model_sha256(self):
        return bytes(range(32))


class QueryTimingMemory:
    def __init__(self):
        self.active = True
        self.prepared_tail = None
        self.discard_count = 0
        self.clear_count = 0

    def discard_captured(self):
        self.discard_count += 1

    def clear_query_read_override(self):
        self.clear_count += 1

    def prepare_query_read_override(self, tail_tokens):
        self.prepared_tail = tail_tokens

    def prepare_query_read_context(self, tail_tokens):
        self.prepared_tail = tail_tokens


class QueryTimingBackbone:
    def __init__(self, memory):
        self.memory = memory
        self.calls = []

    def __call__(self, tokens, **kwargs):
        self.calls.append((tokens.clone(), self.memory.active, kwargs))
        return tokens.float()


class ReadOnlyQueryMemory:
    def __init__(self):
        self.active = True
        self.state_mode = "delta"
        self.fusion_mode = "fixed"
        self.commit_count = 0
        self.pending_slot_labels = None
        self.pending_slot_label = -1
        self.commit_labels = []

    def reset_state(self):
        self.commit_count = 0
        self.commit_labels = []

    def set_pending_slot_labels(self, labels):
        self.pending_slot_labels = labels

    def set_pending_memory_id(self, memory_id):
        del memory_id

    def commit_all(self):
        self.commit_count += 1
        self.commit_labels.append(list(self.pending_slot_labels))

    def commit_all_grad_enabled(self):
        self.commit_count += 1
        self.commit_labels.append(list(self.pending_slot_labels))

    def discard_captured(self):
        pass

    def clear_query_read_override(self):
        pass


class ReadOnlyQueryBackbone:
    def __init__(self):
        self.cfg = SimpleNamespace(rms_eps=1e-5)
        self.device = torch.device("cpu")

    def __call__(self, tokens, logits_all=False, **_kwargs):
        if logits_all:
            return torch.zeros(tokens.shape[0], 16)
        return torch.zeros(tokens.shape[0], 8)


class ReadOnlyQueryTokenizer:
    def encode(self, text, add_bos):
        del add_bos
        if text == "target-a":
            return [1]
        if text == "target-b":
            return [2]
        if "payload-a" in text:
            return [3, 1, 4]
        if "payload-b" in text:
            return [3, 2, 4]
        return [3, 4]


def test_multi_query_training_never_commits_query_or_gold_answer():
    trainer = MemoryTrainer.__new__(MemoryTrainer)
    trainer.mem = ReadOnlyQueryMemory()
    trainer.backbone = ReadOnlyQueryBackbone()
    trainer.device = torch.device("cpu")
    trainer.tok = ReadOnlyQueryTokenizer()
    trainer.eos = 0
    trainer.decoder = None
    trainer.self_prefix_used = 0
    trainer.contrastive_pairs_used = 0
    trainer.write_selection_loss_sum = 0.0
    trainer.write_selection_loss_count = 0
    trainer.args = SimpleNamespace(
        query_read_mode="inline",
        retrieval_windows=0,
        self_prefix_prob=0.0,
        fusion_gate_lambda=0.0,
        evidence_lambda=0.0,
        first_token_lambda=0.0,
        prefix_token_lambda=0.0,
        prefix_token_count=1,
        contrastive_lambda=0.0,
        contrastive_negatives=1,
        contrastive_margin=1.0,
        write_selection_lambda=0.0,
        evidence_label_mode="chunk",
    )
    sample = {
        "messages": [
            [
                {"role": "user", "content": "write-a"},
                {"role": "assistant", "content": "ok"},
            ],
            [
                {"role": "user", "content": "query-a"},
                {"role": "assistant", "content": "target-a"},
            ],
            [
                {"role": "user", "content": "write-b"},
                {"role": "assistant", "content": "ok"},
            ],
            [
                {"role": "user", "content": "query-b"},
                {"role": "assistant", "content": "target-b"},
            ],
        ],
        "query_turn_id": [1, 3],
        "evidence_message_indices": [0, 2],
        "distractor_message_indices": [],
        "metadata": {"type": "remember"},
    }

    trainer.run_sample(sample, False, {})

    assert trainer.mem.commit_count == 2


def test_write_labels_come_from_each_message_not_future_query():
    trainer = MemoryTrainer.__new__(MemoryTrainer)
    trainer.mem = ReadOnlyQueryMemory()
    trainer.backbone = ReadOnlyQueryBackbone()
    trainer.device = torch.device("cpu")
    trainer.tok = ReadOnlyQueryTokenizer()
    trainer.eos = 0
    trainer.decoder = None
    trainer.self_prefix_used = 0
    trainer.contrastive_pairs_used = 0
    trainer.write_selection_loss_sum = 0.0
    trainer.write_selection_loss_count = 0
    trainer.args = SimpleNamespace(
        query_read_mode="inline",
        retrieval_windows=0,
        self_prefix_prob=0.0,
        fusion_gate_lambda=0.0,
        evidence_lambda=0.0,
        first_token_lambda=0.0,
        prefix_token_lambda=0.0,
        prefix_token_count=1,
        contrastive_lambda=0.0,
        contrastive_negatives=1,
        contrastive_margin=1.0,
        write_selection_lambda=0.0,
        evidence_label_mode="answer_tokens",
    )
    sample = {
        "messages": [
            [
                {"role": "user", "content": "write payload-a"},
                {"role": "assistant", "content": "ok"},
            ],
            [
                {"role": "user", "content": "query-a"},
                {"role": "assistant", "content": "target-a"},
            ],
            [
                {"role": "user", "content": "write payload-b"},
                {"role": "assistant", "content": "ok"},
            ],
            [
                {"role": "user", "content": "query-b"},
                {"role": "assistant", "content": "target-b"},
            ],
        ],
        "query_turn_id": [1, 3],
        "memory_targets_by_message": {
            "0": "target-a",
            "2": "target-b",
        },
        # Deliberately wrong legacy evidence metadata: the per-write payload
        # map must take precedence over future-query-derived supervision.
        "evidence_message_indices": [0],
        "distractor_message_indices": [2],
        "metadata": {"type": "remember"},
    }

    trainer.run_sample(sample, False, {})

    assert trainer.mem.commit_labels == [
        [0, 1, 0],
        [0, 1, 0],
    ]


def test_two_pass_first_prefill_never_sees_answer_prefix():
    trainer = MemoryTrainer.__new__(MemoryTrainer)
    trainer.args = SimpleNamespace(
        query_read_mode="two_pass",
        two_pass_query_tokens=4,
    )
    trainer.mem = QueryTimingMemory()
    trainer.backbone = QueryTimingBackbone(trainer.mem)
    tokens = torch.tensor([10, 11, 12, 90, 91])

    output = trainer._query_forward(
        tokens, logits_all=True, query_token_count=3)

    assert trainer.backbone.calls[0][0].tolist() == [10, 11, 12]
    assert trainer.backbone.calls[0][1] is False
    assert trainer.backbone.calls[1][0].tolist() == tokens.tolist()
    assert trainer.backbone.calls[1][1] is True
    assert trainer.mem.prepared_tail == 4
    torch.testing.assert_close(output, tokens.float())


def test_contextual_first_prefill_never_sees_answer_prefix():
    trainer = MemoryTrainer.__new__(MemoryTrainer)
    trainer.args = SimpleNamespace(
        query_read_mode="contextual",
        two_pass_query_tokens=4,
    )
    trainer.mem = QueryTimingMemory()
    trainer.backbone = QueryTimingBackbone(trainer.mem)
    tokens = torch.tensor([10, 11, 12, 90, 91])

    output = trainer._query_forward(
        tokens, logits_all=True, query_token_count=3)

    assert trainer.backbone.calls[0][0].tolist() == [10, 11, 12]
    assert trainer.backbone.calls[0][1] is False
    assert trainer.backbone.calls[1][0].tolist() == tokens.tolist()
    assert trainer.backbone.calls[1][1] is True
    assert trainer.mem.prepared_tail == 4
    torch.testing.assert_close(output, tokens.float())


def test_straight_through_alpha_top_p():
    scores = torch.tensor(
        [3.0, 2.0, 0.5, -1.0], dtype=torch.float64,
        requires_grad=True)
    probs = scores.softmax(dim=-1)
    weights = straight_through_alpha_top_p(probs, rho=0.8, k_min=1)

    # Forward is the normalized sparse top-p selection.
    assert torch.count_nonzero(weights).item() == 2
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=weights.dtype))

    # Backward follows the dense distribution: an unselected token must
    # still influence a non-constant weighted objective.
    objective = (weights * torch.tensor(
        [1.0, -0.5, 0.25, 2.0], dtype=weights.dtype)).sum()
    objective.backward()
    assert scores.grad is not None
    assert torch.all(scores.grad[2:].abs() > 0)


def test_alpha_top_p_matches_official_cumulative_rule():
    """Official Metis keeps count(cumsum <= rho) + 1 tokens."""
    probs = torch.tensor([0.50, 0.30, 0.15, 0.05])
    for rho in (0.5, 0.8, 0.9, 1.0):
        actual = straight_through_alpha_top_p(
            probs, rho=rho, k_min=1).detach()
        cumulative = probs.sort(descending=True).values.cumsum(0)
        official_k = min(
            int((cumulative <= rho).sum().item()) + 1,
            probs.numel())
        expected = torch.zeros_like(probs)
        selected = probs.topk(official_k).indices
        expected[selected] = probs[selected] / probs[selected].sum()
        torch.testing.assert_close(actual, expected)


def test_gated_delta_scales_both_states_by_alpha():
    M = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    S = torch.tensor([2.0, -1.0])
    K = torch.tensor([[0.5, -0.25], [0.2, 0.4]])
    V = torch.tensor([[1.5, -0.5], [0.75, 2.0]])
    beta_weight = torch.tensor([0.3, 0.6])
    alpha = torch.tensor(0.7)

    new_M, new_S = gated_delta_update(
        M, S, K, V, beta_weight, alpha)

    erase_M = K.t() @ (beta_weight[:, None] * (K @ M))
    add_M = K.t() @ (beta_weight[:, None] * V)
    expected_M = alpha * (M - erase_M) + add_M
    erase_S = K.t() @ (beta_weight * (K @ S))
    add_S = K.t() @ beta_weight
    expected_S = alpha * (S - erase_S) + add_S

    torch.testing.assert_close(new_M, expected_M)
    torch.testing.assert_close(new_S, expected_S)


def test_rms_norm_applies_weight_after_normalization():
    x = torch.tensor([[1.0, 2.0, -3.0, 4.0]], dtype=torch.float64)
    weight = torch.tensor([0.5, 2.0, 1.5, 0.25], dtype=torch.float64)
    eps = 1e-6
    actual = metis_rms_norm(x, weight, eps)
    expected = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        * weight.float()
    )
    xw = x.float() * weight.float()
    wrong_weight_first = (
        xw * torch.rsqrt(xw.square().mean(-1, keepdim=True) + eps)
    )
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, wrong_weight_first)


def test_balanced_svd_reconstructs_full_rank_matrix():
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(4, 7, generator=generator)
    a, b = balanced_truncated_svd(weight, rank=4)
    torch.testing.assert_close(a @ b, weight, atol=1e-5, rtol=1e-5)


def test_structural_gate_is_binary_with_sigmoid_gradient():
    logits = torch.tensor([-2.0, 2.0], requires_grad=True)
    gates = straight_through_binary_gate(logits)
    torch.testing.assert_close(gates.detach(), torch.tensor([0.0, 1.0]))
    gates.sum().backward()
    assert logits.grad is not None
    assert torch.all(logits.grad > 0)


def test_stale_answer_margin_prefers_current_value():
    positive = torch.tensor(2.0, requires_grad=True)
    stale = torch.tensor(1.5, requires_grad=True)
    loss = stale_answer_margin_loss(positive, [stale], margin=1.0)
    torch.testing.assert_close(loss, torch.tensor(1.5))
    loss.backward()
    assert positive.grad.item() > 0.0
    assert stale.grad.item() < 0.0
    separated = stale_answer_margin_loss(
        torch.tensor(0.5), [torch.tensor(2.0)], margin=1.0)
    torch.testing.assert_close(separated, torch.tensor(0.0))


def test_retention_floor_only_penalizes_excess_global_decay():
    low = torch.tensor(0.90, requires_grad=True)
    loss = retention_floor_loss(low, 0.99)
    assert loss.item() > 0.0
    loss.backward()
    assert low.grad.item() < 0.0
    torch.testing.assert_close(
        retention_floor_loss(torch.tensor(0.995), 0.99),
        torch.tensor(0.0),
    )


def test_read_preservation_detects_interfering_not_orthogonal_write():
    zero_memory = torch.zeros(2, 2)
    zero_normalizer = torch.zeros(2)
    first_key = torch.tensor([[1.0, 0.0]])
    first_value = torch.tensor([[0.0, 1.0]])
    old_memory, old_normalizer = gated_delta_update(
        zero_memory, zero_normalizer,
        first_key, first_value, torch.ones(1), torch.tensor(1.0))

    interfering_memory, interfering_normalizer = gated_delta_update(
        old_memory, old_normalizer,
        first_key, torch.tensor([[0.0, -1.0]]),
        torch.ones(1), torch.tensor(1.0))
    orthogonal_memory, orthogonal_normalizer = gated_delta_update(
        old_memory, old_normalizer,
        torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]]),
        torch.ones(1), torch.tensor(1.0))

    interfering_loss = memory_read_preservation_loss(
        old_memory, old_normalizer,
        interfering_memory, interfering_normalizer,
        first_key, "signed_plus_one")
    orthogonal_loss = memory_read_preservation_loss(
        old_memory, old_normalizer,
        orthogonal_memory, orthogonal_normalizer,
        first_key, "signed_plus_one")
    assert interfering_loss > 1.0
    torch.testing.assert_close(
        orthogonal_loss, torch.zeros_like(orthogonal_loss))


def test_write_orthogonalization_protects_distinct_but_allows_update():
    basis = torch.tensor([[1.0, 0.0]])
    weights = torch.tensor([1.0])
    distinct = torch.tensor([[1.0, 1.0]])
    protected = orthogonalize_write_keys(
        torch.nn.functional.normalize(distinct, dim=-1),
        weights, basis, strength=1.0,
        update_similarity_threshold=0.85)
    torch.testing.assert_close(
        protected, torch.tensor([[0.0, 1.0]]), atol=1e-6, rtol=0.0)

    same_address = torch.tensor([[1.0, 0.0]])
    update = orthogonalize_write_keys(
        same_address, weights, basis, strength=1.0,
        update_similarity_threshold=0.85)
    torch.testing.assert_close(update, same_address)


def test_write_orthogonalization_tracks_unlabelled_runtime_commits():
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone, [0], query_rank=0,
        query_mode="independent", kv_rank=0,
        write_orthogonalization=1.0,
        device="cpu")
    memory.init_from_backbone()
    memory.reset_state()
    memory.capture([torch.randn(4, memory.d_model)])
    assert memory.commit_all()
    assert len(memory.write_address_basis) == 1


def test_immutable_episode_commits_preserve_previous_state():
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone, [0], query_rank=0,
        query_mode="independent", kv_rank=0,
        state_mode="episodes", max_memory_episodes=4,
        episode_address_rank=4,
        device="cpu")
    memory.init_from_backbone()
    memory.reset_state()
    memory.capture([torch.randn(
        4, memory.d_model,
        generator=torch.Generator().manual_seed(41))])
    assert memory.commit_all()
    first_memory = memory.episode_M[:, 0].clone()
    first_normalizer = memory.episode_S[:, 0].clone()
    first_address = memory.episode_address[:, 0].clone()
    first_token_address = memory.episode_token_address[:, 0].clone()
    first_token_weight = memory.episode_token_weight[:, 0].clone()

    memory.capture([torch.randn(
        5, memory.d_model,
        generator=torch.Generator().manual_seed(43))])
    assert memory.commit_all()
    assert memory.episode_M.shape[1] == 2
    torch.testing.assert_close(memory.episode_M[:, 0], first_memory)
    torch.testing.assert_close(memory.episode_S[:, 0], first_normalizer)
    torch.testing.assert_close(
        memory.episode_address[:, 0], first_address)
    torch.testing.assert_close(
        memory.episode_token_address[:, 0], first_token_address)
    torch.testing.assert_close(
        memory.episode_token_weight[:, 0], first_token_weight)


def test_token_metric_episode_router_receives_address_gradients():
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone, [0], query_rank=0,
        query_mode="independent", kv_rank=0,
        state_mode="episodes",
        episode_address_mode="shared_query",
        episode_address_granularity="tokens",
        episode_address_tokens=2,
        episode_router_mode="token_metric",
        episode_router_rank=2,
        device="cpu")
    memory.reset_state()
    memory.episode_token_address = torch.tensor([[
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
    ]])
    memory.episode_token_weight = torch.ones(1, 2, 2)
    query_groups = torch.tensor([
        [[0.6, 0.2, 0.7, 0.1],
         [0.1, 0.8, 0.2, 0.4]],
        [[0.4, 0.3, 0.1, 0.8],
         [0.7, 0.2, 0.5, 0.1]],
    ])
    logits = memory._episode_route_logits(0, query_groups)
    torch.nn.functional.cross_entropy(
        logits.unsqueeze(0), torch.tensor([1])).backward()
    assert memory.episode_match_q_up.grad is not None
    assert torch.count_nonzero(
        memory.episode_match_q_up.grad).item() > 0
    assert memory.episode_match_k_up.grad is not None
    assert torch.count_nonzero(
        memory.episode_match_k_up.grad).item() > 0


def test_embedding_episode_router_activates_matching_identity():
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone, [0], query_rank=0,
        query_mode="independent", kv_rank=0,
        state_mode="episodes",
        episode_address_granularity="tokens",
        episode_address_tokens=2,
        episode_address_source="embedding",
        episode_router_rank=2,
        device="cpu")
    memory.reset_state()
    memory.episode_identity_address = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
    ])
    memory.episode_identity_weight = torch.ones(2, 2)
    query_identity = torch.tensor([
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ])
    logits = memory._episode_identity_route_logits(query_identity)
    assert int(logits.argmax()) == 1


def test_embedding_ngram_router_preserves_token_order():
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone, [0], query_rank=0,
        query_mode="independent", kv_rank=0,
        state_mode="episodes",
        episode_address_granularity="tokens",
        episode_address_tokens=3,
        episode_address_source="embedding",
        episode_identity_ngram=2,
        episode_router_rank=2,
        device="cpu")
    memory.reset_state()
    common = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    zero = torch.tensor(
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    one = torch.tensor(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    memory.episode_identity_address = torch.stack([
        torch.stack([common, zero, zero]),
        torch.stack([common, zero, one]),
    ])
    memory.episode_identity_weight = torch.ones(2, 3)
    memory.episode_identity_positions = torch.tensor([
        [0, 1, 2],
        [0, 1, 2],
    ])
    logits = memory._episode_identity_route_logits(
        torch.stack([common, zero, one]))
    assert int(logits.argmax()) == 1


def test_embedding_span_router_matches_full_identifier_sequence():
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone, [0], query_rank=0,
        query_mode="independent", kv_rank=0,
        state_mode="episodes",
        episode_address_granularity="tokens",
        episode_address_tokens=4,
        episode_address_source="embedding",
        episode_identity_route_mode="span",
        episode_identity_match_scale=0.02,
        episode_router_rank=2,
        device="cpu")
    memory.reset_state()
    vectors = torch.eye(8)[:5]
    memory.episode_identity_address = torch.stack([
        torch.stack([vectors[0], vectors[1], vectors[2], vectors[3]]),
        torch.stack([vectors[0], vectors[1], vectors[2], vectors[4]]),
    ])
    memory.episode_identity_weight = torch.ones(2, 4)
    memory.episode_identity_positions = torch.tensor([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ])
    query = torch.stack([
        vectors[4], vectors[0], vectors[1], vectors[2], vectors[4],
    ])
    logits = memory._episode_identity_route_logits(query)
    assert int(logits.argmax()) == 1


def test_embedding_episode_selector_receives_key_label_gradients():
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone, [0], query_rank=0,
        query_mode="independent", kv_rank=0,
        state_mode="episodes",
        episode_address_granularity="tokens",
        episode_address_tokens=2,
        episode_address_source="embedding",
        episode_router_rank=2,
        write_mode="dual_tokens",
        device="cpu")
    memory.reset_state()
    rows = torch.randn(
        4, 8, generator=torch.Generator().manual_seed(47))
    memory.capture([rows])
    memory.capture_token_identity(torch.randn(
        4, 8, generator=torch.Generator().manual_seed(49)))
    memory.capture_token_ids(torch.arange(4))
    memory.set_pending_slot_labels([1, 1, 0, 0])
    memory.set_pending_factorized_labels(
        [1, 1, 0, 0], [0, 0, 1, 1])
    memory.begin_write_selection_supervision()
    assert memory.commit_all_grad_enabled()
    loss = memory.end_write_selection_supervision()
    loss.backward()
    assert memory.episode_identity_selector_w.grad is not None
    assert torch.count_nonzero(
        memory.episode_identity_selector_w.grad).item() > 0


def test_neural_address_loss_prefers_correct_write_key():
    keys = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ], requires_grad=True)
    good_query = torch.tensor(
        [[[1.0, 0.0]]], requires_grad=True)
    bad_query = torch.tensor([[[0.0, 1.0]]])
    good = memory_address_contrastive_loss(
        good_query, keys, target_index=0, temperature=0.1)
    bad = memory_address_contrastive_loss(
        bad_query, keys.detach(), target_index=0, temperature=0.1)
    assert good < bad
    good.backward()
    assert good_query.grad is not None
    assert keys.grad is not None


def test_autoregressive_supervision_stops_at_first_mismatch():
    target = [10, 20, 30, 2]
    assert autoregressive_supervision_length(
        [99, 20, 30], target) == 1
    assert autoregressive_supervision_length(
        [10, 99, 30], target) == 2
    assert autoregressive_supervision_length(
        [10, 20, 30], target) == 4
    assert matching_prefix_length([10, 20, 99], target) == 2
    assert matching_prefix_length([99, 20], target) == 0
    assert matching_prefix_length([10, 20], target) == 2


def test_generation_query_evidence_map_accepts_legacy_parallel_lists():
    assert generation_query_evidence_map(
        {
            "query_turn_id": [2, 5],
            "evidence_message_indices": [0, 3],
        },
        [2, 5],
    ) == {"2": 0, "5": 3}
    assert generation_query_evidence_map(
        {
            "query_evidence_message_indices": {
                "2": 1,
            },
            "evidence_message_indices": [0],
        },
        [2],
    ) == {"2": 1}
    assert generation_query_evidence_map(
        {
            "evidence_message_indices": [0, 3],
        },
        [5],
    ) == {"5": [0, 3]}


def test_query_targets_are_mapped_only_to_their_own_evidence():
    assert target_ids_by_evidence(
        [2, 5, 7],
        {
            "2": 0,
            "5": [1, 3],
            "7": 3,
        },
        {
            2: [10, 11],
            5: [20, 21],
            7: [30, 31],
        },
    ) == {
        0: [[10, 11]],
        1: [[20, 21]],
        3: [[20, 21], [30, 31]],
    }


def test_ordered_episode_continuation_stays_inside_contiguous_payload():
    payload = [90, 10, 20, 30, 77, 40, 50, -1]
    positions = [1, 5, 6, 7, 12, 13, 14, -1]
    assert ordered_episode_continuation(
        payload, positions, [10]) == 20
    assert ordered_episode_continuation(
        payload, positions, [10, 20]) == 30
    assert ordered_episode_continuation(
        payload, positions, [10, 20, 30]) is None
    assert ordered_episode_continuation(
        payload, positions, [20]) == 30
    assert ordered_episode_continuation(
        payload, positions, [999, 10, 20]) == 30
    assert ordered_episode_continuation(
        payload, positions, [30]) is None


def test_ordered_episode_continuation_rejects_ambiguous_matches():
    assert ordered_episode_continuation(
        [10, 20, 10, 30],
        [0, 1, 2, 3],
        [10],
    ) is None
    assert ordered_episode_continuation(
        [10, 20, 10, 20],
        [0, 1, 2, 3],
        [10],
    ) is None
    assert ordered_episode_continuation(
        [10, 20, 30],
        [0, 1, 2],
        [],
    ) is None


def test_straight_through_bank_router_is_hard_with_soft_gradients():
    logits = torch.tensor([0.1, 0.8, -0.2], requires_grad=True)
    probabilities = logits.softmax(dim=0)
    route = straight_through_top1(probabilities)
    torch.testing.assert_close(
        route.detach(), torch.tensor([0.0, 1.0, 0.0]))
    (route * torch.tensor([1.0, 2.0, 3.0])).sum().backward()
    assert logits.grad is not None
    assert bool((logits.grad.abs() > 0).any())
    top_two = straight_through_topk(probabilities, 2)
    assert int((top_two.detach() > 0).sum()) == 2
    torch.testing.assert_close(top_two.detach().sum(), torch.tensor(2.0))


def test_factorial_curriculum_separates_length_from_memory_age():
    rng = random.Random(7)
    single_natural = long_memory_sample(
        10_000_001, 29, rng, profile=(1, 60),
        value_kind="natural", style="single_natural_value")
    assert single_natural["metadata"]["commits"] == 1
    assert len(single_natural["query_turn_id"]) == 1
    assert " " in single_natural["messages"][-1][1]["content"]

    long_opaque = long_memory_sample(
        20_000_001, 29, rng, profile=(8, 48),
        value_kind="opaque", style="long_context_opaque")
    assert long_opaque["metadata"]["commits"] == 8
    assert len(long_opaque["query_turn_id"]) == 3
    for query_index in long_opaque["query_turn_id"]:
        answer = long_opaque["messages"][query_index][1]["content"]
        assert " " not in answer
        assert "-" in answer
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "task3_long_opaque.jsonl"
        path.write_text(
            json.dumps(long_opaque) + "\n",
            encoding="utf-8")
        strata = load_dataset(directory)
        assert len(strata) == 1
    assert strata[0][0] == "task3_long_opaque"


def test_banked_memory_routes_each_write_to_one_latent_bank():
    memory = MetisMemory(
        TinyBackbone(), [0], query_rank=0, kv_rank=0,
        state_mode="banked", memory_banks=3,
        bank_temperature=0.2, device="cpu")
    with torch.no_grad():
        memory.wk[0].normal_()
        memory.wv[0].normal_()
        memory.w_agg[0].normal_()
    memory.reset_state()
    rows = torch.randn(6, memory.d_model)
    new_memory, new_normalizer = memory._commit_math(
        0, rows, memory.rms_eps)
    assert new_memory.shape == (3, memory.kv_dim, memory.kv_dim)
    assert new_normalizer.shape == (3, memory.kv_dim)
    changed = (
        new_memory.abs().sum(dim=(1, 2)) > 0
    ).to(torch.int32)
    assert int(changed.sum()) == 1
    route = memory.last_commit_bank_route_by_layer[0]
    assert route is not None
    torch.testing.assert_close(route.detach().sum(), torch.tensor(1.0))


def test_sequential_banks_preserve_previous_bank_state():
    memory = MetisMemory(
        TinyBackbone(), [0], query_rank=0, kv_rank=0,
        state_mode="banked", memory_banks=3,
        bank_router_mode="sequential",
        bank_temperature=0.2, device="cpu")
    with torch.no_grad():
        memory.wk[0].normal_()
        memory.wv[0].normal_()
        memory.w_agg[0].normal_()
    memory.reset_state()
    first_rows = torch.randn(5, memory.d_model)
    memory._captured = [first_rows]
    memory.pending_memory_id = 0
    memory.commit_all_grad_enabled()
    first_bank = memory.M[0, 0].detach().clone()
    assert int((memory.M[0].abs().sum(dim=(1, 2)) > 0).sum()) == 1
    assert memory.next_bank_index == 1

    second_rows = torch.randn(5, memory.d_model)
    memory._captured = [second_rows]
    memory.pending_memory_id = 1
    memory.commit_all()
    torch.testing.assert_close(memory.M[0, 0], first_bank)
    assert bool((memory.M[0, 1].abs() > 0).any())
    assert memory.next_bank_index == 2


def test_banked_router_sidecar_round_trip():
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone, [0], query_rank=0, query_mode="independent",
        kv_rank=0,
        state_mode="banked", memory_banks=3,
        bank_temperature=0.2, bank_write_top_k=2,
        device="cpu", seed=11)
    with torch.no_grad():
        memory.bank_router_w.add_(0.25)
    expected = memory.bank_router_w.detach().clone()
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "banked.bnmem")
        memory.export(path)
        restored = MetisMemory(
            backbone, [0], query_rank=0,
            query_mode="independent", kv_rank=0,
            state_mode="banked", memory_banks=3,
            bank_temperature=0.2, bank_write_top_k=2,
            device="cpu", seed=99)
        restored.load_checkpoint(path)
        torch.testing.assert_close(restored.bank_router_w, expected)


def test_factorized_value_selector_trains_and_round_trips():
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone, [0], query_rank=0,
        query_mode="independent", kv_rank=0,
        write_mode="factorized", device="cpu")
    memory.init_from_backbone()
    memory.reset_state()
    rows = torch.randn(4, memory.d_model, requires_grad=True)
    memory.capture([rows])
    memory.set_pending_slot_labels([1, 1, 1, 0])
    memory.set_pending_factorized_labels(
        [1, 1, 0, 0], [0, 0, 1, 0])
    memory.begin_write_selection_supervision()
    assert memory.commit_all_grad_enabled()
    loss = memory.end_write_selection_supervision()
    loss.backward()
    assert memory.w_value_agg.grad is not None
    assert bool((memory.w_value_agg.grad.abs() > 0).any())

    with torch.no_grad():
        memory.w_value_agg.add_(0.5)
    expected = memory.w_value_agg.detach().clone()
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "factorized.bnmem")
        memory.export(path)
        restored = MetisMemory(
            backbone, [0], query_rank=0,
            query_mode="independent", kv_rank=0,
            write_mode="factorized", device="cpu")
        restored.load_checkpoint(path)
        torch.testing.assert_close(restored.w_value_agg, expected)


def test_balanced_token_selection_supervises_every_payload_token():
    logits = torch.zeros(5, requires_grad=True)
    labels = torch.tensor([1, 1, 0, 0, -1], dtype=torch.int8)
    loss = balanced_token_selection_loss(logits, labels)
    loss.backward()
    assert logits.grad[0] < 0
    assert logits.grad[1] < 0
    assert logits.grad[2] > 0
    assert logits.grad[3] > 0
    assert logits.grad[4] == 0


def test_dual_token_selector_preserves_legacy_update_at_initialization():
    backbone = TinyBackbone()
    paired = MetisMemory(
        backbone, [0], query_rank=0,
        query_mode="independent", kv_rank=0,
        write_mode="paired_tokens", device="cpu", seed=17)
    dual = MetisMemory(
        backbone, [0], query_rank=0,
        query_mode="independent", kv_rank=0,
        write_mode="dual_tokens", device="cpu", seed=17)
    paired.init_from_backbone()
    dual.init_from_backbone()
    with torch.no_grad():
        dual.w_value_agg.copy_(dual.w_agg)
    paired.reset_state()
    dual.reset_state()
    rows = torch.randn(
        7, paired.d_model,
        generator=torch.Generator().manual_seed(31))
    paired.capture([rows])
    dual.capture([rows])
    assert paired.commit_all_grad_enabled()
    assert dual.commit_all_grad_enabled()
    torch.testing.assert_close(dual.M, paired.M)
    torch.testing.assert_close(dual.S, paired.S)


def test_neural_address_loss_rejects_one_lucky_query_group():
    keys = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    consistent = torch.tensor([[
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]])
    one_lucky_group = torch.tensor([[
        [1.0, 0.0],
        [0.6, 0.8],
        [0.6, 0.8],
        [0.6, 0.8],
    ]])
    assert memory_address_contrastive_loss(
        consistent, keys, target_index=0, temperature=0.1
    ) < memory_address_contrastive_loss(
        one_lucky_group, keys, target_index=0, temperature=0.1
    )


def test_write_key_diversity_penalizes_collisions():
    previous = torch.tensor([[
        [1.0, 0.0],
        [0.0, 1.0],
    ]])
    colliding = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    distinct = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.0],
    ])
    assert memory_key_diversity_loss(
        colliding, previous, margin=0.1
    ) > memory_key_diversity_loss(
        distinct, previous, margin=0.1
    )


def test_memory_address_transform_matches_gated_delta_state():
    old_key = torch.tensor([0.6, 0.8])
    old_value = torch.tensor([1.5, -0.5])
    old_m = old_key.unsqueeze(1) @ old_value.unsqueeze(0)
    old_s = old_key.clone()
    keys = torch.tensor([
        [0.8, 0.6],
        [-0.6, 0.8],
    ])
    values = torch.zeros(2, 2)
    beta = torch.tensor([0.4, 0.2])
    alpha = torch.tensor(0.95)
    new_m, new_s = gated_delta_update(
        old_m, old_s, keys, values, beta, alpha)
    transform = memory_address_transform(keys, beta, alpha)
    transformed_key = transform @ old_key
    expected_m = transformed_key.unsqueeze(1) @ old_value.unsqueeze(0)
    torch.testing.assert_close(new_m, expected_m)
    torch.testing.assert_close(
        new_s - keys.t() @ beta,
        transform @ old_s)


def test_evidence_attention_loss_rewards_positive_mass():
    labels = torch.tensor([1, 1, 0, 0], dtype=torch.int8)
    good = torch.tensor([[[0.45, 0.45, 0.05, 0.05]]])
    bad = torch.tensor([[[0.05, 0.05, 0.45, 0.45]]])
    assert evidence_attention_loss(good, labels) < evidence_attention_loss(
        bad, labels)
    no_distractor = torch.tensor([1, 1, -1, -1], dtype=torch.int8)
    torch.testing.assert_close(
        evidence_attention_loss(good, no_distractor), torch.tensor(0.0))


def test_answer_token_slot_labels_marks_longest_contiguous_match():
    labels = answer_token_slot_labels(
        [101, 7, 8, 9, 4, 7, 8, 3],
        [7, 8, 55],
        eos_id=55)
    assert labels == [0, 1, 1, 0, 0, 1, 1, 0]
    assert answer_token_slot_labels(
        [1, 2, 3], [2, 9, 55], eos_id=55) == [-1, -1, -1]
    assert answer_token_slot_labels(
        [1, 42, 3], [42, 55], eos_id=55) == [0, 1, 0]


def test_memory_target_slot_labels_merges_multiple_payloads():
    labels = memory_target_slot_labels(
        [8, 1, 2, 9, 3, 4, 7],
        [[1, 2, 55], [3, 4, 55]],
        eos_id=55,
    )
    assert labels == [0, 1, 1, 0, 1, 1, 0]
    assert memory_target_slot_labels(
        [1, 2, 3], [[8, 9, 55]], eos_id=55
    ) == [-1, -1, -1]


def test_full_rank_projection_initialization_matches_reference():
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone,
        [0],
        query_rank=0,
        query_mode="independent",
        kv_rank=0,
        query_gate_lambda=0.0,
        kv_gate_lambda=0.0,
        layer_gate_lambda=0.0,
        device="cpu",
    )
    memory.init_from_backbone()
    torch.testing.assert_close(
        memory.query_proj[0], backbone.layers[0]["q"])
    torch.testing.assert_close(memory.wk[0], backbone.layers[0]["k"])
    torch.testing.assert_close(memory.wv[0], backbone.layers[0]["v"])
    assert memory.query_a is None
    assert memory.wk_a is None


def test_learned_query_projects_raw_hidden_then_normalizes_heads():
    """Match official NormedReweightLearnedQueryMetisBlock ordering."""
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone,
        [0],
        query_rank=0,
        query_mode="independent",
        kv_rank=0,
        query_gate_lambda=0.0,
        kv_gate_lambda=0.0,
        layer_gate_lambda=0.0,
        device="cpu",
    )
    memory.reset_state()
    hidden = torch.tensor([[
        1.0, -2.0, 3.0, 0.5, -1.5, 2.5, 0.25, -0.75,
    ]])
    with torch.no_grad():
        memory.query_proj[0].copy_(torch.eye(8))
        memory.query_norm[0].copy_(
            torch.tensor([0.5, 1.0, 1.5, 2.0]))
        memory.M[0].copy_(torch.eye(4))
        memory.S[0].zero_()
    actual = memory.read(0, hidden)
    projected = hidden.view(1, 2, 4)
    expected = metis_rms_norm(
        projected, memory.query_norm[0], 1e-6)
    expected = torch.nn.functional.normalize(
        expected, dim=-1, eps=1e-12).reshape(1, 8)
    torch.testing.assert_close(actual, expected)


def test_two_pass_query_read_is_frozen_and_does_not_write_state():
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone,
        [0],
        query_rank=0,
        query_mode="independent",
        kv_rank=0,
        query_gate_lambda=0.0,
        kv_gate_lambda=0.0,
        layer_gate_lambda=0.0,
        device="cpu",
    )
    memory.init_from_backbone()
    memory.reset_state()
    generator = torch.Generator().manual_seed(31)
    with torch.no_grad():
        memory.M[0].copy_(
            torch.randn(4, 4, generator=generator))
        memory.S[0].copy_(
            torch.randn(4, generator=generator) * 0.1)
    rows = torch.randn(5, 8, generator=generator)
    expected = memory.read(0, rows[-2:]).mean(dim=0, keepdim=True)
    before_M = memory.M.clone()
    before_S = memory.S.clone()

    memory.capture([rows])
    memory.prepare_query_read_override(tail_tokens=2)

    torch.testing.assert_close(memory.M, before_M)
    torch.testing.assert_close(memory.S, before_S)
    torch.testing.assert_close(
        memory.query_read_override_by_layer[0], expected)
    assert memory.take_captured() is None

    output = memory.fuse(
        0, torch.zeros(3, 8), torch.randn(3, 8, generator=generator))
    torch.testing.assert_close(output[0], output[1])
    torch.testing.assert_close(output[1], output[2])
    memory.clear_query_read_override()
    assert memory.query_read_override_by_layer == [None]


def test_contextual_query_read_stays_dynamic_across_tokens():
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone,
        [0],
        query_rank=0,
        query_mode="independent",
        kv_rank=0,
        query_gate_lambda=0.0,
        kv_gate_lambda=0.0,
        layer_gate_lambda=0.0,
        device="cpu",
    )
    memory.init_from_backbone()
    memory.reset_state()
    with torch.no_grad():
        memory.query_proj[0].copy_(torch.eye(8))
        memory.query_norm[0].fill_(1.0)
        memory.M[0].copy_(torch.eye(4))
        memory.S[0].zero_()
    rows = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ])
    memory.capture([rows])
    memory.prepare_query_read_context(tail_tokens=2)
    torch.testing.assert_close(
        memory.query_read_context_by_layer[0],
        rows.mean(dim=0, keepdim=True))

    current = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    output = memory.read(0, current)
    assert not torch.allclose(output[0], output[1])
    memory.clear_query_read_override()
    assert memory.query_read_context_by_layer == [None]


def test_full_rank_backbone_delta_starts_aligned_and_fusion_is_identity():
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone,
        [0],
        query_rank=0,
        query_mode="backbone_delta",
        kv_rank=0,
        query_gate_lambda=0.0,
        kv_gate_lambda=0.0,
        layer_gate_lambda=0.0,
        fusion_mode="residual_gate",
        device="cpu",
    )
    memory.init_from_backbone()
    torch.testing.assert_close(
        memory.query_proj, torch.zeros_like(memory.query_proj))
    memory.reset_state()
    attention = torch.randn(3, 8)
    hidden = torch.randn(3, 8)
    output = memory.fuse(0, attention, hidden)
    torch.testing.assert_close(output, attention)


def test_backbone_delta_export_folds_raw_query_operator(tmp_path):
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone,
        [0],
        query_rank=0,
        query_mode="backbone_delta",
        kv_rank=0,
        query_gate_lambda=0.0,
        kv_gate_lambda=0.0,
        layer_gate_lambda=0.0,
        fusion_mode="residual_gate",
        device="cpu",
    )
    memory.init_from_backbone()
    with torch.no_grad():
        memory.query_proj.copy_(
            torch.arange(memory.query_proj.numel()).reshape_as(
                memory.query_proj) / memory.query_proj.numel())
    expected = memory.query_proj.detach() + backbone.layers[0]["q"]
    output = tmp_path / "folded.bnmem"
    memory.export(str(output))

    from bnmem_export import load_bnmem_v1
    checkpoint = load_bnmem_v1(str(output))
    assert checkpoint["query_add_backbone"] is False
    torch.testing.assert_close(
        checkpoint["tensors"]["query_proj"], expected)


def test_write_selection_supervision_reaches_selector_and_beta_gate():
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone,
        [0],
        query_rank=0,
        query_mode="backbone_delta",
        kv_rank=0,
        query_gate_lambda=0.0,
        kv_gate_lambda=0.0,
        layer_gate_lambda=0.0,
        device="cpu",
    )
    memory.init_from_backbone()
    memory.reset_state()
    rows = torch.randn(3, 8, requires_grad=True)
    memory.capture([rows])
    memory.set_pending_slot_labels([1, 0, 0])
    memory.begin_write_selection_supervision()
    assert memory.commit_all_grad_enabled()
    loss = memory.end_write_selection_supervision()
    loss.backward()
    assert memory.w_agg.grad is not None
    assert torch.count_nonzero(memory.w_agg.grad).item() > 0
    assert memory.gdu_bw.grad is not None
    assert torch.count_nonzero(memory.gdu_bw.grad).item() > 0


def test_memory_commit_preserves_source_activation_graph():
    backbone = TinyBackbone()
    memory = MetisMemory(
        backbone,
        [0],
        query_rank=0,
        query_mode="independent",
        kv_rank=0,
        query_gate_lambda=0.0,
        kv_gate_lambda=0.0,
        layer_gate_lambda=0.0,
        device="cpu",
    )
    memory.init_from_backbone()
    memory.reset_state()
    rows = torch.randn(3, 8, requires_grad=True)
    memory.capture([rows])
    assert memory.commit_all_grad_enabled()
    (memory.M.sum() + memory.S.sum()).backward()
    assert rows.grad is not None
    assert torch.count_nonzero(rows.grad).item() > 0


def test_official_five_task_mapping_and_schedule():
    assert parse_task_filter("all") is None
    assert parse_task_filter("0,2,4") == {0, 2, 4}
    assert memory_task_id("reconstruction", {"metadata": {}}) == 0
    assert memory_task_id("update_explicit", {"metadata": {}}) == 1
    assert memory_task_id("remember_distract", {"metadata": {}}) == 2
    assert memory_task_id("multi_entity", {"metadata": {}}) == 3
    assert memory_task_id(
        "anything", {"metadata": {"v2_task": "task4_normal"}}) == 4
    starts = {task: pair[0] for task, pair in {
        0: (0.25, 0.10), 1: (0.35, 0.25), 2: (0.20, 0.30),
        3: (0.10, 0.20), 4: (0.10, 0.15)}.items()}
    ends = {task: pair[1] for task, pair in {
        0: (0.25, 0.10), 1: (0.35, 0.25), 2: (0.20, 0.30),
        3: (0.10, 0.20), 4: (0.10, 0.15)}.items()}
    initial = scheduled_task_weights(0.0, [0, 1, 2, 3, 4], starts, ends)
    final = scheduled_task_weights(1.0, [0, 1, 2, 3, 4], starts, ends)
    torch.testing.assert_close(
        torch.tensor(sum(initial.values())), torch.tensor(1.0))
    torch.testing.assert_close(
        torch.tensor(sum(final.values())), torch.tensor(1.0))
    assert initial[1] > final[1]
    assert final[2] > initial[2]


if __name__ == "__main__":
    test_multi_query_training_never_commits_query_or_gold_answer()
    test_write_labels_come_from_each_message_not_future_query()
    test_two_pass_first_prefill_never_sees_answer_prefix()
    test_contextual_first_prefill_never_sees_answer_prefix()
    test_straight_through_alpha_top_p()
    test_alpha_top_p_matches_official_cumulative_rule()
    test_gated_delta_scales_both_states_by_alpha()
    test_rms_norm_applies_weight_after_normalization()
    test_balanced_svd_reconstructs_full_rank_matrix()
    test_structural_gate_is_binary_with_sigmoid_gradient()
    test_stale_answer_margin_prefers_current_value()
    test_retention_floor_only_penalizes_excess_global_decay()
    test_read_preservation_detects_interfering_not_orthogonal_write()
    test_write_orthogonalization_protects_distinct_but_allows_update()
    test_write_orthogonalization_tracks_unlabelled_runtime_commits()
    test_immutable_episode_commits_preserve_previous_state()
    test_token_metric_episode_router_receives_address_gradients()
    test_embedding_episode_router_activates_matching_identity()
    test_embedding_ngram_router_preserves_token_order()
    test_embedding_span_router_matches_full_identifier_sequence()
    test_embedding_episode_selector_receives_key_label_gradients()
    test_autoregressive_supervision_stops_at_first_mismatch()
    test_generation_query_evidence_map_accepts_legacy_parallel_lists()
    test_query_targets_are_mapped_only_to_their_own_evidence()
    test_ordered_episode_continuation_stays_inside_contiguous_payload()
    test_ordered_episode_continuation_rejects_ambiguous_matches()
    test_straight_through_bank_router_is_hard_with_soft_gradients()
    test_factorial_curriculum_separates_length_from_memory_age()
    test_banked_memory_routes_each_write_to_one_latent_bank()
    test_sequential_banks_preserve_previous_bank_state()
    test_banked_router_sidecar_round_trip()
    test_factorized_value_selector_trains_and_round_trips()
    test_balanced_token_selection_supervises_every_payload_token()
    test_dual_token_selector_preserves_legacy_update_at_initialization()
    test_neural_address_loss_prefers_correct_write_key()
    test_neural_address_loss_rejects_one_lucky_query_group()
    test_write_key_diversity_penalizes_collisions()
    test_memory_address_transform_matches_gated_delta_state()
    test_evidence_attention_loss_rewards_positive_mass()
    test_answer_token_slot_labels_marks_longest_contiguous_match()
    test_memory_target_slot_labels_merges_multiple_payloads()
    test_full_rank_projection_initialization_matches_reference()
    test_learned_query_projects_raw_hidden_then_normalizes_heads()
    test_two_pass_query_read_is_frozen_and_does_not_write_state()
    test_contextual_query_read_stays_dynamic_across_tokens()
    test_full_rank_backbone_delta_starts_aligned_and_fusion_is_identity()
    test_write_selection_supervision_reaches_selector_and_beta_gate()
    test_memory_commit_preserves_source_activation_graph()
    test_official_five_task_mapping_and_schedule()
    print("metis training formula tests: PASS")
