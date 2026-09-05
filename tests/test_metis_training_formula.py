#!/usr/bin/env python3
"""Formula-level regression tests for the Metis training implementation."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from train_memory import (  # noqa: E402
    MetisMemory,
    answer_token_slot_labels,
    balanced_truncated_svd,
    evidence_attention_loss,
    gated_delta_update,
    memory_task_id,
    metis_rms_norm,
    scheduled_task_weights,
    stale_answer_margin_loss,
    straight_through_binary_gate,
    straight_through_alpha_top_p,
)


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
            "attn_norm": torch.ones(8),
        }]

    def model_sha256(self):
        return bytes(range(32))


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


def test_full_rank_projection_initialization_is_exact():
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
    torch.testing.assert_close(memory.query_proj[0], backbone.layers[0]["q"])
    torch.testing.assert_close(memory.wk[0], backbone.layers[0]["k"])
    torch.testing.assert_close(memory.wv[0], backbone.layers[0]["v"])
    assert memory.query_a is None
    assert memory.wk_a is None


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
    test_straight_through_alpha_top_p()
    test_gated_delta_scales_both_states_by_alpha()
    test_rms_norm_applies_weight_after_normalization()
    test_balanced_svd_reconstructs_full_rank_matrix()
    test_structural_gate_is_binary_with_sigmoid_gradient()
    test_stale_answer_margin_prefers_current_value()
    test_evidence_attention_loss_rewards_positive_mass()
    test_answer_token_slot_labels_marks_longest_contiguous_match()
    test_full_rank_projection_initialization_is_exact()
    test_memory_commit_preserves_source_activation_graph()
    test_official_five_task_mapping_and_schedule()
    print("metis training formula tests: PASS")
