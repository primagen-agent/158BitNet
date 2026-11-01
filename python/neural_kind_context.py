"""Fit-only contextual kind residual with fixed, bounded ridge calibration."""

import torch
from torch.nn import functional as F


RIDGE = .001
TARGET_SCALE = 3.


def calibrate_kind_context(model, means, contexts, labels, *, split="fit"):
    if split != "fit":
        raise ValueError("contextual kind calibration requires fit inputs")
    if (means.ndim != 2 or means.shape[1] != 64 or not len(means)
            or contexts.shape != (len(means), 4096)
            or not means.is_floating_point() or contexts.dtype != means.dtype
            or contexts.device != means.device
            or means.device != model.kind.weight.device
            or means.dtype != model.kind.weight.dtype
            or not torch.isfinite(means).all() or not torch.isfinite(contexts).all()
            or labels.shape != (len(means),) or labels.device != means.device
            or labels.dtype != torch.long or not ((labels >= 0) & (labels < 3)).all()):
        raise ValueError("invalid contextual kind calibration inputs")
    norms = contexts.norm(dim=-1)
    if not ((norms == 0) | ((norms - 1).abs() < 1e-5)).all():
        raise ValueError("contextual kind inputs must use the model's normalization")
    old_weight = model.kind_context.weight.detach().clone()
    old_bias = model.kind_context.bias.detach().clone()
    with torch.no_grad():
        base = model.kind(means).detach().cpu().double()
        original = model.kind_logits_from_mean(means, contexts).detach().cpu().double()
    x, y = contexts.detach().cpu().double(), labels.detach().cpu()
    counts = torch.stack([(y == k).sum() for k in range(3)]).double()
    if (counts == 0).any():
        raise ValueError("contextual kind calibration requires all three source classes")
    # Labels and fixed constants alone determine this fit solution.
    target = TARGET_SCALE * F.one_hot(y, 3).double() - (base - base.mean(-1, keepdim=True))
    weights = 1 / (3 * counts[y])
    mean_x = (x * weights[:, None]).sum(0)
    mean_y = (target * weights[:, None]).sum(0)
    xs = (x - mean_x) * weights.sqrt()[:, None]
    ys = (target - mean_y) * weights.sqrt()[:, None]
    lhs = xs @ xs.T + RIDGE * torch.eye(len(x), dtype=x.dtype)
    dual = torch.linalg.solve(lhs, ys)
    weight = xs.T @ dual
    bias = mean_y - mean_x @ weight
    converted_weight = weight.T.to(model.kind_context.weight).contiguous()
    converted_bias = bias.to(model.kind_context.bias).contiguous()
    if not torch.isfinite(converted_weight).all() or not torch.isfinite(converted_bias).all():
        raise ValueError("nonfinite contextual kind calibration parameters")

    def describe(logits):
        predicted = logits.argmax(-1)
        gap = logits.gather(1, y[:, None]).squeeze(1) - logits.masked_fill(
            F.one_hot(y, 3).bool(), -torch.inf).max(-1).values
        return dict(samples=len(y), correct=int((predicted == y).sum()),
                    confusion=[[int(((y == g) & (predicted == p)).sum())
                                for p in range(3)] for g in range(3)],
                    minimum_gold_margin=float(gap.min()))

    with torch.no_grad():
        candidate = (model.kind(means) + F.linear(contexts, converted_weight,
                                                 converted_bias)).cpu().double()
    result = dict(accepted=False, fit_only=True, runtime_qualified=False,
                  ridge=RIDGE, target_scale=TARGET_SCALE,
                  optimizer="CPU.float64.class-balanced.ridge.unpenalized-intercept",
                  initial=describe(original), final=describe(candidate),
                  class_sample_counts=counts.to(torch.long).tolist(),
                  residual_weight_norm=float(weight.norm()),
                  residual_bias_norm=float(bias.norm()),
                  solve_relative_residual=float((lhs @ dual - ys).norm() / ys.norm().clamp_min(1e-30)),
                  original_precision=str(means.dtype), original_device=str(means.device),
                  minimum_accepted_gold_margin=1e-5, no_weights_or_models_saved=True)
    if result["final"]["correct"] != len(y) or result["final"]["minimum_gold_margin"] < 1e-5:
        return result
    try:
        with torch.no_grad():
            model.kind_context.weight.copy_(converted_weight)
            model.kind_context.bias.copy_(converted_bias)
    except Exception:
        with torch.no_grad():
            model.kind_context.weight.copy_(old_weight)
            model.kind_context.bias.copy_(old_bias)
        raise
    result["accepted"] = True
    return result
