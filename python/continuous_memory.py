"""DG-003 output-residual architecture control, not an approved deployment model."""
import torch
from torch.nn import functional as F


class _BasePlusCorrection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, correction):
        if base.requires_grad or base.shape != correction.shape or base.dtype != correction.dtype or base.device != correction.device:
            raise ValueError("base/correction contract mismatch")
        if not torch.isfinite(base).all() or not torch.isfinite(correction).all(): raise ValueError("nonfinite logits")
        # Preserve signed-zero bits for a disabled/zero branch, without cutting
        # the ordinary addition derivative that lets a zero gain learn to open.
        return base.clone() if not torch.count_nonzero(correction) else base + correction

    @staticmethod
    def backward(ctx, gradient):
        return None, gradient


def continuous_logits(base, residual, frozen_output_weight, logit_scale):
    if frozen_output_weight.requires_grad: raise ValueError("GGUF output projection must be frozen")
    if residual.dtype != torch.float32 or frozen_output_weight.dtype != torch.float32:
        raise ValueError("continuous path requires FP32")
    correction = F.linear(residual, frozen_output_weight)
    if logit_scale: correction = correction / logit_scale
    return _BasePlusCorrection.apply(base, correction), correction
