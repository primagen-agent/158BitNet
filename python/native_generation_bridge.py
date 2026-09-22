"""Explicit diagnostic surrogate: exact native values, approximate backward.

Not the derivative of a quantized program; never enable in a trainer implicitly.
"""
import torch
from torch.nn import functional as F

from diagnose_neural_memory_features import c_q8_activation_reference
from torch_backbone import rms_norm


class _NativeValueSurrogateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native, surrogate):
        if native.requires_grad or native.shape != surrogate.shape or native.dtype != surrogate.dtype or native.device != surrogate.device:
            raise ValueError("native/surrogate geometry or ownership mismatch")
        if not native.is_floating_point() or not torch.isfinite(native).all() or not torch.isfinite(surrogate).all():
            raise ValueError("finite floating point logits required")
        return native.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output


class _QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return c_q8_activation_reference(value)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def native_value_with_surrogate_gradient(native, surrogate):
    return _NativeValueSurrogateGradient.apply(native, surrogate)


def suffix_surrogate(backbone, after_attention, variant):
    """Last-layer FFN + output only. Frozen weights still propagate input gradients."""
    if variant not in ("dense_suffix", "q8_ste_suffix"): raise ValueError("unregistered backward variant")
    cfg, layer = backbone.cfg, backbone.layers[-1]
    quant = _QuantizeSTE.apply if variant == "q8_ste_suffix" else lambda x: x
    for weight in tuple(layer.values()) + (backbone.out_norm, backbone.out_proj):
        if weight.requires_grad: raise ValueError("backbone must remain frozen")
    h = rms_norm(after_attention, layer["ffn_norm"], cfg.rms_eps)
    g, u = (F.linear(quant(h), layer[key]) for key in ("gate", "up"))
    h = after_attention + F.linear(quant(F.silu(g) * u), layer["down"]) * cfg.residual_scale
    hidden = rms_norm(h, backbone.out_norm, cfg.rms_eps)
    logits = F.linear(quant(hidden), backbone.out_proj)
    return logits / cfg.logit_scale if cfg.logit_scale else logits
