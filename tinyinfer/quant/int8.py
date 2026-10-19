"""int8 weight-only quantisation, per output channel, symmetric (milestone M6).

Each row of a weight matrix (one output channel) gets its own scale
s = max|w_row| / 127, and is stored as round(w / s) in int8. The forward pass
dequantises on the fly (w_int8 * s, in the activation dtype) and then runs an
ordinary matmul.

This halves weight memory and the bytes read per decode step compared with
fp16. It does not reduce FLOPs: the matmul still runs in fp16/fp32. Decode is
limited by reading weights from memory, which is why fewer bytes help.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (int8 weight, per-row scale) for a [out, in] float weight."""
    w = w.float()
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


class Int8Linear(nn.Module):
    """Drop-in replacement for a bias-free nn.Linear with int8 weights."""

    def __init__(self, weight_q: torch.Tensor, scale: torch.Tensor, scale_dtype: torch.dtype):
        super().__init__()
        self.out_features, self.in_features = weight_q.shape
        self.register_buffer("weight_q", weight_q)
        self.register_buffer("scale", scale.to(scale_dtype))

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> Int8Linear:
        if linear.bias is not None:
            raise ValueError("Int8Linear supports bias-free layers only (as in Llama)")
        q, scale = quantize_weight(linear.weight.data)
        return cls(q, scale, linear.weight.dtype)

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        """Float weight [out, in] reconstructed from int8 values and scales."""
        return self.weight_q.to(dtype) * self.scale.to(dtype)[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantize(x.dtype))

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int8 per-channel"


def quantize_model(model: nn.Module, skip: tuple[str, ...] = ("lm_head",)) -> nn.Module:
    """Replace every nn.Linear (except names containing a skip entry) in place.

    lm_head is skipped by default: it is the last projection to logits and the
    most sensitive to rounding. Pass skip=() to quantise it too.
    """
    if getattr(getattr(model, "cfg", None), "tie_word_embeddings", False):
        raise ValueError("tied embeddings are not supported with quantisation")
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                setattr(module, child_name, Int8Linear.from_linear(child))
    return model


def weight_bytes(model: nn.Module) -> int:
    """Bytes held by weights: parameters plus quantised buffers (not RoPE tables)."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    for name, buf in model.named_buffers():
        if name.endswith(("weight_q", "scale")):
            total += buf.numel() * buf.element_size()
    return total
