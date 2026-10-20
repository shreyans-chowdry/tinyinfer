"""A Llama-architecture decoder written from scratch in PyTorch.

Components: RMSNorm, rotary position embeddings (rotate-half convention, as in
HF Llama), grouped-query attention, SwiGLU MLP, pre-norm decoder layers and an
LM head. No transformers modelling code is used here.

Positions are always passed explicitly so later milestones (KV cache, paged
cache, preemption) can run the same code on arbitrary token positions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import LlamaConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer norm: x / rms(x) * weight, no mean subtraction."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise in fp32 for stability, then cast back (matches HF Llama).
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(dtype)


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for RoPE and looks them up by position."""

    def __init__(self, head_dim: int, max_positions: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_positions).float()
        freqs = torch.outer(t, inv_freq)  # [max_pos, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # rotate-half layout
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos, sin of shape [..., head_dim] for integer positions."""
        return self.cos[positions], self.sin[positions]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map [x1, x2] to [-x2, x1] over the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate q/k heads laid out as [B, T, heads, D] by position-dependent angles."""
    cos = cos.unsqueeze(2).to(x.dtype)  # [B, T, 1, D]
    sin = sin.unsqueeze(2).to(x.dtype)
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    """Causal self-attention with grouped-query attention (GQA).

    Q, K and V come from one fused projection, and RoPE is applied to the query
    and key heads in a single call. K/V are never copied per query head: the
    query heads that share a KV head are folded into the query-length axis, so
    attention reads each KV head once.
    """

    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.groups = cfg.num_kv_groups
        self.head_dim = cfg.head_dim
        qkv_heads = self.n_heads + 2 * self.n_kv
        self.qkv_proj = nn.Linear(cfg.hidden_size, qkv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        cache=None,
        layer: int = 0,
        state: dict | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        h, hkv, g, d = self.n_heads, self.n_kv, self.groups, self.head_dim
        qkv = self.qkv_proj(x).view(b, t, h + 2 * hkv, d)
        qk = apply_rope(qkv[:, :, : h + hkv], cos, sin)  # query and key heads together
        q = qk[:, :, :h].transpose(1, 2)  # [B, H, T, D]
        k = qk[:, :, h:].transpose(1, 2)  # [B, Hkv, T, D]
        v = qkv[:, :, h + hkv :].transpose(1, 2)
        if cache is None:
            key_pos = positions
        else:
            # Write this step's keys/values, then attend over everything stored.
            k, v, key_pos = cache.update(layer, k, v, positions)
        state = {} if state is None else state
        if "mask" not in state:
            # Causal mask from explicit positions: a query sees keys at or before it.
            # Padding queries (position -1) see key 0, so no row is fully masked.
            mask = key_pos.unsqueeze(1) <= positions.clamp(min=0).unsqueeze(2)  # [B, T, S]
            s = mask.shape[-1]
            # One copy per query head in a group, matching the folded query layout.
            state["mask"] = mask.unsqueeze(1).expand(b, g, t, s).reshape(b, 1, g * t, s)
        # Query head i uses KV head i // g, so [B, H, T, D] = [B, Hkv, g, T, D]:
        # fold g into the query-length axis and attend once per KV head.
        q = q.reshape(b, hkv, g * t, d)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=state["mask"])
        # reshape, not view: CUDA SDPA kernels may return a non-contiguous output.
        out = out.reshape(b, h, t, d).transpose(1, 2).reshape(b, t, h * d)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x)), gate and up fused into one matmul."""

    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DecoderLayer(nn.Module):
    """Pre-norm block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, positions, cache=None, layer: int = 0, state=None):
        attn = self.self_attn(self.input_layernorm(x), cos, sin, positions, cache, layer, state)
        x = x + attn
        return x + self.mlp(self.post_attention_layernorm(x))


class Llama(nn.Module):
    """Full decoder: embeddings, N layers, final norm, LM head."""

    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        cache=None,
        logits_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits [B, T, vocab] for input_ids [B, T].

        positions defaults to 0..T-1; pass it explicitly when tokens do not
        start at position 0. Position -1 marks a padding token: it is neither
        written to the cache nor attended to. With a cache, keys/values are
        written at the given positions and attention covers everything stored.

        logits_index ([B] token indices) computes the LM head only at those
        positions and returns [B, 1, vocab]; prefill needs just the last token.
        """
        if positions is None:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            positions = positions.unsqueeze(0).expand(input_ids.shape[0], -1)
        cos, sin = self.rope(positions.clamp(min=0))
        x = self.embed_tokens(input_ids)
        state: dict = {}  # per-forward values shared by all layers (the attention mask)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, positions, cache, i, state)
        if logits_index is not None:
            rows = torch.arange(x.shape[0], device=x.device)
            x = x[rows, logits_index].unsqueeze(1)
        return self.lm_head(self.norm(x))
