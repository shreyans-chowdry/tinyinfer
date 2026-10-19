"""Model configuration for Llama-architecture models.

LlamaConfig mirrors the fields of a Hugging Face Llama config.json that the
forward pass needs. tiny_config() gives a model small enough for CPU tests
that run in milliseconds with random weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class LlamaConfig:
    """Hyper-parameters of a Llama-style decoder (names follow HF config.json)."""

    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    bos_token_id: int = 1
    eos_token_id: int = 2

    @property
    def head_dim(self) -> int:
        """Size of one attention head; hidden_size must divide evenly."""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        """How many query heads share each key/value head (GQA group size)."""
        return self.num_attention_heads // self.num_key_value_heads

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    @classmethod
    def from_json(cls, path: str | Path) -> LlamaConfig:
        """Read a HF config.json, keeping only the fields this engine uses."""
        raw = json.loads(Path(path).read_text())
        if raw.get("rope_scaling"):
            raise NotImplementedError("rope_scaling is not supported")
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def tiny_config(**overrides) -> LlamaConfig:
    """A 2-layer model (hidden 64, 4 heads, 2 KV heads) for fast CPU tests."""
    base = {
        "vocab_size": 1000,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 512,
    }
    base.update(overrides)
    return LlamaConfig(**base)
