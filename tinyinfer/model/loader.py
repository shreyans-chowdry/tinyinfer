"""Load HF Llama safetensors checkpoints into our modules.

HF names look like model.layers.3.self_attn.q_proj.weight; ours drop the
"model." prefix. HF stores q/k/v and gate/up as separate matrices; we
concatenate them into qkv_proj and gate_up_proj (one matmul instead of three
or two). Every checkpoint tensor must be used exactly once and every
parameter must be filled, so a silent mismatch cannot load half a model.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from .config import LlamaConfig
from .llama import Llama


def hf_to_ours(name: str) -> str:
    """Map a HF Llama tensor name to this engine's parameter name."""
    return name.removeprefix("model.")


FUSED = {
    "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
}


def fuse_projections(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Concatenate per-layer q/k/v and gate/up weights along the output axis.

    A layer must provide all parts of a fused group or none of them.
    """
    out = dict(tensors)
    prefixes = {n.rsplit(".self_attn.", 1)[0] for n in tensors if ".self_attn." in n}
    prefixes |= {n.rsplit(".mlp.", 1)[0] for n in tensors if ".mlp." in n}
    for prefix in sorted(prefixes):
        for fused, parts in FUSED.items():
            names = [f"{prefix}.{part}.weight" for part in parts]
            present = [n in out for n in names]
            if not any(present):
                continue
            if not all(present):
                missing = [n for n, ok in zip(names, present, strict=True) if not ok]
                raise KeyError(f"parameters not in checkpoint: {missing}")
            out[f"{prefix}.{fused}.weight"] = torch.cat([out.pop(n) for n in names], dim=0)
    return out


def load_state(model: Llama, tensors: dict[str, torch.Tensor]) -> None:
    """Copy tensors into model, checking names, shapes and full coverage."""
    tensors = fuse_projections(tensors)
    params = dict(model.named_parameters())
    used: set[str] = set()
    for hf_name, tensor in tensors.items():
        if "rotary_emb" in hf_name:  # older checkpoints store inv_freq; we recompute it
            continue
        name = hf_to_ours(hf_name)
        if name not in params:
            raise KeyError(f"checkpoint tensor {hf_name!r} has no matching parameter")
        if params[name].shape != tensor.shape:
            want, got = tuple(params[name].shape), tuple(tensor.shape)
            raise ValueError(f"{hf_name}: shape {got} != {want}")
        if name in used:
            raise ValueError(f"{hf_name} provided twice")
        with torch.no_grad():
            params[name].copy_(tensor)
        used.add(name)
    missing = set(params) - used
    if model.cfg.tie_word_embeddings:
        missing.discard("lm_head.weight")
    if missing:
        raise KeyError(f"parameters not in checkpoint: {sorted(missing)}")


def load_model(
    model_dir: str | Path, device: str = "cpu", dtype: torch.dtype | None = None
) -> Llama:
    """Build a Llama from a HF model directory (config.json + *.safetensors).

    Default dtype: fp16 on CUDA, fp32 on CPU.
    """
    model_dir = Path(model_dir)
    cfg = LlamaConfig.from_json(model_dir / "config.json")
    dtype = dtype or (torch.float16 if device.startswith("cuda") else torch.float32)
    tensors: dict[str, torch.Tensor] = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        tensors.update(load_file(str(f)))
    if not tensors:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    model = Llama(cfg)
    load_state(model, tensors)
    return model.to(device=device, dtype=dtype).eval()
