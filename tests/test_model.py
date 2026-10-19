"""M1 tests: shapes, causality, RoPE, GQA and logit parity with Hugging Face.

The parity test builds a random tiny HF LlamaForCausalLM, saves it as
safetensors, loads it with our loader and compares logits. It runs on CPU in
seconds; the real TinyLlama fixture comparison is marked gpu.
"""

import pytest
import torch

from tinyinfer.model.config import tiny_config
from tinyinfer.model.llama import Llama, RotaryEmbedding, apply_rope
from tinyinfer.model.loader import load_model

torch.manual_seed(0)


def test_logits_shape():
    cfg = tiny_config()
    model = Llama(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 7))
    assert model(ids).shape == (2, 7, cfg.vocab_size)


def test_causal_mask_future_tokens_do_not_change_past_logits():
    cfg = tiny_config()
    model = Llama(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 10))
    changed = ids.clone()
    changed[0, 6:] = (changed[0, 6:] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = model(ids), model(changed)
    torch.testing.assert_close(a[:, :6], b[:, :6])
    assert not torch.allclose(a[:, 6:], b[:, 6:])


def test_rope_dot_product_depends_only_on_relative_distance():
    head_dim = 16
    rope = RotaryEmbedding(head_dim, 256, 10000.0)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def score(pq: int, pk: int) -> float:
        cq, sq = rope(torch.tensor([[pq]]))
        ck, sk = rope(torch.tensor([[pk]]))
        return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum().item()

    assert score(10, 3) == pytest.approx(score(110, 103), rel=1e-4, abs=1e-4)
    assert score(10, 3) != pytest.approx(score(10, 4), rel=1e-3)


def test_explicit_positions_match_default():
    cfg = tiny_config()
    model = Llama(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        torch.testing.assert_close(model(ids), model(ids, torch.arange(5).unsqueeze(0)))


@pytest.fixture(scope="module")
def hf_pair(tmp_path_factory):
    """A random tiny HF Llama saved to disk, plus our model loaded from it."""
    transformers = pytest.importorskip("transformers")
    cfg = tiny_config()
    hf_cfg = transformers.LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=False,
    )
    torch.manual_seed(1)
    hf = transformers.LlamaForCausalLM(hf_cfg).eval()
    path = tmp_path_factory.mktemp("tiny_llama")
    hf.save_pretrained(path, safe_serialization=True)
    return hf, load_model(path)


def test_logits_match_hugging_face(hf_pair):
    hf, ours = hf_pair
    ids = torch.randint(0, ours.cfg.vocab_size, (2, 33))
    with torch.no_grad():
        ref = hf(ids).logits
        got = ours(ids)
    assert (ref - got).abs().max().item() < 1e-4
    assert torch.equal(ref.argmax(-1), got.argmax(-1))


def test_loader_rejects_missing_tensors(tmp_path, hf_pair):
    import shutil

    from safetensors.torch import save_file
    hf, _ = hf_pair
    hf.save_pretrained(tmp_path / "m", safe_serialization=True)
    state = {k: v.contiguous() for k, v in hf.state_dict().items() if "q_proj" not in k}
    for f in (tmp_path / "m").glob("*.safetensors"):
        f.unlink()
    save_file(state, str(tmp_path / "m" / "model.safetensors"))
    with pytest.raises(KeyError, match="not in checkpoint"):
        load_model(tmp_path / "m")
    shutil.rmtree(tmp_path / "m")


@pytest.mark.real
def test_tinyllama_matches_reference_fixture():
    """Real TinyLlama vs tests/fixtures/reference.pt.

    Set TINYINFER_MODEL_DIR to the downloaded model. Runs in fp32 on CPU
    (tolerance 1e-3) or fp16 on CUDA (tolerance 2e-2).
    """
    import os
    from pathlib import Path
    fixture = Path(__file__).parent / "fixtures" / "reference.pt"
    model_dir = os.environ.get("TINYINFER_MODEL_DIR")
    if not fixture.exists() or not model_dir:
        pytest.skip("set TINYINFER_MODEL_DIR and build tests/fixtures/reference.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tol = 2e-2 if device == "cuda" else 1e-3
    ref = torch.load(fixture)
    model = load_model(model_dir, device=device)
    for case in ref["cases"]:
        ids = case["input_ids"].to(device).unsqueeze(0)
        with torch.no_grad():
            logits = model(ids)[0, -1].float().cpu()
        diff = (logits - case["last_logits"]).abs().max().item()
        print(f"{ids.shape[1]:>4} tokens: max |diff| = {diff:.2e}")
        assert diff < tol
        assert logits.argmax() == case["last_logits"].argmax()
