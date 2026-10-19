"""M2 tests: KV-cache decoding equals full recomputation; sampling is correct."""

import os
from pathlib import Path

import pytest
import torch

from tinyinfer.cache.contiguous import ContiguousKVCache
from tinyinfer.engine.engine import LLMEngine
from tinyinfer.engine.sampling import SamplingParams, filter_logits, sample
from tinyinfer.model.config import tiny_config
from tinyinfer.model.llama import Llama

FIXTURE = Path(
    os.environ.get("TINYINFER_FIXTURE", Path(__file__).parent / "fixtures" / "reference.pt")
)


@pytest.fixture(scope="module")
def engine():
    torch.manual_seed(0)
    return LLMEngine(Llama(tiny_config()), eos_token_id=-1)  # -1: never stop early


def test_cached_greedy_equals_uncached_for_64_tokens(engine):
    prompt = torch.randint(0, 1000, (17,)).tolist()
    params = SamplingParams(max_tokens=64)
    cached = engine.generate(prompt, params)
    assert len(cached) == 64
    assert cached == engine.generate_uncached(prompt, params)


def test_cached_sampling_equals_uncached_with_same_seed(engine):
    prompt = torch.randint(0, 1000, (9,)).tolist()
    params = SamplingParams(temperature=0.8, top_k=50, top_p=0.9, max_tokens=32, seed=7)
    assert engine.generate(prompt, params) == engine.generate_uncached(prompt, params)


def test_cache_logits_match_full_forward(engine):
    model = engine.model
    ids = torch.randint(0, 1000, (1, 12))
    with torch.no_grad():
        full = model(ids)
        cache = ContiguousKVCache(model.cfg, 1, 12)
        pre = model(ids[:, :8], torch.arange(8).unsqueeze(0), cache)
        steps = [model(ids[:, i : i + 1], torch.tensor([[i]]), cache) for i in range(8, 12)]
    torch.testing.assert_close(pre, full[:, :8], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.cat(steps, 1), full[:, 8:], atol=1e-5, rtol=1e-5)


def test_stop_token_ends_generation_and_is_excluded(engine):
    prompt = [5, 6, 7]
    first = engine.generate(prompt, SamplingParams(max_tokens=1))[0]
    assert engine.generate(prompt, SamplingParams(max_tokens=10, stop_token_ids=[first])) == []


def test_cache_overflow_raises():
    cfg = tiny_config()
    cache = ContiguousKVCache(cfg, 1, 4)
    k = torch.zeros(1, cfg.num_key_value_heads, 5, cfg.head_dim)
    with pytest.raises(ValueError, match="cache full"):
        cache.update(0, k, k, torch.arange(5).unsqueeze(0))


def test_top_k_keeps_only_k_largest():
    logits = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
    kept = torch.isfinite(filter_logits(logits, top_k=2, top_p=1.0))
    assert kept.tolist() == [False, True, False, True, False]


def test_top_p_keeps_smallest_nucleus():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    kept = torch.isfinite(filter_logits(probs.log(), top_k=0, top_p=0.75))
    assert kept.tolist() == [True, True, False, False]  # 0.5 + 0.3 >= 0.75
    kept = torch.isfinite(filter_logits(probs.log(), top_k=0, top_p=0.4))
    assert kept.tolist() == [True, False, False, False]  # top token always kept


def test_sampling_is_reproducible_and_follows_distribution():
    logits = torch.tensor([0.0, 1.0, 2.0])
    params = SamplingParams(temperature=1.0)
    draws = [sample(logits, params, torch.Generator().manual_seed(s)) for s in range(3000)]
    freq = torch.bincount(torch.tensor(draws), minlength=3).float() / 3000
    assert torch.allclose(freq, logits.softmax(-1), atol=0.03)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    a = [sample(logits, params, g1) for _ in range(20)]
    assert a == [sample(logits, params, g2) for _ in range(20)]


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        SamplingParams(top_p=0)
    with pytest.raises(ValueError):
        SamplingParams(temperature=-1)


@pytest.mark.real
def test_tinyllama_greedy_matches_hugging_face_fixture():
    """Cached greedy decoding reproduces HF generate() tokens on the real model."""
    from tinyinfer.model.loader import load_model

    fixture = FIXTURE
    model_dir = os.environ.get("TINYINFER_MODEL_DIR")
    if not fixture.exists() or not model_dir:
        pytest.skip("set TINYINFER_MODEL_DIR and build tests/fixtures/reference.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = LLMEngine(load_model(model_dir, device=device))
    for case in torch.load(fixture)["cases"]:
        want = case["greedy_tokens"].tolist()
        if want and want[-1] == engine.eos:
            want = want[:-1]  # HF keeps the EOS token; we exclude stop tokens
        got = engine.generate(case["input_ids"].tolist(), SamplingParams(max_tokens=32))
        assert got == want, f"prompt {case['prompt'][:30]!r}: {got} != {want}"
