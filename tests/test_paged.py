"""M3 tests: the paged KV cache is equivalent to the contiguous cache and never leaks."""

import os
import random
from pathlib import Path

import pytest
import torch

from tinyinfer.cache.block_manager import BlockManager, OutOfBlocks
from tinyinfer.cache.contiguous import ContiguousKVCache
from tinyinfer.cache.paged import PagedKVCache, bytes_per_block
from tinyinfer.engine.engine import LLMEngine
from tinyinfer.engine.sampling import SamplingParams
from tinyinfer.model.config import tiny_config
from tinyinfer.model.llama import Llama

FIXTURE = Path(
    os.environ.get("TINYINFER_FIXTURE", Path(__file__).parent / "fixtures" / "reference.pt")
)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Llama(tiny_config()).eval()


@pytest.mark.parametrize("prompt_len", [1, 15, 16, 17, 40])
def test_paged_greedy_equals_contiguous(model, prompt_len):
    """Prompt lengths straddle the 16-token block boundary on purpose."""
    gen = torch.Generator().manual_seed(prompt_len)
    prompt = torch.randint(0, 1000, (prompt_len,), generator=gen)
    params = SamplingParams(max_tokens=40)
    contiguous = LLMEngine(model, eos_token_id=-1).generate(prompt.tolist(), params)
    paged = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=8).generate(
        prompt.tolist(), params
    )
    assert paged == contiguous
    assert len(paged) == 40


def test_paged_engine_frees_blocks_after_generate(model):
    engine = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=8)
    engine.generate([1, 2, 3, 4, 5], SamplingParams(max_tokens=30))
    assert engine.paged_cache.blocks.num_free == 8


def test_batched_rows_of_different_lengths_match_single_rows(model):
    """Two sequences in one batch (one decode token each) give the same logits as alone.

    This exercises padding: the shorter row's padded keys must be masked out.
    """
    cfg = model.cfg
    lens = [21, 6]
    seqs = [torch.randint(0, 1000, (n,), generator=torch.Generator().manual_seed(n)) for n in lens]
    cache = PagedKVCache(cfg, num_blocks=10, block_size=4)
    singles = []
    with torch.no_grad():
        for sid, s in enumerate(seqs):
            cache.blocks.allocate(sid, len(s) - 1)
            cache.bind([sid])
            model(s[None, :-1], torch.arange(len(s) - 1)[None], cache)
        for s in seqs:  # reference: last token alone, fresh contiguous cache
            ref = ContiguousKVCache(cfg, 1, len(s))
            singles.append(model(s[None], torch.arange(len(s))[None], ref)[0, -1])
        for sid in range(2):
            cache.blocks.append_slot(sid)
        cache.bind([0, 1])
        last = torch.stack([s[-1:] for s in seqs])  # [2, 1]
        pos = torch.tensor([[n - 1] for n in lens])
        batched = model(last, pos, cache)[:, -1]
    for row in range(2):
        torch.testing.assert_close(batched[row], singles[row], atol=1e-5, rtol=1e-5)


def test_block_accounting_never_leaks():
    rng = random.Random(0)
    bm = BlockManager(num_blocks=64, block_size=16)
    live: dict[int, int] = {}
    for sid in range(1000):
        n = rng.randint(1, 120)
        if bm.can_allocate(n):
            bm.allocate(sid, n)
            live[sid] = n
        for s in list(live):
            if rng.random() < 0.3 and bm.can_append(s):
                bm.append_slot(s)
                live[s] += 1
            if rng.random() < 0.4:
                bm.free(s)
                del live[s]
        owned = sum(len(bm.tables[s]) for s in live)
        assert owned + bm.num_free == 64
    for s in list(live):
        bm.free(s)
    assert bm.num_free == 64 and sorted(bm.free_blocks) == list(range(64))
    assert all(r == 0 for r in bm.ref_counts)


def test_fragmentation_is_under_one_block_per_sequence():
    bm = BlockManager(num_blocks=200, block_size=16)
    rng = random.Random(1)
    lengths = [rng.randint(1, 300) for _ in range(10)]
    for sid, n in enumerate(lengths):
        bm.allocate(sid, n)
    stored = sum(lengths)
    capacity = bm.used_blocks() * 16
    assert capacity - stored < 16 * len(lengths)
    for sid, n in enumerate(lengths):
        assert len(bm.tables[sid]) == -(-n // 16)


def test_out_of_blocks_and_slot_errors():
    bm = BlockManager(num_blocks=2, block_size=4)
    assert not bm.can_allocate(9)
    with pytest.raises(OutOfBlocks):
        bm.allocate(0, 9)
    bm.allocate(0, 8)
    assert not bm.can_append(0)
    with pytest.raises(OutOfBlocks):
        bm.append_slot(0)
    assert bm.slot(0, 5) == (bm.tables[0][1], 1)
    with pytest.raises(IndexError):
        bm.slot(0, 8)


def test_bytes_per_block_tinyllama_fp16():
    from tinyinfer.model.config import LlamaConfig

    # 22 layers * 2 (K,V) * 4 kv heads * 16 tokens * 64 dims = 180,224 values * 2 bytes
    assert bytes_per_block(LlamaConfig(), 16, torch.float16) == 360_448


@pytest.mark.real
def test_tinyllama_paged_greedy_matches_hugging_face_fixture():
    from tinyinfer.model.loader import load_model

    fixture = FIXTURE
    model_dir = os.environ.get("TINYINFER_MODEL_DIR")
    if not fixture.exists() or not model_dir:
        pytest.skip("set TINYINFER_MODEL_DIR and build tests/fixtures/reference.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = LLMEngine(load_model(model_dir, device=device), paged=True, num_blocks=64)
    for case in torch.load(fixture)["cases"]:
        want = case["greedy_tokens"].tolist()
        if want and want[-1] == engine.eos:
            want = want[:-1]
        got = engine.generate(case["input_ids"].tolist(), SamplingParams(max_tokens=32))
        assert got == want, f"prompt {case['prompt'][:30]!r}"
    assert engine.paged_cache.blocks.num_free == 64
