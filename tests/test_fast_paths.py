"""M8 tests: batched padded prefill, logits_index, GQA folding, fused weights."""

import pytest
import torch
import torch.nn.functional as F

from tinyinfer.cache.contiguous import ContiguousKVCache
from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.engine.engine import LLMEngine
from tinyinfer.engine.runner import ModelRunner
from tinyinfer.engine.sampling import SamplingParams
from tinyinfer.engine.sequence import Sequence
from tinyinfer.model.config import tiny_config
from tinyinfer.model.llama import Llama


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Llama(tiny_config()).eval()


def prompts(lengths):
    return [torch.randint(0, 1000, (n,), generator=torch.Generator().manual_seed(n)).tolist()
            for n in lengths]


def test_batched_padded_prefill_matches_single_prefills(model):
    """Prompts of 3, 17 and 9 tokens in one padded pass give each prompt's own logits."""
    lengths = [3, 17, 9]
    ps = prompts(lengths)
    cache = PagedKVCache(model.cfg, num_blocks=16, block_size=4)
    seqs = [Sequence(i, p, SamplingParams()) for i, p in enumerate(ps)]
    for s in seqs:
        cache.blocks.allocate(s.seq_id, len(s.prompt_ids))
    batched = ModelRunner(model, cache).prefill(seqs)
    for p, row in zip(ps, batched, strict=True):
        with torch.no_grad():
            alone = model(torch.tensor([p]))[0, -1]
        torch.testing.assert_close(row, alone, atol=1e-5, rtol=1e-5)


def test_padding_does_not_corrupt_cache_and_decoding_continues(model):
    """After a padded prefill, every row decodes exactly as it would alone."""
    ps = prompts([2, 21, 7, 13])
    params = SamplingParams(max_tokens=12)
    single = LLMEngine(model, eos_token_id=-1)
    expected = [single.generate(p, params) for p in ps]
    engine = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=64, block_size=4)
    for p in ps:
        engine.add_request(p, params)
    plan_sizes = []
    while engine.has_unfinished():
        before = len(engine.scheduler.waiting)
        engine.step()
        plan_sizes.append(before - len(engine.scheduler.waiting))
    assert plan_sizes[0] == 4  # all four prompts prefilled together in the first step
    assert [engine.sequences[i + 1].output_ids for i in range(4)] == expected
    assert engine.paged_cache.blocks.num_free == 64


def test_logits_index_matches_full_forward(model):
    ids = torch.randint(0, 1000, (3, 11))
    idx = torch.tensor([10, 4, 0])
    with torch.no_grad():
        full = model(ids)
        picked = model(ids, logits_index=idx)
    assert picked.shape == (3, 1, model.cfg.vocab_size)
    torch.testing.assert_close(picked[:, 0], full[torch.arange(3), idx])


def test_folded_gqa_equals_repeated_kv_attention():
    """Folding query groups into the length axis equals repeat_interleave on K/V."""
    torch.manual_seed(1)
    b, hkv, g, t, s, d = 2, 3, 4, 5, 9, 8
    q = torch.randn(b, hkv * g, t, d)
    k, v = torch.randn(b, hkv, s, d), torch.randn(b, hkv, s, d)
    mask = torch.rand(b, t, s) > 0.3
    mask[..., 0] = True
    ref = F.scaled_dot_product_attention(
        q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), attn_mask=mask.unsqueeze(1)
    )
    folded_mask = mask.unsqueeze(1).expand(b, g, t, s).reshape(b, 1, g * t, s)
    out = F.scaled_dot_product_attention(q.reshape(b, hkv, g * t, d), k, v, attn_mask=folded_mask)
    torch.testing.assert_close(out.view(b, hkv * g, t, d), ref)


def test_scheduler_budget_counts_padding(model):
    engine = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=64,
                       max_num_batched_tokens=50)
    engine.add_request(list(range(10)), SamplingParams(max_tokens=2))
    engine.add_request(list(range(30)), SamplingParams(max_tokens=2))
    plan = engine.scheduler.schedule()
    assert len(plan.prefills) == 1  # 2 rows x 30 padded tokens = 60 > 50


def test_contiguous_cache_path_unchanged(model):
    """Single-request generate() and a manual contiguous-cache run agree."""
    p = prompts([14])[0]
    out = LLMEngine(model, eos_token_id=-1).generate(p, SamplingParams(max_tokens=6))
    cache = ContiguousKVCache(model.cfg, 1, 20)
    with torch.no_grad():
        logits = model(torch.tensor([p]), torch.arange(14)[None], cache)[0, -1]
        toks = []
        for i in range(6):
            tok = int(logits.argmax())
            toks.append(tok)
            logits = model(torch.tensor([[tok]]), torch.tensor([[14 + i]]), cache)[0, -1]
    assert out == toks
