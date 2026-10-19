"""M4 tests: continuous batching gives the same outputs as running each request alone."""

import os
import random
from pathlib import Path

import pytest
import torch

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


def make_requests(n: int, seed: int):
    rng = random.Random(seed)
    prompts = [[rng.randrange(1000) for _ in range(rng.randint(1, 60))] for _ in range(n)]
    params = [SamplingParams(max_tokens=rng.randint(1, 40)) for _ in range(n)]
    arrivals = sorted(rng.randint(0, 30) for _ in range(n))
    return prompts, params, arrivals


def run_staggered(engine: LLMEngine, prompts, params, arrivals) -> dict[int, list[int]]:
    """Add each request at its arrival step while the engine keeps stepping."""
    ids: dict[int, int] = {}
    step, nxt = 0, 0
    while nxt < len(prompts) or engine.has_unfinished():
        while nxt < len(prompts) and arrivals[nxt] <= step:
            ids[nxt] = engine.add_request(prompts[nxt], params[nxt])
            nxt += 1
        engine.step()
        step += 1
    return {i: engine.sequences[sid].output_ids for i, sid in ids.items()}


def test_20_staggered_requests_match_single_runs(model):
    prompts, params, arrivals = make_requests(20, seed=0)
    single = LLMEngine(model, eos_token_id=-1)
    expected = [single.generate(p, sp) for p, sp in zip(prompts, params, strict=True)]
    engine = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=128)
    got = run_staggered(engine, prompts, params, arrivals)
    assert [got[i] for i in range(20)] == expected
    assert engine.paged_cache.blocks.num_free == 128


def test_batch_actually_runs_sequences_together(model):
    engine = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=128)
    for i in range(6):
        engine.add_request([i + 1] * 5, SamplingParams(max_tokens=10))
    engine.step()  # admits all six (prefill)
    plan = engine.scheduler.schedule()
    assert len(plan.decodes) == 6


def test_forced_preemption_still_gives_identical_outputs(model):
    prompts, params, arrivals = make_requests(12, seed=3)
    single = LLMEngine(model, eos_token_id=-1)
    expected = [single.generate(p, sp) for p, sp in zip(prompts, params, strict=True)]
    # 26 blocks of 4 tokens = 104 slots: far too few to run all 12 requests at once.
    engine = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=26, block_size=4)
    got = run_staggered(engine, prompts, params, [0] * 12)
    assert engine.scheduler.num_preemptions > 0
    assert [got[i] for i in range(12)] == expected
    assert engine.paged_cache.blocks.num_free == 26
    assert sum(engine.sequences[s].preemptions for s in engine.sequences) == (
        engine.scheduler.num_preemptions
    )


def test_seeded_sampling_matches_single_runs(model):
    prompts, _, _ = make_requests(8, seed=5)
    params = [SamplingParams(temperature=0.9, top_p=0.9, max_tokens=20, seed=i) for i in range(8)]
    single = LLMEngine(model, eos_token_id=-1)
    expected = [single.generate(p, sp) for p, sp in zip(prompts, params, strict=True)]
    engine = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=128)
    assert engine.generate_many(prompts, params) == expected


def test_token_budget_limits_prefill_per_step(model):
    engine = LLMEngine(
        model, eos_token_id=-1, paged=True, num_blocks=128, max_num_batched_tokens=50
    )
    for _ in range(5):
        engine.add_request(list(range(20)), SamplingParams(max_tokens=2))
    plan = engine.scheduler.schedule()
    assert len(plan.prefills) == 2  # 20 + 20 <= 50 < 60


def test_stop_token_finishes_and_frees(model):
    engine = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=64)
    first = LLMEngine(model, eos_token_id=-1).generate([3, 4, 5], SamplingParams(max_tokens=1))[0]
    out = engine.generate_many([[3, 4, 5]], SamplingParams(max_tokens=10, stop_token_ids=[first]))
    assert out == [[]]
    assert engine.sequences[1].finish_reason == "stop"
    assert engine.paged_cache.blocks.num_free == 64


def test_requests_that_can_never_fit_are_rejected(model):
    engine = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=2, block_size=4)
    with pytest.raises(ValueError, match="never fit"):
        engine.add_request(list(range(6)), SamplingParams(max_tokens=10))
    with pytest.raises(RuntimeError):
        LLMEngine(model).add_request([1], SamplingParams())


def test_timestamps_recorded(model):
    engine = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=64)
    engine.generate_many([[1, 2, 3]], SamplingParams(max_tokens=5))
    seq = engine.sequences[1]
    assert seq.arrival_time <= seq.first_token_time <= seq.finish_time
    assert len(seq.token_times) == 5


@pytest.mark.real
def test_tinyllama_batched_greedy_matches_hugging_face_fixture():
    """All fixture prompts decoded together in one continuous batch."""
    from tinyinfer.model.loader import load_model

    fixture = FIXTURE
    model_dir = os.environ.get("TINYINFER_MODEL_DIR")
    if not fixture.exists() or not model_dir:
        pytest.skip("set TINYINFER_MODEL_DIR and build tests/fixtures/reference.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = LLMEngine(load_model(model_dir, device=device), paged=True, num_blocks=128)
    cases = torch.load(fixture)["cases"]
    outs = engine.generate_many(
        [c["input_ids"].tolist() for c in cases], SamplingParams(max_tokens=32)
    )
    for case, got in zip(cases, outs, strict=True):
        want = case["greedy_tokens"].tolist()
        if want and want[-1] == engine.eos:
            want = want[:-1]
        assert got == want, f"prompt {case['prompt'][:30]!r}"
    assert engine.paged_cache.blocks.num_free == 128
