"""M7 tests: the benchmark harness runs end to end on tiny models (CPU)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from bench.report import pct, summarise, write_markdown
from bench.throughput import make_requests, run_hf, run_tinyinfer
from tinyinfer.engine.engine import LLMEngine
from tinyinfer.model.config import tiny_config
from tinyinfer.model.llama import Llama

ROOT = Path(__file__).resolve().parents[1]


def test_requests_are_deterministic_and_in_range():
    a = make_requests(20, seed=0, vocab=(10, 900), prompt_len=(5, 9), gen_len=(2, 4))
    assert a == make_requests(20, seed=0, vocab=(10, 900), prompt_len=(5, 9), gen_len=(2, 4))
    assert all(5 <= len(r.prompt) <= 9 and 2 <= r.max_tokens <= 4 for r in a)
    assert all(10 <= t < 900 for r in a for t in r.prompt)


def test_closed_loop_respects_concurrency_and_counts_tokens():
    torch.manual_seed(0)
    engine = LLMEngine(Llama(tiny_config()).eval(), eos_token_id=-1, paged=True, num_blocks=128)
    reqs = make_requests(10, vocab=(10, 900), prompt_len=(3, 20), gen_len=(2, 9))
    seen = []
    step = engine.step

    def watched():
        seen.append(len(engine.scheduler.running) + len(engine.scheduler.waiting))
        return step()

    engine.step = watched
    records, wall = run_tinyinfer(engine, reqs, concurrency=3)
    assert max(seen) <= 3
    assert sorted(r["tokens"] for r in records) == sorted(r.max_tokens for r in reqs)
    assert all(r["ttft_s"] >= 0 for r in records) and wall > 0


def test_percentile_nearest_rank():
    vals = list(range(1, 101))
    assert pct(vals, 50) == 50 and pct(vals, 99) == 99 and pct([None], 50) is None


def test_summary_ratio_and_headline(tmp_path):
    def row(system, conc, wall, tokens=1000):
        rec = {"tokens": tokens, "ttft_s": 0.2, "tpot_s": 0.01, "latency_s": 1, "preemptions": 0}
        return {"system": system, "concurrency": conc, "repeat": 1, "wall_s": wall,
                "peak_mem_bytes": 2**30, "env": {"gpu": "T4"}, "records": [rec]}

    table, env = summarise([row("tinyinfer", 16, 1.0), row("hf-static", 16, 4.0),
                            row("hf-sequential", 1, 8.0)])
    assert table[("tinyinfer", 16)]["x_vs_hf_static"] == pytest.approx(4.0)
    assert table[("tinyinfer", 16)]["x_vs_hf_sequential"] == pytest.approx(8.0)
    text = write_markdown(table, env, tmp_path / "summary.md")
    assert "4.00x" in text and "TTFT p99" in text


@pytest.fixture(scope="module")
def tiny_hf_dir(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    cfg = tiny_config()
    hf_cfg = transformers.LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size, num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
        max_position_embeddings=cfg.max_position_embeddings, tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_hf")
    transformers.LlamaForCausalLM(hf_cfg).save_pretrained(path, safe_serialization=True)
    return path


def test_hf_static_counts_only_useful_tokens(tiny_hf_dir):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_hf_dir).eval()
    reqs = make_requests(5, vocab=(10, 900), prompt_len=(3, 12), gen_len=(2, 6))
    records, _ = run_hf(model, reqs, batch=2, device="cpu")
    assert [r["tokens"] for r in records] == [r.max_tokens for r in reqs]


def test_full_pipeline_runs_and_writes_report(tiny_hf_dir, tmp_path):
    raw = tmp_path / "raw.jsonl"
    cmd = [sys.executable, "bench/throughput.py", "--model-dir", str(tiny_hf_dir),
           "--device", "cpu", "--concurrency", "1", "2", "--repeats", "1",
           "--min-requests", "4", "--prompt-len", "4", "12", "--gen-len", "2", "5",
           "--num-blocks", "64", "--out", str(raw)]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}  # works without `pip install -e .`
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True, timeout=300)
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    systems = {(r["system"], r["concurrency"]) for r in rows}
    assert ("tinyinfer", 2) in systems and ("tinyinfer-int8", 1) in systems
    assert ("hf-static", 2) in systems and ("hf-sequential", 1) in systems
    subprocess.run([sys.executable, "bench/report.py", "--raw", str(raw), "--out", str(tmp_path),
                    "--no-plots"], cwd=ROOT, env=env, check=True, capture_output=True,
                   timeout=120)
    assert "| tinyinfer | 2 |" in (tmp_path / "summary.md").read_text()
