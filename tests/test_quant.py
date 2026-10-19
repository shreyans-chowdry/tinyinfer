"""M6 tests: int8 per-channel quantisation is accurate and halves weight memory."""

import pytest
import torch
from torch import nn

from tinyinfer.engine.engine import LLMEngine
from tinyinfer.engine.sampling import SamplingParams
from tinyinfer.model.config import tiny_config
from tinyinfer.model.llama import Llama
from tinyinfer.quant.int8 import Int8Linear, quantize_model, quantize_weight, weight_bytes


def test_rounding_error_is_at_most_half_a_step_per_channel():
    torch.manual_seed(0)
    w = torch.randn(64, 128) * torch.logspace(-2, 1, 64)[:, None]  # very different row ranges
    q, scale = quantize_weight(w)
    assert q.dtype == torch.int8 and q.abs().max() <= 127
    err = (q.float() * scale[:, None] - w).abs()
    assert (err <= scale[:, None] / 2 + 1e-7).all()
    # per-channel scales keep small rows accurate; one global scale would not
    assert scale.max() / scale.min() > 100


def test_int8_linear_close_to_float_linear():
    torch.manual_seed(1)
    lin = nn.Linear(256, 128, bias=False)
    x = torch.randn(4, 256)
    q = Int8Linear.from_linear(lin)
    rel = (q(x) - lin(x)).norm() / lin(x).norm()
    assert rel < 0.01


def test_quantize_model_replaces_all_linears_except_lm_head():
    model = quantize_model(Llama(tiny_config()))
    linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    int8 = [n for n, m in model.named_modules() if isinstance(m, Int8Linear)]
    assert linears == ["lm_head"]
    assert len(int8) == tiny_config().num_hidden_layers * 7  # q,k,v,o,gate,up,down


def test_weight_memory_drops_and_logits_stay_close():
    torch.manual_seed(2)
    ref = Llama(tiny_config()).eval()
    q = Llama(tiny_config()).eval()
    q.load_state_dict(ref.state_dict())
    before = weight_bytes(ref)
    lin = [m for n, m in ref.named_modules() if isinstance(m, nn.Linear) and n != "lm_head"]
    quantize_model(q)
    after = weight_bytes(q)
    # each fp32 linear weight (4 bytes) becomes int8 (1 byte) plus one fp32 scale per row
    saved = sum(m.weight.numel() * 3 - m.out_features * 4 for m in lin)
    assert after == before - saved
    ids = torch.randint(0, 1000, (2, 20))
    with torch.no_grad():
        a, b = ref(ids), q(ids)
    cos = torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1)).min()
    assert cos > 0.999


def test_quantized_model_runs_through_the_batching_engine():
    torch.manual_seed(3)
    model = quantize_model(Llama(tiny_config()).eval())
    engine = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=64)
    out = engine.generate_many([[1, 2, 3], [4, 5, 6, 7]], SamplingParams(max_tokens=8))
    single = LLMEngine(model, eos_token_id=-1)
    assert out == [single.generate([1, 2, 3], SamplingParams(max_tokens=8)),
                   single.generate([4, 5, 6, 7], SamplingParams(max_tokens=8))]


def test_bias_layers_rejected():
    with pytest.raises(ValueError):
        Int8Linear.from_linear(nn.Linear(4, 4, bias=True))
