"""The AWQ identity and the scale folding must be EXACT before any rounding happens."""
import torch

from awq.pipeline import _fold


def test_scaling_identity_exact():
    torch.manual_seed(0)
    x = torch.randn(64, 128, dtype=torch.float64)
    w = torch.randn(32, 128, dtype=torch.float64)
    s = torch.rand(128, dtype=torch.float64) * 10 + 0.1
    assert torch.allclose((x / s) @ (w * s).T, x @ w.T, atol=1e-10)


def _llama_block():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=4,
                      num_hidden_layers=1, vocab_size=100)
    torch.manual_seed(0)
    blk = LlamaDecoderLayer(cfg, 0).double().eval()
    for p in blk.parameters():
        p.data.normal_(0, 0.2)
    rope = LlamaRotaryEmbedding(cfg)
    return blk, rope


def _run(blk, rope, x):
    pos = torch.arange(x.shape[1]).unsqueeze(0)
    cos, sin = rope(x.float(), pos)
    out = blk(x, position_embeddings=(cos.double(), sin.double()), position_ids=pos)
    return out[0] if isinstance(out, tuple) else out


def test_folding_keeps_block_output_identical():
    blk, rope = _llama_block()
    x = torch.randn(2, 8, 64, dtype=torch.float64)
    ref = _run(blk, rope, x)
    groups = {
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"): ("norm", "input_layernorm"),
        ("mlp.gate_proj", "mlp.up_proj"): ("norm", "post_attention_layernorm"),
        ("mlp.down_proj",): ("linear_out", "mlp.up_proj"),
        ("self_attn.o_proj",): ("linear_out", "self_attn.v_proj"),
    }
    for names, target in groups.items():
        in_f = blk.get_submodule(names[0]).in_features
        s = torch.rand(in_f, dtype=torch.float64) * 4 + 0.25
        for n in names:
            blk.get_submodule(n).weight.data *= s
        _fold(blk, target, s)
    assert torch.allclose(_run(blk, rope, x), ref, atol=1e-9)


def test_gemma_style_norm_offset_fold():
    class OffsetNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(16, dtype=torch.float64) * 0.1)

        def forward(self, x):
            return x * (1 + self.weight)

    blk = torch.nn.Module()
    blk.norm = OffsetNorm()
    x = torch.randn(4, 16, dtype=torch.float64)
    s = torch.rand(16, dtype=torch.float64) + 0.5
    ref = blk.norm(x) / s
    _fold(blk, ("norm_offset", "norm"), s)
    assert torch.allclose(blk.norm(x), ref, atol=1e-12)
