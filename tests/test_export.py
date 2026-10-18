import json

import pytest
import torch

from awq.export import export_autoawq, gemm_dequant, gemm_pack_codes, gemm_unpack_codes, to_gemm
from awq.quantizer import compute_qparams, dequantize, pack_int4, quantize
from awq.pipeline import quantize_model


def test_gemm_pack_roundtrip():
    codes = torch.randint(0, 16, (64, 96))
    assert torch.equal(gemm_unpack_codes(gemm_pack_codes(codes)), codes)


def test_gemm_known_layout():
    # one int32 holds outputs [0..7] in the AWQ order [0,2,4,6,1,3,5,7] at nibbles 0..7
    codes = torch.arange(8).view(1, 8)
    packed = gemm_pack_codes(codes).item() & 0xFFFFFFFF
    nibbles = [(packed >> (4 * i)) & 0xF for i in range(8)]
    assert nibbles == [0, 2, 4, 6, 1, 3, 5, 7]


def test_to_gemm_dequant_matches_native():
    torch.manual_seed(0)
    w = torch.randn(48, 256)
    s, z = compute_qparams(w, 4, 128, True)
    q = quantize(w, s, z, 4, 128)
    native = dequantize(q, s, z, 128, torch.float32)
    qw, qz, sc = to_gemm(pack_int4(q), z.to(torch.uint8), s.half())
    assert qw.dtype == torch.int32 and qw.shape == (256, 6) and qz.shape == (2, 6) and sc.shape == (2, 48)
    assert torch.allclose(gemm_dequant(qw, qz, sc, 128), native, rtol=2e-3, atol=2e-3)


def test_export_folder(tiny_llama, make_rc, calib_tokens, tmp_path):
    qdir = quantize_model(make_rc(tiny_llama), calib_tokens=calib_tokens)
    out = export_autoawq(qdir, tmp_path / "gemm")
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["quantization_config"]["quant_method"] == "awq"
    assert cfg["quantization_config"]["version"] == "gemm"
    from safetensors import safe_open
    with safe_open(str(out / "model.safetensors"), framework="pt") as h:
        keys = set(h.keys())
    assert "model.layers.0.self_attn.q_proj.qweight" in keys
    assert "model.layers.0.self_attn.q_proj.weight" not in keys


def test_export_refuses_unfolded(tiny_llama, make_rc, calib_tokens, tmp_path):
    qdir = quantize_model(make_rc(tiny_llama, "quant.fold_scales=never"), calib_tokens=calib_tokens)
    with pytest.raises(RuntimeError, match="1/s"):
        export_autoawq(qdir, tmp_path / "gemm")
