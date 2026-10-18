import torch
import torch.nn as nn
import torch.nn.functional as F

from awq.awq_linear import AwqLinear
from awq.quantizer import compute_qparams, dequantize, fake_quant, pack_int4, quantize, unpack_int4


def test_pack_unpack_roundtrip_is_exact():
    q = torch.randint(0, 16, (64, 256), dtype=torch.uint8)
    p = pack_int4(q)
    assert p.shape == (64, 128) and p.dtype == torch.uint8
    assert torch.equal(unpack_int4(p), q)


def test_error_is_bounded_by_half_a_step():
    torch.manual_seed(0)
    w = torch.randn(32, 256)
    for zp in (True, False):
        scales, zeros = compute_qparams(w, 4, 128, zp)
        q = quantize(w, scales, zeros, 4, 128)
        assert q.min() >= 0 and q.max() <= 15
        w_hat = dequantize(q, scales, zeros, 128, torch.float32)
        step = scales.repeat_interleave(128, dim=1)
        assert ((w - w_hat).abs() <= step / 2 + 1e-6).all(), f"zero_point={zp}"


def test_per_channel_group():
    w = torch.randn(8, 96)
    out = fake_quant(w, 4, -1, True)
    assert out.shape == w.shape and torch.isfinite(out).all()


def test_awq_linear_matches_fake_quant():
    torch.manual_seed(0)
    lin = nn.Linear(256, 64, bias=True)
    x = torch.randn(10, 256)
    m = AwqLinear.from_float(lin, 4, 128, True)
    ref = F.linear(x, fake_quant(lin.weight.data, 4, 128, True), lin.bias)
    assert torch.allclose(m(x), ref, atol=1e-5)
    # storage really is 4-bit: 0.5 byte per weight
    assert m.qweight.numel() * m.qweight.element_size() == 64 * 256 // 2


def test_awq_linear_with_inv_s_undoes_the_scale():
    torch.manual_seed(0)
    lin = nn.Linear(128, 32, bias=False)
    s = torch.rand(128) + 0.5
    scaled = nn.Linear(128, 32, bias=False)
    scaled.weight.data = lin.weight.data * s
    m = AwqLinear.from_float(scaled, 4, 64, True, inv_s=1 / s)
    x = torch.randn(4, 128)
    expected = F.linear(x / s, fake_quant(lin.weight.data * s, 4, 64, True))
    assert torch.allclose(m(x), expected, atol=1e-5)
