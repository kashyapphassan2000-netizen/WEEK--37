"""
quantizer.py — the arithmetic of 4-bit group quantization. No AWQ yet, just numbers.

Plain words:
  * A weight matrix W has shape (out_features, in_features).
  * We cut every row into groups of `group_size` consecutive input channels.
  * Each group gets one `scale` (and one `zero` in asymmetric mode).
  * Every weight becomes an integer q in [0, 15]:      q = round(w / scale) + zero
  * To get the weight back (dequantize):               w ≈ (q - zero) * scale
  * Two 4-bit integers are packed into one uint8 byte, so storage is 0.5 byte/weight.

`fake_quant` = quantize then immediately dequantize. It returns a float matrix that
already contains the rounding error — that is what the AWQ search measures.
"""
from __future__ import annotations

import torch


def _group(in_features: int, group_size: int) -> int:
    """group_size=-1 means one group per whole row (per-channel)."""
    g = in_features if group_size <= 0 else group_size
    if in_features % g != 0:
        raise ValueError(f"in_features={in_features} is not divisible by group_size={g}")
    return g


@torch.no_grad()
def compute_qparams(w: torch.Tensor, w_bit: int = 4, group_size: int = 128, zero_point: bool = True):
    """Return (scales, zeros), both float32 with shape (out, n_groups).

    Asymmetric (zero_point=True): use min and max of each group  -> the 16 levels cover [min, max].
    Symmetric  (zero_point=False): use max|w| of each group       -> levels cover [-max, +max], zero fixed at 8.
    """
    out_f, in_f = w.shape
    g = _group(in_f, group_size)
    wg = w.float().reshape(out_f, in_f // g, g)
    qmax = 2 ** w_bit - 1                                   # 15 for 4-bit
    if zero_point:
        w_max = wg.amax(dim=-1)
        w_min = wg.amin(dim=-1)
        scales = (w_max - w_min).clamp(min=1e-5) / qmax
        zeros = (-torch.round(w_min / scales)).clamp(0, qmax)
    else:
        half = 2 ** (w_bit - 1)                             # 8 for 4-bit
        scales = wg.abs().amax(dim=-1).clamp(min=1e-5) / (half - 1)
        zeros = torch.full_like(scales, float(half))
    return scales, zeros


@torch.no_grad()
def quantize(w: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
             w_bit: int = 4, group_size: int = 128) -> torch.Tensor:
    """Float weights -> integer codes q in [0, 2^bits - 1] (uint8, shape (out, in))."""
    out_f, in_f = w.shape
    g = _group(in_f, group_size)
    wg = w.float().reshape(out_f, in_f // g, g)
    q = torch.round(wg / scales.unsqueeze(-1)) + zeros.unsqueeze(-1)
    q = q.clamp(0, 2 ** w_bit - 1)
    return q.reshape(out_f, in_f).to(torch.uint8)


def dequantize(q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
               group_size: int = 128, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Integer codes -> float weights:  w = (q - zero) * scale."""
    out_f, in_f = q.shape
    g = _group(in_f, group_size)
    qg = q.reshape(out_f, in_f // g, g).to(scales.dtype)
    w = (qg - zeros.to(scales.dtype).unsqueeze(-1)) * scales.unsqueeze(-1)
    return w.reshape(out_f, in_f).to(dtype)


@torch.no_grad()
def fake_quant(w: torch.Tensor, w_bit: int = 4, group_size: int = 128, zero_point: bool = True) -> torch.Tensor:
    """Quantize + dequantize in one go. Same shape and dtype as the input."""
    scales, zeros = compute_qparams(w, w_bit, group_size, zero_point)
    q = quantize(w, scales, zeros, w_bit, group_size)
    return dequantize(q, scales, zeros, group_size, dtype=w.dtype)


# ---------------------------------------------------------------------------
# Packing: two 4-bit codes per byte.  byte = low nibble (even column) | high nibble (odd column) << 4
# ---------------------------------------------------------------------------
def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """(..., in) uint8 codes in [0,15] -> (..., in/2) uint8."""
    if q.shape[-1] % 2:
        raise ValueError("pack_int4 needs an even number of columns")
    q = q.to(torch.uint8)
    return (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()


def unpack_int4(p: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_int4: (..., in/2) uint8 -> (..., in) uint8 codes."""
    low = p & 0x0F
    high = (p >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(*p.shape[:-1], p.shape[-1] * 2)
