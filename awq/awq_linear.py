"""
awq_linear.py — the modules that hold 4-bit weights inside the model.

AwqLinear
    Drop-in replacement for nn.Linear. Stores packed 4-bit codes + per-group scales/zeros.
    forward(x): (optionally x * inv_s) -> dequantize weight -> ordinary matmul.
    `inv_s` (= 1/s) is only present when the AWQ scale could NOT be folded into the
    previous layer; then the layer divides its own input by s at runtime.

AwqExpertsWeight
    Replacement for a fused 3-D Mixture-of-Experts weight (num_experts, out, in), the way
    transformers >= 5 stores experts. The experts' forward code does `W[e]` to get expert e's
    matrix; this module answers `W[e]` with the dequantized matrix, so the original
    (eager) expert code runs unchanged.

Honest note: dequantizing in Python on every forward is the slow-but-correct reference.
Fast inference uses fused int4 kernels (vLLM / SGLang) via scripts/07_export.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import compute_qparams, dequantize, pack_int4, quantize, unpack_int4


class AwqLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, w_bit: int = 4,
                 group_size: int = 128, has_inv_s: bool = False, dtype=torch.float16, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = in_features if group_size <= 0 else group_size
        n_groups = in_features // self.group_size
        f = dict(device=device)
        self.register_buffer("qweight", torch.zeros(out_features, in_features // 2, dtype=torch.uint8, **f))
        self.register_buffer("qzeros", torch.zeros(out_features, n_groups, dtype=torch.uint8, **f))
        self.register_buffer("scales", torch.zeros(out_features, n_groups, dtype=dtype, **f))
        if has_inv_s:
            self.register_buffer("inv_s", torch.ones(in_features, dtype=dtype, **f))
        else:
            self.inv_s = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, **f), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    @torch.no_grad()
    def from_float(cls, linear: nn.Linear, w_bit: int, group_size: int, zero_point: bool,
                   inv_s: torch.Tensor | None = None) -> "AwqLinear":
        """`linear.weight` must ALREADY be W * diag(s) (scaled) and clipped."""
        w = linear.weight.data
        m = cls(linear.in_features, linear.out_features, linear.bias is not None, w_bit, group_size,
                has_inv_s=inv_s is not None, dtype=w.dtype, device=w.device)
        scales, zeros = compute_qparams(w, w_bit, m.group_size, zero_point)
        q = quantize(w, scales, zeros, w_bit, m.group_size)
        m.qweight.copy_(pack_int4(q))
        m.qzeros.copy_(zeros.to(torch.uint8))
        m.scales.copy_(scales.to(w.dtype))
        if inv_s is not None:
            m.inv_s.copy_(inv_s.to(w.dtype))
        if linear.bias is not None:
            m.bias.data.copy_(linear.bias.data)
        return m

    def dequant_weight(self, dtype=None) -> torch.Tensor:
        q = unpack_int4(self.qweight)
        return dequantize(q, self.scales.float(), self.qzeros, self.group_size,
                          dtype=dtype or self.scales.dtype)

    @property
    def weight(self) -> torch.Tensor:
        """Some model code reads `.weight` (e.g. for its dtype). Return the dequantized matrix."""
        return self.dequant_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inv_s is not None:
            x = x * self.inv_s.to(x.dtype)            # the "x / s" half of the AWQ identity
        w = self.dequant_weight(dtype=x.dtype)
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, bits={self.w_bit}, "
                f"group={self.group_size}, inv_s={self.inv_s is not None}, bias={self.bias is not None}")


class AwqExpertsWeight(nn.Module):
    """Quantized stand-in for a fused expert parameter of shape (E, out, in)."""

    def __init__(self, num_experts: int, out_features: int, in_features: int, w_bit: int = 4,
                 group_size: int = 128, dtype=torch.float16, device=None):
        super().__init__()
        self.num_experts, self.out_features, self.in_features = num_experts, out_features, in_features
        self.w_bit = w_bit
        self.group_size = in_features if group_size <= 0 else group_size
        ng = in_features // self.group_size
        f = dict(device=device)
        self.register_buffer("qweight", torch.zeros(num_experts, out_features, in_features // 2,
                                                    dtype=torch.uint8, **f))
        self.register_buffer("qzeros", torch.zeros(num_experts, out_features, ng, dtype=torch.uint8, **f))
        self.register_buffer("scales", torch.zeros(num_experts, out_features, ng, dtype=dtype, **f))
        self.register_buffer("inv_s", torch.ones(num_experts, in_features, dtype=dtype, **f))

    @torch.no_grad()
    def set_expert(self, e: int, w_scaled: torch.Tensor, inv_s: torch.Tensor | None, zero_point: bool):
        scales, zeros = compute_qparams(w_scaled, self.w_bit, self.group_size, zero_point)
        q = quantize(w_scaled, scales, zeros, self.w_bit, self.group_size)
        self.qweight[e].copy_(pack_int4(q))
        self.qzeros[e].copy_(zeros.to(torch.uint8))
        self.scales[e].copy_(scales.to(self.scales.dtype))
        if inv_s is not None:
            self.inv_s[e].copy_(inv_s.to(self.inv_s.dtype))

    def expert_weight(self, e: int, dtype=None) -> torch.Tensor:
        """Dequantized W_e with 1/s folded into its input columns:  Q(W s) diag(1/s)."""
        w = dequantize(unpack_int4(self.qweight[e]), self.scales[e].float(), self.qzeros[e],
                       self.group_size, dtype=torch.float32)
        w = w * self.inv_s[e].float().unsqueeze(0)
        return w.to(dtype or self.scales.dtype)

    # --- tensor-like interface used by eager expert forward code -----------------
    def __getitem__(self, e):
        return self.expert_weight(int(e))

    @property
    def shape(self):
        return torch.Size((self.num_experts, self.out_features, self.in_features))

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    @property
    def dtype(self):
        return self.scales.dtype

    @property
    def device(self):
        return self.scales.device

    @property
    def ndim(self):
        return 3

    def extra_repr(self) -> str:
        return (f"experts={self.num_experts}, out={self.out_features}, in={self.in_features}, "
                f"bits={self.w_bit}, group={self.group_size}")
