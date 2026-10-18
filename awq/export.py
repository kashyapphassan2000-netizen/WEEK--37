"""
export.py — turn our native output into the STANDARD AWQ "GEMM" format that vLLM / SGLang load.

Plain words:
  Our native format stores codes as (out, in/2) uint8. The ecosystem format (AutoAWQ "GEMM",
  understood by vLLM's awq / awq_marlin kernels) stores, per Linear:
      qweight : int32 (in, out/8)   eight 4-bit codes per int32, packed along OUTPUT channels,
                                    in the interleaved order [0, 2, 4, 6, 1, 3, 5, 7]
      qzeros  : int32 (in/g, out/8) zero points, packed the same way
      scales  : fp16  (in/g, out)
  plus `quantization_config` in config.json. Dequant rule:  W[o, i] = (q[i, o] - z[i/g, o]) * s[i/g, o].

  This only works when every AWQ scale was folded into the previous layer (no `inv_s` left),
  because vLLM has no slot for a per-input-channel 1/s. Use `quant.fold_scales: auto` on a
  supported family. Fused MoE experts are not exported (stay native).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from .quantizer import unpack_int4

AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _to_int32(x64: torch.Tensor) -> torch.Tensor:
    x64 = x64 & 0xFFFFFFFF
    return torch.where(x64 >= 2 ** 31, x64 - 2 ** 32, x64).to(torch.int32)


def gemm_pack_codes(codes_in_out: torch.Tensor) -> torch.Tensor:
    """(rows, out) integer codes 0..15 -> (rows, out/8) int32 in AWQ interleaved order."""
    rows, out_f = codes_in_out.shape
    if out_f % 8:
        raise ValueError(f"out_features={out_f} must be divisible by 8 for the AWQ GEMM format")
    c = codes_in_out.to(torch.int64).view(rows, out_f // 8, 8)[..., AWQ_ORDER]
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    return _to_int32((c << shifts).sum(-1))


def gemm_unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of gemm_pack_codes: (rows, out/8) int32 -> (rows, out) int64 codes."""
    rows, cols = packed.shape
    p = packed.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    nib = (p.unsqueeze(-1) >> shifts) & 0xF                      # (rows, cols, 8) in packed order
    return nib[..., AWQ_REVERSE_ORDER].reshape(rows, cols * 8)


def to_gemm(qweight_native: torch.Tensor, qzeros_native: torch.Tensor, scales_native: torch.Tensor):
    """Native (out, in/2) uint8 / (out, ng) / (out, ng) -> GEMM qweight, qzeros, scales."""
    codes = unpack_int4(qweight_native)                          # (out, in)
    qweight = gemm_pack_codes(codes.t().contiguous())            # (in, out/8)
    qzeros = gemm_pack_codes(qzeros_native.t().contiguous())     # (ng, out/8)
    scales = scales_native.t().contiguous().to(torch.float16)    # (ng, out)
    return qweight, qzeros, scales


def gemm_dequant(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group_size: int):
    """Reference dequantization of the GEMM format -> (out, in) float32. Used by tests."""
    codes = gemm_unpack_codes(qweight).float()                   # (in, out)
    zeros = gemm_unpack_codes(qzeros).float()                    # (ng, out)
    in_f = codes.shape[0]
    g = in_f if group_size <= 0 else group_size
    idx = torch.arange(in_f) // g
    w = (codes - zeros[idx]) * scales.float()[idx]
    return w.t().contiguous()


def export_autoawq(qdir: str | Path, out_dir: str | Path, shard_gb: float = 4.0, log=print) -> Path:
    from safetensors import safe_open
    from safetensors.torch import save_file

    qdir, out_dir = Path(qdir), Path(out_dir)
    qc = json.loads((qdir / "quant_config.json").read_text())
    if qc.get("n_linears_with_inv_s", 0):
        raise RuntimeError(
            f"{qc['n_linears_with_inv_s']} layers keep a runtime 1/s (scale could not be folded). The standard "
            "AWQ format has no place for it. Re-quantize with quant.fold_scales=auto on a supported family "
            "(llama, mistral, qwen2/3, gemma*, ...) or ship the native format.")
    if qc.get("n_expert_tensors_quantized", 0):
        raise RuntimeError("Fused MoE experts were quantized natively; the GEMM export does not cover them. "
                           "Ship the native format, or re-run with quant.quantize_experts=false.")
    if qc.get("group_size", 128) <= 0:
        raise RuntimeError("Per-channel (group_size=-1) is not supported by the GEMM export.")
    out_dir.mkdir(parents=True, exist_ok=True)
    quant_names = set(qc["linears"])
    tensors, shards, cur_bytes = {}, [], 0
    limit = int(shard_gb * 1e9)

    def flush():
        nonlocal tensors, cur_bytes
        if tensors:
            shards.append(dict(tensors))
            tensors, cur_bytes = {}, 0

    for f in sorted(qdir.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as h:
            keys = list(h.keys())
            done = set()
            for k in keys:
                base, _, leaf = k.rpartition(".")
                if base in quant_names:
                    if base in done:
                        continue
                    qw, qz, sc = to_gemm(h.get_tensor(base + ".qweight"), h.get_tensor(base + ".qzeros"),
                                         h.get_tensor(base + ".scales"))
                    new = {base + ".qweight": qw, base + ".qzeros": qz, base + ".scales": sc}
                    if base + ".bias" in keys:
                        new[base + ".bias"] = h.get_tensor(base + ".bias").to(torch.float16)
                    done.add(base)
                else:
                    t = h.get_tensor(k)
                    new = {k: t.to(torch.float16) if t.is_floating_point() else t}
                for nk, nv in new.items():
                    tensors[nk] = nv.contiguous()
                    cur_bytes += nv.numel() * nv.element_size()
                if cur_bytes > limit:
                    flush()
    flush()
    weight_map = {}
    for i, sd in enumerate(shards):
        name = "model.safetensors" if len(shards) == 1 else f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sd, str(out_dir / name), metadata={"format": "pt"})
        weight_map.update({k: name for k in sd})
    if len(shards) > 1:
        index = {"metadata": {}, "weight_map": weight_map}
        (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    for f in qdir.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in ("quant_config.json", "config.json"):
            shutil.copy2(f, out_dir / f.name)
    cfg = json.loads((qdir / "config.json").read_text())
    skipped = sorted({k.rsplit(".", 1)[-1] for k in qc.get("skipped", {})})
    cfg["quantization_config"] = {
        "quant_method": "awq", "bits": qc["w_bit"], "group_size": qc["group_size"],
        "zero_point": qc["zero_point"], "version": "gemm",
        "modules_to_not_convert": skipped or None,
    }
    cfg["torch_dtype"] = "float16"
    cfg["dtype"] = "float16"
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    log(f"[export] AutoAWQ-GEMM format -> {out_dir}  ({len(quant_names)} quantized linears, {len(shards)} shard(s))")
    return out_dir
