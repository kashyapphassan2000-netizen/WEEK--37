"""
hardware.py — "which GPU, how many, how long, how much?" (Option B of the playbook, as code).

Plain words:
  1. Read the model's config.json (a few KB — no weights are downloaded).
  2. Build the model on the "meta" device to count EXACTLY how many parameters sit in
     quantizable Linear layers, in MoE experts, and outside the blocks (embeddings/head).
     If the skeleton cannot be built, fall back to formulas from config fields.
  3. Apply the sizing formulas for the three phases:
        A. quantize      -> one block + calibration activations
        B. FP16 baseline -> the whole 16-bit model
        C. serve AWQ     -> the 4-bit model + KV cache
  4. For each phase pick the cheapest (GPU type x count) from configs/gpus.yaml.

Every number printed is an ESTIMATE from formulas; the quantize-speed constant is a guess
until calibrated by a real GPU run (see docs/).
"""
from __future__ import annotations

import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
QUANT_MIN_PER_BPARAM_A100 = 3.0     # ESTIMATE: minutes of quantization per billion params on an A100
DOWNLOAD_MB_S = 200.0               # typical cloud download speed from the HF Hub


def load_gpus(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or ROOT / "configs" / "gpus.yaml").read_text())


def _cfg_get(cfg, *names, default=None):
    for n in names:
        v = getattr(cfg, n, None)
        if v is not None:
            return v
    return default


def config_dims(config) -> dict:
    t = config.get_text_config() if hasattr(config, "get_text_config") else config
    h = _cfg_get(t, "hidden_size", "d_model", "n_embd")
    heads = _cfg_get(t, "num_attention_heads", "n_head")
    d = {
        "model_type": getattr(t, "model_type", None),
        "layers": _cfg_get(t, "num_hidden_layers", "n_layer", "num_layers"),
        "hidden": h,
        "intermediate": _cfg_get(t, "intermediate_size", "ffn_dim", "n_inner", default=4 * (h or 0)),
        "heads": heads,
        "kv_heads": _cfg_get(t, "num_key_value_heads", "num_kv_heads", default=heads),
        "head_dim": _cfg_get(t, "head_dim", default=(h // heads) if h and heads else None),
        "vocab": _cfg_get(t, "vocab_size"),
        "tie_word_embeddings": bool(getattr(t, "tie_word_embeddings", getattr(config, "tie_word_embeddings", False))),
        "experts": _cfg_get(t, "num_local_experts", "num_experts", "n_routed_experts", default=0) or 0,
        "experts_per_tok": _cfg_get(t, "num_experts_per_tok", "moe_topk", default=0) or 0,
        "moe_intermediate": _cfg_get(t, "moe_intermediate_size", default=0) or 0,
        "max_context": _cfg_get(t, "max_position_embeddings", "n_positions", "seq_length"),
        "dtype": str(_cfg_get(t, "dtype", "torch_dtype", default=_cfg_get(config, "dtype", "torch_dtype"))),
    }
    # kv_lora (DeepSeek MLA) caches a compressed latent instead of full K/V
    if getattr(t, "kv_lora_rank", None):
        d["kv_cache_width"] = int(t.kv_lora_rank + getattr(t, "qk_rope_head_dim", 0))
    return d


def count_params(model) -> dict:
    """Exact parameter accounting from a (meta) skeleton."""
    import torch.nn as nn

    from .blocks import find_blocks, fused_experts, quantizable_linears

    name, blocks = find_blocks(model)
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    per_block, lin_total, exp_total, n_lin, transposed_experts = [], 0, 0, 0, 0
    for b in blocks:
        lin, _ = quantizable_linears(b, 128)
        exps = fused_experts(b)
        lp = sum(m.weight.numel() for m in lin.values())
        ep = sum(getattr(m, pn).numel() for _, pn, m in exps)
        transposed_experts += sum(1 for _, m in b.named_modules() if getattr(m, "is_transposed", False))
        per_block.append(sum(p.numel() for p in b.parameters()))
        lin_total += lp
        exp_total += ep
        n_lin += len(lin)
    block_total = sum(per_block)
    # tied embeddings: the head re-uses the embedding matrix -> count it once
    tie = getattr(model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config,
                  "tie_word_embeddings", False)
    out_emb, in_emb = model.get_output_embeddings(), model.get_input_embeddings()
    if tie and out_emb is not None and in_emb is not None and out_emb.weight is not in_emb.weight:
        total -= out_emb.weight.numel()
    return {
        "blocks_path": name,
        "n_blocks": len(blocks),
        "total_params": total,
        "block_params": block_total,
        "non_block_params": total - block_total,
        "max_block_params": max(per_block),
        "quant_linear_params": lin_total,
        "quant_expert_params": exp_total,
        "quantizable_params": lin_total + exp_total,
        "n_quant_linears": n_lin,
        "coverage_of_blocks": (lin_total + exp_total) / max(1, block_total),
        "transposed_expert_modules": transposed_experts,
        "has_nn_linear": any(isinstance(m, nn.Linear) for m in blocks[0].modules()),
    }


def formula_params(d: dict) -> dict:
    """Fallback when no skeleton can be built: standard decoder formulas from config fields."""
    h, L, inter = d["hidden"], d["layers"], d["intermediate"]
    hd = d["head_dim"] or h // d["heads"]
    attn = h * d["heads"] * hd * 2 + h * d["kv_heads"] * hd * 2
    if d["experts"]:
        mlp = d["experts"] * 3 * h * (d["moe_intermediate"] or inter)
    else:
        mlp = 3 * h * inter
    block = attn + mlp
    emb = d["vocab"] * h * (1 if d["tie_word_embeddings"] else 2)
    return {"blocks_path": "?", "n_blocks": L, "total_params": block * L + emb, "block_params": block * L,
            "non_block_params": emb, "max_block_params": block,
            "quant_linear_params": attn * L + (0 if d["experts"] else mlp * L),
            "quant_expert_params": mlp * L if d["experts"] else 0, "quantizable_params": block * L,
            "n_quant_linears": None, "coverage_of_blocks": 1.0, "transposed_expert_modules": 0,
            "has_nn_linear": True, "estimated_from_formula": True}


def _pick(vram_gb: float, gpus: list[dict], allowed=(1, 2, 3, 4, 5, 6, 7, 8)):
    """Cheapest (gpu, count) whose usable memory (90%) holds vram_gb, with count in `allowed`."""
    options = []
    for g in gpus:
        need = max(1, math.ceil(vram_gb / (0.9 * g["vram_gb"])))
        n = next((a for a in allowed if a >= need), None)
        if n is not None:
            options.append({"gpu": g["name"], "count": n, "usd_per_hr": round(n * g["usd_per_hr"], 2),
                            "speed": g.get("speed", 1.0)})
    options.sort(key=lambda o: (o["usd_per_hr"], o["count"]))
    return options


def estimate(counts: dict, dims: dict, ckpt_gb: float | None = None, n_samples: int = 128,
             seq_len: int = 512, context: int = 8192, batch: int = 1, gpus_cfg: dict | None = None) -> dict:
    gpus_cfg = gpus_cfg or load_gpus()
    gpus = gpus_cfg["gpus"]
    P = counts["total_params"] / 1e9
    Pq = counts["quantizable_params"] / 1e9
    fp16_gb = 2 * P
    awq_gb = 0.52 * Pq + 2 * (P - Pq)
    block_gb = 2 * counts["max_block_params"] / 1e9
    width = max(dims["intermediate"] or 0, (dims["experts_per_tok"] or 0) * (dims["moe_intermediate"] or 0),
                dims["hidden"] or 0)
    calib_gb = n_samples * seq_len * width * 2 / 1e9
    vram_quant = 2 * block_gb + 6 * calib_gb + 2
    vram_fp16 = 1.2 * fp16_gb
    kv_width = dims.get("kv_cache_width") or 2 * (dims["kv_heads"] or 0) * (dims["head_dim"] or 0)
    kv_gb = (dims["layers"] or 0) * kv_width * context * batch * 2 / 1e9
    vram_serve = 1.15 * (awq_gb + kv_gb)
    ckpt_gb = ckpt_gb if ckpt_gb is not None else fp16_gb
    disk_gb = ckpt_gb + awq_gb + 20
    ram_gb = max(16, 3 * block_gb + 2 * calib_gb + 8)
    download_h = ckpt_gb * 1000 / DOWNLOAD_MB_S / 3600

    phases = {}
    # A: one block must fit ONE GPU (the pipeline does not split a block across cards).
    # B: transformers device_map="auto" spreads layers over any number of GPUs.
    # C: vLLM tensor parallelism wants 1, 2, 4 or 8 GPUs.
    allowed = {"A_quantize": (1,), "B_fp16_eval": (1, 2, 3, 4, 5, 6, 7, 8), "C_serve_awq": (1, 2, 4, 8)}
    for key, vram in (("A_quantize", vram_quant), ("B_fp16_eval", vram_fp16), ("C_serve_awq", vram_serve)):
        opts = _pick(vram, gpus, allowed[key])
        best = opts[0] if opts else None
        if key == "A_quantize" and best:
            hours = P * QUANT_MIN_PER_BPARAM_A100 / 60 / best["speed"] + download_h + 0.5
        elif key == "B_fp16_eval" and best:
            hours = 0.3 + P / 100 + download_h * 0          # weights already on disk from phase A
        else:
            hours = 0.3
        phases[key] = {
            "vram_gb": round(vram, 1),
            "best": best,
            "alternatives": opts[1:4],
            "hours_est": round(hours, 2) if best else None,
            "cost_usd_est": round(best["usd_per_hr"] * hours * 1.25, 2) if best else None,
            "feasible_single_node": bool(opts),
        }
    return {
        "params_b": round(P, 3),
        "quantizable_params_b": round(Pq, 3),
        "fp16_gb": round(fp16_gb, 2),
        "awq_gb": round(awq_gb, 2),
        "compression_x": round(fp16_gb / awq_gb, 2) if awq_gb else None,
        "checkpoint_gb": round(ckpt_gb, 2),
        "block_gb": round(block_gb, 3),
        "calib_act_gb": round(calib_gb, 3),
        "kv_cache_gb": round(kv_gb, 3),
        "context": context,
        "disk_gb": round(disk_gb, 1),
        "cpu_ram_gb": round(ram_gb, 1),
        "download_hours": round(download_h, 2),
        "phases": phases,
        "prices_checked": gpus_cfg.get("checked"),
        "price_source": gpus_cfg.get("source"),
        "note": "ESTIMATES. Quantize speed constant is uncalibrated until a real GPU run; prices change monthly.",
    }


def native_size_gb(counts: dict, dtype_bytes: int, group_size: int = 128) -> float:
    """Exact size of OUR native output: 4-bit codes + per-group (scale in run dtype + 1-byte zero),
    everything outside the quantized weights kept in the run dtype (fp32 on CPU, bf16/fp16 on GPU)."""
    pq, p = counts["quantizable_params"], counts["total_params"]
    g = group_size if group_size > 0 else 1
    return (pq * 0.5 + pq / g * (dtype_bytes + 1) + (p - pq) * dtype_bytes) / 1e9


def checkpoint_size_gb(model_id: str, revision: str = "main") -> float | None:
    p = Path(model_id).expanduser()
    try:
        if p.is_dir():
            files = list(p.glob("*.safetensors")) or list(p.glob("*.bin"))
            return sum(f.stat().st_size for f in files) / 1e9 if files else None
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
        sizes = [s.size for s in info.siblings if s.rfilename.endswith(".safetensors") and s.size]
        if not sizes:
            sizes = [s.size for s in info.siblings if s.rfilename.endswith(".bin") and s.size]
        return sum(sizes) / 1e9 if sizes else None
    except Exception:
        return None


def analyze_model(model_id: str, revision: str = "main", trust_remote_code: bool = False,
                  n_samples: int = 128, seq_len: int = 512, context: int = 8192, log=print) -> dict:
    import torch

    from .model_io import already_quantized_method, build_skeleton, checkpoint_is_fp8, load_hf_config

    config = load_hf_config(model_id, trust_remote_code)
    dims = config_dims(config)
    try:
        model = build_skeleton(model_id, torch.bfloat16, trust_remote_code)
        counts = count_params(model)
        del model
    except Exception as e:                                   # custom code that will not build, etc.
        log(f"[inspect] skeleton build failed ({type(e).__name__}: {str(e)[:160]}) -> formula estimate")
        counts = formula_params(dims)
    est = estimate(counts, dims, checkpoint_size_gb(model_id, revision), n_samples, seq_len, context)

    verdict, reasons = "PASS", []
    aq = already_quantized_method(config)
    if aq:
        verdict, reasons = "FAIL", [f"checkpoint is already quantized ({aq}). Use the original 16-bit/FP8 weights."]
    if counts["coverage_of_blocks"] < 0.5 and verdict != "FAIL":
        verdict = "WARN"
        reasons.append(f"only {counts['coverage_of_blocks']:.0%} of block weights are quantizable "
                       "(non-Linear compute or transposed experts) -> small memory saving")
    if counts.get("transposed_expert_modules"):
        reasons.append("some expert modules store transposed weights; they stay 16-bit")
    if checkpoint_is_fp8(config):
        reasons.append("FP8 checkpoint: dequantized to 16-bit block by block while loading (supported)")
    if not est["phases"]["B_fp16_eval"]["feasible_single_node"]:
        reasons.append("FP16 baseline does not fit one 8-GPU node -> skip it (--skip-fp16) and compare "
                       "against published scores")
    return {"model": model_id, "revision": revision, "verdict": verdict, "reasons": reasons,
            "dims": dims, "counts": counts, "estimate": est}


def format_report(r: dict) -> str:
    e, c, d = r["estimate"], r["counts"], r["dims"]
    lines = [
        f"MODEL      {r['model']}   [{d['model_type']}]   verdict: {r['verdict']}",
        *[f"           - {x}" for x in r["reasons"]],
        f"PARAMS     total {e['params_b']} B | quantizable {e['quantizable_params_b']} B "
        f"({c['coverage_of_blocks']:.0%} of block weights) | blocks {c['n_blocks']} | "
        f"{'MoE ' + str(d['experts']) + ' experts, top-' + str(d['experts_per_tok']) if d['experts'] else 'dense'}",
        f"SIZE       FP16 {e['fp16_gb']} GB -> AWQ-4bit {e['awq_gb']} GB  ({e['compression_x']}x)   "
        f"checkpoint on disk {e['checkpoint_gb']} GB",
        f"MEMORY     largest block {e['block_gb']} GB | calib activations {e['calib_act_gb']} GB | "
        f"KV@{e['context']} {e['kv_cache_gb']} GB | CPU RAM >= {e['cpu_ram_gb']} GB | disk >= {e['disk_gb']} GB",
        "",
        f"{'PHASE':<14}{'VRAM need':>11}  {'RENT (cheapest)':<24}{'$/hr':>7}{'hours*':>8}{'cost*':>9}   alternatives",
    ]
    for k, label in (("A_quantize", "A quantize"), ("B_fp16_eval", "B FP16 eval"), ("C_serve_awq", "C serve AWQ")):
        p = r["estimate"]["phases"][k]
        if p["best"]:
            b = p["best"]
            alts = ", ".join(f"{a['count']}x {a['gpu']} (${a['usd_per_hr']})" for a in p["alternatives"])
            lines.append(f"{label:<14}{p['vram_gb']:>9} GB  {str(b['count']) + 'x ' + b['gpu']:<24}"
                         f"{b['usd_per_hr']:>7}{p['hours_est']:>8}{'$' + str(p['cost_usd_est']):>9}   {alts}")
        else:
            why = ("one block does not fit the largest single GPU" if k == "A_quantize"
                   else "needs > 8 GPUs of the largest type (multi-node)")
            lines.append(f"{label:<14}{p['vram_gb']:>9} GB  {why}")
    lines += ["", f"* ESTIMATES. Prices: {e['price_source']} (checked {e['prices_checked']}). "
              "Edit configs/gpus.yaml for your provider."]
    return "\n".join(lines)
