#!/usr/bin/env python
"""
05_eval.py — the headline numbers for the quantized model + pass/fail gates vs. baseline.json.

  perplexity on eval.ppl_datasets (same windows as 02_baseline), generations side by side
  (generations.md), tokens/s, weight footprint, optional lm-eval tasks and your custom prompts.

    python scripts/05_eval.py --config configs/run.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from awq.calib import get_eval_tokens  # noqa: E402
from awq.cli import parser, run_config  # noqa: E402
from awq.evaluate import (DEFAULT_PROMPTS, generations, model_bytes, perplexity, read_custom_jsonl,  # noqa: E402
                          run_lm_eval, speed)
from awq.model_io import load_quantized  # noqa: E402
from awq.utils import Logger, env_info, read_json, write_json  # noqa: E402


def gate_limits(params_b: float):
    if params_b <= 3.5:
        return 0.10, 3.0, "<=3B"
    if params_b < 60:
        return 0.05, 1.5, "7-34B"
    return 0.03, 1.0, ">=70B"


def custom_accuracy(model, tok, path):
    rows = read_custom_jsonl(path)
    gens = generations(model, tok, [r["prompt"] for r in rows], max_new_tokens=64)
    hits = sum(1 for r, g in zip(rows, gens) if r["expected"].strip().lower() in g.lower())
    return hits / max(1, len(rows))


def main():
    p = parser(__doc__)
    p.add_argument("--no-lm-eval", action="store_true")
    p.add_argument("--no-gen", action="store_true")
    args = p.parse_args()
    rc = run_config(args)
    log = Logger(rc.work_dir / "eval.log")
    from transformers import AutoTokenizer

    qdir = rc.work_dir / "quantized"
    tok = AutoTokenizer.from_pretrained(str(qdir))
    if rc.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = load_quantized(qdir, rc.device)
    res = {"model": rc.model, "env": env_info(), "quant_config": {
        k: v for k, v in read_json(qdir / "quant_config.json").items() if k not in ("linears", "experts", "skipped")}}
    res["awq_weight_bytes"] = model_bytes(model)
    for d in rc.eval.ppl_datasets:
        t = get_eval_tokens(tok, d, rc.eval.ppl_seq_len, rc.eval.max_windows)
        res[f"awq_ppl_{d}"] = perplexity(model, t, log, desc=f"awq {d}")
        res[f"eval_windows_{d}"] = int(t.shape[0])
        log(f"[eval] awq {d} ppl = {res[f'awq_ppl_{d}']:.4f}  ({t.shape[0]} windows)")
    if rc.device == "cuda":
        res["awq_peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    if not args.no_gen:
        res["awq_generations"] = generations(model, tok, max_new_tokens=rc.eval.gen_max_new_tokens)
        res["awq_speed"] = speed(model, tok)
        log(f"[eval] awq speed {res['awq_speed']}  (reference dequant-per-call kernels; export for fast serving)")
    if rc.eval.custom_jsonl:
        res["awq_custom_accuracy"] = custom_accuracy(model, tok, rc.eval.custom_jsonl)
    if rc.eval.lm_eval_tasks and not args.no_lm_eval:
        try:
            res["awq_lm_eval"] = run_lm_eval(model, tok, rc.eval.lm_eval_tasks, rc.eval.lm_eval_limit,
                                             rc.eval.lm_eval_batch_size)
            log(f"[eval] awq lm-eval {res['awq_lm_eval']}")
        except Exception as e:
            log(f"[eval] lm-eval skipped: {type(e).__name__}: {e}")

    # ---- compare with baseline + gates --------------------------------------------------------
    base_f = rc.work_dir / "baseline.json"
    gates = {}
    if base_f.exists():
        base = read_json(base_f)
        params_b = (base.get("fp16_weight_bytes") or 0) / 2 / 1e9
        mr = rc.work_dir / "model_report.json"
        if mr.exists():
            params_b = read_json(mr)["estimate"]["params_b"]
        ppl_gate, acc_gate, cls = gate_limits(params_b)
        res["size_class"] = cls
        for d in rc.eval.ppl_datasets:
            fp, rtn, awq = base.get(f"fp16_ppl_{d}"), base.get(f"rtn_ppl_{d}"), res[f"awq_ppl_{d}"]
            if fp:
                res[f"awq_vs_fp16_{d}"] = (awq - fp) / fp
                gates[f"{d}: AWQ ppl within {ppl_gate:.0%} of FP16"] = res[f"awq_vs_fp16_{d}"] <= ppl_gate
            if rtn:
                res[f"rtn_vs_fp16_{d}"] = (rtn - fp) / fp if fp else None
                gates[f"{d}: AWQ ppl < RTN ppl"] = awq < rtn
        if base.get("fp16_weight_bytes"):
            res["weight_memory_reduction_x"] = round(base["fp16_weight_bytes"] / res["awq_weight_bytes"], 3)
        if base.get("fp16_lm_eval") and res.get("awq_lm_eval"):
            drops = []
            for task, m in res["awq_lm_eval"].items():
                key = "acc_norm" if "acc_norm" in m else "acc"
                if task in base["fp16_lm_eval"] and key in m:
                    drops.append(100 * (base["fp16_lm_eval"][task][key] - m[key]))
            if drops:
                res["lm_eval_avg_drop_points"] = sum(drops) / len(drops)
                gates[f"lm-eval avg drop <= {acc_gate} points"] = res["lm_eval_avg_drop_points"] <= acc_gate
        if not args.no_gen:
            gates["no empty generations"] = all(g.strip() for g in res["awq_generations"])
            fp_g = base.get("fp16_generations") or [""] * len(DEFAULT_PROMPTS)
            md = ["# Generations: FP16 vs AWQ (greedy, same prompts)\n",
                  f"Model: `{rc.model}` | AWQ: {rc.quant.w_bit}-bit g{rc.quant.group_size} | "
                  f"max_new_tokens={rc.eval.gen_max_new_tokens}\n"]
            for pr, a, b in zip(DEFAULT_PROMPTS, fp_g, res["awq_generations"]):
                md += [f"## Prompt\n```\n{pr}\n```", f"**FP16:**\n```\n{a.strip()}\n```",
                       f"**AWQ:**\n```\n{b.strip()}\n```\n"]
            (rc.work_dir / "generations.md").write_text("\n".join(md))
    else:
        log("[eval] no baseline.json -> run 02_baseline.py for comparisons and gates")
    res["gates"] = gates
    res["pass"] = bool(gates) and all(gates.values())
    write_json(rc.work_dir / "results.json", res)
    log("[eval] ---------------- RESULTS ----------------")
    for d in rc.eval.ppl_datasets:
        b = read_json(base_f) if base_f.exists() else {}
        fp, rtn = b.get(f"fp16_ppl_{d}", float("nan")), b.get(f"rtn_ppl_{d}", float("nan"))
        log(f"[eval] {d:<10} original {fp:.4f} | RTN {rtn:.4f}"
            f" | AWQ {res[f'awq_ppl_{d}']:.4f}")
    for k, v in gates.items():
        log(f"[eval] {'PASS' if v else 'FAIL'}  {k}")
    log("[eval] OVERALL " + ("PASS" if res["pass"] else "FAIL"))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
