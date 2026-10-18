#!/usr/bin/env python
"""
02_baseline.py — the numbers to beat. Measures the ORIGINAL model (16-bit) and naive
round-to-nearest 4-bit (RTN, no AWQ) on the same eval data that 05_eval.py uses.

Writes <work_dir>/baseline.json:  fp16_ppl / rtn_ppl per dataset, fp16 weight bytes, generations,
optional lm-eval scores.

    python scripts/02_baseline.py --config configs/run.yaml
    python scripts/02_baseline.py --config configs/run.yaml --skip-fp16     # model too big for FP16 eval
    python scripts/02_baseline.py --config configs/run.yaml --skip-fp16 --set quant.group_size=64 --out rtn_g64.json
"""
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from awq.calib import get_eval_tokens  # noqa: E402
from awq.cli import parser, run_config  # noqa: E402
from awq.evaluate import (apply_rtn_, generations, load_fp_model, model_bytes, perplexity,  # noqa: E402
                          run_lm_eval, speed)
from awq.model_io import resolve_model_path  # noqa: E402
from awq.utils import Logger, env_info, write_json  # noqa: E402


def main():
    p = parser(__doc__)
    p.add_argument("--skip-fp16", action="store_true")
    p.add_argument("--skip-rtn", action="store_true")
    p.add_argument("--no-gen", action="store_true", help="skip generations / speed")
    p.add_argument("--no-lm-eval", action="store_true")
    p.add_argument("--out", default="baseline.json", help="file name inside work_dir (e.g. rtn_g64.json)")
    args = p.parse_args()
    rc = run_config(args)
    log = Logger(rc.work_dir / "baseline.log")
    from transformers import AutoTokenizer

    path = resolve_model_path(rc.model, rc.revision, rc.trust_remote_code)
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=rc.trust_remote_code)
    data = {d: get_eval_tokens(tok, d, rc.eval.ppl_seq_len, rc.eval.max_windows) for d in rc.eval.ppl_datasets}
    for d, t in data.items():
        log(f"[baseline] eval set {d}: {t.shape[0]} windows x {t.shape[1]} tokens")
    out = {"model": rc.model, "env": env_info(), "eval_windows": {d: int(t.shape[0]) for d, t in data.items()},
           "ppl_seq_len": rc.eval.ppl_seq_len}
    do_lm = bool(rc.eval.lm_eval_tasks) and not args.no_lm_eval

    for variant in ("fp16", "rtn"):
        if (variant == "fp16" and args.skip_fp16) or (variant == "rtn" and args.skip_rtn):
            continue
        log(f"[baseline] loading {variant} model ...")
        if rc.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = load_fp_model(path, rc.torch_dtype, rc.device, rc.trust_remote_code)
        if rc.device == "cpu":
            model.to("cpu")
        if variant == "fp16":
            out["fp16_weight_bytes"] = model_bytes(model)
        if variant == "rtn":
            n = apply_rtn_(model, rc.quant.w_bit, rc.quant.group_size, rc.quant.zero_point, rc.quant.skip_modules)
            log(f"[baseline] RTN applied to {n} weight tensors")
        for d, t in data.items():
            ppl = perplexity(model, t, log, desc=f"{variant} {d}")
            out[f"{variant}_ppl_{d}"] = ppl
            log(f"[baseline] {variant} {d} ppl = {ppl:.4f}")
        if rc.device == "cuda":
            out[f"{variant}_peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        if variant == "fp16" and not args.no_gen:
            out["fp16_generations"] = generations(model, tok, max_new_tokens=rc.eval.gen_max_new_tokens)
            out["fp16_speed"] = speed(model, tok)
            log(f"[baseline] fp16 speed {out['fp16_speed']}")
        if variant == "fp16" and rc.eval.custom_jsonl:
            from awq.evaluate import read_custom_jsonl
            rows = read_custom_jsonl(rc.eval.custom_jsonl)
            gens = generations(model, tok, [r["prompt"] for r in rows], max_new_tokens=64)
            hits = sum(r["expected"].lower() in g.lower() for r, g in zip(rows, gens))
            out["fp16_custom_accuracy"] = hits / len(rows)
        if do_lm:
            try:
                out[f"{variant}_lm_eval"] = run_lm_eval(model, tok, rc.eval.lm_eval_tasks, rc.eval.lm_eval_limit,
                                                         rc.eval.lm_eval_batch_size)
                log(f"[baseline] {variant} lm-eval {out[f'{variant}_lm_eval']}")
            except Exception as e:
                log(f"[baseline] lm-eval skipped: {type(e).__name__}: {e}")
                do_lm = False
        del model
        gc.collect()
        if rc.device == "cuda":
            torch.cuda.empty_cache()
        write_json(rc.work_dir / args.out, out)
    log(f"[baseline] wrote {rc.work_dir / args.out}")


if __name__ == "__main__":
    main()
