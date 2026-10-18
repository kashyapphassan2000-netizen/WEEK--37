#!/usr/bin/env python
"""
06_tune.py — stage-wise tuning of the quantization knobs ("hyperparameters").

Rules (strict):
  * the tuning metric is perplexity on eval.tune_dataset (default: WikiText-2 VALIDATION split),
    which is never used by 05_eval.py — the final test numbers are not tuned on;
  * one knob at a time (configs/tune_space.yaml), the winner is carried into the next stage;
  * a change must improve the metric by >= min_improvement (default 0.5%) to be accepted;
  * stops at budget_trials or --budget-hours.

Outputs: <work_dir>/tuning/leaderboard.csv and configs/run.best.yaml (use it for the final run).

    python scripts/06_tune.py --config configs/run.yaml --space configs/tune_space.yaml --budget-hours 3
"""
import csv
import gc
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml  # noqa: E402

from awq.calib import get_eval_tokens  # noqa: E402
from awq.cli import parser  # noqa: E402
from awq.config import load_config  # noqa: E402
from awq.evaluate import perplexity  # noqa: E402
from awq.model_io import load_quantized  # noqa: E402
from awq.pipeline import quantize_model  # noqa: E402
from awq.utils import Logger  # noqa: E402


def get_value(rc, dotted):
    node = rc
    for p in dotted.split("."):
        node = getattr(node, p)
    return node


def main():
    p = parser(__doc__)
    p.add_argument("--space", default="configs/tune_space.yaml")
    p.add_argument("--budget-hours", type=float, default=None)
    p.add_argument("--keep-trials", action="store_true", help="keep every trial's quantized weights")
    p.add_argument("--out-config", default="configs/run.best.yaml")
    args = p.parse_args()
    base_rc = load_config(args.config, overrides=args.set)
    space = yaml.safe_load(Path(args.space).read_text())
    budget = space.get("budget_trials", 8)
    min_imp = space.get("min_improvement", 0.005)
    tdir = base_rc.work_dir / "tuning"
    tdir.mkdir(parents=True, exist_ok=True)
    log = Logger(tdir / "tune.log")
    from transformers import AutoTokenizer

    from awq.model_io import resolve_model_path
    path = resolve_model_path(base_rc.model, base_rc.revision, base_rc.trust_remote_code)
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=base_rc.trust_remote_code)
    tune_tokens = get_eval_tokens(tok, base_rc.eval.tune_dataset, base_rc.eval.ppl_seq_len,
                                  base_rc.eval.tune_max_windows)
    log(f"[tune] metric: ppl on {base_rc.eval.tune_dataset} ({tune_tokens.shape[0]} windows) — NOT the test set")
    t_start, rows, cache = time.time(), [], {}
    best_over: dict = {}

    def trial(over: dict):
        key = tuple(sorted((k, str(v)) for k, v in over.items()))
        if key in cache:
            return cache[key]
        k = len(rows)
        wd = tdir / f"trial_{k:02d}"
        rc = load_config(args.config, overrides=list(args.set) + [f"{kk}={vv}" for kk, vv in over.items()]
                         + [f"io.work_dir={wd}"])
        t = time.time()
        quantize_model(rc, log=Logger(wd / "quantize.log"))
        model = load_quantized(wd / "quantized", rc.device)
        ppl = perplexity(model, tune_tokens)
        del model
        gc.collect()
        row = {"trial": k, **{kk: str(vv) for kk, vv in over.items()}, "tune_ppl": round(ppl, 5),
               "minutes": round((time.time() - t) / 60, 2)}
        rows.append(row)
        cache[key] = ppl
        log(f"[tune] trial {k}: {over or '(base config)'} -> ppl {ppl:.5f}  ({row['minutes']} min)")
        if not args.keep_trials:
            shutil.rmtree(wd / "quantized", ignore_errors=True)
            shutil.rmtree(wd / "state", ignore_errors=True)
        return ppl

    best = trial({})
    stalls = 0
    for stage in space["stages"]:
        cur = best_over.get(stage["key"], get_value(base_rc, stage["key"]))
        log(f"[tune] stage '{stage['name']}': {stage['key']} in {stage['values']} (current {cur})")
        stage_best, stage_val = best, cur
        for v in stage["values"]:
            if str(v) == str(cur):
                continue
            if len(rows) >= budget or (args.budget_hours and (time.time() - t_start) / 3600 > args.budget_hours):
                log("[tune] budget reached")
                break
            ppl = trial({**best_over, stage["key"]: v})
            if ppl < stage_best * (1 - min_imp):
                stage_best, stage_val = ppl, v
        if stage_val != cur:
            best_over[stage["key"]] = stage_val
            log(f"[tune] stage winner: {stage['key']}={stage_val}  ppl {best:.5f} -> {stage_best:.5f}")
            best, stalls = stage_best, 0
        else:
            stalls += 1
            log(f"[tune] stage kept {stage['key']}={cur}")
        if stalls >= space.get("patience_stages", 3) or len(rows) >= budget:
            break

    fields = sorted({k for r in rows for k in r}, key=lambda x: (x not in ("trial",), x))
    with open(tdir / "leaderboard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["tune_ppl"]))
    final = load_config(args.config, overrides=list(args.set) + [f"{kk}={vv}" for kk, vv in best_over.items()])
    final.save(args.out_config)
    log(f"[tune] best overrides {best_over} -> {args.out_config}  (tune ppl {best:.5f})")
    log(f"[tune] leaderboard -> {tdir / 'leaderboard.csv'}")
    log("[tune] NEXT: run 03_quantize.py + 05_eval.py with the best config for the final, untuned-on test numbers.")



if __name__ == "__main__":
    main()
