#!/usr/bin/env python
"""
04_verify.py — GATE 4. Is the quantized model sane before we spend time on full evaluation?

  * loads the quantized model ALONE first and measures its real footprint (weights actually allocated,
    and peak VRAM on GPU) — the honest version of the "VRAM drop" claim
  * size on disk vs. the formula estimate from 01_inspect_model
  * logits on 10 probe prompts vs. the original model: relative error + top-1 agreement
  * every packed layer decodes to finite numbers

    python scripts/04_verify.py --config configs/run.yaml
"""
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from awq.cli import parser, run_config  # noqa: E402
from awq.evaluate import DEFAULT_PROMPTS, input_device, load_fp_model, model_bytes  # noqa: E402
from awq.awq_linear import AwqLinear  # noqa: E402
from awq.hardware import native_size_gb  # noqa: E402
from awq.model_io import load_quantized, resolve_model_path  # noqa: E402
from awq.utils import Logger, dir_size_gb, read_json, write_json  # noqa: E402

GATE_TOP1 = 0.75          # >= 75% of next-token predictions identical to the original model
GATE_SIZE_TOL = 0.15      # on-disk size within 15% of the formula estimate


@torch.no_grad()
def logits_for(model, tok):
    dev = input_device(model)
    outs = []
    for pr in DEFAULT_PROMPTS:
        ids = tok(pr, return_tensors="pt").input_ids.to(dev)
        outs.append(model(input_ids=ids, use_cache=False).logits[0].float().cpu())
    return outs


def main():
    p = parser(__doc__)
    p.add_argument("--skip-fp16", action="store_true", help="no FP16 comparison (model too big)")
    args = p.parse_args()
    rc = run_config(args)
    log = Logger(rc.work_dir / "verify.log")
    from transformers import AutoTokenizer

    qdir = rc.work_dir / "quantized"
    path = resolve_model_path(rc.model, rc.revision, rc.trust_remote_code)
    tok = AutoTokenizer.from_pretrained(str(qdir))
    out, gates = {}, {}

    if rc.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    qm = load_quantized(qdir, rc.device)
    out["awq_weight_bytes"] = model_bytes(qm)
    if rc.device == "cuda":
        out["awq_load_peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    out["awq_disk_gb"] = dir_size_gb(qdir)
    bad = [n for n, m in qm.named_modules()
           if isinstance(m, AwqLinear) and not torch.isfinite(m.dequant_weight()).all()]
    gates["all_layers_finite"] = not bad
    report = rc.work_dir / "model_report.json"
    if report.exists():
        counts = read_json(report)["counts"]
        dtype_bytes = torch.tensor([], dtype=rc.torch_dtype).element_size()
        est = native_size_gb(counts, dtype_bytes, rc.quant.group_size)   # formula for THIS run's dtype
        out["awq_gb_formula"] = round(est, 4)
        out["awq_gb_formula_16bit"] = read_json(report)["estimate"]["awq_gb"]
        out["size_vs_formula"] = round(out["awq_disk_gb"] / est, 3)
        gates["size_matches_formula"] = abs(out["size_vs_formula"] - 1) <= GATE_SIZE_TOL
    log(f"[verify] AWQ weights {out['awq_weight_bytes'] / 1e9:.3f} GB in memory, {out['awq_disk_gb']} GB on disk")
    q_logits = logits_for(qm, tok)
    del qm
    gc.collect()

    if not args.skip_fp16:
        fp = load_fp_model(path, rc.torch_dtype, rc.device, rc.trust_remote_code)
        if rc.device == "cpu":
            fp.to("cpu")
        out["fp16_weight_bytes"] = model_bytes(fp)
        f_logits = logits_for(fp, tok)
        del fp
        rel, agree, n = [], 0, 0
        for a, b in zip(f_logits, q_logits):
            rel.append(((a - b).norm() / a.norm()).item())
            agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
            n += a.shape[0]
        out["logit_rel_err_mean"] = sum(rel) / len(rel)
        out["top1_agreement"] = agree / n
        out["memory_reduction_x"] = round(out["fp16_weight_bytes"] / out["awq_weight_bytes"], 3)
        gates["top1_agreement>=0.75"] = out["top1_agreement"] >= GATE_TOP1
        log(f"[verify] logit rel err {out['logit_rel_err_mean']:.4f} | top-1 agreement {out['top1_agreement']:.3f} "
            f"| weights {out['memory_reduction_x']}x smaller in memory")
    out["gates"] = gates
    out["pass"] = all(gates.values())
    write_json(rc.work_dir / "verify.json", out)
    for k, v in gates.items():
        log(f"[verify] {'PASS' if v else 'FAIL'}  {k}")
    log("[verify] PASS" if out["pass"] else "[verify] FAIL")
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
