#!/usr/bin/env python
"""
08_publish.py — write a model card with ONLY measured numbers and upload to the Hugging Face Hub.

    python scripts/08_publish.py --config configs/run.yaml --repo you/Model-AWQ            # private (default)
    python scripts/08_publish.py --config configs/run.yaml --repo you/Model-AWQ --public
    python scripts/08_publish.py --config configs/run.yaml --repo you/Model-AWQ --dry-run  # card only

Needs a WRITE token: `hf auth login`. The base model's LICENSE file travels with the weights
(it was copied into the quantized folder) — a quantized model is a derivative work.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awq.cli import parser, run_config  # noqa: E402
from awq.utils import Logger, read_json  # noqa: E402


def fmt(x, nd=4):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def model_card(rc, folder: Path) -> str:
    wd = rc.work_dir
    res = read_json(wd / "results.json") if (wd / "results.json").exists() else {}
    base = read_json(wd / "baseline.json") if (wd / "baseline.json").exists() else {}
    ver = read_json(wd / "verify.json") if (wd / "verify.json").exists() else {}
    is_gemm = (folder / "config.json").exists() and "quantization_config" in (folder / "config.json").read_text()
    rows = []
    for d in rc.eval.ppl_datasets:
        rows.append(f"| {d} perplexity (lower is better) | {fmt(base.get(f'fp16_ppl_{d}'))} | "
                    f"{fmt(base.get(f'rtn_ppl_{d}'))} | {fmt(res.get(f'awq_ppl_{d}'))} |")
    gates = "\n".join(f"- {'PASS' if v else 'FAIL'}: {k}" for k, v in res.get("gates", {}).items()) or "- not run"
    usage = ("""```bash
vllm serve <this-repo> --quantization awq_marlin --dtype float16
```""" if is_gemm else """```python
# pip install git+https://github.com/kashyapphassan2000-netizen/WEEK--37
from awq import load_quantized
model = load_quantized("<local snapshot of this repo>", device="cuda")
```""")
    zp = ", zero-point" if rc.quant.zero_point else ""
    mem_fp = fmt((base.get("fp16_weight_bytes") or 0) / 1e9, 3)
    mem_q = fmt((res.get("awq_weight_bytes") or 0) / 1e9, 3)
    return f"""---
base_model: {rc.model}
tags: [awq, quantization, int4, awq-from-scratch]
---

# {rc.model.split('/')[-1]} — AWQ {rc.quant.w_bit}-bit (group {rc.quant.group_size}{zp})

4-bit AWQ (Lin et al. 2023, arXiv:2306.00978) quantization of [`{rc.model}`](https://huggingface.co/{rc.model}),
produced with the from-scratch, block-streaming implementation at
https://github.com/kashyapphassan2000-netizen/WEEK--37 . Every number below was measured, not copied.

## Results (measured)

| metric | original 16-bit | naive RTN 4-bit | **AWQ 4-bit (this)** |
|---|---|---|---|
{chr(10).join(rows)}
| weight memory (GB) | {mem_fp} | — | {mem_q} |

Next-token agreement with the original on probe prompts: {fmt(ver.get('top1_agreement'), 3)}.
Eval: {rc.eval.ppl_seq_len}-token windows; windows per dataset: {base.get('eval_windows', 'n/a')}.

Gates:
{gates}

## Method
- Calibration: {rc.calib.dataset}, {rc.calib.n_samples} x {rc.calib.seq_len} tokens.
- Per-group activation-aware scale search ({rc.search.n_grid} grid points, duo scaling={rc.search.duo_scaling}),
  weight-clip search={rc.search.clip}, scales folded into preceding norms/linears where possible.
- Embeddings and output head kept in 16-bit (standard AWQ practice).

## Use
{usage}

## License
Same license and use policy as the base model `{rc.model}` (see LICENSE / USE_POLICY files in this repo).
"""


def main():
    p = parser(__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--folder", help="folder to upload (default: export/autoawq, else export/native)")
    p.add_argument("--public", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    rc = run_config(args)
    log = Logger(rc.work_dir / "publish.log")
    folder = Path(args.folder) if args.folder else next(
        (f for f in (rc.work_dir / "export" / "autoawq", rc.work_dir / "export" / "native", rc.work_dir / "quantized")
         if f.exists()), None)
    if folder is None:
        sys.exit("nothing to publish: run 03_quantize.py / 07_export.py first")
    (folder / "README.md").write_text(model_card(rc, folder))
    log(f"[publish] model card written: {folder / 'README.md'}")
    if args.dry_run:
        log("[publish] dry run — nothing uploaded")
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, private=not args.public, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=args.repo, commit_message="AWQ 4-bit (awq-from-scratch)")
    log(f"[publish] uploaded -> https://huggingface.co/{args.repo}  ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
