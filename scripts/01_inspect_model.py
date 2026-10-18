#!/usr/bin/env python
"""
01_inspect_model.py — GATE 1. Before renting anything: is the model compatible, and what GPU,
how many, how long, how much? Downloads only config.json (+ custom code if trust_remote_code).

    python scripts/01_inspect_model.py --model Qwen/Qwen2.5-7B-Instruct
    python scripts/01_inspect_model.py --config configs/run.yaml
    python scripts/01_inspect_model.py --model deepseek-ai/DeepSeek-V3 --context 32768 --out report.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awq.cli import parser, run_config  # noqa: E402
from awq.hardware import analyze_model, format_report  # noqa: E402
from awq.utils import write_json  # noqa: E402


def main():
    p = parser(__doc__, need_config=False)
    p.add_argument("--model", help="HF id or local folder (instead of --config)")
    p.add_argument("--revision", default="main")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--context", type=int, default=8192, help="serving context length for the KV-cache estimate")
    p.add_argument("--out", help="where to write the JSON report")
    args = p.parse_args()
    if args.config:
        rc = run_config(args)
        model, rev, trc, ns, sl = rc.model, rc.revision, rc.trust_remote_code, rc.calib.n_samples, rc.calib.seq_len
        out = Path(args.out) if args.out else rc.work_dir / "model_report.json"
    elif args.model:
        model, rev, trc, ns, sl = args.model, args.revision, args.trust_remote_code, 128, 512
        out = Path(args.out) if args.out else Path("model_report.json")
    else:
        p.error("give --model or --config")
    r = analyze_model(model, rev, trc, ns, sl, args.context)
    print(format_report(r))
    write_json(out, r)
    print(f"\nwrote {out}")
    sys.exit(1 if r["verdict"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
