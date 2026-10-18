#!/usr/bin/env python
"""
03_quantize.py — run AWQ, block by block. Output: <work_dir>/quantized/ (native format) and
<work_dir>/search_log.jsonl (per-block alphas, losses, block error).

    python scripts/03_quantize.py --config configs/run.yaml
    python scripts/03_quantize.py --config configs/run.yaml --resume      # after a crash / reclaimed pod
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awq.cli import parser, run_config  # noqa: E402
from awq.pipeline import quantize_model  # noqa: E402
from awq.utils import Logger  # noqa: E402


def main():
    p = parser(__doc__)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    rc = run_config(args)
    log = Logger(rc.work_dir / "quantize.log")
    log(f"[quantize] config: {rc.to_dict()}")
    quantize_model(rc, resume=args.resume, log=log)


if __name__ == "__main__":
    main()
