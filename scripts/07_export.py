#!/usr/bin/env python
"""
07_export.py — package the result.

  native  : our format (any architecture; load with awq.load_quantized)       -> <work_dir>/export/native
  autoawq : standard AWQ GEMM format for vLLM / SGLang (needs folded scales)  -> <work_dir>/export/autoawq

    python scripts/07_export.py --config configs/run.yaml --format native,autoawq
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safetensors import safe_open  # noqa: E402

from awq.cli import parser, run_config  # noqa: E402
from awq.export import export_autoawq, gemm_dequant  # noqa: E402
from awq.quantizer import dequantize, unpack_int4  # noqa: E402
from awq.utils import Logger, read_json  # noqa: E402


def check_autoawq(qdir: Path, out: Path, log, n: int = 4) -> float:
    """Decode a few layers from both formats; they must give identical weights."""
    qc = read_json(qdir / "quant_config.json")
    names = list(qc["linears"])[:n]
    worst = 0.0
    nat, gem = {}, {}
    for f in qdir.glob("*.safetensors"):
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                nat[k] = f
    for f in out.glob("*.safetensors"):
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                gem[k] = f
    for name in names:
        with safe_open(str(nat[name + ".qweight"]), framework="pt") as h:
            w_nat = dequantize(unpack_int4(h.get_tensor(name + ".qweight")), h.get_tensor(name + ".scales").float(),
                               h.get_tensor(name + ".qzeros"), qc["group_size"], dtype=__import__("torch").float32)
        with safe_open(str(gem[name + ".qweight"]), framework="pt") as h:
            w_gem = gemm_dequant(h.get_tensor(name + ".qweight"), h.get_tensor(name + ".qzeros"),
                                 h.get_tensor(name + ".scales"), qc["group_size"])
        d = (w_nat - w_gem).abs().max().item() / (w_nat.abs().max().item() + 1e-12)
        worst = max(worst, d)
        log(f"[export] check {name}: max rel diff native vs GEMM = {d:.2e}")
    return worst


def main():
    p = parser(__doc__)
    p.add_argument("--format", default="native,autoawq")
    args = p.parse_args()
    rc = run_config(args)
    log = Logger(rc.work_dir / "export.log")
    qdir = rc.work_dir / "quantized"
    fmts = [f.strip() for f in args.format.split(",") if f.strip()]
    ok = True
    if "native" in fmts:
        dst = rc.work_dir / "export" / "native"
        shutil.copytree(qdir, dst, dirs_exist_ok=True)
        log(f"[export] native -> {dst}")
    if "autoawq" in fmts:
        dst = rc.work_dir / "export" / "autoawq"
        try:
            export_autoawq(qdir, dst, log=log)
            worst = check_autoawq(qdir, dst, log)
            ok &= worst < 1e-2           # fp16 rounding of scales only
            log(f"[export] GEMM layout check {'PASS' if worst < 1e-2 else 'FAIL'} (worst rel diff {worst:.2e})")
        except RuntimeError as e:
            log(f"[export] autoawq NOT produced: {e}")
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
