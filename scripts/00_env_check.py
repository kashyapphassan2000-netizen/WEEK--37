#!/usr/bin/env python
"""
00_env_check.py — GATE 0. Is this machine ready? Prints every version that matters and the exact
fix for anything that is wrong. Ends with "ENV OK" or "ENV FAIL".

    python scripts/00_env_check.py                       # generic check
    python scripts/00_env_check.py --config configs/run.yaml   # also checks disk vs. this model
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awq.cli import parser, run_config  # noqa: E402
from awq.utils import env_info, write_json  # noqa: E402


def main():
    p = parser(__doc__, need_config=False)
    args = p.parse_args()
    rc = run_config(args) if args.config else None
    problems, warnings = [], []
    info = env_info()
    for k, v in info.items():
        print(f"  {k:<16} {v}")

    import torch
    want_cuda = rc is None or rc.device == "cuda"
    if want_cuda and not torch.cuda.is_available():
        (problems if rc else warnings).append(
            "CUDA not available. On a GPU box: check `nvidia-smi`, then reinstall torch with "
            "`pip install torch --index-url https://download.pytorch.org/whl/cu128` (cuXYZ <= driver's CUDA Version).")
    if torch.cuda.is_available():
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(20):
            x @ x
        torch.cuda.synchronize()
        tflops = 20 * 2 * 4096 ** 3 / (time.time() - t) / 1e12
        info["fp16_matmul_tflops"] = round(tflops, 1)
        print(f"  fp16 matmul      {tflops:.1f} TFLOP/s")
        if rc and rc.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            problems.append("GPU has no bfloat16 (pre-Ampere, e.g. T4/V100): set dtype: float16 in run.yaml.")

    work = Path(rc.io.work_dir if rc else ".").resolve()
    probe = work if work.exists() else next(p for p in [work, *work.parents] if p.exists())
    free_gb = shutil.disk_usage(probe).free / 1e9
    info["disk_free_gb"] = round(free_gb, 1)
    print(f"  disk free        {free_gb:.1f} GB at {probe}")
    if rc:
        try:
            from awq.hardware import analyze_model
            r = analyze_model(rc.model, rc.revision, rc.trust_remote_code, rc.calib.n_samples, rc.calib.seq_len,
                              log=lambda *a: None)
            need = r["estimate"]["disk_gb"]
            info["disk_needed_gb_est"] = need
            print(f"  disk needed      ~{need} GB (estimate for {rc.model})")
            if free_gb < need:
                problems.append(f"Not enough disk: {free_gb:.0f} GB free, ~{need:.0f} GB needed. "
                                "Use a bigger volume and put HF_HOME + work_dir on it.")
        except Exception as e:
            warnings.append(f"could not size the model ({type(e).__name__}: {e})")

    try:
        from huggingface_hub import whoami
        user = whoami()["name"]
        info["hf_user"] = user
        print(f"  HF login         {user}")
    except Exception:
        warnings.append("Not logged in to Hugging Face (fine for public models). Gated models: `hf auth login`.")

    for mod in ("transformers", "accelerate", "datasets", "safetensors", "huggingface_hub", "yaml"):
        try:
            __import__(mod)
        except ImportError:
            problems.append(f"missing package {mod}: pip install -e .")
    try:
        import lm_eval  # noqa: F401
        info["lm_eval"] = True
    except ImportError:
        warnings.append("lm-eval not installed (downstream tasks skipped): pip install -e '.[eval]'")

    for w in warnings:
        print("  WARN  ", w)
    for pr in problems:
        print("  FAIL  ", pr)
    out = (rc.work_dir if rc else Path(".")) / "env.json"
    info["problems"], info["warnings"] = problems, warnings
    write_json(out, info)
    print("ENV OK" if not problems else "ENV FAIL", f"(wrote {out})")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
