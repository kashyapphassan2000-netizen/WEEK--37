"""
utils.py — small helpers shared by every script: logging, JSON, environment info, timing.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


class Logger:
    """print() that also appends to a log file, with elapsed time."""

    def __init__(self, path: str | Path | None = None):
        self.t0 = time.time()
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *msg):
        line = f"[{time.time() - self.t0:8.1f}s] " + " ".join(str(m) for m in msg)
        print(line, flush=True)
        if self.path:
            with open(self.path, "a") as fh:
                fh.write(line + "\n")


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def env_info() -> dict:
    import torch

    info = {
        "time_utc": now_iso(),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpus"] = [
            {"name": torch.cuda.get_device_name(i),
             "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
            for i in range(torch.cuda.device_count())
        ]
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


def dir_size_gb(path: str | Path, pattern: str = "*.safetensors") -> float:
    return round(sum(f.stat().st_size for f in Path(path).glob(pattern)) / 1e9, 4)
