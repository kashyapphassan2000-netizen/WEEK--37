"""
config.py — read configs/run.yaml into typed Python objects.

Plain words: every script in this repo calls `load_config("configs/run.yaml")`.
This file fills in defaults for anything you did not write, refuses obviously
wrong values *before* you spend GPU money, and lets you override any value
from the command line with `--set quant.group_size=64`.
"""
from __future__ import annotations

import copy
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class QuantConfig:
    w_bit: int = 4                      # bits per weight. v1 packs 4-bit only.
    group_size: int = 128               # weights sharing one scale (128 = standard AWQ)
    zero_point: bool = True             # True = asymmetric (standard AWQ format), False = symmetric
    skip_modules: list[str] = field(default_factory=list)  # name substrings kept in 16-bit
    fold_scales: str = "auto"           # auto = fold 1/s into previous op when the family is known
                                        # never = keep 1/s inside every layer (any architecture)
    quantize_experts: bool = True       # quantize MoE expert weights (fused 3D or nn.Linear)


@dataclass
class SearchConfig:
    n_grid: int = 20                    # alpha values tried in [0, 1)
    duo_scaling: bool = True            # s = x_mean^a / w_mean^(1-a)  (official AWQ default)
    max_search_tokens: int = 0          # 0 = use every calibration token; >0 = random subsample (CPU speed)
    clip: bool = True                   # AWQ weight-clipping search
    clip_grid: int = 20
    max_shrink: float = 0.5             # clip search tries max|w| * (1 .. 1-max_shrink)
    clip_tokens: int = 512              # tokens used by the clip search (official value)
    propagate: str = "fp"               # next block input: fp (official AWQ) | quantized (GPTQ-style)


@dataclass
class CalibConfig:
    dataset: str = "pileval"            # pileval | c4 | wikitext2 | /path/to/file.jsonl ({"text": ...})
    n_samples: int = 128
    seq_len: int = 512
    batch_size: int = 8                 # sequences per forward pass through a block


@dataclass
class EvalConfig:
    ppl_datasets: list[str] = field(default_factory=lambda: ["wikitext2", "c4"])
    ppl_seq_len: int = 2048
    max_windows: int = 0                # 0 = full dataset; >0 = first N windows (CPU / quick checks)
    lm_eval_tasks: list[str] = field(
        default_factory=lambda: ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"])
    lm_eval_limit: int | None = None    # docs per task (None = all)
    lm_eval_batch_size: str = "auto"
    gen_max_new_tokens: int = 48
    custom_jsonl: str | None = None     # your own {"prompt":..., "expected":...} lines
    tune_dataset: str = "wikitext2_val"  # held-out split used ONLY by 06_tune.py
    tune_max_windows: int = 40


@dataclass
class IOConfig:
    work_dir: str = "runs/default"


@dataclass
class RunConfig:
    model: str = ""
    revision: str = "main"
    trust_remote_code: bool = False
    dtype: str = "bfloat16"             # compute dtype (use float32 on CPU)
    device: str = "cuda"                # cuda | cpu
    plugin: str | None = None           # path to your own-model plugin .py (see awq/plugin.py)
    seed: int = 42
    quant: QuantConfig = field(default_factory=QuantConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    io: IOConfig = field(default_factory=IOConfig)

    # ---- helpers -----------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @property
    def work_dir(self) -> Path:
        return Path(self.io.work_dir)

    @property
    def torch_dtype(self):
        import torch
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[self.dtype]


def _build(cls, data: dict | None):
    """Turn a plain dict into the dataclass `cls`, recursing into nested dataclasses."""
    data = dict(data or {})
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown config key(s) in [{cls.__name__}]: {sorted(unknown)}")
    kwargs = {}
    for name, f in known.items():
        if name not in data:
            continue
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        if is_dataclass(default):
            kwargs[name] = _build(type(default), data[name])
        else:
            kwargs[name] = data[name]
    return cls(**kwargs)


def _parse_value(text: str) -> Any:
    """'64' -> 64, 'true' -> True, '[a,b]' -> ['a','b'] (YAML rules)."""
    return yaml.safe_load(text)


def apply_overrides(cfg_dict: dict, overrides: dict[str, Any] | list[str] | None) -> dict:
    """Apply {'quant.group_size': 64} or ['quant.group_size=64'] onto a nested dict."""
    out = copy.deepcopy(cfg_dict)
    if not overrides:
        return out
    if isinstance(overrides, list):
        pairs = {}
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"--set expects key=value, got {item!r}")
            k, v = item.split("=", 1)
            pairs[k.strip()] = _parse_value(v)
        overrides = pairs
    for dotted, value in overrides.items():
        node = out
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def validate(cfg: RunConfig) -> RunConfig:
    """Fail fast, with a message that says what to change."""
    if not cfg.model:
        raise ValueError("config: `model:` is empty. Put a Hugging Face id or a local folder path.")
    if cfg.quant.w_bit != 4:
        raise ValueError("config: quant.w_bit must be 4 (this implementation packs 4-bit only).")
    if cfg.quant.group_size not in (-1, 32, 64, 128, 256):
        raise ValueError("config: quant.group_size must be one of -1 (per-channel), 32, 64, 128, 256.")
    if cfg.quant.fold_scales not in ("auto", "never"):
        raise ValueError("config: quant.fold_scales must be 'auto' or 'never'.")
    if cfg.search.propagate not in ("fp", "quantized"):
        raise ValueError("config: search.propagate must be 'fp' or 'quantized'.")
    if cfg.dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("config: dtype must be float32, float16 or bfloat16.")
    if cfg.device not in ("cuda", "cpu"):
        raise ValueError("config: device must be cuda or cpu.")
    if cfg.device == "cpu" and cfg.dtype != "float32":
        raise ValueError("config: on CPU use dtype: float32 (half-precision matmuls on CPU are slow/unsupported).")
    if cfg.calib.n_samples % cfg.calib.batch_size != 0:
        raise ValueError("config: calib.n_samples must be divisible by calib.batch_size.")
    if not 0 < cfg.search.max_shrink < 1:
        raise ValueError("config: search.max_shrink must be in (0, 1).")
    return cfg


def load_config(path: str | Path | None = None, overrides=None, **top_level) -> RunConfig:
    """Load YAML -> apply --set overrides -> build dataclasses -> validate."""
    raw: dict = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    raw.update({k: v for k, v in top_level.items() if v is not None})
    raw = apply_overrides(raw, overrides)
    return validate(_build(RunConfig, raw))
