"""
pipeline.py — quantize a whole model, ONE decoder block at a time.

Plain words, the loop:
  1. Tokenize calibration text -> (n_samples, seq_len) tokens.
  2. Run embeddings and stop right before block 0 ("Catcher") -> inputs of block 0.
  3. For block i = 0 .. L-1:
       a. load block i's weights (only this block is on the GPU)
       b. run it once with hooks: record each Linear's input; Linears that receive the very
          same tensor form a "group" (q/k/v, gate/up, ...). MoE experts record their routed tokens.
       c. AWQ scale search per group, apply the scales (fold 1/s upstream when possible)
       d. AWQ clip search per Linear
       e. replace each Linear with a 4-bit AwqLinear (experts -> AwqExpertsWeight)
       f. run the quantized block -> block error; outputs become inputs of block i+1
       g. save block i to disk + progress file  (so a killed pod can --resume here)
  4. Save embeddings/head (16-bit), config, tokenizer and quant_config.json.
"""
from __future__ import annotations

import json
import shutil
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .awq_linear import AwqExpertsWeight, AwqLinear
from .blocks import (FOLD_FAMILIES, allow_clip, fold_target, fused_experts, model_type,
                     quantizable_linears)
from .calib import get_calib_tokens
from .config import RunConfig
from .model_io import ModelSource, resolve_model_path
from .search import apply_clip, search_clip, search_scale
from .utils import Logger, git_commit, now_iso, write_json


class _Stop(Exception):
    pass


class _Catcher(nn.Module):
    """Stands in for block 0: stores its inputs, then stops the forward pass."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.hidden: list[torch.Tensor] = []
        self.kwargs: dict | None = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["block"], name)   # e.g. `.attention_type` read by the model

    def forward(self, hidden_states, *args, **kwargs):
        self.hidden.append(hidden_states.detach())
        if self.kwargs is None:
            self.kwargs = kwargs
        raise _Stop


def _clean_kwargs(kw: dict) -> dict:
    kw = dict(kw)
    for k in ("past_key_values", "past_key_value", "layer_past"):
        kw.pop(k, None)
    kw["use_cache"] = False
    return kw


@torch.no_grad()
def capture_block0_inputs(model: nn.Module, blocks: nn.ModuleList, batches, device):
    """batches: iterable of model-input dicts. Returns (inputs (N, T, h), layer_kwargs)."""
    is_hf = hasattr(model, "config") and hasattr(model, "generate")
    extra = {"use_cache": False} if is_hf else {}
    catcher = _Catcher(blocks[0])
    blocks[0] = catcher
    try:
        for batch in batches:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            try:
                model(**batch, **extra)
            except _Stop:
                pass
    finally:
        blocks[0] = catcher.block
    kw = _clean_kwargs(catcher.kwargs or {}) if is_hf else dict(catcher.kwargs or {})
    return torch.cat(catcher.hidden, dim=0), kw


@torch.no_grad()
def run_block(block: nn.Module, inps: torch.Tensor, kwargs: dict, bs: int) -> torch.Tensor:
    outs = []
    for i in range(0, inps.shape[0], bs):
        o = block(inps[i:i + bs], **kwargs)
        outs.append(o[0] if isinstance(o, (tuple, list)) else o)
    return torch.cat(outs, dim=0)


class _ExpertRecorder:
    """While active, F.linear calls whose weight is expert e of a fused 3-D parameter are recorded."""

    def __init__(self, experts):
        self.map = {}
        self.feats = defaultdict(list)
        for mod_name, pname, mod in experts:
            p = getattr(mod, pname)
            for e in range(p.shape[0]):
                self.map[p[e].data_ptr()] = (f"{mod_name}.{pname}", e, tuple(p.shape[1:]))
        self._orig = None

    def __enter__(self):
        self._orig = F.linear
        rec = self

        def linear(input, weight, bias=None):
            hit = rec.map.get(weight.data_ptr()) if weight.dim() == 2 else None
            if hit is not None and tuple(weight.shape) == hit[2]:
                rec.feats[(hit[0], hit[1])].append(input.detach().reshape(-1, input.shape[-1]))
            return rec._orig(input, weight, bias)

        torch.nn.functional.linear = linear
        return self

    def __exit__(self, *exc):
        torch.nn.functional.linear = self._orig


@torch.no_grad()
def collect(block, inps, kwargs, bs, linears: dict, experts):
    """Run the FP block once. Returns (fp_outputs, {name: input tensor list}, groups, expert_feats)."""
    feats, first_key, handles = defaultdict(list), {}, []

    def make_hook(name):
        def hook(mod, args, kw, out):
            x = args[0] if args else kw.get("input")
            if name not in first_key:
                first_key[name] = (x.data_ptr(), tuple(x.shape), x.stride())
            feats[name].append(x.detach().reshape(-1, x.shape[-1]))
        return hook

    for name, m in linears.items():
        handles.append(m.register_forward_hook(make_hook(name), with_kwargs=True))
    rec = _ExpertRecorder(experts)
    try:
        with rec:
            outs = run_block(block, inps, kwargs, bs)
    finally:
        for h in handles:
            h.remove()
    groups: OrderedDict[tuple, list[str]] = OrderedDict()
    for name in linears:                      # module order = execution order for standard blocks
        if name in first_key:
            groups.setdefault(first_key[name], []).append(name)
    return outs, feats, list(groups.values()), rec.feats


def _fold(block, target, s: torch.Tensor):
    """Absorb x/s into the op that produces x (so no runtime cost)."""
    kind, name = target
    mod = block.get_submodule(name)
    w = mod.weight.data
    hp = torch.float64 if w.dtype == torch.float64 else torch.float32     # compute in >= fp32
    s = s.to(w.device, hp)
    if kind == "norm":
        mod.weight.data = (w.to(hp) / s).to(w.dtype)
    elif kind == "norm_offset":                    # Gemma RMSNorm multiplies by (1 + w)
        mod.weight.data = ((1 + w.to(hp)) / s - 1).to(w.dtype)
    elif kind == "linear_out":                     # previous Linear: scale its output rows
        mod.weight.data = (w.to(hp) / s.view(-1, 1)).to(w.dtype)
    if getattr(mod, "bias", None) is not None:
        mod.bias.data = (mod.bias.data.to(hp) / s).to(mod.bias.dtype)


@torch.no_grad()
def quantize_block(block: nn.Module, idx: int, inps: torch.Tensor, kwargs: dict, rc: RunConfig,
                   family: str, log) -> tuple[torch.Tensor, dict, dict]:
    q, sc = rc.quant, rc.search
    bs = rc.calib.batch_size
    linears, skipped = quantizable_linears(block, q.group_size, q.skip_modules)
    experts = fused_experts(block) if q.quantize_experts else []
    t0 = time.time()
    outs_fp, feats, groups, expert_feats = collect(block, inps, kwargs, bs, linears, experts)
    report = {"block": idx, "groups": [], "experts": {}, "skipped": skipped}

    # ---- (c) scale search for every group, then apply -------------------------------------
    inv_s_of: dict[str, torch.Tensor] = {}
    group_x: dict[str, torch.Tensor] = {}
    plans = []
    for names in groups:
        x = torch.cat(feats[names[0]], dim=0)
        for n in names:
            feats.pop(n, None)
        target = fold_target(block, names, family) if q.fold_scales == "auto" else None
        searchable = target is not None or q.fold_scales == "never" or family not in FOLD_FAMILIES
        ws = [linears[n].weight.data for n in names]
        if searchable:
            res = search_scale(ws, [linears[n].bias for n in names], x, q.w_bit, q.group_size,
                               q.zero_point, sc.n_grid, sc.duo_scaling, sc.max_search_tokens)
        else:
            res = None
        plans.append((names, target, res, x))
        report["groups"].append({
            "linears": names,
            "fold": None if target is None else f"{target[0]}:{target[1]}",
            "mode": "folded" if target else ("inv_s" if searchable else "no-scale (export-compatible)"),
            "ratio": None if res is None or res["ratio"] is None else round(res["ratio"], 3),
            "loss_awq": None if res is None else res["loss"],
            "loss_rtn": None if res is None else res["loss_rtn"],
        })
    for names, target, res, x in plans:
        if res is not None and res["ratio"] is not None:
            s = res["scales"].to(x.device)
            for n in names:
                w = linears[n].weight
                w.data = (w.data.float() * s.view(1, -1)).to(w.dtype)
            if target is not None:
                _fold(block, target, s)
            else:
                for n in names:
                    inv_s_of[n] = (1.0 / s)
            x = (x.float() / s.view(1, -1)).to(x.dtype)
        for n in names:
            group_x[n] = x

    # ---- (d) clip search + (e) quantize ------------------------------------------------------
    for name, lin in linears.items():
        if name not in group_x:
            skipped[name] = "never executed during calibration (kept 16-bit)"
            continue
        if sc.clip and allow_clip(name):
            mv = search_clip(lin.weight.data, group_x[name], q.w_bit, q.group_size, q.zero_point,
                             sc.clip_grid, sc.max_shrink, sc.clip_tokens)
            lin.weight.data = apply_clip(lin.weight.data, mv)
        new = AwqLinear.from_float(lin, q.w_bit, q.group_size, q.zero_point, inv_s_of.get(name))
        parent_name, _, attr = name.rpartition(".")
        setattr(block.get_submodule(parent_name) if parent_name else block, attr, new)
    group_x.clear()

    # ---- MoE fused experts: per-expert AWQ (scale kept inside, never folded) -------------------
    for mod_name, pname, mod in experts:
        key = f"{mod_name}.{pname}"
        p = getattr(mod, pname)
        E, out_f, in_f = p.shape
        if in_f % (in_f if q.group_size <= 0 else q.group_size) != 0:
            skipped[key] = "expert in_features not divisible by group_size"
            continue
        qmod = AwqExpertsWeight(E, out_f, in_f, q.w_bit, q.group_size, dtype=p.dtype, device=p.device)
        stats = {"experts": E, "with_tokens": 0, "tokens_min": None, "tokens_max": 0}
        for e in range(E):
            w = p[e].data
            xs = expert_feats.get((key, e))
            if xs:
                x = torch.cat(xs, dim=0)
                n_tok = x.shape[0]
                stats["with_tokens"] += 1
                stats["tokens_max"] = max(stats["tokens_max"], n_tok)
                stats["tokens_min"] = n_tok if stats["tokens_min"] is None else min(stats["tokens_min"], n_tok)
                res = search_scale([w], [None], x, q.w_bit, q.group_size, q.zero_point, sc.n_grid,
                                   sc.duo_scaling, sc.max_search_tokens)
                inv = None
                if res["ratio"] is not None:
                    s = res["scales"].to(w.device)
                    w = (w.float() * s.view(1, -1)).to(w.dtype)
                    x = (x.float() / s.view(1, -1)).to(x.dtype)
                    inv = 1.0 / s
                if sc.clip:
                    w = apply_clip(w, search_clip(w, x, q.w_bit, q.group_size, q.zero_point,
                                                  sc.clip_grid, sc.max_shrink, sc.clip_tokens))
                qmod.set_expert(e, w, inv, q.zero_point)
            else:                                   # expert never routed to: plain RTN
                qmod.set_expert(e, w, None, q.zero_point)
        mod._parameters.pop(pname, None)      # a Parameter slot cannot hold a Module
        setattr(mod, pname, qmod)
        report["experts"][key] = stats
    expert_feats.clear()

    # ---- (f) quantized forward -> block error + next inputs ----------------------------------
    outs_q = run_block(block, inps, kwargs, bs)
    denom = outs_fp.float().pow(2).mean().item() + 1e-12
    report["block_rel_mse"] = (outs_q.float() - outs_fp.float()).pow(2).mean().item() / denom
    report["seconds"] = round(time.time() - t0, 2)
    meta = {
        "linears": {n: {"inv_s": n in inv_s_of} for n, m in block.named_modules() if isinstance(m, AwqLinear)},
        "experts": {n: {"shape": [m.num_experts, m.out_features, m.in_features]}
                    for n, m in block.named_modules() if isinstance(m, AwqExpertsWeight)},
        "skipped": skipped,
    }
    nxt = outs_fp if sc.propagate == "fp" else outs_q
    del outs_q
    return nxt, report, meta


# ---------------------------------------------------------------------------
def _copy_side_files(src: Path, dst: Path, trust_remote_code: bool) -> None:
    pats = ["tokenizer*", "*.model", "*.tiktoken", "vocab*", "merges*", "special_tokens_map.json",
            "generation_config.json", "chat_template*", "LICENSE*", "NOTICE*", "USE_POLICY*"]
    if trust_remote_code:
        pats.append("*.py")
    for pat in pats:
        for f in src.glob(pat):
            if f.is_file():
                shutil.copy2(f, dst / f.name)
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))


def quantize_model(rc: RunConfig, resume: bool = False, log: Logger | None = None,
                   calib_tokens: torch.Tensor | None = None) -> Path:
    """Quantize rc.model into <work_dir>/quantized. `calib_tokens` overrides the dataset (tests)."""
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    work = rc.work_dir
    log = log or Logger(work / "quantize.log")
    torch.manual_seed(rc.seed)
    device = torch.device(rc.device)
    qdir, state = work / "quantized", work / "state"
    qdir.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    rc.save(work / "run_config.yaml")

    if rc.plugin:
        from .plugin import PluginSource
        src = PluginSource(rc, log)
        path, tok = None, src.tokenizer
    else:
        path = resolve_model_path(rc.model, rc.revision, rc.trust_remote_code)
        log(f"[quantize] model files: {path}")
        src = ModelSource(path, rc.torch_dtype, rc.trust_remote_code, log=log)
        tok = None if calib_tokens is not None else AutoTokenizer.from_pretrained(
            str(path), trust_remote_code=rc.trust_remote_code)
    family = model_type(src.model)
    n_blocks = len(src.blocks)
    log(f"[quantize] family={family} blocks={n_blocks} fold_scales={rc.quant.fold_scales} "
        f"(fold-capable family: {family in FOLD_FAMILIES})")

    progress_f = state / "progress.json"
    start = 0
    if resume and progress_f.exists():
        start = json.loads(progress_f.read_text())["next_block"]
        inps = torch.load(state / "inps.pt", map_location=device)
        kwargs = torch.load(state / "kwargs.pt", map_location=device, weights_only=False)
        log(f"[quantize] RESUME from block {start}/{n_blocks}")
    else:
        for f in qdir.glob("block-*.safetensors"):
            f.unlink()
        (work / "search_log.jsonl").unlink(missing_ok=True)
        src.load_non_block(device)
        if rc.plugin:
            batches = src.calib_batches(rc.calib.n_samples, rc.calib.seq_len, rc.calib.batch_size)
        else:
            calib = calib_tokens if calib_tokens is not None else get_calib_tokens(
                tok, rc.calib.dataset, rc.calib.n_samples, rc.calib.seq_len, rc.seed)
            src_name = "custom tokens" if calib_tokens is not None else rc.calib.dataset
            log(f"[quantize] calibration: {src_name} {tuple(calib.shape)} tokens")
            batches = ({"input_ids": calib[i:i + rc.calib.batch_size]}
                       for i in range(0, calib.shape[0], rc.calib.batch_size))
        inps, kwargs = capture_block0_inputs(src.model, src.blocks, batches, device)
        src.unload_non_block()
        torch.save(kwargs, state / "kwargs.pt")
        log(f"[quantize] block-0 inputs {tuple(inps.shape)} {inps.dtype}")

    for i in range(start, n_blocks):
        t = time.time()
        block = src.load_block(i, device)
        inps, report, meta = quantize_block(block, i, inps, kwargs, rc, family, log)
        prefix = f"{src.blocks_name}.{i}."
        sd = {prefix + k: v.detach().cpu().contiguous() for k, v in block.state_dict().items()}
        save_file(sd, str(qdir / f"block-{i:05d}.safetensors"))
        meta_full = {"linears": {prefix + k: v for k, v in meta["linears"].items()},
                     "experts": {prefix + k: v for k, v in meta["experts"].items()},
                     "skipped": {prefix + k: v for k, v in meta["skipped"].items()}}
        write_json(state / f"meta-{i:05d}.json", meta_full)
        src.unload_block(i)
        with open(work / "search_log.jsonl", "a") as fh:
            fh.write(json.dumps(report) + "\n")
        torch.save(inps, state / "inps.pt")
        progress_f.write_text(json.dumps({"next_block": i + 1, "n_blocks": n_blocks}))
        ratios = [g["ratio"] for g in report["groups"] if g["ratio"] is not None]
        log(f"[quantize] block {i + 1}/{n_blocks}  rel_mse={report['block_rel_mse']:.2e}  "
            f"alphas={ratios}  experts={len(report['experts'])}  {time.time() - t:.1f}s")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- non-block weights + metadata ---------------------------------------------------------
    nb = src.non_block_state_dict()
    save_file(nb, str(qdir / "non_block.safetensors"))
    linears, experts, skipped = {}, {}, {}
    for f in sorted(state.glob("meta-*.json")):
        m = json.loads(f.read_text())
        linears.update(m["linears"])
        experts.update(m["experts"])
        skipped.update(m["skipped"])
    if path is not None:
        _copy_side_files(path, qdir, rc.trust_remote_code)
    else:
        src.save_side_files(qdir)
    qc = {
        "format": "awq-from-scratch-native-v1",
        "source_model": rc.model,
        "revision": rc.revision,
        "family": family,
        "w_bit": rc.quant.w_bit,
        "group_size": rc.quant.group_size,
        "zero_point": rc.quant.zero_point,
        "fold_scales": rc.quant.fold_scales,
        "dtype": rc.dtype,
        "trust_remote_code": rc.trust_remote_code,
        "tie_word_embeddings": bool(src.tied),
        "plugin": rc.plugin,
        "calibration": {"dataset": rc.calib.dataset, "n_samples": rc.calib.n_samples,
                        "seq_len": rc.calib.seq_len},
        "search": rc.to_dict()["search"],
        "n_linears_quantized": len(linears),
        "n_linears_with_inv_s": sum(1 for v in linears.values() if v["inv_s"]),
        "n_expert_tensors_quantized": len(experts),
        "linears": linears,
        "experts": experts,
        "skipped": skipped,
        "created_utc": now_iso(),
        "git_commit": git_commit(),
    }
    write_json(qdir / "quant_config.json", qc)
    log(f"[quantize] DONE -> {qdir}  ({len(linears)} linears, {len(experts)} expert tensors, "
        f"{len(skipped)} skipped)")
    return qdir
