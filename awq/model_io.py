"""
model_io.py — getting weights in and out WITHOUT ever needing the whole model in memory.

Plain words:
  1. `resolve_model_path`  turns "Qwen/Qwen2.5-7B" or "/workspace/models/x" into a local folder
     (downloading only the files we need: configs, tokenizer, *.safetensors).
  2. `build_skeleton`      builds the model's Python structure on the "meta" device: every layer
     exists, but no weight memory is allocated. A 1-trillion-parameter skeleton costs ~0 bytes.
  3. `WeightStore`         opens the safetensors shards lazily and reads ONE tensor at a time.
     It also handles: FP8 checkpoints (DeepSeek-style block scales), renamed keys, and the
     per-expert -> fused-experts layout that transformers >= 5 uses for MoE models.
  4. `ModelSource`         ties it together: load block i to the GPU, quantize it, throw it away.
     If the checkpoint layout cannot be streamed, it falls back to a normal full CPU load
     (then you need CPU RAM >= model size) and says so loudly.
  5. `load_quantized`      rebuilds a quantized model from our native output folder.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import torch
import torch.nn as nn

from .blocks import find_blocks

WEIGHT_PATTERNS = ["*.json", "*.safetensors", "tokenizer*", "*.model", "*.tiktoken", "*.txt",
                   "LICENSE*", "*.md", "NOTICE*", "USE_POLICY*"]


# ---------------------------------------------------------------------------
# 1. where are the files?
# ---------------------------------------------------------------------------
def resolve_model_path(model: str, revision: str = "main", trust_remote_code: bool = False,
                       weights: bool = True) -> Path:
    p = Path(model).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import list_repo_files, snapshot_download

    patterns = list(WEIGHT_PATTERNS) if weights else ["*.json", "tokenizer*", "*.model", "*.tiktoken", "*.txt"]
    if trust_remote_code:
        patterns.append("*.py")
    if weights:
        files = list_repo_files(model, revision=revision)
        if not any(f.endswith(".safetensors") for f in files):
            patterns.append("*.bin")                      # old checkpoints without safetensors
    return Path(snapshot_download(model, revision=revision, allow_patterns=patterns))


# ---------------------------------------------------------------------------
# 2. the empty skeleton
# ---------------------------------------------------------------------------
def load_hf_config(path: Path | str, trust_remote_code: bool = False):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(path), trust_remote_code=trust_remote_code)
    return config


def checkpoint_is_fp8(config) -> bool:
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return False
    method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
    return str(method).lower() == "fp8"


def already_quantized_method(config) -> str | None:
    """Return 'awq'/'gptq'/'bitsandbytes'/... if the checkpoint is already low-bit (wrong source)."""
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return None
    method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
    method = str(method).lower()
    return None if method == "fp8" else method


def _strip_quant_config(config):
    if hasattr(config, "quantization_config"):
        try:
            delattr(config, "quantization_config")
        except AttributeError:
            config.quantization_config = None
    return config


def force_eager_experts(model: nn.Module) -> None:
    """Fused-expert models must run the plain per-expert loop so quantized weights can be served."""
    for cfg in {id(m.config): m.config for m in model.modules() if hasattr(m, "config")}.values():
        if hasattr(cfg, "_experts_implementation"):
            cfg._experts_implementation = "eager"


def build_skeleton(path: Path | str, dtype: torch.dtype, trust_remote_code: bool = False):
    """Model structure on the meta device (no weight memory). Buffers like rotary tables are real."""
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    config = _strip_quant_config(load_hf_config(path, trust_remote_code))
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype, trust_remote_code=trust_remote_code)
    model.eval()
    force_eager_experts(model)
    return model


# ---------------------------------------------------------------------------
# 3. lazy weight reader
# ---------------------------------------------------------------------------
_EXPERT_PAIRS = [("w1", "w3", "w2"), ("gate_proj", "up_proj", "down_proj")]
_RENAMES = [(".mlp.", ".block_sparse_moe."), (".mlp.", ".feed_forward.")]


class WeightStore:
    """Reads individual tensors from a checkpoint folder on demand."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.key_to_file: dict[str, str] = {}
        self._handles: dict[str, object] = {}
        self._bin_cache: dict[str, dict] = {}
        idx = self.path / "model.safetensors.index.json"
        st_files = sorted(glob.glob(str(self.path / "*.safetensors")))
        if idx.exists():
            self.key_to_file = {k: str(self.path / v) for k, v in json.loads(idx.read_text())["weight_map"].items()}
            self.kind = "safetensors"
        elif st_files:
            from safetensors import safe_open
            for f in st_files:
                with safe_open(f, framework="pt") as h:
                    for k in h.keys():
                        self.key_to_file[k] = f
            self.kind = "safetensors"
        else:
            bidx = self.path / "pytorch_model.bin.index.json"
            if bidx.exists():
                wmap = json.loads(bidx.read_text())["weight_map"]
                self.key_to_file = {k: str(self.path / v) for k, v in wmap.items()}
            else:
                for f in sorted(glob.glob(str(self.path / "*.bin"))):
                    sd = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
                    self._bin_cache[f] = sd
                    for k in sd:
                        self.key_to_file[k] = f
            self.kind = "bin"
        if not self.key_to_file:
            raise FileNotFoundError(f"No *.safetensors or *.bin weights found in {self.path}")

    def keys(self):
        return self.key_to_file.keys()

    def has(self, key: str) -> bool:
        return key in self.key_to_file

    def _raw(self, key: str) -> torch.Tensor:
        f = self.key_to_file[key]
        if self.kind == "safetensors":
            if f not in self._handles:
                from safetensors import safe_open
                self._handles[f] = safe_open(f, framework="pt")
            return self._handles[f].get_tensor(key)
        if f not in self._bin_cache:
            self._bin_cache[f] = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
        return self._bin_cache[f][key]

    def get(self, key: str) -> torch.Tensor:
        """One tensor, FP8 block-dequantized to float32 if needed."""
        t = self._raw(key)
        if t.dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
            scale_key = key + "_scale_inv" if key.endswith("weight") else key.replace(".weight", ".weight_scale_inv")
            t = t.float()
            if self.has(scale_key):
                s = self._raw(scale_key).float()
                bo = -(-t.shape[0] // s.shape[0])          # block size along rows (128 for DeepSeek)
                bi = -(-t.shape[1] // s.shape[1])
                s = s.repeat_interleave(bo, 0)[: t.shape[0]].repeat_interleave(bi, 1)[:, : t.shape[1]]
                t = t * s
        return t

    # --- map a model parameter name to checkpoint tensor(s) -------------------
    def resolve(self, model_key: str) -> torch.Tensor:
        for cand in self._candidates(model_key):
            if self.has(cand):
                return self.get(cand)
        fused = self._fused_expert(model_key)
        if fused is not None:
            return fused
        raise KeyError(model_key)

    def _candidates(self, key: str):
        yield key
        for new, old in _RENAMES:
            if new in key:
                yield key.replace(new, old)

    def _fused_expert(self, key: str) -> torch.Tensor | None:
        m = re.match(r"(.*\.experts)\.(gate_up_proj|down_proj|up_proj)$", key)
        if not m:
            return None
        base, kind = m.groups()
        for prefix in self._candidates(base):
            for g_name, u_name, d_name in _EXPERT_PAIRS:
                first = f"{prefix}.0.{g_name}.weight"
                if not self.has(first):
                    continue
                n = 0
                while self.has(f"{prefix}.{n}.{g_name}.weight"):
                    n += 1
                if kind == "gate_up_proj":
                    gate = torch.stack([self.get(f"{prefix}.{e}.{g_name}.weight") for e in range(n)])
                    up = torch.stack([self.get(f"{prefix}.{e}.{u_name}.weight") for e in range(n)])
                    return torch.cat([gate, up], dim=1)
                name = d_name if kind == "down_proj" else u_name
                return torch.stack([self.get(f"{prefix}.{e}.{name}.weight") for e in range(n)])
        return None


def _set_tensor(module: nn.Module, name: str, value: torch.Tensor, device, dtype) -> None:
    from accelerate.utils import set_module_tensor_to_device

    if value.is_floating_point():
        value = value.to(dtype)
    set_module_tensor_to_device(module, name, device, value=value)


def materialize(module: nn.Module, prefix: str, store: WeightStore, device, dtype,
                skip_prefixes: tuple[str, ...] = ()) -> list[str]:
    """Fill every meta parameter of `module` from the checkpoint. Returns names that were missing."""
    missing = []
    for name, p in list(module.named_parameters()) + list(module.named_buffers()):
        full = prefix + name
        if any(full.startswith(s) for s in skip_prefixes):
            continue
        is_meta = p.device.type == "meta"
        try:
            value = store.resolve(full)
        except KeyError:
            if is_meta:
                missing.append(full)
            elif p.device != torch.device(device):
                _set_tensor(module, name, p.data, device, p.dtype if not p.is_floating_point() else dtype)
            continue
        _set_tensor(module, name, value, device, dtype)
    return missing


def _module_to_meta(module: nn.Module) -> None:
    """Release memory: turn every tensor of `module` back into a meta tensor."""
    module.to_empty(device="meta")


# ---------------------------------------------------------------------------
# 4. the streaming source used by the pipeline
# ---------------------------------------------------------------------------
class ModelSource:
    """Hands out one decoder block at a time.

    mode='stream': skeleton on meta + per-tensor reads (RAM ~ one block).
    mode='full'  : transformers.from_pretrained on CPU, blocks moved to GPU one by one
                   (fallback when the checkpoint layout cannot be mapped; RAM ~ model size).
    """

    def __init__(self, path: Path, dtype: torch.dtype, trust_remote_code: bool = False,
                 mode: str = "auto", log=print):
        self.path, self.dtype, self.trust_remote_code, self.log = Path(path), dtype, trust_remote_code, log
        self.config = load_hf_config(path, trust_remote_code)
        self.is_fp8 = checkpoint_is_fp8(self.config)
        self.mode = mode
        if mode in ("auto", "stream"):
            self.model = build_skeleton(path, dtype, trust_remote_code)
            self.store = WeightStore(path)
            self.blocks_name, self.blocks = find_blocks(self.model)
            if mode == "auto" and not self._layout_streamable():
                self.log("[model_io] checkpoint keys do not match the model layout -> FULL CPU LOAD fallback "
                         "(needs CPU RAM >= model size).")
                self.mode = "full"
            else:
                self.mode = "stream"
        if self.mode == "full":
            self.model = self._full_load()
            self.store = None
            self.blocks_name, self.blocks = find_blocks(self.model)
        self.log(f"[model_io] mode={self.mode}  blocks={self.blocks_name} x{len(self.blocks)}  fp8_ckpt={self.is_fp8}")

    def _layout_streamable(self) -> bool:
        """Check that every tensor of block 0 and the non-block parts can be found in the checkpoint."""
        try:
            for name, p in self.model.named_parameters():
                if name.startswith(self.blocks_name + ".") and not name.startswith(self.blocks_name + ".0."):
                    continue
                if self.store.has(name):
                    continue
                if self.tied and name.endswith("lm_head.weight"):
                    continue
                self.store.resolve(name)               # raises KeyError if not resolvable
            return True
        except KeyError as e:
            self.log(f"[model_io] unresolvable key: {e}")
            return False

    @property
    def tied(self) -> bool:
        cfg = self.config.get_text_config() if hasattr(self.config, "get_text_config") else self.config
        return bool(getattr(cfg, "tie_word_embeddings", False))

    def _full_load(self):
        from transformers import AutoModelForCausalLM

        kwargs = dict(dtype=self.dtype, trust_remote_code=self.trust_remote_code, low_cpu_mem_usage=True)
        if self.is_fp8:
            from transformers import FineGrainedFP8Config
            kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        model = AutoModelForCausalLM.from_pretrained(str(self.path), **kwargs)
        model.eval()
        force_eager_experts(model)
        return model

    # non-block = embeddings, final norm, lm_head, rotary tables ...
    def _non_block_modules(self):
        b = self.blocks_name
        for name, mod in self.model.named_modules():
            if name == b or name.startswith(b + "."):
                continue
            yield name, mod

    def load_non_block(self, device) -> None:
        if self.mode == "stream":
            missing = materialize(self.model, "", self.store, device, self.dtype,
                                  skip_prefixes=(self.blocks_name + ".",))
            if self.tied:
                self._retie()
                missing = [m for m in missing if not m.endswith("lm_head.weight")]
            if missing:
                raise KeyError(f"Missing non-block weights in checkpoint: {missing[:5]}")
        for _, mod in self._non_block_modules():
            for n, p in list(mod._parameters.items()):
                if p is not None and p.device.type != "meta":
                    mod._parameters[n] = nn.Parameter(p.data.to(device), requires_grad=False)
            for n, buf in list(mod._buffers.items()):
                if buf is not None:
                    mod._buffers[n] = buf.to(device)
        if self.tied:
            self._retie()

    def _retie(self) -> None:
        out = self.model.get_output_embeddings()
        inp = self.model.get_input_embeddings()
        if out is not None and inp is not None:
            out.weight = inp.weight

    def unload_non_block(self) -> None:
        """Free embeddings/lm_head memory while blocks are being processed."""
        target = "meta" if self.mode == "stream" else "cpu"
        for _, mod in self._non_block_modules():
            for n, p in list(mod._parameters.items()):
                if p is None:
                    continue
                t = torch.empty_like(p, device="meta") if target == "meta" else p.data.to("cpu")
                mod._parameters[n] = nn.Parameter(t, requires_grad=False)

    def load_block(self, i: int, device) -> nn.Module:
        block = self.blocks[i]
        if self.mode == "stream":
            missing = materialize(block, f"{self.blocks_name}.{i}.", self.store, device, self.dtype)
            if missing:
                raise KeyError(f"Block {i}: weights missing from checkpoint: {missing[:5]}")
        block.to(device)
        return block

    def unload_block(self, i: int) -> None:
        if self.mode == "stream":
            _module_to_meta(self.blocks[i])
        else:
            self.blocks[i].to("cpu")

    def non_block_state_dict(self) -> dict[str, torch.Tensor]:
        """Original (unquantized) weights outside the blocks, as CPU tensors, for saving."""
        out = {}
        for name, p in self.model.named_parameters():
            if name.startswith(self.blocks_name + "."):
                continue
            if self.mode == "stream":
                try:
                    t = self.store.resolve(name)
                except KeyError:
                    if self.tied and name.endswith("lm_head.weight"):
                        continue
                    raise
            else:
                t = p.detach().cpu()
            out[name] = (t.to(self.dtype) if t.is_floating_point() else t).contiguous()
        return out


# ---------------------------------------------------------------------------
# 5. load our native quantized folder back
# ---------------------------------------------------------------------------
def _get_parent(root: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def load_quantized(qdir: str | Path, device="cpu", dtype: torch.dtype | None = None):
    """Rebuild the model from <work_dir>/quantized/. Only packed weights are ever allocated,
    so peak memory == real quantized footprint (this is how VRAM is measured honestly)."""
    from safetensors import safe_open

    from .awq_linear import AwqExpertsWeight, AwqLinear

    qdir = Path(qdir)
    qc = json.loads((qdir / "quant_config.json").read_text())
    if qc.get("plugin"):
        from .plugin import load_quantized_plugin
        return load_quantized_plugin(qdir, qc["plugin"], device)
    dtype = dtype or {"float32": torch.float32, "float16": torch.float16,
                      "bfloat16": torch.bfloat16}[qc["dtype"]]
    model = build_skeleton(qdir, dtype, trust_remote_code=qc.get("trust_remote_code", False))
    for name, info in qc["linears"].items():
        parent, attr = _get_parent(model, name)
        old = getattr(parent, attr)
        new = AwqLinear(old.in_features, old.out_features, old.bias is not None, qc["w_bit"],
                        qc["group_size"], has_inv_s=info["inv_s"], dtype=dtype, device="meta")
        setattr(parent, attr, new)
    for name, info in qc.get("experts", {}).items():
        parent, attr = _get_parent(model, name)
        e, o, i = info["shape"]
        parent._parameters.pop(attr, None)
        setattr(parent, attr, AwqExpertsWeight(e, o, i, qc["w_bit"], qc["group_size"], dtype=dtype, device="meta"))

    from accelerate.utils import set_module_tensor_to_device

    loaded = set()
    for f in sorted(qdir.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                t = h.get_tensor(k)
                if t.is_floating_point():
                    t = t.to(dtype)
                parent, attr = _get_parent(model, k)
                set_module_tensor_to_device(parent, attr, device, value=t)
                loaded.add(k)
    if qc.get("tie_word_embeddings"):
        model.get_output_embeddings().weight = model.get_input_embeddings().weight
    still_meta = [n for n, p in list(model.named_parameters()) + list(model.named_buffers())
                  if p.device.type == "meta"]
    if still_meta:
        raise RuntimeError(f"Quantized folder is incomplete, missing: {still_meta[:5]}")
    model.to(device)
    force_eager_experts(model)
    model.eval()
    return model
