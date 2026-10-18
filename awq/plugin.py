"""
plugin.py — quantize YOUR OWN PyTorch model (not on the Hugging Face Hub, not a transformers class).

Write a small file, e.g. my_plugin.py, with three functions and point `plugin:` in run.yaml at it:

    def load():                                   # -> (model, tokenizer_or_None), FP weights loaded
        ...
    def blocks(model):                            # -> the nn.ModuleList of repeated blocks
        return model.layers
    def calib_batches(tokenizer, n_samples, seq_len, batch_size):
        # yield dicts of keyword arguments for model(**batch), batch_size samples each
        ...

Requirements (honest): the block's heavy compute must be nn.Linear layers, and each block must be
callable as block(hidden_states, **kwargs) returning a tensor (or a tuple whose first item is it).
The plugin model is loaded fully in CPU RAM (no streaming): fine up to ~tens of GB.
See tests/tiny_plugin.py for a complete working example.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


def load_plugin(path: str | Path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"awq_user_plugin_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("load", "blocks", "calib_batches"):
        if not hasattr(mod, fn):
            raise AttributeError(f"plugin {path} must define `{fn}()`")
    return mod


class PluginSource:
    """Same interface as model_io.ModelSource, backed by the user's plugin."""

    mode = "full"
    tied = False

    def __init__(self, rc, log=print):
        self.rc, self.log = rc, log
        self.mod = load_plugin(rc.plugin)
        self.model, self.tokenizer = self.mod.load()
        self.model.eval()
        self.blocks = self.mod.blocks(self.model)
        self.blocks_name = next(n for n, m in self.model.named_modules() if m is self.blocks)
        self.log(f"[plugin] {rc.plugin}: blocks={self.blocks_name} x{len(self.blocks)}")

    def _non_block(self):
        b = self.blocks_name
        for name, mod in self.model.named_modules():
            if name == b or name.startswith(b + "."):
                continue
            yield mod

    def load_non_block(self, device):
        for mod in self._non_block():
            for n, p in list(mod._parameters.items()):
                if p is not None:
                    mod._parameters[n] = nn.Parameter(p.data.to(device), requires_grad=False)
            for n, b in list(mod._buffers.items()):
                if b is not None:
                    mod._buffers[n] = b.to(device)

    def unload_non_block(self):
        self.load_non_block("cpu")

    def load_block(self, i, device):
        return self.blocks[i].to(device)

    def unload_block(self, i):
        self.blocks[i].to("cpu")

    def calib_batches(self, n_samples, seq_len, batch_size):
        return self.mod.calib_batches(self.tokenizer, n_samples, seq_len, batch_size)

    def non_block_state_dict(self):
        b = self.blocks_name + "."
        return {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()
                if not k.startswith(b)}

    def save_side_files(self, qdir):
        (Path(qdir) / "PLUGIN.txt").write_text(f"Built with plugin: {Path(self.rc.plugin).resolve()}\n")


def load_quantized_plugin(qdir, plugin_path, device="cpu"):
    """Rebuild a plugin model and swap in the quantized layers from <qdir>."""
    import json

    from safetensors import safe_open

    from .awq_linear import AwqExpertsWeight, AwqLinear
    from .model_io import _get_parent

    qc = json.loads((Path(qdir) / "quant_config.json").read_text())
    model, _ = load_plugin(plugin_path).load()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[qc["dtype"]]
    for name, info in qc["linears"].items():
        parent, attr = _get_parent(model, name)
        old = getattr(parent, attr)
        setattr(parent, attr, AwqLinear(old.in_features, old.out_features, old.bias is not None,
                                        qc["w_bit"], qc["group_size"], info["inv_s"], dtype))
    for name, info in qc.get("experts", {}).items():
        parent, attr = _get_parent(model, name)
        parent._parameters.pop(attr, None)
        setattr(parent, attr, AwqExpertsWeight(*info["shape"], qc["w_bit"], qc["group_size"], dtype))
    sd = {}
    for f in sorted(Path(qdir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                sd[k] = h.get_tensor(k)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()
