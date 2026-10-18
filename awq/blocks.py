"""
blocks.py — find the repeated decoder blocks and the weights inside them, for ANY architecture.

Plain words:
  * Every LLM is "embedding -> N identical blocks -> norm -> output head".
    `find_blocks` finds that list of N blocks by looking for the biggest nn.ModuleList
    whose children are all the same class. No model names are hard-coded.
  * Only weights INSIDE blocks are quantized. The embedding and the output head stay 16-bit
    (that is the AWQ standard, and it is why we never need to know the head's name).
  * `quantizable_linears` lists every nn.Linear in a block, skipping MoE routers, tiny layers,
    and anything you listed in `quant.skip_modules`.
  * `fused_experts` finds transformers>=5 style MoE weights: one 3-D tensor (experts, out, in).
  * `fold_target` answers: "after scaling this group's weights by s, which earlier layer can
    absorb the 1/s for free?"  (RMSNorm/LayerNorm before attention or MLP, or up_proj before
    down_proj.) If none, the scale is stored inside the quantized layer instead.
"""
from __future__ import annotations

import re

import torch.nn as nn

ROUTER_NAMES = {"gate", "router", "shared_expert_gate", "gate_proj_router"}
# Linears whose outputs feed attention scores: the official AWQ skips weight clipping for them.
NO_CLIP_PATTERNS = ("q_", "k_", "query", "key", "Wqkv", "qkv", "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj")
FOLD_FAMILIES = {"llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "gemma3_text", "granite",
                 "mixtral", "qwen2_moe", "qwen3_moe", "olmo", "smollm3", "ministral"}


def _numel(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def find_blocks(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """Return (dotted_name, ModuleList) of the repeated decoder blocks."""
    best, best_n = None, -1
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 1:
            classes = {type(c).__name__ for c in mod}
            if len(classes) != 1:
                continue
            n = _numel(mod)
            if n > best_n:
                best, best_n = (name, mod), n
    if best is None:
        raise RuntimeError("Could not find a list of repeated decoder blocks in this model.")
    return best


def model_type(model_or_config) -> str:
    cfg = getattr(model_or_config, "config", model_or_config)
    text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    return str(getattr(text, "model_type", getattr(cfg, "model_type", "")))


def quantizable_linears(block: nn.Module, group_size: int, skip_patterns=()) -> tuple[dict, dict]:
    """Return ({name: nn.Linear} to quantize, {name: reason} skipped)."""
    keep, skipped = {}, {}
    for name, m in block.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        g = m.in_features if group_size <= 0 else group_size
        if any(p in name for p in skip_patterns):
            skipped[name] = "listed in quant.skip_modules"
        elif leaf in ROUTER_NAMES or m.out_features < 32:
            skipped[name] = "MoE router / tiny layer (kept 16-bit on purpose)"
        elif m.in_features % g != 0 or m.in_features % 2 != 0:
            skipped[name] = f"in_features={m.in_features} not divisible by group_size={g}"
        else:
            keep[name] = m
    return keep, skipped


def fused_experts(block: nn.Module) -> list[tuple[str, str, nn.Module]]:
    """[(module_name, param_name, module)] for 3-D expert weights used as W[e] in eager forward."""
    out = []
    for name, m in block.named_modules():
        if getattr(m, "is_transposed", False):
            continue          # (E, in, out) layout: not supported, stays 16-bit
        for pname, p in m.named_parameters(recurse=False):
            if p.ndim == 3 and pname in ("gate_up_proj", "down_proj", "up_proj"):
                out.append((name, pname, m))
    return out


def allow_clip(name: str) -> bool:
    leaf = name.split(".")[-1]
    return not any(leaf.startswith(p) or p == leaf for p in NO_CLIP_PATTERNS)


def _get(block: nn.Module, dotted: str):
    m = block
    for p in dotted.split("."):
        if not hasattr(m, p):
            return None
        m = getattr(m, p)
    return m


def fold_target(block: nn.Module, group: list[str], family: str):
    """Where can 1/s be absorbed for this group of linears (names relative to block)?

    Returns (kind, module_name) with kind in {"norm", "norm_offset", "linear_out"} or None.
    Only for families whose block structure we know (FOLD_FAMILIES); everything else keeps 1/s
    inside the quantized layer, which is always correct.
    """
    if family not in FOLD_FAMILIES:
        return None
    leaves = {g.split(".")[-1] for g in group}
    parent = group[0].rsplit(".", 1)[0] if "." in group[0] else ""

    def norm(name):
        m = _get(block, name)
        if m is None or not hasattr(m, "weight") or m.weight is None:
            return None
        offset = "Gemma" in type(m).__name__ or family.startswith("gemma")
        return ("norm_offset" if offset else "norm", name)

    if leaves == {"q_proj", "k_proj", "v_proj"}:
        return norm("input_layernorm")
    if leaves == {"gate_proj", "up_proj"} and re.fullmatch(r"(mlp|feed_forward)", parent or ""):
        # Gemma2/3 put a norm AFTER attention ("post_attention_layernorm") and a separate
        # "pre_feedforward_layernorm" before the MLP — the MLP input comes from the latter.
        return norm("pre_feedforward_layernorm") or norm("post_attention_layernorm")
    if leaves == {"down_proj"}:
        up = _get(block, f"{parent}.up_proj") if parent else None
        gate = _get(block, f"{parent}.gate_proj") if parent else None
        if isinstance(up, nn.Linear) and isinstance(gate, nn.Linear):
            return ("linear_out", f"{parent}.up_proj")
        return None
    if leaves == {"o_proj"}:
        v = _get(block, f"{parent}.v_proj")
        o = _get(block, group[0])
        if isinstance(v, nn.Linear) and v.out_features == o.in_features:
            return ("linear_out", f"{parent}.v_proj")
        return None
    return None
