"""
evaluate.py — measure what quantization cost you, honestly.

  perplexity()      : how surprised the model is by real text (lower = better). Standard metric.
  generations()     : same prompts, greedy decoding, FP16 vs AWQ side by side (catches broken output).
  speed()           : tokens/second for a fixed prompt (this repo's reference kernels are SLOW by design;
                      fast serving = export to vLLM).
  model_bytes()     : real memory footprint of all weights/buffers actually allocated.
  run_lm_eval()     : downstream accuracy through EleutherAI lm-evaluation-harness (optional dep).
  apply_rtn_()      : naive round-to-nearest baseline (no AWQ) — the number AWQ must beat.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .blocks import find_blocks, fused_experts, quantizable_linears
from .model_io import checkpoint_is_fp8, force_eager_experts, load_hf_config
from .quantizer import fake_quant

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue:",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Translate to French: 'Good morning, how are you?' ->",
    "Q: If a train travels 60 km in 1.5 hours, what is its average speed?\nA:",
    "The three primary colors are",
    "Write a haiku about the ocean:\n",
    "In machine learning, overfitting means",
    "import numpy as np\n# compute the mean of each column of matrix X\n",
    "The theory of relativity was proposed by",
]


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def load_fp_model(path, dtype, device: str, trust_remote_code: bool = False):
    from transformers import AutoModelForCausalLM

    kwargs = dict(dtype=dtype, trust_remote_code=trust_remote_code, low_cpu_mem_usage=True)
    if device == "cuda":
        kwargs["device_map"] = "auto"
    if checkpoint_is_fp8(load_hf_config(path, trust_remote_code)):
        from transformers import FineGrainedFP8Config
        kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    force_eager_experts(model)
    return model.eval()


@torch.no_grad()
def apply_rtn_(model, w_bit: int, group_size: int, zero_point: bool, skip_patterns=()) -> int:
    """In-place naive RTN on exactly the weights AWQ would quantize. Returns #tensors changed."""
    _, blocks = find_blocks(model)
    n = 0
    for block in blocks:
        lin, _ = quantizable_linears(block, group_size, skip_patterns)
        for m in lin.values():
            m.weight.data = fake_quant(m.weight.data, w_bit, group_size, zero_point)
            n += 1
        for _, pname, mod in fused_experts(block):
            p = getattr(mod, pname)
            for e in range(p.shape[0]):
                p.data[e] = fake_quant(p.data[e], w_bit, group_size, zero_point)
            n += 1
    return n


@torch.no_grad()
def perplexity(model, tokens: torch.Tensor, log=None, desc="ppl") -> float:
    dev = input_device(model)
    nll, count = 0.0, 0
    for i in range(tokens.shape[0]):
        ids = tokens[i:i + 1].to(dev)
        logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
        target = ids[:, 1:].to(logits.device)
        nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="sum").item()
        count += target.numel()
        if log and (i + 1) % max(1, tokens.shape[0] // 5) == 0:
            log(f"  [{desc}] {i + 1}/{tokens.shape[0]} windows  running ppl={math.exp(nll / count):.4f}")
    return math.exp(nll / count)


@torch.no_grad()
def generations(model, tok, prompts=None, max_new_tokens: int = 48) -> list[str]:
    prompts = prompts or DEFAULT_PROMPTS
    dev = input_device(model)
    outs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        g = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        outs.append(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))
    return outs


@torch.no_grad()
def speed(model, tok, new_tokens: int = 64) -> dict:
    dev = input_device(model)
    ids = tok("The history of computing", return_tensors="pt").input_ids.to(dev)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=pad)        # warm-up
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    g = model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, pad_token_id=pad)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t
    n = g.shape[1] - ids.shape[1]
    return {"new_tokens": int(n), "seconds": round(dt, 3), "tokens_per_s": round(n / dt, 2)}


def model_bytes(model) -> int:
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.data_ptr() in seen or t.device.type == "meta":
            continue
        seen.add(t.data_ptr())
        total += t.numel() * t.element_size()
    return total


def first_token_agreement(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    return (logits_a.argmax(-1) == logits_b.argmax(-1)).float().mean().item()


def run_lm_eval(model, tok, tasks, limit=None, batch_size="auto") -> dict:
    """Downstream accuracy via lm-evaluation-harness. Returns {task: {metric: value}}."""
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise RuntimeError("lm-eval not installed: pip install -e '.[eval]'") from e
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size)
    res = simple_evaluate(model=lm, tasks=list(tasks), limit=limit)
    out = {}
    for task, metrics in res["results"].items():
        out[task] = {k.split(",")[0]: v for k, v in metrics.items()
                     if isinstance(v, (int, float)) and not k.startswith("alias") and "stderr" not in k}
    return out


def read_custom_jsonl(path: str | Path) -> list[dict]:
    import json
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
