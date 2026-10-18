"""Fused 3-D MoE experts (transformers >= 5 layout) are quantized per expert."""
import json

import torch

from awq.awq_linear import AwqExpertsWeight
from awq.model_io import load_quantized
from awq.pipeline import quantize_model


def test_mixtral_experts_quantized(tiny_mixtral, make_rc, calib_tokens):
    qdir = quantize_model(make_rc(tiny_mixtral), calib_tokens=calib_tokens)
    qc = json.loads((qdir / "quant_config.json").read_text())
    assert qc["n_expert_tensors_quantized"] == 2 * 2          # gate_up_proj + down_proj per layer
    log = [json.loads(line) for line in (qdir.parent / "search_log.jsonl").read_text().splitlines()]
    stats = list(log[0]["experts"].values())[0]
    assert stats["with_tokens"] >= 1
    qm = load_quantized(qdir, "cpu")
    assert any(isinstance(m, AwqExpertsWeight) for m in qm.modules())
    from transformers import AutoModelForCausalLM
    fp = AutoModelForCausalLM.from_pretrained(str(tiny_mixtral), dtype=torch.float32).eval()
    with torch.no_grad():
        a, b = fp(calib_tokens[:2]).logits, qm(calib_tokens[:2]).logits
    assert torch.isfinite(b).all()
    assert ((a - b).norm() / a.norm()).item() < 0.3
