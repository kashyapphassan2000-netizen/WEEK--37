"""End-to-end on tiny models: stream -> AWQ -> pack -> reload, plus resume and AWQ-vs-RTN."""
import json

import pytest
import torch

from awq.awq_linear import AwqLinear
from awq.evaluate import apply_rtn_
from awq.model_io import load_quantized
from awq.pipeline import quantize_model


def _fp(path):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32).eval()


@torch.no_grad()
def _err(a, b):
    return ((a - b).norm() / a.norm()).item()


def test_end_to_end_llama_folded(tiny_llama, make_rc, calib_tokens):
    rc = make_rc(tiny_llama)
    qdir = quantize_model(rc, calib_tokens=calib_tokens)
    qc = json.loads((qdir / "quant_config.json").read_text())
    assert qc["n_linears_quantized"] == 3 * 7
    assert qc["n_linears_with_inv_s"] == 0, "Llama (MHA) groups should all fold"
    qm = load_quantized(qdir, "cpu")
    assert all(isinstance(m, AwqLinear) for n, m in qm.named_modules() if n.endswith(("_proj",)))
    fp = _fp(tiny_llama)
    with torch.no_grad():
        a, b = fp(calib_tokens[:4]).logits, qm(calib_tokens[:4]).logits
    assert torch.isfinite(b).all()
    assert _err(a, b) < 0.25


def test_awq_beats_rtn_end_to_end(tiny_llama, make_rc, calib_tokens):
    qdir = quantize_model(make_rc(tiny_llama), calib_tokens=calib_tokens)
    qm, fp, rtn = load_quantized(qdir, "cpu"), _fp(tiny_llama), _fp(tiny_llama)
    apply_rtn_(rtn, 4, 64, True)
    with torch.no_grad():
        ref = fp(calib_tokens).logits
        e_awq, e_rtn = _err(ref, qm(calib_tokens).logits), _err(ref, rtn(calib_tokens).logits)
    assert e_awq < e_rtn, (e_awq, e_rtn)


def test_never_fold_keeps_inv_s(tiny_llama, make_rc, calib_tokens):
    qdir = quantize_model(make_rc(tiny_llama, "quant.fold_scales=never"), calib_tokens=calib_tokens)
    qc = json.loads((qdir / "quant_config.json").read_text())
    assert qc["n_linears_with_inv_s"] > 0
    qm = load_quantized(qdir, "cpu")
    with torch.no_grad():
        assert _err(_fp(tiny_llama)(calib_tokens[:2]).logits, qm(calib_tokens[:2]).logits) < 0.25


def test_gqa_tied_qwen(tiny_qwen_gqa, make_rc, calib_tokens):
    qdir = quantize_model(make_rc(tiny_qwen_gqa), calib_tokens=calib_tokens)
    qc = json.loads((qdir / "quant_config.json").read_text())
    assert qc["tie_word_embeddings"] is True
    groups = [json.loads(line) for line in (qdir.parent / "search_log.jsonl").read_text().splitlines()]
    o_proj = [g for g in groups[0]["groups"] if g["linears"] == ["self_attn.o_proj"]][0]
    assert o_proj["mode"].startswith("no-scale"), "GQA o_proj cannot fold into v_proj"
    qm = load_quantized(qdir, "cpu")
    assert qm.lm_head.weight is qm.model.embed_tokens.weight
    with torch.no_grad():
        assert _err(_fp(tiny_qwen_gqa)(calib_tokens[:2]).logits, qm(calib_tokens[:2]).logits) < 0.25


def test_resume_gives_identical_result(tiny_llama, make_rc, calib_tokens, monkeypatch):
    import awq.pipeline as P

    rc = make_rc(tiny_llama)
    full = quantize_model(rc, calib_tokens=calib_tokens)
    ref = {f.name: f.read_bytes() for f in full.glob("block-*.safetensors")}
    for f in full.glob("block-*.safetensors"):
        f.unlink()

    real = P.quantize_block
    def crash_on_block_2(block, idx, *a, **k):
        if idx == 2:
            raise RuntimeError("simulated pod reclaim")
        return real(block, idx, *a, **k)
    monkeypatch.setattr(P, "quantize_block", crash_on_block_2)
    with pytest.raises(RuntimeError):
        quantize_model(rc, calib_tokens=calib_tokens)
    monkeypatch.setattr(P, "quantize_block", real)
    quantize_model(rc, resume=True, calib_tokens=calib_tokens)
    got = {f.name: f.read_bytes() for f in full.glob("block-*.safetensors")}
    assert got == ref
