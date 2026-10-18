"""The sizing formulas, checked against the hand calculation in the playbook (Llama-3.1-8B)."""
import torch

from awq.hardware import config_dims, count_params, estimate


def test_llama_8b_numbers_match_playbook():
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM, LlamaConfig

    cfg = LlamaConfig(vocab_size=128256, hidden_size=4096, intermediate_size=14336, num_hidden_layers=32,
                      num_attention_heads=32, num_key_value_heads=8, tie_word_embeddings=False)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    counts = count_params(m)
    assert abs(counts["total_params"] / 1e9 - 8.03) < 0.01
    est = estimate(counts, config_dims(cfg), ckpt_gb=16.06)
    assert abs(est["fp16_gb"] - 16.06) < 0.05
    assert abs(est["awq_gb"] - 5.73) < 0.05
    assert abs(est["phases"]["A_quantize"]["vram_gb"] - 14.1) < 0.2
    assert abs(est["phases"]["B_fp16_eval"]["vram_gb"] - 19.3) < 0.1
    assert abs(est["kv_cache_gb"] - 1.07) < 0.01
    assert est["phases"]["A_quantize"]["best"]["gpu"] == "RTX 4090"
