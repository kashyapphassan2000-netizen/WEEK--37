"""Tiny random models built on the fly — the test-suite needs no downloads and no GPU."""
import pytest
import torch


def _save(model, path):
    model.save_pretrained(str(path), safe_serialization=True)
    return path


def add_outliers(model, channels=(3, 17, 40), factor=25.0):
    """Real LLMs have a few input channels with huge activations; recreate that so AWQ has work to do."""
    for name, p in model.named_parameters():
        if name.endswith("layernorm.weight"):
            with torch.no_grad():
                p[list(channels)] *= factor


@pytest.fixture(scope="session")
def tiny_llama(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                      tie_word_embeddings=False)
    m = LlamaForCausalLM(cfg).eval()
    add_outliers(m)
    return _save(m, tmp_path_factory.mktemp("tiny_llama"))


@pytest.fixture(scope="session")
def tiny_qwen_gqa(tmp_path_factory):
    """GQA + tied embeddings + qkv bias: the Qwen2 shape of things."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(1)
    cfg = Qwen2Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=True)
    m = Qwen2ForCausalLM(cfg).eval()
    add_outliers(m)
    return _save(m, tmp_path_factory.mktemp("tiny_qwen"))


@pytest.fixture(scope="session")
def tiny_mixtral(tmp_path_factory):
    from transformers import MixtralConfig, MixtralForCausalLM

    torch.manual_seed(2)
    cfg = MixtralConfig(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, num_local_experts=4,
                        num_experts_per_tok=2, max_position_embeddings=256, tie_word_embeddings=False)
    m = MixtralForCausalLM(cfg).eval()
    add_outliers(m)
    return _save(m, tmp_path_factory.mktemp("tiny_mixtral"))


@pytest.fixture
def make_rc(tmp_path):
    from awq.config import load_config

    def _make(model_path, *extra):
        return load_config(None, overrides=[
            f"model={model_path}", "device=cpu", "dtype=float32", f"io.work_dir={tmp_path / 'run'}",
            "calib.n_samples=16", "calib.seq_len=64", "calib.batch_size=4", "quant.group_size=64",
            "search.n_grid=10", *extra])
    return _make


@pytest.fixture(scope="session")
def calib_tokens():
    g = torch.Generator().manual_seed(123)
    return torch.randint(0, 512, (16, 64), generator=g)
