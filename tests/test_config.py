import pytest

from awq.config import load_config


def test_overrides_are_typed():
    rc = load_config(None, overrides=["model=x", "device=cpu", "dtype=float32", "quant.group_size=64",
                                      "quant.zero_point=false", "eval.ppl_datasets=[wikitext2]"])
    assert rc.quant.group_size == 64 and rc.quant.zero_point is False and rc.eval.ppl_datasets == ["wikitext2"]


@pytest.mark.parametrize("bad", ["quant.w_bit=3", "quant.group_size=100", "search.propagate=maybe", "dtype=float16"])
def test_bad_values_fail_fast(bad):
    with pytest.raises(ValueError):
        load_config(None, overrides=["model=x", "device=cpu", "dtype=float32", bad])


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        load_config(None, overrides=["model=x", "device=cpu", "dtype=float32", "quant.groupsize=64"])
