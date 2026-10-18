from pathlib import Path

import torch

from awq.awq_linear import AwqLinear
from awq.model_io import load_quantized
from awq.pipeline import quantize_model

PLUGIN = str(Path(__file__).with_name("tiny_plugin.py"))


def test_custom_model_via_plugin(make_rc):
    rc = make_rc("custom-tiny-lm", f"plugin={PLUGIN}")
    qdir = quantize_model(rc)
    qm = load_quantized(qdir)
    assert sum(isinstance(m, AwqLinear) for m in qm.modules()) == 4      # up/down x 2 blocks, head stays FP
    import importlib.util
    spec = importlib.util.spec_from_file_location("tp", PLUGIN)
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    fp, _ = tp.load()
    x = torch.randint(0, 300, (2, 32))
    with torch.no_grad():
        a, b = fp(x), qm(x)
    assert ((a - b).norm() / a.norm()).item() < 0.2
