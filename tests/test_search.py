import torch

from awq.quantizer import fake_quant
from awq.search import apply_clip, search_clip, search_scale


def _outlier_data(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2048, 256, generator=g)
    x[:, [5, 77, 130]] *= 40.0                    # ~1% salient channels, like real LLMs
    w = torch.randn(128, 256, generator=g) * 0.05
    return x, w


def test_awq_beats_rtn_on_salient_channels():
    x, w = _outlier_data()
    res = search_scale([w], [None], x, 4, 128, True, n_grid=20)
    assert res["ratio"] is not None, "AWQ should find a scale better than plain RTN here"
    assert res["loss"] < 0.7 * res["loss_rtn"], (res["loss"], res["loss_rtn"])
    # measure independently: Q(W*s)/s on fresh data from the same distribution
    xt, _ = _outlier_data(seed=1)
    s = res["scales"]
    err_awq = (xt @ (w - fake_quant(w * s, 4, 128, True) / s).T).pow(2).mean()
    err_rtn = (xt @ (w - fake_quant(w, 4, 128, True)).T).pow(2).mean()
    assert err_awq < err_rtn


def test_search_never_worse_than_rtn():
    torch.manual_seed(3)
    x, w = torch.randn(512, 128), torch.randn(64, 128)     # no outliers: RTN may already be optimal
    res = search_scale([w], [None], x, 4, 64, True, n_grid=10)
    assert res["loss"] <= res["loss_rtn"] + 1e-12


def test_clip_search_does_not_increase_error():
    x, w = _outlier_data()
    w = w.clone()
    w[0, 3] = 3.0                                           # one outlier weight stretches its group
    mv = search_clip(w, x, 4, 128, True, clip_grid=20, max_shrink=0.5, n_tokens=512)
    wc = apply_clip(w, mv)
    err_clip = (x @ (w - fake_quant(wc, 4, 128, True)).T).pow(2).mean()
    err_none = (x @ (w - fake_quant(w, 4, 128, True)).T).pow(2).mean()
    assert err_clip <= err_none * 1.001
