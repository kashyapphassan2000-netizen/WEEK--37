"""
search.py — the AWQ algorithm itself (Lin et al. 2023, arXiv:2306.00978).

Plain words, scale search (`search_scale`):
  For a group of layers that read the same input x (e.g. q/k/v):
    1. importance of input channel j:   x_mean[j] = mean |x[:, j]|
    2. try ratio a = 0, 1/n, 2/n, ...:   s = x_mean^a / w_mean^(1-a)        (duo scaling)
                                         s = s / sqrt(max(s) * min(s))      (keep magnitudes sane)
    3. pretend-quantize the scaled weight and undo the scale:  W_test = Q(W * s) / s
    4. error = || x W^T - x W_test^T ||^2   (how much the layer output changed)
    5. keep the s with the smallest error.
  s = 1 (plain round-to-nearest) is always tried first, so on calibration data AWQ can never be
  worse than RTN — it only replaces RTN when some scale measurably lowers the error.

Plain words, clip search (`search_clip`):
  One huge weight in a group stretches that group's scale, wasting the 16 levels.
  Try shrinking each group's max |w| to 100%, 95%, ... 50% and keep the cut that minimises
  the group's output error. Skipped for q/k projections (attention scores are sensitive).
"""
from __future__ import annotations

import math

import torch

from .quantizer import fake_quant


def _chunks(n: int, size: int):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


@torch.no_grad()
def _subsample(x: torch.Tensor, max_tokens: int, seed: int = 0) -> torch.Tensor:
    if max_tokens and x.shape[0] > max_tokens:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=g)[:max_tokens].to(x.device)
        return x.index_select(0, idx)
    return x


@torch.no_grad()
def search_scale(weights: list[torch.Tensor], biases: list[torch.Tensor | None], x: torch.Tensor,
                 w_bit: int, group_size: int, zero_point: bool, n_grid: int = 20,
                 duo_scaling: bool = True, max_tokens: int = 0, token_chunk: int = 4096):
    """weights: list of (out_i, in) matrices sharing input x (n, in).

    Returns dict(scales=(in,) float32, ratio=float or None (None = plain RTN won),
                 loss=float, loss_rtn=float, losses=list).
    """
    x = _subsample(x, max_tokens)
    dev, dt = weights[0].device, weights[0].dtype
    x = x.to(dev, dt)
    in_f = x.shape[-1]
    x_mean = x.abs().float().mean(0)                                     # (in,)

    if duo_scaling:
        wcat = torch.cat(weights, dim=0).float()
        g = in_f if group_size <= 0 else group_size
        wg = wcat.abs().view(-1, g)
        w_scale = (wg / (wg.amax(dim=1, keepdim=True) + 1e-6)).view(wcat.shape)
        w_mean = w_scale.mean(0)                                         # (in,)
        del wcat, wg, w_scale
    else:
        w_mean = None

    # output error of a candidate:  || x W^T - x W_test^T ||^2 = || x (W - W_test)^T ||^2
    # -> ONE matmul per candidate, computed chunk by chunk to bound memory (bias cancels).
    def outputs_error(test_weights):
        diffs = [(w_ref - w_t) for w_ref, w_t in zip(weights, test_weights)]
        err, count = 0.0, 0
        for sl in _chunks(x.shape[0], token_chunk):
            xb = x[sl]
            for d in diffs:
                err += (xb @ d.T).float().pow(2).sum().item()
                count += xb.shape[0] * d.shape[0]
        return err / max(count, 1)

    # candidate 0: s = 1 (plain round-to-nearest). AWQ only replaces it if a scale is better.
    loss_rtn = outputs_error([fake_quant(w, w_bit, group_size, zero_point) for w in weights])
    best = dict(loss=loss_rtn, ratio=None, scales=torch.ones(in_f, device=dev))
    losses = []
    for i in range(n_grid):
        ratio = i / n_grid
        if w_mean is not None:
            s = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
        else:
            s = x_mean.pow(ratio).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        if not torch.isfinite(s).all():
            losses.append(math.inf)
            continue
        s_dt = s.to(dt)
        tests = [fake_quant(w * s_dt.view(1, -1), w_bit, group_size, zero_point) / s_dt.view(1, -1)
                 for w in weights]
        loss = outputs_error(tests)
        losses.append(loss)
        if loss < best["loss"]:
            best = dict(loss=loss, ratio=ratio, scales=s.clone())
    best["loss_rtn"] = loss_rtn
    best["losses"] = losses
    best["scales"] = best["scales"].float()
    return best


@torch.no_grad()
def search_clip(w: torch.Tensor, x: torch.Tensor, w_bit: int, group_size: int, zero_point: bool,
                clip_grid: int = 20, max_shrink: float = 0.5, n_tokens: int = 512,
                elem_budget: int = 2 ** 27) -> torch.Tensor:
    """Return best max|w| per (out, group) — shape (out, n_groups, 1). `w` is already scaled.
    `x` must already be divided by the AWQ scale (it is the input the scaled weight will see)."""
    out_f, in_f = w.shape
    g = in_f if group_size <= 0 else group_size
    ng = in_f // g
    # evenly spaced token subsample (official AWQ uses 512)
    if x.shape[0] > n_tokens:
        step = max(1, x.shape[0] // n_tokens)
        x = x[::step][:n_tokens]
    x = x.to(w.device, w.dtype)
    n = x.shape[0]
    xg = x.view(n, ng, g)                                                  # (n, ng, g)
    oc_batch = max(1, min(out_f, elem_budget // max(1, n * ng)))
    best_max_all = []
    for sl in _chunks(out_f, oc_batch):
        wb = w[sl].view(-1, ng, g)                                         # (ob, ng, g)
        # partial output of every group: out[o, t, k] = sum_j x[t, k, j] * w[o, k, j]  (batched matmul)
        org_out = torch.einsum("nkg,okg->onk", xg, wb)                     # (ob, n, ng)
        org_max = wb.abs().amax(dim=-1, keepdim=True)                      # (ob, ng, 1)
        best_max = org_max.clone()
        min_err = torch.full((wb.shape[0], ng), float("inf"), device=w.device)
        for i in range(int(max_shrink * clip_grid)):
            max_val = org_max * (1 - i / clip_grid)
            cur = torch.clamp(wb, -max_val, max_val)
            q = fake_quant(cur.reshape(wb.shape[0], in_f), w_bit, g, zero_point).view(-1, ng, g)
            cur_out = torch.einsum("nkg,okg->onk", xg, q)
            err = (cur_out - org_out).float().pow(2).mean(dim=1)           # (ob, ng)
            better = err < min_err
            min_err[better] = err[better]
            best_max[better] = max_val[better]
        best_max_all.append(best_max)                                      # (ob, ng, 1)
        del org_out, wb
    return torch.cat(best_max_all, dim=0)


@torch.no_grad()
def apply_clip(w: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    out_f, in_f = w.shape
    ng = max_val.shape[1]
    g = in_f // ng
    wg = w.view(out_f, ng, g)
    mv = max_val.to(w.dtype)
    return torch.clamp(wg, -mv, mv).view(out_f, in_f)
