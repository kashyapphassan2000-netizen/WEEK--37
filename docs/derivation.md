# AWQ derivation (Week 37)

Reference: Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, arXiv:2306.00978.

## 1. Linear layer
Every matmul in a transformer block is `y = x · Wᵀ`, with `x ∈ ℝ^{B×d_in}` and `W ∈ ℝ^{d_out×d_in}`.

## 2. 4-bit group quantization
Cut each row of `W` into groups of `g = 128` input channels. Each group gets a scale `Δ` (and a zero point `z` in asymmetric mode):

- asymmetric: `Δ = (max − min) / 15`, `z = round(−min / Δ)`, `q = clamp(round(w/Δ) + z, 0, 15)`, `ŵ = (q − z)·Δ`
- symmetric: `Δ = max|w| / 7`, `z = 8`

Per-element error: `|w − ŵ| ≤ Δ/2`. **Δ is set by the largest |w| in the group.** One big weight inflates Δ and pushes rounding error onto every other weight in the group (`tests/test_quantizer.py::test_error_is_bounded_by_half_a_step`).

## 3. Salient channels
Weight `w_ij` contributes `w_ij · x_j` to the output. A small weight multiplied by a huge activation still moves the output, so the importance of input channel `j` is measured by the **activation** `m_j = mean_b |x_bj|`, not by `|w|`. Real LLMs have a handful of channels with 10–100× larger activations (see the alpha/importance logs in `proof/`).

## 4. The scaling identity (exact in floating point)
For any positive vector `s ∈ ℝ^{d_in}`:

```
y = x Wᵀ = (x / s) · (W · diag(s))ᵀ
```

Quantize `W' = W·diag(s)` instead of `W`, and feed `x' = x/s`. Inference computes `Q(W·diag(s)) · (x/s)ᵀ` (`tests/test_scaling.py::test_scaling_identity_exact`).

## 5. Where does the error go?
The error now sits on `Q(W·diag(s))`. Making `s_j` large for a salient channel makes column `j` large **before** the group scale is computed. Its rounding error *relative to its new magnitude* shrinks. Dividing by `s_j` afterwards shrinks that error again in absolute terms. Non-salient channels pay a little more error. That's a good trade because their activations are small.

The `1/s` costs nothing at inference: it folds into the operation that produces `x`:

| group | `x` is produced by | fold |
|---|---|---|
| q, k, v | `input_layernorm` | `γ ← γ / s` (Gemma: `γ ← (1+γ)/s − 1`) |
| gate, up | `post_attention_layernorm` (Gemma: `pre_feedforward_layernorm`) | `γ ← γ / s` |
| down | `act(gate) ⊙ up` | `W_up rows ← W_up / s` (valid because the product is linear in `up`) |
| o | attention output = softmax · V | `W_v rows ← W_v / s`, **only without GQA** (with GQA, V rows are shared across heads) |

`tests/test_scaling.py::test_folding_keeps_block_output_identical` checks every one of these to 1e-9 in float64.

## 6. Choosing s
Parameterise with one scalar per group (the official "duo scaling"):

```
s = mean|x|^α / mean_norm|w|^(1−α),   s ← s / sqrt(max(s)·min(s))
α* = argmin_{α ∈ {0, 1/20, …, 19/20}}  || x Wᵀ − (x/s) Q(W·diag(s))ᵀ ||²
```

Plain round-to-nearest (`s = 1`) is always tried as candidate 0, so on calibration data AWQ can never be worse than RTN. The loss is computed as `|| x (W − Q(Ws)/s)ᵀ ||²`: one matmul per candidate.

## 7. Why a one-dimensional search is enough
Uniform scaling of all channels cancels out of the loss (the normalisation removes it). What matters is the *direction* of `s` in log-space. The family `log s = α·log m − (1−α)·log w̄` slides between "ignore activations" and "follow activations". One scalar captures the trade-off between protecting salient channels and stretching groups. A full per-channel optimisation would overfit 128 calibration samples and would need gradients.

## 8. Weight clipping
After scaling, a group can still contain one outlier weight. For each (output row, group) try `max|w| × (1 − i/20)` for `i = 0..9` and keep the clip that minimises that group's partial output error. This is skipped for q/k, because attention logits are sensitive.

## 9. Open choices (and what this repo does)
- group 128 vs 64: smaller groups mean more scales, which costs more memory but gives better accuracy. Default is 128 (paper parity). The tuner tries 64.
- symmetric vs asymmetric: asymmetric is the default because it's the standard ecosystem format and suits weights whose groups have a non-zero mean.
- next-block input: the official AutoAWQ feeds FP outputs forward (`search.propagate: fp`, default). GPTQ-style quantized propagation is a tuning knob.
- calibration: pile-val, 128 × 512 tokens (official). The domain matters more than the amount.
