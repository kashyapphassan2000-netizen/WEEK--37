# AWQ from the paper, rebuilt for any model size

I reimplemented AWQ (Activation-aware Weight Quantization, Lin et al. 2023) from scratch in pure PyTorch. I built it so the same code can quantize a 0.5B model on a laptop CPU or, in principle, a 1-trillion-parameter MoE on a single rented H200. This post covers what I built, what I measured, and what I haven't proven yet.

Code: https://github.com/kashyapphassan2000-netizen/WEEK--37

## The problem
At 4 bits a weight can take 16 values. Round every weight to the nearest level (RTN) and a 0.5B model's WikiText-2 perplexity goes from 12.70 to 15.08 (+18.7%), measured below. The reason is that inside each group of 128 weights, the largest |w| sets the step size. Meanwhile a handful of input channels carry 10–100× larger activations than the rest. Errors on the weights that multiply those channels dominate the output.

## The trick
For any positive vector `s`: `x Wᵀ = (x/s)(W·diag(s))ᵀ`. Quantize `W·diag(s)` instead of `W`. Salient columns (large mean |x|) get scaled up before rounding, so their relative error shrinks. The `1/s` folds into the RMSNorm (or the previous Linear) that produces `x`, so it costs nothing at inference. `s = mean|x|^α / mean|w|^(1−α)`, with α picked by a 20-point grid search that minimises the layer's output error on 128 calibration sequences. It needs no gradients and no Hessian. A weight-clipping search runs afterwards.

The derivation is in [`docs/derivation.md`](derivation.md). The folding table (norm → q/k/v, norm → gate/up, up → down, v → o for non-GQA) is checked in float64 by `tests/test_scaling.py`.

## What I built differently from the usual tutorial
1. **Block streaming.** The model is built on PyTorch's `meta` device (zero bytes). Weights are read from safetensors one decoder block at a time. Only one block plus its calibration activations ever sits on the GPU, so a 70B model needs a ~28 GB GPU to quantize, not 140 GB.
2. **Architecture-agnostic.** Decoder blocks are found as "the biggest ModuleList of identical modules". Linears that share an input are grouped by detecting that they received *the same tensor in memory*. MoE experts stored as fused 3-D tensors (transformers ≥ 5) are calibrated with the tokens actually routed to them.
3. **Honest measurement.** The quantized model is rebuilt from packed weights only, so the memory number is the real footprint, not an FP16 peak.
4. **Resumable.** Every block is a checkpoint. A test kills the run mid-way, resumes it, and asserts byte-identical output.
5. **Gated workflow.** Scripts 00–08 each write a JSON file and exit non-zero on failure: environment, sizing, baseline, quantize, verify, eval, tune, export (vLLM GEMM layout), publish.

## Numbers (CPU run, Qwen2.5-0.5B, first 20×2048 tokens per dataset)

| | Original (FP32) | RTN 4-bit | AWQ 4-bit |
|---|---|---|---|
| WikiText-2 PPL | 12.701 | 15.081 | **14.733** |
| C4 PPL | 19.156 | 22.987 | **21.999** |
| Weights in memory | 1.98 GB | — | 0.74 GB |

AWQ beats RTN on both datasets, but on a 0.5B model it closes only 15–26% of the gap and **fails my own ≤3B gate** (within 10% of the original). That's the honest result. Tiny models have the least redundancy. The 7B+ claim is for the first GPU run to settle.

## Tuning, done strictly
`06_tune.py` changes one knob per stage and scores it on WikiText-2 **validation**, never on the test set. A change must win by ≥ 0.5% to be accepted.

| change | validation PPL |
|---|---|
| base (group 128) | 13.844 |
| keep runtime 1/s everywhere (`fold_scales=never`) | 13.829, rejected (+ would lose the vLLM export) |
| **group 64** | **13.198, accepted** |
| + GPTQ-style quantized propagation | 13.219, rejected |

Then one final run, evaluated once on test: **WikiText-2 14.057 (+10.7%), C4 21.149 (+10.4%)**. The fair comparison is RTN at the same group 64: 14.481 / 21.894. So group size did most of the work, and AWQ removes another ~25% of the remaining gap. That's still just over my 10% gate for a 0.5B model.

## What I learned
- The scale search is cheap if you write the error as `‖x (W − Q(Ws)/s)ᵀ‖²`: one matmul per candidate instead of two.
- The clip search went from 150 s to 5 s per block on CPU by turning an element-wise product into a batched einsum.
- The dangerous bugs are not in the math. They're in plumbing: padding tokens leaking into calibration (crashes on Llama tokenizers), measuring "quantized VRAM" after loading FP16 first, and missing the output head because it isn't called `lm_head` in every model.
- Under GQA, `o_proj` cannot fold its scale into `v_proj`. You either keep a runtime `1/s` (native format, slightly better) or skip scaling it (vLLM-exportable). The tuner measures that trade-off.

## What's next
- First GPU run: Qwen2.5-7B on one RTX 4090 (≈ $1–3), load the export in vLLM, calibrate the sizing tool's speed constant.
- Fused dequant-matmul kernel (Triton) so the native format is fast too.
