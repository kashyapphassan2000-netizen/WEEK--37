# Proof run: Qwen2.5-0.5B, CPU only

Real, end-to-end run of this repository on the build machine: **4 CPU cores, 16.9 GB RAM, no GPU**, torch 2.14.1+cpu, transformers 5.20.1.
Config: [`configs/examples/qwen2.5-0.5b-cpu.yaml`](../../configs/examples/qwen2.5-0.5b-cpu.yaml).

```bash
bash scripts/run_all.sh configs/examples/qwen2.5-0.5b-cpu.yaml
python scripts/06_tune.py --config configs/examples/qwen2.5-0.5b-cpu.yaml --space configs/tune_space_cpu.yaml
```

On CPU the "original" model runs in **FP32**, so every `fp16_*` key in `baseline.json` means "original precision" (FP32 here).

## Results (test windows: first 20 × 2048 tokens of each dataset)

| | Original (FP32) | Naive RTN 4-bit | **AWQ 4-bit** |
|---|---|---|---|
| WikiText-2 PPL | 12.701 | 15.081 (+18.7%) | **14.733 (+16.0%)** |
| C4 PPL | 19.156 | 22.987 (+20.0%) | **21.999 (+14.8%)** |
| ARC-easy acc (300 ex.) | 0.597 | 0.590 | 0.573 |
| PIQA acc (300 ex.) | 0.710 | 0.663 | 0.680 |
| Weights in memory | 1.976 GB (FP32) | — | 0.738 GB (2.68× smaller) |
| Next-token agreement with original | — | — | 0.786 |
| Speed (CPU, greedy) | 9.23 tok/s | — | 1.64 tok/s (reference Python dequant kernel) |

## Gates
| Gate | Result |
|---|---|
| verify: all layers finite, size = formula (ratio 1.00), top-1 agreement ≥ 0.75 | PASS |
| AWQ PPL < RTN PPL (both datasets) | PASS |
| lm-eval avg drop ≤ 3 points | PASS |
| no empty generations | PASS |
| **AWQ PPL within 10% of original (≤3B gate)** | **FAIL** (+16.0% / +14.8%) |

## What this says, honestly
- AWQ works as designed: on every dataset it beats plain rounding, and on 69 of 72 searched groups the chosen scale beat RTN on calibration output (`search_log.jsonl`). The other 24 groups are the GQA `o_proj` layers, which cannot fold, so they are left unscaled to keep the vLLM export possible.
- On a **0.5B** model it closes only 15–26% of the RTN gap at 4-bit/g128. Tiny models have little redundancy, so 4-bit hurts them most. That's why the ≤3B gate fails. Expect a much smaller gap at 7B+, but that's a claim for **your GPU run** to verify, not something proven here.
- The lm-eval differences (±3 points) are inside the noise of 300-example samples.
- The reference runtime is ~5.6× slower than FP32 on CPU. Speed comes from the vLLM export, not from this kernel.

## History (nothing hidden)
1. `run.log`: the first `run_all.sh` passed gates 00–03 and **stopped at gate 04**. The size gate compared against a 16-bit formula, but the CPU run stores the unquantized embedding in FP32. That was a bug in the gate, not in the quantization: 0.738 GB measured vs 0.737 GB for the correct FP32 formula.
2. The gate was fixed (`awq/hardware.py::native_size_gb`); `04_verify.py` was re-run → `verify.log` PASS; then `05_eval.py` → `eval.log`.
3. Tuning (`tuning/`), then a final run with the tuned config (`tuned-g64/`). Details below.

## Tuning (06_tune.py): metric = WikiText-2 *validation* PPL, never the test set

| trial | change | validation PPL | accepted? |
|---|---|---|---|
| 0 | base (g128, fold auto, propagate fp) | 13.844 | — |
| 1 | `quant.fold_scales=never` | 13.829 (−0.1%) | no (< 0.5% bar; would also lose the vLLM export) |
| 2 | `quant.group_size=64` | **13.198 (−4.7%)** | **yes** |
| 3 | g64 + `search.propagate=quantized` | 13.219 | no |

## Final run with the tuned config (group 64), evaluated once on the test windows (`tuned-g64/`)

| | Original | RTN g128 | AWQ g128 | RTN g64 | **AWQ g64** |
|---|---|---|---|---|---|
| WikiText-2 PPL | 12.701 | 15.081 | 14.733 | 14.481 | **14.057 (+10.7%)** |
| C4 PPL | 19.156 | 22.987 | 21.999 | 21.894 | **21.149 (+10.4%)** |

- Next-token agreement 0.786 → **0.830**. Weights 0.752 GB (FP32 run). lm-eval: ARC-easy 0.570, PIQA 0.683 (gate PASS).
- **Brutal truth:** most of the gain came from the smaller group size, which helps plain RTN too. At equal group size, AWQ removes a further ~24% (WikiText-2) / ~27% (C4) of the remaining gap. The 0.5B model still **fails the strict ≤10% gate** by 0.4–0.7 percentage points.
- Export: `07_export.py --format native,autoawq` → standard AWQ GEMM folder. Layout check PASS (worst rel. diff 3.3e-4 = fp16 rounding of scales). `autoawq_export_config.json` shows the `quantization_config` vLLM reads. `MODEL_CARD.md` was produced by `08_publish.py --dry-run` (nothing uploaded: no HF write token was used).

## Files
`env.json` · `model_report.json` (sizing before the run) · `baseline.json` · `search_log.jsonl` (per block: α, AWQ vs RTN loss, block error) · `quant_config.json` · `verify.json` · `results.json` · `generations.md` · logs.
