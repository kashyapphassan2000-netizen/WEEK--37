# awq-from-scratch: universal 4-bit AWQ quantization for any open LLM

A from-scratch, pure-PyTorch implementation of **AWQ** (Activation-aware Weight Quantization, Lin et al. 2023, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)), built to quantize models of any size. It streams **one decoder block at a time**, so a 70B model quantizes on one 80 GB GPU and a 1T-parameter MoE on one H200. Hardware sizing, evaluation gates, tuning, export to vLLM and publishing are all included.

> Week 37 of the build log. The full step-by-step operating manual (choosing a model → sizing and renting a GPU → SSH/VS Code/WinSCP → install → run → evaluate → tune → export → shut down) is **[docs/Universal_AWQ_Playbook.pdf](docs/Universal_AWQ_Playbook.pdf)**.

## What is real here (no sugarcoating)

| Claim | Status |
|---|---|
| AWQ scale search + clip search + scale folding + INT4 packing | Implemented, unit-tested (`tests/`, 30 tests, CPU) |
| End-to-end on a real model | **Run here on CPU** with Qwen2.5-0.5B. Real logs and numbers in [`proof/cpu-qwen2.5-0.5b/`](proof/cpu-qwen2.5-0.5b/) |
| Any size via block streaming | Tested on tiny models and Qwen2.5-0.5B. **Not yet run on a GPU at 7B+.** Your GPU run produces those numbers ([template](proof/GPU_RUN_TEMPLATE.md)) |
| MoE (fused 3-D experts, transformers ≥ 5) | Per-expert AWQ implemented and tested on a tiny Mixtral. Untested on real MoE weights |
| FP8 checkpoints (DeepSeek-V3 style) | Block-dequantization implemented. Untested at scale |
| Export to standard AWQ GEMM (vLLM / SGLang) | Implemented. Layout verified against a reference unpacker. **Not yet loaded in vLLM here (no GPU)** |
| Fast inference | **No.** The `AwqLinear` runtime dequantizes in Python (correct, slow). Fast = export → vLLM |

## Measured (CPU proof run, Qwen2.5-0.5B, first 20×2048 test tokens per dataset)

| | Original (FP32) | RTN g128 | AWQ g128 | RTN g64 | **AWQ g64 (tuned)** |
|---|---|---|---|---|---|
| WikiText-2 PPL | 12.701 | 15.081 | 14.733 | 14.481 | **14.057** (+10.7%) |
| C4 PPL | 19.156 | 22.987 | 21.999 | 21.894 | **21.149** (+10.4%) |

AWQ beats RTN at every setting, and tuning (group 64, chosen on the *validation* split) brought the gap from +16% to +10.7%. A 0.5B model still **fails** the strict ≤10% gate; the details and every raw file are in [`proof/cpu-qwen2.5-0.5b/`](proof/cpu-qwen2.5-0.5b/README.md).

## Quick start

```bash
git clone https://github.com/kashyapphassan2000-netizen/WEEK--37.git && cd WEEK--37
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CPU: .../whl/cpu
pip install -e ".[eval,dev]"

# 1. size the hardware BEFORE renting anything (downloads only config.json)
python scripts/01_inspect_model.py --model Qwen/Qwen2.5-7B-Instruct

# 2. edit configs/run.yaml (model + work_dir), then run every gate
bash scripts/run_all.sh configs/run.yaml 2>&1 | tee run.log

# 3. optional tuning, export, publish
python scripts/06_tune.py   --config configs/run.yaml --space configs/tune_space.yaml
python scripts/07_export.py --config configs/run.yaml --format native,autoawq
python scripts/08_publish.py --config configs/run.yaml --repo <you>/<model>-AWQ        # private by default
```

CPU-only proof (no GPU needed, about 1 hour on 4 cores):

```bash
bash scripts/run_all.sh configs/examples/qwen2.5-0.5b-cpu.yaml
```

## Pipeline

| Script | Gate / output |
|---|---|
| `00_env_check.py` | GPU, torch/CUDA, disk vs. model size, HF login → `ENV OK` |
| `01_inspect_model.py` | compatibility verdict + GPU type/count/hours/cost per phase (quantize, FP16 eval, serve) |
| `02_baseline.py` | original-precision and naive-RTN perplexity, generations, lm-eval → `baseline.json` |
| `03_quantize.py` | block-by-block AWQ, resumable (`--resume`) → `quantized/`, `search_log.jsonl` |
| `04_verify.py` | real memory footprint, size vs. formula, top-1 agreement with the original |
| `05_eval.py` | PPL, lm-eval, generations side by side, speed; pass/fail gates vs. baseline |
| `06_tune.py` | stage-wise tuning on a held-out split (never the test set) → `configs/run.best.yaml` |
| `07_export.py` | native format + standard AWQ GEMM format (vLLM), with layout check |
| `08_publish.py` | model card with measured numbers only + upload to the HF Hub |

## Library layout

```
awq/config.py      run.yaml -> typed config, validation, --set overrides
awq/hardware.py    sizing formulas + GPU picker (configs/gpus.yaml prices)
awq/model_io.py    meta-device skeleton, lazy safetensors reader, FP8 dequant, fused-expert mapping, load_quantized
awq/blocks.py      finds decoder blocks / quantizable Linears / fused experts / fold targets in any architecture
awq/calib.py       calibration + eval token windows (no padding)
awq/quantizer.py   4-bit group quantization, pack/unpack
awq/search.py      AWQ scale search (duo scaling) + clip search
awq/awq_linear.py  AwqLinear and AwqExpertsWeight runtime modules
awq/pipeline.py    the block-streaming loop, checkpoints/resume
awq/evaluate.py    perplexity, RTN baseline, generations, speed, lm-eval bridge
awq/export.py      AutoAWQ-GEMM packing (vLLM-compatible)
awq/plugin.py      bring your own non-transformers model
```

## Tests

```bash
pytest -q        # 30 tests, CPU, ~3 min. Tiny random Llama / Qwen2 (GQA, tied) / Mixtral (MoE) / custom plugin model
```

## License
Apache-2.0 for this code. Quantized models inherit their base model's license.
