# GPU run record — fill this in after your rented-GPU run

Copy this file to `proof/gpu-<model-name>/README.md` and paste the real outputs. Don't round numbers in your favour, and don't delete failing gates.

## Setup
- Provider / instance: (e.g. RunPod community, 1x RTX 4090 24 GB)
- Hourly price paid: $
- Total wall time / total cost: h / $
- `git rev-parse --short HEAD`:
- Config used (paste `runs/<name>/run_config.yaml`):

## Files to copy from `runs/<name>/` into `proof/gpu-<model-name>/`
- [ ] `env.json` (GPU, driver, torch versions)
- [ ] `model_report.json` (sizing estimate made BEFORE the run)
- [ ] `baseline.json`
- [ ] `search_log.jsonl`
- [ ] `verify.json`
- [ ] `results.json`
- [ ] `generations.md`
- [ ] `run.log` (full output of `bash scripts/run_all.sh ... | tee run.log`)
- [ ] `quantize.log` (it contains the measured minutes per block, which calibrates the speed constant in `awq/hardware.py`)

## Headline table (copy from results.json / baseline.json)

| | original 16-bit | RTN 4-bit | AWQ 4-bit |
|---|---|---|---|
| WikiText-2 PPL | | | |
| C4 PPL | | | |
| weight memory GB | | — | |
| peak VRAM GB (eval) | | | |
| lm-eval avg | | | |

## Estimate vs reality
| | 01_inspect_model estimate | measured |
|---|---|---|
| quantize VRAM GB | | |
| quantize hours | | |
| AWQ size GB | | |
