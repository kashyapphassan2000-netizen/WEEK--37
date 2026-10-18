#!/usr/bin/env bash
# run_all.sh — gates 0 -> 5 for one config. Stops at the first failure.
#   bash scripts/run_all.sh configs/run.yaml 2>&1 | tee run.log
set -euo pipefail
CFG="${1:?usage: bash scripts/run_all.sh configs/run.yaml [extra --set args]}"
shift || true
cd "$(dirname "$0")/.."
step() { echo; echo "================ $* ================"; }
step "00 env check";      python scripts/00_env_check.py   --config "$CFG" "$@"
step "01 inspect model";  python scripts/01_inspect_model.py --config "$CFG" "$@"
step "02 baseline";       python scripts/02_baseline.py    --config "$CFG" "$@"
step "03 quantize";       python scripts/03_quantize.py    --config "$CFG" "$@"
step "04 verify";         python scripts/04_verify.py      --config "$CFG" "$@"
step "05 eval";           python scripts/05_eval.py        --config "$CFG" "$@"
echo; echo "ALL GATES PASSED. Next: 06_tune.py (optional), 07_export.py, 08_publish.py"
