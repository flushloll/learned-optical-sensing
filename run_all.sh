#!/usr/bin/env bash
# Full sweep. ~35 min on a 15-core CPU; set JOBS / THREADS for your machine.
# On a Mac, PyTorch uses the MPS GPU automatically (run fewer JOBS, e.g. 2).
set -euo pipefail
cd "$(dirname "$0")"
JOBS=${JOBS:-5}
THREADS=${THREADS:-3}
STEPS=${STEPS:-4000}

python3 kalman.py > results_kalman.log

{
  for m in frames events diff oracle; do echo "--method $m --seed 0"; done
  for seed in 0 1 2; do
    for k in 2 4 8 16 64 256; do
      echo "--method mask --k $k --seed $seed"
      echo "--method random_mask --k $k --seed $seed"
    done
  done
} | xargs -P "$JOBS" -I{} sh -c "python3 train.py {} --steps $STEPS --threads $THREADS >> train.log 2>&1"

python3 analyze.py
