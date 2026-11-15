#!/usr/bin/env bash
# Follow-ups: (1) total-light normalisation detector, (2) training on mixed conditions.
set -euo pipefail
cd "$(dirname "$0")"
JOBS=${JOBS:-5}; THREADS=${THREADS:-3}; STEPS=${STEPS:-4000}
{
  echo "--method frames --train-cond mixed --seed 0"
  for k in 16 64; do
    for seed in 0 1 2; do echo "--method mask_norm --k $k --seed $seed"; done
    echo "--method mask --k $k --train-cond mixed --seed 0"
    echo "--method mask_norm --k $k --train-cond mixed --seed 0"
  done
} | xargs -P "$JOBS" -I{} sh -c "python3 train.py {} --steps $STEPS --threads $THREADS >> train.log 2>&1"
python3 analyze.py
