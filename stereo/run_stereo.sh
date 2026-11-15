#!/usr/bin/env bash
# Full stereo sweep (~1 h on 15 CPU cores). On a Mac: JOBS=2 ./run_stereo.sh
set -euo pipefail
cd "$(dirname "$0")"
JOBS=${JOBS:-5}; THREADS=${THREADS:-3}; STEPS=${STEPS:-4000}

python3 classical3d.py > classical.log

{
  # slowest first
  echo "--setup ssl_mono --size random"
  echo "--setup ssl_stereo --size random"
  for size in fixed random; do
    echo "--setup mono_frames --size $size"
    echo "--setup stereo_frames --size $size"
  done
  for b in 0.03 0.3; do echo "--setup stereo_frames --size random --baseline $b"; done
  for seed in 0 1; do
    for size in fixed random; do
      for r in 32 128; do
        for s in mono_mask sep_mask bino_mask; do
          echo "--setup $s --readings $r --size $size --seed $seed"
        done
      done
    done
  done
} | xargs -P "$JOBS" -I{} sh -c "python3 train3d.py {} --steps $STEPS --threads $THREADS >> train.log 2>&1"

python3 analyze3d.py
