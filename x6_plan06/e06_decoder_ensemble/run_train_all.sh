#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
PY=${PY:-/d/Programs/anaconda_new/envs/py11/python.exe}
for M in CR3 C1 C2 C3; do for REP in 1 2; do
  $PY x6_plan06/e06_decoder_ensemble/train.py $M $REP 10 0 40 base >> outputs/x6_plan06/e06_train.log 2>&1
done; done
echo finished >> outputs/x6_plan06/e06_train.log
