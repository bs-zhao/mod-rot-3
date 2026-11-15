#!/usr/bin/env bash
# C1-C3 confirmation of the CR3 smoothing result (historical 300-subject outer measure, sigma = 0, .25, .5)
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
PY=${PY:-/d/Programs/anaconda_new/envs/py11/python.exe}
L=outputs/three_gaps/exp20_wave07/logs
until [ -f $L/exp20a_c.flag ]; do sleep 20; done
for M in C1 C2 C3; do for R in 1 2; do
  $PY three_gaps/exp22_smoothed_hessian/run.py $M $R prob cache a0 16 0,0.25,0.5 > $L/exp22c_cache_${M}_${R}.log 2>&1
  $PY three_gaps/exp22_smoothed_hessian/run.py $M $R prob eval a0 16 0,0.25,0.5 > $L/exp22c_eval_${M}_${R}.log 2>&1
done; done
echo ok > $L/exp22c.flag
