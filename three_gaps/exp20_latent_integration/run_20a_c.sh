#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
PY=${PY:-/d/Programs/anaconda_new/envs/py11/python.exe}
for M in C1 C2 C3; do for R in 1 2; do
  $PY three_gaps/exp20_latent_integration/run_20a.py $M $R prob 300,1000 2 > outputs/three_gaps/exp20_wave07/logs/exp20a_${M}_${R}_prob.log 2>&1
done; done
echo ok > outputs/three_gaps/exp20_wave07/logs/exp20a_c.flag
