#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
PY=${PY:-/d/Programs/anaconda_new/envs/py11/python.exe}
for SPEC in "CR3 1 prob" "CR3 2 prob" "CR3 1 logit" "CR3 2 logit"; do
  set -- $SPEC
  $PY three_gaps/exp20_latent_integration/run_20a.py $1 $2 $3 > outputs/three_gaps/exp20_wave07/logs/exp20a_$1_$2_$3.log 2>&1
done
echo finished > outputs/three_gaps/exp20_wave07/logs/exp20a_cr3.flag
