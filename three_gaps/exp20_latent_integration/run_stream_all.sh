#!/usr/bin/env bash
# sequential GPU jobs (never in parallel): per-subject moments for all subjects, prob first, CR3 first.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
PY=${PY:-/d/Programs/anaconda_new/envs/py11/python.exe}
for SPEC in "CR3 1 prob" "CR3 2 prob" "CR3 1 logit" "CR3 2 logit" "C1 1 prob" "C1 2 prob" "C2 1 prob" "C2 2 prob" "C3 1 prob" "C3 2 prob"; do
  set -- $SPEC
  $PY three_gaps/exp20_latent_integration/stream_moments.py $1 $2 $3 >> outputs/three_gaps/exp20_wave07/logs/stream_$1_$2_$3.log 2>&1
done
echo finished > outputs/three_gaps/exp20_wave07/logs/stream_all.flag
