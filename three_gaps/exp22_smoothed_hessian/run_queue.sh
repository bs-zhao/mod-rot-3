#!/usr/bin/env bash
# sequential GPU queue: wait for the running EXP20B node job, then CR3 rep2 nodes, then EXP22 per-subject smoothed moments (A0 300 subjects).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
PY=${PY:-/d/Programs/anaconda_new/envs/py11/python.exe}
L=outputs/three_gaps/exp20_wave07/logs
while pgrep -f "run_20b.py CR3 1 prob nodes" > /dev/null 2>&1 || tasklist | grep -qi "run_20b"; do sleep 20; done
until [ -f outputs/three_gaps/exp20_wave07/nodes_CR3_beta5em05_lambda0_ctx0p25to0p8_L4_prob/COMPLETE.json ]; do sleep 20; done
$PY three_gaps/exp20_latent_integration/run_20b.py CR3 2 prob nodes > $L/exp20b_nodes_CR3_2_prob.log 2>&1
for REP in 1 2; do
  $PY three_gaps/exp22_smoothed_hessian/run.py CR3 $REP prob cache a0 16 >> $L/exp22_cache_CR3_${REP}_prob.log 2>&1
done
echo finished > $L/queue1.flag
