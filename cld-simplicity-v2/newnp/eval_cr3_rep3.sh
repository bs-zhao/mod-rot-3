#!/usr/bin/env bash
# Evaluate on the NEW NP (CR3 repeat 3): frozen EXP13, the phase-1 top candidates, and the closest alternatives (Sobolev MLP seeds 0-2, shared-basis, local linear).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 CLD_MODELS=CR3
PY=${PY:-/d/Programs/anaconda_new/envs/py11/python.exe}
L=outputs/cld-simplicity-v2
until [ -f $L/np_cr3_rep3.flag ]; do sleep 30; done
CLD_CANDS="ctx+uu|axisneff|K4|f2.0;ctx+ux+uu|tail2|K8|f2.0;ctx+uu|cf1|K8|f2.0;ctx+ux+uu|varimax|K8|f2.0;frozen|-|K0|f0.0" $PY cld-simplicity-v2/phase1_ctx/run.py 0 1 3 24 250 newnp > $L/newnp_p1.log 2>&1
for SD in 0 1 2; do ONLY=sobolev_mlp SEED=$SD TAGOUT=newnp_mlp_seed$SD $PY cld-simplicity-v2/phase6_xdep/run.py CR3 3 24 250 > $L/newnp_mlp_$SD.log 2>&1; done
ONLY=sharedbasis_r16,localLinear,chord_knn TAGOUT=newnp_other $PY cld-simplicity-v2/phase6_xdep/run.py CR3 3 24 250 > $L/newnp_other.log 2>&1
echo ok > $L/newnp_eval.flag
