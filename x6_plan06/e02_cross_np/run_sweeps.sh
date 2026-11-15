#!/usr/bin/env bash
# E2: run the (unmodified) exp13 tune.py on repeat 1 and repeat 2; selection is done on one repeat, evaluation on the other (analyze.py).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
PY=${PY:-/d/Programs/anaconda_new/envs/py11/python.exe}
CFG="neff,share,0,0;neff,share,0.25,0;neff,share,0.5,0;neff,share,1,0;neff,share,2,0;neff,share,4,0;neff,share,8,0;neff,sum,0.25,0;neff,sum,0.5,0;neff,sum,1,0;neff,sum,2,0;neff,sum,4,0;neff,sum,8,0;neff,share,0.5,1;neff,share,1,1;neff,share,2,1;neff,share,4,1;l1l2,share,1,0;l1l2,share,2,0;l1l2,share,1,1;quartimax,share,1,0;softL0,share,1,0"
for REP in 1 2; do
  $PY three_gaps/exp13_pure_simplicity/tune.py $REP outputs/x6_plan06/e02_tune_rep$REP.csv "$CFG" 12 > outputs/x6_plan06/e02_tune_rep$REP.log 2>&1
done
echo finished > outputs/x6_plan06/e02_done.flag
