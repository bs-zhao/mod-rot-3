#!/usr/bin/env bash
# New NP for CR3: independent training seed (repeat 3) on the SAME S1 dataset (a new S1 dataset would need np/s1_generate with a new seed; not done here).
# train -> S5 latent -> EXP00 oracle -> EXP13 cache2 (p channel).  One job at a time (parallel trainings froze the laptop before).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
PY=${PY:-/d/Programs/anaconda_new/envs/py11/python.exe}
L=outputs/cld-simplicity-v2
$PY np/s3_train.py --model CR3 --beta 0.00005 --lambda-sparse 0 --latent-dim 4 --context-min 0.25 --context-max 0.80 --batch-size 128 --learning-rate 0.0005 --weight-decay 0 --max-epochs 500 --patience 50 --repeat 3 --device auto > $L/np_train_CR3_rep3.log 2>&1 || exit 1
$PY np/s5_extract_z.py --model CR3 --beta 0.00005 --lambda-sparse 0 --latent-dim 4 --repeat 3 > $L/np_s5_CR3_rep3.log 2>&1 || exit 1
$PY three_gaps/exp00_oracle/run.py --model CR3 --beta 0.00005 --zdim 4 --pca 0 --repeat 3 > $L/np_exp00_CR3_rep3.log 2>&1 || exit 1
$PY three_gaps/exp13_pure_simplicity/cache2.py --model CR3 --beta 0.00005 --zdim 4 --pca 0 --repeat 3 --channel prob > $L/np_cache2_CR3_rep3.log 2>&1 || exit 1
echo ok > $L/np_cr3_rep3.flag
