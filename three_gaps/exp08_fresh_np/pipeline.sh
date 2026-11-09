#!/usr/bin/env bash
# EXP08 sequential driver for the fresh NP (repeat 2).  ONE job at a time, 4 CPU
# threads (running several trainings in parallel froze the laptop).
# For each model: S3 train (skipped if its summary exists; waits if another
# process is already training it) -> S5 all-trial Z -> EXP00 oracle -> EXP06.
# usage: bash three_gaps/exp08_fresh_np/pipeline.sh <python> [MODEL ...]
set -e
PY=$1; shift
MODELS=${*:-"C1 C2 C3 CR3"}
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p outputs/three_gaps/logs
for M in $MODELS; do
  case $M in
    C1|C2) B=0.0005;  Z=4; PAT=20; COND=beta0p0005_lambda0_ctx0p25to0p8_L4 ;;
    C3)    B=0.0005;  Z=6; PAT=20; COND=beta0p0005_lambda0_ctx0p25to0p8_L6 ;;
    CR3)   B=0.00005; Z=4; PAT=50; COND=beta5em05_lambda0_ctx0p25to0p8_L4 ;;
  esac
  LOG=outputs/three_gaps/logs/exp08_$M.log
  SUM=outputs/choice/$M/s3_train/$COND/repeat02/s3_train__summary.json
  if [ ! -f "$SUM" ]; then
    if grep -q "epoch=" outputs/three_gaps/logs/s3_rep2_$M.log 2>/dev/null && \
       [ $(( $(date +%s) - $(stat -c %Y outputs/three_gaps/logs/s3_rep2_$M.log) )) -lt 300 ]; then
      until [ -f "$SUM" ]; do sleep 30; done      # already being trained by another process
    else
      "$PY" np/s3_train.py --model $M --beta $B --lambda-sparse 0 --latent-dim $Z --context-min 0.25 \
          --context-max 0.80 --batch-size 128 --learning-rate 0.0005 --weight-decay 0 --max-epochs 500 \
          --patience $PAT --repeat 2 --device auto > outputs/three_gaps/logs/s3_rep2_$M.log 2>&1
    fi
  fi
  echo "[$(date +%T)] $M S3 done" > "$LOG"
  "$PY" np/s5_extract_z.py --model $M --beta $B --lambda-sparse 0 --latent-dim $Z --repeat 2 >> "$LOG" 2>&1
  echo "[$(date +%T)] $M S5 done" >> "$LOG"
  "$PY" three_gaps/exp00_oracle/run.py --model $M --beta $B --zdim $Z --pca 0 --repeat 2 >> "$LOG" 2>&1
  echo "[$(date +%T)] $M EXP00 done" >> "$LOG"
  "$PY" three_gaps/exp06_two_stage/run.py --model $M --beta $B --zdim $Z --pca 0 --repeat 2 \
      --view GCat --losses absL1,rmsL1 --sources decoder_ad,decoder_sieve --tag confirm >> "$LOG" 2>&1
  echo "[$(date +%T)] $M EXP06 done" >> "$LOG"
done
echo "[$(date +%T)] ALL DONE" >> outputs/three_gaps/logs/exp08_all.log
