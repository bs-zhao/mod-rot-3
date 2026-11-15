"""EXP20A streaming per-subject moments.

For EVERY subject (all N; TRAIN/VAL/TEST ids stored) the autograd second moments over the X bank:
  python stream_moments.py MODEL REPEAT CHANNEL [BATCH=16]
writes outputs/three_gaps/exp20_wave07/moments_<model>_<cond>_<ch>.npz and an atomic COMPLETE.json.
Subsets / TRAIN-only / transductive aggregation are weighted averages afterwards: no Hessian is recomputed.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x7core as X

if __name__ == "__main__":
    model, rep, ch = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    bs = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    st = X.Setting(model, rep)
    out = X.OUT / f"moments_{model}_{st.cond}_{ch}.npz"
    done = X.OUT / f"moments_{model}_{st.cond}_{ch}.COMPLETE.json"
    if done.exists():
        print("already complete", out)
        sys.exit(0)
    t0 = time.time()
    SX, T4 = X.centre_moments(st, ch, st.V, bs=bs)
    np.savez(out, SX=SX, T4=T4, V=st.V, train_ids=st.tr, val_ids=st.va, test_ids=st.te)
    json.dump(dict(model=model, rep=rep, channel=ch, N=len(st.V), K=len(st.X), secs=time.time() - t0, device=str(X.DEV),
                   dtype_accum="float64", checkpoint_sha256=X.sha256(st.data["paths"]["checkpoint"]),
                   latent_sha256=X.sha256(st.data["paths"]["latent"]), x_bank_source=str(st.cache_path),
                   moments_sha256=X.sha256(out)), open(done, "w"), indent=1)
    print(f"done {model} rep{rep} {ch}: N={len(st.V)} K={len(st.X)} in {time.time() - t0:.0f}s")
