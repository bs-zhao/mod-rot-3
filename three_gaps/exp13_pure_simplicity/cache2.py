"""EXP13: cache NN-decoder derivative fields up to second order in the frozen-Z coordinates V.
g (S,K,d), Htt (S,K,d,d), Htx (S,K,d,p), h (S,K), with the grid subjects, stimuli, V, Tstd (post-hoc only), split ids.
python three_gaps/exp13_pure_simplicity/cache2.py --model CR3 --beta 0.00005 --zdim 4 --pca 0 [--repeat 1] [--channel logit]"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import io  # noqa: E402
from common.analytic import X_DIM  # noqa: E402
from common.decoder import decoder_h  # noqa: E402
from common.derivs import base_derivatives  # noqa: E402

OUT = io.ROOT / "outputs" / "three_gaps" / "exp13"


def build(model, beta, zdim, repeat=1, S=300, channel="logit", seed=20261108):
    dev = io.pick_device("auto"); rng = np.random.default_rng(seed)
    data, truth = io.load_all(model, beta, zdim, 0, repeat)
    net, _ = io.load_decoder(data["paths"]["checkpoint"], dev)
    ids = np.sort(rng.choice(data["train_ids"], S, replace=False))
    if model == "CR3":
        X = torch.tensor(data["x"][data["subject_index"] == 0], dtype=torch.float64)
    else:
        X0 = rng.uniform(-1, 1, (200, X_DIM[model])); X = torch.tensor(np.concatenate([X0, -X0]), dtype=torch.float64)
    h = decoder_h(net, data["mu_z"], data["Minv"], channel)
    grid = torch.tensor(data["V"][ids], dtype=torch.float64, device=dev)
    b = base_derivatives(h, grid, X.to(dev), which=("g", "Htt", "Htx"), chunk=2048)
    return dict(g=b["g"].cpu(), Htt=b["Htt"].cpu(), Htx=b["Htx"].cpu(), h=b["h"].cpu(), ids=ids, X=X, V=data["V"],
                Tst=truth.std(), te=data["test_ids"], tr=data["train_ids"], cond=data["condition"])


if __name__ == "__main__":
    ap = io.add_common_args(argparse.ArgumentParser()); ap.add_argument("--channel", default="logit"); a = ap.parse_args()
    c = build(a.model, a.beta, a.zdim, a.repeat, channel=a.channel)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(c, OUT / f"cache2_{a.model}_{c['cond']}_{a.channel}.pt")
    print("saved", {k: tuple(v.shape) for k, v in c.items() if torch.is_tensor(v)})
