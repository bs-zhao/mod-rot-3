"""Build and cache decoder autodiff gradient fields for a model (shared by EXP11 scripts).

python three_gaps/exp11_cr3_search/cache.py --model CR3 --beta 0.00005 --zdim 4 --pca 0 [--repeat 1]
Saves outputs/three_gaps/exp11/cache_<cond>.pt : g (S,K,d), h (S,K), ids, V, Tstd (grid subjects), X.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import io  # noqa: E402
from common.analytic import X_DIM, make_h  # noqa: E402
from common.decoder import decoder_h  # noqa: E402
from common.derivs import base_derivatives  # noqa: E402


def build(model, beta, zdim, repeat=1, S=300, source="decoder", channel="logit", seed=20261108):
    dev = io.pick_device("auto")
    rng = np.random.default_rng(seed)
    data, truth = io.load_all(model, beta, zdim, 0, repeat)
    net, _ = io.load_decoder(data["paths"]["checkpoint"], dev)
    tr = data["train_ids"]
    ids = np.sort(rng.choice(tr, S, replace=False))
    bank = model == "CR3"
    m0 = data["subject_index"] == 0
    if bank:
        X = torch.tensor(data["x"][m0], dtype=torch.float64)
    else:
        X0 = rng.uniform(-1, 1, (200, X_DIM[model]))
        X = torch.tensor(np.concatenate([X0, -X0]), dtype=torch.float64)
    Tst = truth.std()
    if source == "decoder":
        h = decoder_h(net, data["mu_z"], data["Minv"], channel); coords = data["V"]
    else:
        h = make_h(model, channel, truth.mu, truth.sd); coords = Tst
    grid = torch.tensor(coords[ids], dtype=torch.float64, device=dev)
    base = base_derivatives(h, grid, X.to(dev), which=("g",), chunk=4096)
    return dict(g=base["g"].cpu(), h=base["h"].cpu(), ids=ids, X=X, V=data["V"], Tst=Tst,
                te=data["test_ids"], tr=tr, cond=data["condition"], mu_z=data["mu_z"])


if __name__ == "__main__":
    ap = io.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--source", default="decoder")
    ap.add_argument("--channel", default="logit")
    a = ap.parse_args()
    c = build(a.model, a.beta, a.zdim, a.repeat, source=a.source, channel=a.channel)
    out = io.ROOT / "outputs" / "three_gaps" / "exp11"; out.mkdir(parents=True, exist_ok=True)
    torch.save(c, out / f"cache_{a.model}_{c['cond']}_{a.source}_{a.channel}.pt")
    print("saved", c["g"].shape)
