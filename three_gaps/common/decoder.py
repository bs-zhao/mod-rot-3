"""Frozen NP decoder queried through the exact inverse map z = mu_Z + C^T v."""
from __future__ import annotations

import numpy as np
import torch


def decoder_h(net, mu_z, Minv, channel):
    """z = mu_z + Minv @ v   (pass data['Minv'])."""
    mu = torch.as_tensor(mu_z)
    Ct = torch.as_tensor(Minv).T

    def h(v, x):
        z = mu.to(v) + Ct.to(v).T @ v
        lg = net.decoder(torch.cat([x, z]).to(torch.float32)).to(v.dtype)
        e = lg[1] - lg[0]
        return e if channel == "logit" else torch.sigmoid(e)
    return h


@torch.no_grad()
def decoder_eta(net, z, x, si, dev, chunk=400000):
    out = np.empty(len(x))
    for i in range(0, len(x), chunk):
        zz = torch.tensor(z[si[i:i + chunk]], dtype=torch.float32, device=dev)
        xx = torch.tensor(x[i:i + chunk], dtype=torch.float32, device=dev)
        lg = net.decoder(torch.cat([xx, zz], -1))
        out[i:i + chunk] = (lg[:, 1] - lg[:, 0]).double().cpu().numpy()
    return out
