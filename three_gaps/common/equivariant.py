"""Stage A: output-equivariant axes (shift / gain) detected from gradient fields.

For subject s with stimulus profile of outputs h_s (K,) and gradients G_s (K, d)
w.r.t. base coordinates v, a displacement direction w is
  * a SHIFT axis if  G_s w  is proportional to phi_shift(h_s)
  * a GAIN  axis if  G_s w  is proportional to phi_gain(h_s)
with  logit channel: phi_shift = 1,        phi_gain = h
      prob  channel: phi_shift = h(1-h),   phi_gain = h(1-h) logit(h)
(the prob forms are exactly the chain rule of the logit forms).  The
relative residual  e(w) = sum_s ||P_perp(phi_s) G_s w||^2 / sum_s ||G_s w||^2
is minimised by a generalised eigenproblem.  An axis is accepted only if it is
SALIENT:  e1 < ABS_T  and  e1 < REL_T * e2   (thresholds fixed a priori).
Uses only gradients and outputs of the function under study - never T.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sl

ABS_T = 0.3
REL_T = 0.3


def profiles(h, channel):
    h = np.asarray(h, float)
    if channel == "logit":
        return {"shift": np.ones_like(h), "gain": h}
    p = np.clip(h, 1e-4, 1 - 1e-4)
    w = p * (1 - p)
    return {"shift": w, "gain": w * np.log(p / (1 - p))}


def detect(G, h, channel, exclude=None):
    """G (S,K,d), h (S,K).  Returns list of accepted dicts (type, w unit, e1, e2)."""
    G = np.asarray(G, float)
    B = np.einsum("skd,ske->de", G, G)
    found, report = [], {}
    for name, phi in profiles(h, channel).items():
        P = phi / np.maximum(np.linalg.norm(phi, axis=1, keepdims=True), 1e-300)
        Gp = np.einsum("sk,skd->sd", P, G)
        A = B - np.einsum("sd,se->de", Gp, Gp)
        ev, W = sl.eigh(A, B)
        report[name] = (float(ev[0]), float(ev[1]))
        if ev[0] < ABS_T and ev[0] < REL_T * ev[1]:
            w = W[:, 0] / np.linalg.norm(W[:, 0])
            found.append(dict(type=name, w=w, e1=float(ev[0]), e2=float(ev[1])))
    # keep at most one axis per type; drop near-duplicates (|cos| > 0.9)
    out = []
    for f in sorted(found, key=lambda f: f["e1"]):
        if all(abs(f["w"] @ g["w"]) < 0.9 for g in out):
            out.append(f)
    if len(out) > 1:  # orthogonalise later ones against earlier ones
        for i in range(1, len(out)):
            w = out[i]["w"]
            for j in range(i):
                w = w - (w @ out[j]["w"]) * out[j]["w"]
            out[i]["w"] = w / np.linalg.norm(w)
    return out, report
