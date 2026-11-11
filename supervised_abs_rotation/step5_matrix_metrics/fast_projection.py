"""Exact isotropic polynomial projections with efficient feature generation.

CUDA is a double-precision sufficient-statistic backend, not a neural
network model. Every original row contributes to the final projection.
"""
import numpy as np
from scipy.linalg import solve

import surrogates as sur


class Polynomial(sur.Polynomial):
    def __init__(self, scaler, degree):
        super().__init__(scaler, degree)
        lookup = {tuple(p): i for i, p in enumerate(self.powers)}
        self.blocks = []
        for order in range(1, degree+1):
            ids = np.flatnonzero(self.powers.sum(1) == order)
            parents, variables, ratios = [], [], []
            for row in ids:
                p = self.powers[row].copy()
                j = int(np.flatnonzero(p)[0]); p[j] -= 1
                parent = lookup[tuple(p)]
                parents.append(parent); variables.append(j)
                ratios.append(self.weights[row]/self.weights[parent])
            self.blocks.append((ids, np.asarray(parents), np.asarray(variables), np.asarray(ratios)))

    def features(self, w):
        x = self.scaler(w)
        phi = np.empty((len(x), len(self.powers)), order="F")
        phi[:, 0] = 1.
        for ids, parents, variables, ratios in self.blocks:
            phi[:, ids] = phi[:, parents]*x[:, variables]*ratios[None, :]
        return phi


def _solve_stats(ga, ca, gv, cv, sa, sv, ya, yv, count, nv, ridges):
    p = len(ga); nt = count-nv
    if nt == 0 or nv == 0:
        raise ValueError("Nonempty train and validation sets required")
    gt, ct = ga-gv, ca-cv
    variance = np.maximum((ya-yv)/nt-((sa-sv)/nt)**2, 1e-12)
    penalty = np.eye(p); penalty[0, 0] = 0.

    def estimate(g, cross, ridge):
        scale = max(float(np.trace(g[1:, 1:]))/max(p-1, 1), 1e-12)
        return solve(g+ridge*scale*penalty, cross, assume_a="sym")

    trials = []
    for ridge in ridges:
        coef = estimate(gt, ct, ridge)
        sse = yv-2*np.sum(coef*cv, axis=0)+np.sum(coef*(gv @ coef), axis=0)
        nmse = np.maximum(sse, 0.)/nv/variance
        trials.append((float(nmse.mean()), ridge, nmse))
    _, ridge, nmse = min(trials, key=lambda value: value[0])
    return estimate(ga, ca, ridge), dict(ridge=ridge, validation_nmse=nmse.tolist(),
        validation_rows=nv, train_rows=nt, feature_count=p)


def fit(model, w, y, sid, train_subjects, chunk=8192, device="auto", ridges=(1e-6, 1e-4, .01), statistics_output=None):
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "numpy"
        except ImportError:
            device = "numpy"
    if device == "numpy":
        info = sur.fit_all_rows(model, w, y, sid, train_subjects, chunk=chunk, ridges=ridges)
        info["backend"] = "numpy_float64"
        if statistics_output is not None:
            p = len(model.powers)
            ga, gv = np.zeros((p, p)), np.zeros((p, p))
            train_mask = np.zeros(int(sid.max())+1, dtype=bool); train_mask[train_subjects] = True
            nv = 0
            for a in range(0, len(w), chunk):
                b = min(a+chunk, len(w))
                phi = model.features(w[a:b]); valid = ~train_mask[sid[a:b]]
                ga += phi.T @ phi
                pv = phi[valid]; gv += pv.T @ pv; nv += int(valid.sum())
            np.savez_compressed(statistics_output, g_all=ga, g_validation=gv,
                                n_all=len(w), n_validation=nv)
        return model.coef, info
    import torch
    torch_device = torch.device(device)
    p, channels = len(model.powers), y.shape[1]
    ga = torch.zeros((p, p), dtype=torch.float64, device=torch_device)
    gv = torch.zeros_like(ga)
    ca = torch.zeros((p, channels), dtype=torch.float64, device=torch_device)
    cv = torch.zeros_like(ca)
    sa = torch.zeros(channels, dtype=torch.float64, device=torch_device)
    sv, ya, yv = torch.zeros_like(sa), torch.zeros_like(sa), torch.zeros_like(sa)
    train_mask = np.zeros(int(sid.max())+1, dtype=bool); train_mask[train_subjects] = True
    blocks = [(ids, torch.as_tensor(parents, device=torch_device),
               torch.as_tensor(variables, device=torch_device),
               torch.as_tensor(ratios, dtype=torch.float64, device=torch_device))
              for ids, parents, variables, ratios in model.blocks]
    mean = torch.as_tensor(model.scaler.mean, device=torch_device)
    scale = torch.as_tensor(model.scaler.scale, device=torch_device)
    nv = 0
    with torch.inference_mode():
        for a in range(0, len(w), chunk):
            b = min(a+chunk, len(w))
            x = torch.as_tensor(np.array(w[a:b], dtype=np.float64), device=torch_device)
            x = (x-mean)/scale
            z = torch.as_tensor(np.array(y[a:b], dtype=np.float64), device=torch_device)
            phi = torch.empty((b-a, p), dtype=torch.float64, device=torch_device)
            phi[:, 0] = 1.
            for ids, parents, variables, ratios in blocks:
                phi[:, int(ids[0]):int(ids[-1])+1] = phi.index_select(1, parents)*x.index_select(1, variables)*ratios[None, :]
            ga.addmm_(phi.T, phi); ca.addmm_(phi.T, z)
            sa += z.sum(0); ya += (z*z).sum(0)
            validation = ~train_mask[sid[a:b]]
            if np.any(validation):
                mask = torch.as_tensor(validation, device=torch_device)
                pv, zv = phi[mask], z[mask]
                gv.addmm_(pv.T, pv); cv.addmm_(pv.T, zv)
                sv += zv.sum(0); yv += (zv*zv).sum(0)
                nv += int(validation.sum())
            if (a//chunk+1) % 128 == 0 or b == len(w):
                print(f"[Projection] {model.name}: {b:,}/{len(w):,} original rows", flush=True)
    arrays = [v.cpu().numpy() for v in (ga, ca, gv, cv, sa, sv, ya, yv)]
    if not all(np.isfinite(v).all() for v in arrays):
        raise ArithmeticError("Non-finite double-precision projection statistics")
    if statistics_output is not None:
        np.savez_compressed(statistics_output, g_all=arrays[0], g_validation=arrays[2],
                            n_all=len(w), n_validation=nv)
    coef, info = _solve_stats(*arrays, len(w), nv, ridges)
    info.update(backend=f"torch_{torch_device.type}_float64", final_fit_rows=len(w),
                final_fit_subjects=len(train_mask), train_subjects=len(train_subjects),
                validation_subjects=len(train_mask)-len(train_subjects))
    return coef, info
