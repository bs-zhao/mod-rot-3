"""Historical row-simple-structure and Jacobian-volume objectives; no truths."""
import numpy as np
import torch
from scipy.optimize import minimize


def rotation(v, k):
    a = torch.zeros((k, k), dtype=v.dtype)
    i, j = torch.triu_indices(k, k, 1)
    a[i, j] = v
    a[j, i] = -v
    return torch.matrix_exp(a)


class RowObjective:
    def __init__(self, blocks, kind='geomin', eps=1e-6, weights=None,
                 coverage_lambda=0., coverage_tau=.05):
        self.blocks = []
        self.kind, self.eps = kind, eps
        self.lam, self.tau = coverage_lambda, coverage_tau
        for index, block in enumerate(blocks):
            block = np.asarray(block, float)
            good = np.linalg.norm(block, axis=1) > 1e-12
            if not good.any():
                continue
            w = np.ones(len(block)) if weights is None else np.asarray(weights[index])
            w = w[good].copy()
            w /= w.sum()
            self.blocks.append((torch.tensor(block[good]), torch.tensor(w)))
        if not self.blocks:
            raise ValueError('No nonzero functional signatures')

    def __call__(self, a):
        values = []
        for b, w in self.blocks:
            sq = (b @ a.T).square()
            p = sq / sq.sum(-1, keepdim=True).clamp_min(1e-30)
            if self.kind == 'geomin':
                row = torch.exp(torch.log(p + self.eps).mean(-1))
            elif self.kind == 'quartimin':
                row = .5 * (p.sum(-1).square() - p.square().sum(-1))
            elif self.kind == 'entropy':
                row = -(p * torch.log(p + 1e-12)).sum(-1) / np.log(max(a.shape[0], 2))
            else:
                raise ValueError(self.kind)
            value = (w * row).sum()
            if self.lam:
                share = sq.sum(0) / sq.sum().clamp_min(1e-30)
                value = value + self.lam * ((self.tau-share).clamp_min(0)/self.tau).square().mean()
            values.append(value)
        return torch.stack(values).mean()


def optimize(score, k, cfg, seed_offset=0):
    rng = np.random.default_rng(cfg['seed'] + seed_offset)
    identity = np.eye(k)
    best_a = identity.copy()
    before = float(score(torch.tensor(identity)))
    best = before
    records, matrices = [], []
    for index in range(cfg['starts']):
        initial = np.zeros(k*(k-1)//2) if index == 0 else rng.normal(0., .3, k*(k-1)//2)
        def fun(v):
            t = torch.tensor(v, dtype=torch.float64, requires_grad=True)
            value = score(rotation(t, k))
            value.backward()
            return float(value.detach()), t.grad.numpy().copy()
        result = minimize(fun, initial, method='L-BFGS-B', jac=True,
                          options={'maxiter': cfg['maxiter'], 'gtol': 1e-7, 'ftol': 1e-12})
        a = rotation(torch.tensor(result.x), k).detach().numpy()
        value = float(score(torch.tensor(a)))
        matrices.append(a)
        records.append(dict(start=index, loss=value, success=bool(result.success),
                            message=str(result.message), iterations=int(result.nit), nfev=int(result.nfev)))
        if np.isfinite(value) and value < best:
            best, best_a = value, a
    return best_a, dict(before=before, after=best, starts=records), np.stack(matrices)


def varimax(rows, maxiter=500):
    norms = np.linalg.norm(rows, axis=1)
    good = norms > 1e-12
    a = rows[good] / norms[good, None]
    k = a.shape[1]
    r, old = np.eye(k), 0.
    for iteration in range(maxiter):
        l = a @ r
        target = l**3 - l @ np.diag(np.sum(l*l, axis=0)) / len(a)
        u, s, v = np.linalg.svd(a.T @ target, full_matrices=False)
        r = u @ v
        value = s.sum()
        if old > 0 and abs(value-old) < 1e-7*max(1.,old):
            break
        old = value
    return r, dict(iterations=iteration+1, svd_objective=float(value))


def scaled_jacobian(j):
    # j: subjects x common-X/features x K, exactly the historical feature scaling.
    rms = np.sqrt(np.mean(j*j, axis=(0, 2)))
    keep = rms > max(1e-12, 1e-6*np.max(rms))
    if not keep.any():
        raise ValueError('All fingerprint features are inactive')
    return (j[:, keep]/rms[keep][None, :, None]).reshape(-1,j.shape[-1]), rms, keep


def jvol_score(w, rows, p=16):
    _, logdet = torch.linalg.slogdet(w)
    l1 = (rows @ w).abs().sum(-1).clamp_min(1e-12)
    cp = (torch.logsumexp(p*torch.log(l1), dim=0)-np.log(len(rows)))/p
    return logdet-w.shape[0]*cp


def jvolmax(rows, cfg):
    rows = torch.tensor(rows, dtype=torch.float64)
    k = rows.shape[1]
    rng = np.random.default_rng(cfg['seed']+7001)
    best = -float('inf')
    best_w = None
    records, matrices = [], []
    for start in range(cfg['starts']):
        initial = np.eye(k) if start == 0 else np.linalg.qr(rng.normal(size=(k,k)))[0]
        w = torch.nn.Parameter(torch.tensor(initial))
        optimizer = torch.optim.Adam([w], lr=cfg['jvol_lr'])
        local_best, local_w, wait = -float('inf'), None, 0
        for step in range(cfg['jvol_steps']):
            optimizer.zero_grad()
            score = jvol_score(w, rows, cfg['jvol_p'])
            if not torch.isfinite(score):
                break
            (-score).backward()
            torch.nn.utils.clip_grad_norm_([w], 10.)
            optimizer.step()
            with torch.no_grad():
                sign, ld = torch.linalg.slogdet(w)
                if sign == 0 or not torch.isfinite(ld):
                    break
                w.mul_(torch.exp(-ld/k))
                condition = float(torch.linalg.cond(w))
                if condition > cfg['jvol_condition_limit']:
                    break
                value = float(jvol_score(w, rows, cfg['jvol_p']))
                if value > local_best+1e-8:
                    local_best, local_w, wait = value, w.detach().numpy().copy(), 0
                else:
                    wait += 1
            if wait >= 500:
                break
        if local_w is None:
            raise RuntimeError('JVolmax failed to produce a finite invertible transform')
        records.append(dict(start=start, score=local_best, steps=step+1,
                            condition=float(np.linalg.cond(local_w))))
        matrices.append(local_w)
        if local_best > best:
            best, best_w = local_best, local_w
    # Historical u(row)=q(row) @ inv(W); our column convention u=Aq.
    a = np.linalg.inv(best_w).T
    return a, dict(score=best, starts=records, oblique=True,
                   condition=float(np.linalg.cond(a))), np.stack(matrices)
