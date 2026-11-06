"""Frozen binary-decoder fields at actual X, common X and integration nodes."""
from itertools import product
import numpy as np
import torch
from .decoder import _restore


def representation(source, name, k, floor):
    s = source.copy()
    z, tr = source['z'], source['tr']
    if name == 'pca_whitened':
        w = source['W']
        mu = source['pca'].mean_
        detail = dict(kind=name, eigenvalues=source['pca'].explained_variance_.tolist())
    elif name == 'raw_top_standardized':
        idx = np.argsort(np.var(z[tr],axis=0))[::-1][:k]
        scale = np.maximum(np.std(z[tr,idx],axis=0), np.sqrt(floor))
        mu = z[tr].mean(0)
        w = np.zeros((k,z.shape[1]))
        w[np.arange(k),idx] = 1/scale
        detail = dict(kind=name, selected_indices=idx.tolist(), scale=scale.tolist())
    else:
        raise ValueError(name)
    s.update(W=w, mu=mu, B=np.linalg.pinv(w), representation=detail)
    s['q'] = (z-mu) @ w.T
    return s


def observed_rows(source, cfg, every_trial=False):
    rng = np.random.default_rng(cfg['seed']+1)
    xs, subjects, trial_ids, weights = [], [], [], []
    for i, mask in enumerate(source['mask_fit']):
        indices = np.flatnonzero(mask)
        if not every_trial:
            indices = np.sort(rng.choice(indices,min(len(indices),cfg['trials_per_subject']),replace=False))
        xs.append(source['x_fit'][i,indices])
        subjects.extend([i]*len(indices))
        trial_ids.extend(indices)
        weights.extend([1/(len(source['chosen'])*len(indices))]*len(indices))
    return np.vstack(xs), np.asarray(subjects), np.asarray(trial_ids), np.asarray(weights)


def common_design(source, cfg):
    x, mask = source['x_fit'], source['mask_fit']
    fixed = all(np.array_equal(x[0,mask[0]],x[i,mask[i]]) for i in range(1,len(x)))
    pool = x[0,mask[0]] if fixed else x[mask]
    # Preserve all joint rows: never create impossible CR3 lotteries by product grids.
    pool = np.unique(pool,axis=0)
    rng = np.random.default_rng(cfg['seed']+773)
    ids = np.sort(rng.choice(len(pool), min(cfg['common_x'],len(pool)),replace=False))
    common = pool[ids]
    if fixed:
        integration = pool
        weight = np.full(len(pool),1/len(pool))
        detail = dict(mode='exact_empirical_shared_bank', n_nodes=len(pool),
                      continuous_quadrature_applicable=False,
                      reason='Discrete shared observed design; preserve joint constraints')
    else:
        # A generic product-uniform integration control on the observed FIT bounds.
        # Only X enters; no simulator design/parameter functions are imported.
        lo, hi = pool.min(0), pool.max(0)
        active = np.flatnonzero(hi-lo>1e-10)
        nodes, w = np.polynomial.legendre.leggauss(cfg['quadrature_order'])
        combinations = list(product(range(len(nodes)),repeat=len(active)))
        integration = np.tile((lo+hi)/2,(len(combinations),1))
        weight = np.ones(len(combinations))
        for column,j in enumerate(active):
            ix = np.asarray([c[column] for c in combinations])
            integration[:,j] = (lo[j]+hi[j])/2+(hi[j]-lo[j])/2*nodes[ix]
            weight *= w[ix]/2
        detail = dict(mode='tensor_gauss_legendre_product_uniform_fit_bounds',
                      n_nodes=len(weight), bounds=np.stack([lo,hi]).tolist(),
                      order=cfg['quadrature_order'], continuous_quadrature_applicable=True,
                      assumption='Product-uniform measure is an explicit integration control, not inferred universally')
    return common.astype(float), integration.astype(float), weight/weight.sum(), detail


class DecoderField:
    def __init__(self, source, cfg):
        self.source, self.cfg = source, cfg
        self.model, self.device = _restore(source, {'teacher_device': cfg['device']})
        self.model = self.model.double()
        self.b = torch.tensor(source['B'].T,dtype=torch.float64,device=self.device)

    def values(self, z, x):
        with torch.no_grad():
            zz = torch.tensor(z,dtype=torch.float64,device=self.device)
            xx = torch.tensor(x,dtype=torch.float64,device=self.device).unsqueeze(1)
            out = self.model.decode(xx,zz)[:,0]
            return (out[:,1]-out[:,0]).cpu().numpy()

    def evaluate(self, x, subject, finite_difference=False):
        source, cfg = self.source, self.cfg
        k = source['rotation_dim']
        zs = source['z_fit'][subject]
        outputs = {'eta':[], 'p':[], 'logit':[], 'probability':[]}
        if finite_difference:
            outputs['fd_logit'] = []
        for begin in range(0,len(x),cfg['batch_points']):
            end = min(begin+cfg['batch_points'],len(x))
            z0 = torch.tensor(zs[begin:end],dtype=torch.float64,device=self.device)
            xb = torch.tensor(x[begin:end],dtype=torch.float64,device=self.device).unsqueeze(1)
            delta = torch.zeros((len(z0),k),dtype=torch.float64,device=self.device,requires_grad=True)
            logits = self.model.decode(xb,z0+delta@self.b)[:,0]
            eta = logits[:,1]-logits[:,0]
            p = torch.sigmoid(eta)
            g = torch.autograd.grad(eta.sum(),delta)[0]
            outputs['eta'].append(eta.detach().cpu().numpy())
            outputs['p'].append(p.detach().cpu().numpy())
            outputs['logit'].append(g.detach().cpu().numpy())
            outputs['probability'].append((g*(p*(1-p))[:,None]).detach().cpu().numpy())
            if finite_difference:
                z = zs[begin:end]
                xx = x[begin:end]
                step = cfg['fd_step']
                fd = np.column_stack([(self.values(z+step*source['B'][:,j],xx)-self.values(z-step*source['B'][:,j],xx))/(2*step) for j in range(k)])
                outputs['fd_logit'].append(fd)
        return {key:np.concatenate(value) for key,value in outputs.items()}

    def common(self,x,finite_difference=False):
        n = len(self.source['chosen'])
        result = self.evaluate(np.tile(x,(n,1)),np.repeat(np.arange(n),len(x)),finite_difference)
        return {key:value.reshape(n,len(x),*value.shape[1:]) for key,value in result.items()}


def fisher_rows(field, weights, floor):
    g, p = field['logit'], field['p']
    trace = p*(1-p)*np.sum(g*g,axis=-1)
    valid = trace > floor
    if not valid.any():
        raise ValueError('No informative choice Fisher cells')
    # Binary Fisher p(1-p) gg^T has rank one. After trace normalization its
    # rotated diagonal is exactly the squared projection of unit g.
    rows = g[valid]/np.linalg.norm(g[valid],axis=-1,keepdims=True)
    weight = np.broadcast_to(weights,trace.shape)[valid].copy()
    weight /= weight.sum()
    return rows, weight, dict(cells=int(trace.size), valid_cells=int(valid.sum()),
                             removed_mass=float(np.sum(np.broadcast_to(weights,trace.shape)[~valid])),
                             trace_min=float(trace.min()), trace_max=float(trace.max()))
