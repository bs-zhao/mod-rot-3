"""Independent experiment support. No imports from other research packages."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

DIMS = {'C1': 4, 'C2': 4, 'C3': 6, 'CR3': 4}

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def tag(v):
    return f'{v:g}'.replace('-', 'm').replace('.', 'p').replace('+', '')

def dump(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False), encoding='utf8')

def load(args):
    root = Path(args.repo)
    d = DIMS[args.model]
    if (not args.pca and args.zdim != d) or args.zdim < d:
        raise ValueError('Invalid original zdim / PCA geometry')
    if args.zdim > d:
        raise ValueError('Truncated PCA needs an explicit residual protocol; not supported here')
    cond = f'beta{tag(args.beta)}_lambda0_ctx0p25to0p8_L{args.zdim}'
    base = root/'outputs/choice'/args.model
    cp = base/'s3_train'/cond/'repeat01/s3_train__checkpoint.pt'
    lp = base/'s4_evaluate'/cond/'repeat01/s4_evaluate__latent.npz'
    dp = base/'s1_generate/s1_generate__dataset.npz'
    hashes = {k: sha(v) for k, v in [('checkpoint', cp), ('latent', lp), ('dataset', dp)]}
    latent = np.load(lp, allow_pickle=False)
    if str(latent['checkpoint_sha256']) != hashes['checkpoint']: raise ValueError('Checkpoint hash mismatch')
    if str(latent['dataset_sha256']) != hashes['dataset']: raise ValueError('Dataset hash mismatch')
    if str(latent['context_mode']) != 'all_observed_trials': raise ValueError('Wrong context protocol')
    splits = {k: latent[k+'_ids'].astype(int) for k in ['train', 'validation', 'test']}
    z = latent['z'].astype(float)
    if not np.array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(z))):
        raise ValueError('Splits are not disjoint/exhaustive')
    if z.shape[1] != args.zdim: raise ValueError('Latent dimension mismatch')
    checkpoint = torch.load(cp, map_location='cpu', weights_only=False)
    cfg, tc, extra = [checkpoint[k] for k in ['model_config', 'training_config', 'extra_state']]
    if cfg['latent_dim'] != args.zdim or tc['beta'] != args.beta or extra['condition'] != cond or extra['model_id'] != args.model:
        raise ValueError('Wrong checkpoint condition')
    if extra['dataset_sha256'] != hashes['dataset']: raise ValueError('Checkpoint dataset mismatch')
    if tc['lambda_sparse'] != 0 or tc['context_min_fraction'] != .25 or tc['context_max_fraction'] != .8 or extra['repeat'] != 1:
        raise ValueError('Unsupported training condition')
    for k, ids in splits.items():
        if not np.array_equal(ids, np.asarray(checkpoint['subject_splits'][k])): raise ValueError('Checkpoint split mismatch')
    mean = z[splits['train']].mean(0)
    components = np.eye(args.zdim)
    if args.pca:
        _, _, vt = np.linalg.svd(z[splits['train']]-mean, full_matrices=False)
        components = vt[:d]
    v = (z-mean)@components.T
    data = np.load(dp, allow_pickle=False)
    return dict(root=root, d=d, cond=cond, cp=checkpoint, latent_path=lp,
                data=data, z=z, v=v, mean=mean, components=components,
                splits=splits, hashes=hashes)

def formula(model, t, x):
    if model == 'C1': return t[0]+np.sqrt(3)*(t[1:]*x).sum()
    if model in ('C2', 'C3'):
        f = [x.new_tensor(1.), np.sqrt(3)*(x[0]-x[1]), np.sqrt(5)/2*3*(x[0]**2-x[1]**2), np.sqrt(3)*(x[2]-x[3])]
        if model == 'C3': f += [np.sqrt(5)/2*3*(x[2]**2-x[3]**2), 3*(x[0]*x[2]-x[1]*x[3])]
        return (t*torch.stack(f)).sum()
    a, lam, g, tau = t.unbind()
    def value(v):
        gain, loss, p = v.unbind()
        # clamp before pow avoids NaN derivatives at zero; the value stays zero.
        vg = torch.where(gain > 0, (gain.clamp_min(1e-30)/100)**a, gain.new_tensor(0.))
        vl = torch.where(loss > 0, (loss.clamp_min(1e-30)/100)**a, loss.new_tensor(0.))
        wg = torch.sigmoid(g*torch.logit(p.clamp(1e-12, 1-1e-12)))
        wl = torch.sigmoid(g*torch.logit((1-p).clamp(1e-12, 1-1e-12)))
        return wg*vg-lam*wl*vl
    return tau*(value(x[:3])-value(x[3:]))

def decoder(context):
    cfg = context['cp']['model_config']
    dims = [cfg['x_dim']+cfg['latent_dim'], *cfg['decoder_hidden_dims'], cfg['num_choices']]
    layers=[]
    for i, (a,b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(torch.nn.Linear(a,b))
        if i < len(dims)-2: layers.append(torch.nn.GELU())
    net = torch.nn.Sequential(*layers).double().eval()
    net.load_state_dict({k.removeprefix('decoder.'): v for k,v in context['cp']['model_state_dict'].items() if k.startswith('decoder.')})
    net.requires_grad_(False)
    mean = torch.tensor(context['mean'])
    comp = torch.tensor(context['components'])
    d = context['d']
    def fn(w):
        rawz = w[:d]@comp+mean
        out = net(torch.cat((w[d:], rawz)))
        return out[1]-out[0]
    return fn

def derivatives(fn, w, d, third=False, latent_only=False):
    """F1/F2 and optionally F3; derivatives are in supplied input units."""
    if latent_only:
        def fixed(z,x): return fn(torch.cat((z,x)))
        grad=torch.func.jacrev(fixed,argnums=0)
        hess=torch.func.jacfwd(grad,argnums=0)
        wt=torch.as_tensor(w,dtype=torch.float64)
        return {name:np.concatenate([torch.vmap(op)(batch[:,:d],batch[:,d:]).detach().numpy() for batch in wt.split(32)])
                for name,op in [('g',grad),('h',hess)]}
    grad = torch.func.jacrev(fn)
    hess = torch.func.jacfwd(grad)
    wt = torch.as_tensor(w, dtype=torch.float64)
    fields = {}
    for name, op in [('g', grad), ('h', hess)]+([('third', torch.func.jacfwd(hess))] if third else []):
        parts=[]
        for batch in wt.split(32): parts.append(torch.vmap(op)(batch).detach().numpy())
        fields[name]=np.concatenate(parts)
    return fields

def corr(a,b):
    a=a-a.mean(0); b=b-b.mean(0)
    return (a.T@b)/np.maximum(np.linalg.norm(a,axis=0)[:,None]*np.linalg.norm(b,axis=0)[None,:], 1e-30)

def correspondence(u,t):
    c=corr(u,t); i,j=linear_sum_assignment(-abs(c)); off=np.ones(c.shape,bool); off[i,j]=False
    return dict(signed=c, matching=j, matched_mean=abs(c[i,j]).mean(), matched_min=abs(c[i,j]).min(),
                off_mean=abs(c[off]).mean(), off_max=abs(c[off]).max())
