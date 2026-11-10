"""Build WY-only data for toy A/B and the repository's C1/C2/C3/CR3 simulations.
Truth is stored separately, outside method-facing split files. No generator import
occurs in methods.py, interactions.py or rotation.py.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]


def _toy(model: str, subjects: int, trials: int, seed: int):
    rng = np.random.default_rng(seed)
    z = np.column_stack([rng.normal(0, .8, subjects), rng.normal(0, .8, subjects),
                         rng.uniform(.5, 1.5, subjects)])
    x = rng.uniform(.3, 1.7, (subjects, trials, 2))
    zrow = np.repeat(z, trials, axis=0)
    xrow = x.reshape(-1, 2)
    z1, z2, z3 = zrow.T
    x1, x2 = xrow.T
    if model == 'toy_a':
        eta = z1 * x1**z3 + z1*z2*x2**z3
    else:
        eta = (z1+z2)*x1**z3 + (z1*z1-z2*z2)*x2**z3
    p = expit(eta)
    obs = rng.binomial(1, p).astype(float)
    return zrow, xrow, eta, obs, np.repeat(np.arange(subjects), trials)


def _repository_data(model: str, subjects: int, trials: int, seed: int, latent_file: str | None,
                     allow_leaky: bool):
    path = ROOT / 'outputs' / 'choice' / model / 's1_generate' / 's1_generate__dataset.npz'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. First run python np/s1_generate.py --model {model}')
    with np.load(path, allow_pickle=False) as d:
        sid = np.asarray(d['subject_index'], dtype=np.int64)
        params = np.asarray(d['parameters'], dtype=float)
        X = np.asarray(d['x'], dtype=float)
        obs = np.asarray(d['y'], dtype=float).reshape(-1)
        pp = np.asarray(d['predictive_true'], dtype=float).reshape(-1)
    if np.any(pp < 0) or np.any(pp > 1):
        raise ValueError('Repository predictive_true must be Bernoulli probabilities.')
    target_mask = np.ones(len(sid),dtype=bool)
    if latent_file is not None:
        with np.load(latent_file, allow_pickle=False) as f:
            if 'dataset_sha256' not in f:
                raise ValueError('Latents must include source dataset SHA256')
            import hashlib
            h=hashlib.sha256()
            with open(path,'rb') as handle:
                for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
            if str(f['dataset_sha256'])!=h.hexdigest():
                raise ValueError('Latents belong to a different dataset file')
            if 'z' not in f or 'context_mode' not in f:
                raise ValueError('Latent file needs z and context_mode fields')
            mode = str(f['context_mode'])
            if mode != 'context_only_heldout_targets' and not allow_leaky:
                raise ValueError('All-trial Z leaks target labels: provide context-only latents, '
                                 'or --allow-leaky-latents for an explicitly labeled diagnostic.')
            if 'target_trial_mask' in f:
                target_mask = np.asarray(f['target_trial_mask'], dtype=bool).reshape(-1)
                if len(target_mask) != len(sid):
                    raise ValueError('target_trial_mask must align with original generated row order')
            elif not allow_leaky:
                raise ValueError('Context-only latents must provide target_trial_mask to prevent target leakage')
            params = np.asarray(f['z'], dtype=float)
            if len(params) != len(np.unique(sid)):
                raise ValueError('Latent subject count does not match dataset')
    rng = np.random.default_rng(seed)
    n = min(subjects, len(params))
    chosen = rng.choice(len(params), n, replace=False)
    # Repository files are contiguous subject blocks, with variable length for C-series.
    order = np.argsort(sid, kind='stable')
    counts = np.bincount(sid, minlength=len(params))
    offsets = np.r_[0, counts.cumsum()]
    rows = []
    for s in chosen:
        possible = order[offsets[s]:offsets[s+1]]
        possible = possible[target_mask[possible]]
        if len(possible)==0:raise ValueError(f'No eligible held-out targets for subject {s}')
        rows.append(rng.choice(possible, min(trials, len(possible)), replace=False))
    rows = np.concatenate(rows)
    return params[sid[rows]], X[rows], np.log(np.clip(pp[rows],1e-7,1-1e-7) /
                                           (1-np.clip(pp[rows],1e-7,1-1e-7))), obs[rows], sid[rows]


def build(model: str, output: str, n_subjects: int=180, n_trials: int=30,
          label: str='bernoulli', seed: int=31, latent_file: str | None=None,
          allow_leaky_latents: bool=False):
    if model not in {'toy_a','toy_b','C1','C2','C3','CR3'}:
        raise ValueError(model)
    if label not in {'logit','prob','softplus','bernoulli'}:
        raise ValueError(label)
    if model.startswith('toy_'):
        Z,X,eta,obs,groups = _toy(model,n_subjects,n_trials,seed)
    else:
        Z,X,eta,obs,groups = _repository_data(model,n_subjects,n_trials,seed,latent_file,allow_leaky_latents)
    if label == 'logit':
        y = eta
    elif label == 'prob':
        y = expit(eta)
    elif label == 'softplus':
        y = np.logaddexp(0,eta)
    else:
        y = obs
    W = np.c_[Z,X]
    unique = np.unique(groups)
    rng = np.random.default_rng(seed+713)
    rng.shuffle(unique)
    a = int(.6*len(unique)); b = int(.8*len(unique))
    if a < 1 or b <= a or b >= len(unique):
        raise ValueError('Need at least 6 subjects to form nonempty splits')
    splits = {'train':unique[:a], 'val':unique[a:b], 'test':unique[b:]}
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    for split,ids in splits.items():
        mask = np.isin(groups, ids)
        np.savez_compressed(out / f'{split}.npz', W=W[mask].astype('float32'),
                            y=y[mask].astype('float32'), groups=groups[mask])
    test = np.isin(groups,splits['test'])
    np.savez_compressed(out/'oracle_reference.npz', W=W[test], eta=eta[test],
                        probability=expit(eta[test]), groups=groups[test],
                        z_dim=Z.shape[1], x_dim=X.shape[1])
    meta = {'model':model,'label':label,'seed':seed, 'z_dim':Z.shape[1],
            'x_dim':X.shape[1], 'n_subjects':len(unique), 'n_trials_total':len(y),
            'train_subjects':len(splits['train']), 'val_subjects':len(splits['val']),
            'test_subjects':len(splits['test']), 'latent_source':latent_file or 'oracle_coordinates',
            'leaky_context_diagnostic':bool(latent_file and allow_leaky_latents)}
    (out/'meta.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    return meta


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['toy_a','toy_b','C1','C2','C3','CR3'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--label',choices=['logit','prob','softplus','bernoulli'],default='bernoulli')
    p.add_argument('--subjects',type=int,default=180)
    p.add_argument('--trials',type=int,default=30)
    p.add_argument('--seed',type=int,default=31)
    p.add_argument('--latent-file')
    p.add_argument('--allow-leaky-latents',action='store_true')
    a=p.parse_args()
    print(json.dumps(build(a.model,a.output,a.subjects,a.trials,a.label,a.seed,
                           a.latent_file,a.allow_leaky_latents),indent=2))
