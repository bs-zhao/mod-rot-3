"""One-pass structured transformation-model rotation screen, 14.

The oracle CLI reads true generating function ONLY in oracle mode.
The WY CLI never imports a generator or oracle and reads only train/val/test
W,Y and optional independently fitted binary-choice probability surrogate.

Every angle is refitted with the SAME prespecified sparse basis and penalty,
not tuned to one favorable angle. The unknown link is learned ONCE using the
full (rotation-equivariant) sieve, prior to any angle scan.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from scipy.special import expit
from basis import Dictionary, givens, complexity
from link import learn_monotone_transform, apply

ROOT = Path(__file__).resolve().parents[2]
REV = 'orthogonal_givens_v3_profiled_monotone_sieve_v1'


def _load_splits(data):
    datasets = {}
    for name in ('train', 'val', 'test'):
        with np.load(data / (name + '.npz'), allow_pickle=False) as d:
            datasets[name] = (np.asarray(d['W'], float), np.asarray(d['y'], float).ravel())
    return datasets


def get_targets(data, meta, splits, mode, source_fit=None):
    if mode == 'oracle':
        if meta['latent_source'] != 'oracle_coordinates':
            raise ValueError('oracle requires true parameter coordinates; inferred Z has no analytic oracle')
        # Import the generator only inside this explicitly ORACLE-only branch.
        sys.path.insert(0, str(ROOT / 'math' / '11_wy_nonlinear_interactions'))
        from oracle import logits
        return {k: np.asarray(logits(meta['model'], W, meta['z_dim']), float) for k, (W, _) in splits.items()}, None
    if mode != 'wy':
        raise ValueError(mode)
    if meta['label'] == 'bernoulli':
        if not source_fit:
            raise ValueError('binary choice labels require --source-fit (train-only fitted probability model)')
        sys.path.insert(0, str(ROOT / 'math' / '11_wy_nonlinear_interactions'))
        from models import load_fit
        f, info = load_fit(source_fit)
        if (info.get('model'), info.get('label')) != (meta['model'], meta['label']):
            raise ValueError('source-fit model/label mismatch')
        # Guard against use of a predictor trained on a different data folder.
        if Path(info['data_dir']).resolve() != data.resolve():
            raise ValueError('source-fit must be trained on this exact dataset')
        if info.get('method') == 'ebm':
            return {k: f.predict_proba(W)[:, 1] for k, (W, _) in splits.items()}, 'choice_surrogate'
        return {k: np.asarray(f.predict(W), float) for k, (W, _) in splits.items()}, 'choice_surrogate'
    if source_fit:
        raise ValueError('continuous Y is already observed; do not provide --source-fit')
    return {k: y.copy() for k, (W, y) in splits.items()}, 'continuous_observed_y'


def _clip_sample(W, y, max_rows, seed):
    if max_rows >= len(W): return W, y
    idx = np.random.default_rng(seed).choice(len(W), max_rows, replace=False)
    return W[idx], y[idx]


def _fit_rotation(dic, Wtrain, Wval, Wtest, ytrain, yval, ytest,
                  angle, axis_i, axis_j, alpha, l1_ratio, seed):
    R = givens(dic.zdim, axis_i, axis_j, angle)
    A, (p, q) = dic.features(Wtrain, R)
    B, _ = dic.features(Wval, R)
    C, _ = dic.features(Wtest, R)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                       max_iter=3000, tol=1e-4, random_state=seed, selection='cyclic')
    model.fit(A, ytrain)
    pred_val = model.predict(B); pred_test = model.predict(C)
    interaction = model.coef_[-p*q:]
    strength = dic.groups_from_interactions(interaction)
    score = complexity(strength)
    return {
        'angle_deg': float(angle),
        'train_target_r2': float(r2_score(ytrain, model.predict(A))),
        'val_target_r2': float(r2_score(yval, pred_val)),
        'test_target_r2': float(r2_score(ytest, pred_test)),
        'nonzero_coefficients': int(np.count_nonzero(abs(model.coef_) > 1e-7)),
        'converged': bool(model.n_iter_ < model.max_iter),
        'axis_pair': [axis_i, axis_j],
        **score
    }


def run(data, output, mode, axis_i, axis_j, step=15., zdegree=1, xdegree=2,
        alpha=.002, l1_ratio=.95, max_train=3000, seed=31, source_fit=None,
        n_link_segments=10):
    data=Path(data).resolve(); out=Path(output)
    meta=json.loads((data/'meta.json').read_text(encoding='utf8'))
    if not 0 <= axis_i < meta['z_dim'] or not 0 <= axis_j < meta['z_dim'] or axis_i==axis_j:
        raise ValueError('invalid Z axis pair')
    if not (0 < step <= 90 and abs(90/step-round(90/step)) < 1e-8):
        raise ValueError('step must evenly divide 90')
    if alpha <= 0 or not (0 < l1_ratio <= 1) or max_train < 100:
        raise ValueError('invalid preregistered fitting settings')
    splits = _load_splits(data)
    responses, response_source = get_targets(data, meta, splits, mode, source_fit)
    (Wa, _), (Wv, _), (Wt, _) = (splits[k] for k in ('train','val','test'))
    ya, yv, yt = (responses[k] for k in ('train','val','test'))
    Wa, ya = _clip_sample(Wa, ya, max_train, seed)
    if min(len(ya),len(yv),len(yt)) < 50: raise ValueError('at least 50 rows per split required')
    dic = Dictionary.fit(Wa, meta['z_dim'], zdegree, xdegree)
    # Link profiled against the EXACT full tensor column space. Polynomial
    # span is preserved by Givens rotation, so this is not angle target tuning.
    if mode == 'wy':
        F, _ = dic.features(Wa)
        link = learn_monotone_transform(ya, F, n_segments=n_link_segments)
        ttrain=apply(ya,link);tval=apply(yv,link);ttest=apply(yt,link)
        target='estimated_inverse_monotone_y'
    else:
        link=None;ttrain=ya;tval=yv;ttest=yt
        target='true_internal_logit_oracle_only'
    angles=np.linspace(-90,90,2*int(round(90/step))+1)
    records=[]
    for angle in angles:
        record=_fit_rotation(dic,Wa,Wv,Wt,ttrain,tval,ttest,float(angle),axis_i,axis_j,alpha,l1_ratio,seed)
        records.append(record)
        print(f'[{meta["model"]}/{mode}] angle={angle:+.0f} L21={record["module_l21"]} '
              f'valR2={record["val_target_r2"]:.4f} testR2={record["test_target_r2"]:.4f}',flush=True)
    # Score rotation candidates at comparable validation fidelity. No oracle
    # input and NO best-value hunting on test performance.
    best_val=max(r['val_target_r2'] for r in records)
    for r in records:
        r['eligible'] = bool(r['val_target_r2']>=best_val-.02 and r['converged'] and r['module_l21'] is not None)
    valid=[r for r in records if r['eligible']]
    minimum=min(valid,key=lambda r:r['module_l21']) if valid else None
    out.mkdir(parents=True,exist_ok=True)
    result={
        'experiment':'14_unknown_link_module_decomposition',
        'rotation_matrix_type':REV, 'model':meta['model'],
        'source':mode,'response_source':response_source,
        'label':meta['label'],'latent_source':meta['latent_source'],
        'target':target,'axis_pair':[axis_i,axis_j],'step':step,
        'zdegree':zdegree,'xdegree':xdegree,'alpha':alpha,'l1_ratio':l1_ratio,
        'n_train_used':len(Wa),'n_val':len(Wv),'n_test':len(Wt),
        'link':link,'seed':seed, 'val_fidelity_tolerance':.02,
        'best_val_r2':best_val,
        'winner_angle':minimum['angle_deg'] if minimum else None,
        'winner_l21':minimum['module_l21'] if minimum else None,
        'x_groups':list(dict.fromkeys(dic.xgroup)),
        'records':records,
        'notes':('Each angle fits the same penalized model once on its own rotated Z; '
                 'link learned once before scan; valid X samples only; no parameter search.')}
    (out/'rotation_curve.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    with (out/'rotation_curve.csv').open('w',newline='',encoding='utf8') as h:
        csvout=csv.writer(h);csvout.writerow(['angle','l21','partner_count','val_r2','test_r2','eligible','converged'])
        for r in records:
            csvout.writerow([r['angle_deg'],r['module_l21'],r['module_effective_partners'],
                             r['val_target_r2'],r['test_target_r2'],r['eligible'],r['converged']])
    summary={k:result[k] for k in ('model','source','label','winner_angle','winner_l21','best_val_r2','n_train_used')}
    summary.update({'best_test_target_r2':max(r['test_target_r2'] for r in records),
                    'all_converged':all(r['converged'] for r in records),
                    'link_min_relative_slope':link['min_relative_slope'] if link else None})
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,1,figsize=(8,7),sharex=True)
    axs[0].plot(angles,[r['module_l21'] for r in records],marker='.');axs[0].set_ylabel('Module L21');axs[0].grid(alpha=.2)
    axs[1].plot(angles,[r['val_target_r2'] for r in records],marker='.');axs[1].set_ylabel('Validation target R²');axs[1].set_xlabel('Z rotation angle (degrees)');axs[1].grid(alpha=.2)
    fig.suptitle(f'{meta["model"]} / {mode} / profiled unknown link');fig.tight_layout()
    fig.savefig(out/'rotation_curves.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['oracle','wy'],required=True)
    p.add_argument('--source-fit')
    p.add_argument('--axis-i',type=int,required=True);p.add_argument('--axis-j',type=int,required=True)
    p.add_argument('--step',type=float,default=15)
    p.add_argument('--z-degree',type=int,default=1);p.add_argument('--x-degree',type=int,default=2)
    p.add_argument('--alpha',type=float,default=.002);p.add_argument('--l1-ratio',type=float,default=.95)
    p.add_argument('--max-train',type=int,default=3000);p.add_argument('--seed',type=int,default=31)
    p.add_argument('--link-segments',type=int,default=10)
    a=p.parse_args()
    run(a.data,a.output,a.mode,a.axis_i,a.axis_j,a.step,a.z_degree,a.x_degree,
        a.alpha,a.l1_ratio,a.max_train,a.seed,a.source_fit,a.link_segments)
