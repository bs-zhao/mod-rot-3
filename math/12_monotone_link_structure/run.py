"""Run ONE fixed-predictor or analytic-oracle screen; no oracle leakage into blind fits."""
from __future__ import annotations
import argparse, json, sys, csv
from pathlib import Path
import numpy as np
from core import (orthogonal_plane, transform, inverse, gradient_direction_strength,
                  ordinal_contrast_strength, complexity)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'math'/'11_wy_nonlinear_interactions'


def load_function(data: Path, fit: str | None, oracle_scale: str | None,
                  meta: dict):
    if bool(fit) == bool(oracle_scale):
        raise ValueError('Specify exactly one of --fit and --oracle-scale')
    sys.path.insert(0, str(OLD))
    if oracle_scale is not None:
        if meta.get('latent_source') != 'oracle_coordinates':
            raise ValueError('Oracle not valid for inferred NP Z')
        from oracle import oracle_predict
        return oracle_predict(meta['model'],meta['z_dim'],oracle_scale), 'oracle',oracle_scale
    from models import load_fit
    model, info = load_fit(fit)
    if info.get('model') != meta['model']:
        raise ValueError('Fit and data model mismatch')
    label = meta['label']
    if info.get('label') != label:
        raise ValueError('Fit and data label mismatch')
    if info['method']=='ebm' and label=='bernoulli':
        return lambda W: model.predict_proba(W)[:,1], info['method'], 'prob'
    return model.predict, info['method'], ('prob' if label=='bernoulli' else label)


def run(data, output, method, fit=None, oracle_scale=None,
        axis_i=0, axis_j=1, step=5, n_eval=128, pairs=96,
        z_delta=.18, x_delta=.01, seed=42, standardize=False):
    data=Path(data); output=Path(output); output.mkdir(parents=True, exist_ok=True)
    meta=json.loads((data/'meta.json').read_text(encoding='utf8'))
    model=meta['model']; zdim=int(meta['z_dim'])
    if not (0 <= axis_i < zdim and 0 <= axis_j < zdim and axis_i!=axis_j):
        raise ValueError('Invalid latent rotation axes')
    if method == 'direction' and model=='CR3':
        raise ValueError('CR3 has discrete lottery X; use --method ordinal')
    if step<=0 or 90%step>1e-7:
        raise ValueError('Use a positive step dividing 90 degrees, e.g. 5 or 10')
    if n_eval < 12 or pairs < 12 or z_delta <= 0 or x_delta <= 0:
        raise ValueError('Invalid sample size or difference step')
    pred,source,scale=load_function(data,fit,oracle_scale,meta)
    with np.load(data/'test.npz',allow_pickle=False) as f:
        W=np.asarray(f['W'],dtype=float)
    if len(W)<12:raise ValueError('Insufficient evaluation observations')
    rng=np.random.default_rng(seed)
    # Fixed anchors, same sample and valid stimulus pair selection across angles.
    idx=rng.choice(len(W), min(n_eval,len(W)),replace=False)
    anchors=W[idx]; ztest=anchors[:,:zdim]
    X= W[:,zdim:]
    if method=='ordinal':
        # Every stimulus used is an actual observed vector, never perturbed in X.
        pool=np.unique(X,axis=0)
        if len(pool)<2:raise ValueError('Not enough distinct valid stimuli')
        i=rng.integers(len(pool),size=pairs*len(ztest))
        j=rng.integers(len(pool)-1,size=len(i))
        j=j+(j>=i)
        ZA=np.repeat(ztest,pairs,axis=0)
        XA=pool[i];XB=pool[j]
    std=np.maximum(W[:,:zdim].std(axis=0),1e-6)
    mean=W[:,:zdim].mean(axis=0)
    sd=std if standardize else np.ones_like(std)
    mu=mean if standardize else np.zeros_like(mean)
    nsteps=int(round(90/step)); angles=np.linspace(-90,90,2*nsteps+1)
    records=[]
    for angle in angles:
        R=orthogonal_plane(zdim,axis_i,axis_j,float(angle))
        def rotated_predict(u):return pred(inverse(u,R,zdim,mu,sd))
        U=transform(anchors,R,zdim,mu,sd)
        step_z=np.maximum(z_delta*U[:,:zdim].std(axis=0),1e-5)
        if method=='direction':
            step_x=np.maximum(x_delta*W[:,zdim:].std(axis=0),1e-5)
            strengths, quality=gradient_direction_strength(rotated_predict,U,zdim,step_x,step_z)
        else:
            # Repeated Z rows and matched valid stimuli; no X synthetic combos.
            uq=transform(np.c_[ZA,XA],R,zdim,mu,sd)[:,:zdim]
            strengths,quality=ordinal_contrast_strength(rotated_predict,uq,XA,XB,step_z)
        c=complexity(strengths)
        records.append({'angle_deg':float(angle), 'strengths':strengths.tolist(),
                        **c, **quality})
        print(f'[{method}/{source}] angle={angle:+.0f}  '
              f'effective_axes={c["effective_axes"]}  L1={c["l1"]:.6g}',flush=True)
    result={'experiment':'12_monotone_link_structure', 'model':model,
            'label':meta['label'],'function_scale':scale, 'source':source,
            'method':method,'axis_pair':[axis_i,axis_j],
            'seed':seed,'n_test_rows':int(len(W)),'n_anchor_rows':int(len(anchors)),
            'n_stimulus_pairs':int(pairs) if method=='ordinal' else None,
            'z_delta_fraction':z_delta,'x_delta_fraction':x_delta,
            'standardized_latents':standardize,
            'note': ('These are fixed learned-function scores, not independently retrained'
                     ' regression models at each rotation. Oracle is never used by blind method.'),
            'records':records}
    (output/'rotation_curve.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    with (output/'rotation_curve.csv').open('w',newline='',encoding='utf8') as fh:
        writer=csv.writer(fh)
        writer.writerow(['angle_deg','effective_axes','l1','valid','strengths'])
        for r in records:writer.writerow([r['angle_deg'],r['effective_axes'],r['l1'],
                                          r['score_valid'],';'.join(str(v) for v in r['strengths'])])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name in ('effective_axes','l1'):
        y=[r[name] if r['score_valid'] else np.nan for r in records]
        fig,ax=plt.subplots(figsize=(8,4))
        ax.plot(angles,y,marker='.',lw=1.6)
        ax.set(xlabel='Latent plane rotation (degrees)',ylabel=name,
               title=f'{model} | {source} | {method} | {name}')
        ax.grid(alpha=.3)
        fig.tight_layout();fig.savefig(output/f'{name}.png',dpi=150);plt.close(fig)
    good=[r for r in records if r['score_valid']]
    summary={'model':model,'method':method,'source':source,'output_scale':scale,
             'n_angles':len(records), 'valid_angles':len(good)}
    for key in ('effective_axes','l1'):
        valid=[r for r in good if r[key] is not None]
        if valid:
            best=min(valid,key=lambda r:r[key]);summary[key+'_min_angle']=best['angle_deg']
            summary[key+'_min_value']=best[key]
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--fit');p.add_argument('--oracle-scale',choices=['prob','softplus','logit'])
    p.add_argument('--method',choices=['direction','ordinal'],required=True)
    p.add_argument('--axis-i',type=int,default=0);p.add_argument('--axis-j',type=int,default=1)
    p.add_argument('--step',type=float,default=5)
    p.add_argument('--n-eval',type=int,default=128);p.add_argument('--pairs',type=int,default=96)
    p.add_argument('--z-delta',type=float,default=.18);p.add_argument('--x-delta',type=float,default=.01)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--standardize-z',action='store_true')
    a=p.parse_args()
    run(a.data,a.output,a.method,a.fit,a.oracle_scale,a.axis_i,a.axis_j,
        a.step,a.n_eval,a.pairs,a.z_delta,a.x_delta,a.seed,a.standardize_z)
