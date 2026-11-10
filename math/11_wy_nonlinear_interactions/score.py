"""Compare one blind interaction estimate with its same-estimand oracle estimate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def score(estimated,oracle,out,threshold=.05):
    estimated=Path(estimated);oracle=Path(oracle);out=Path(out)
    E=np.loadtxt(estimated/'interaction_matrix.csv',delimiter=',')
    T=np.loadtxt(oracle/'interaction_matrix.csv',delimiter=',')
    a=json.loads((estimated/'interaction.json').read_text())
    b=json.loads((oracle/'interaction.json').read_text())
    expected={'bernoulli':'prob','prob':'prob','softplus':'softplus','logit':'logit'}
    if a.get('model')!=b.get('model'):
        raise ValueError('Model mismatch')
    if expected.get(a.get('target_scale'))!=b.get('target_scale'):
        raise ValueError('Estimator and oracle use different output scales')
    if a['detector']!=b['detector']:
        raise ValueError('Detector estimands differ: cannot compare Hessian with Friedman/ANOVA')
    if E.shape!=T.shape:raise ValueError('Matrix shapes differ')
    mask=np.triu(np.ones_like(T,dtype=bool),1)&np.isfinite(T)&np.isfinite(E)
    e=E[mask];t=T[mask]
    if len(t)==0: raise ValueError('No valid pairs')
    # Fixed relative oracle threshold in an evaluation metric, never fed to the fitted model.
    cutoff=threshold*max(float(np.max(t)),1e-12)
    true=t>cutoff
    # Match predicted number of edges only for diagnostic ranking; not model selection.
    chosen=np.zeros(len(e),dtype=bool)
    ntrue=int(true.sum())
    if ntrue:chosen[np.argsort(e)[-ntrue:]]=True
    tp=int(np.sum(chosen&true));p=int(np.sum(chosen));r=int(np.sum(true))
    precision=tp/p if p else 1.0 if r==0 else 0.0
    recall=tp/r if r else 1.0
    result={'n_pairs':len(e),'n_oracle_edges':r,'threshold_fraction':threshold,
            'top_k_precision':precision,'top_k_recall':recall,
            'top_k_f1':2*precision*recall/max(precision+recall,1e-12),
            'spearman':float(spearmanr(t,e).statistic) if np.std(t)>1e-12 and np.std(e)>1e-12 else None,
            'rms_strength_error':float(np.sqrt(np.mean((t-e)**2))),
            'note':'Top-K uses oracle K only in scoring; not a deployable edge threshold.'}
    out.mkdir(parents=True,exist_ok=True)
    (out/'score.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--estimated',required=True);p.add_argument('--oracle',required=True)
    p.add_argument('--output',required=True);p.add_argument('--threshold',type=float,default=.05)
    a=p.parse_args();score(a.estimated,a.oracle,a.output,a.threshold)
