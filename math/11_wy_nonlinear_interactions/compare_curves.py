"""Evaluate blind rotation losses against oracle losses; no oracle access during fitting."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def compare(estimated,oracle,output,metric='zx_effective_degree'):
    e=json.loads(Path(estimated).read_text());o=json.loads(Path(oracle).read_text())
    if e['model']!=o['model'] or e['pair']!=o['pair']:
        raise ValueError('Must compare same model and rotation plane')
    expected={'bernoulli':'prob','prob':'prob','logit':'logit','softplus':'softplus'}
    if expected.get(e.get('label'))!=o.get('oracle_scale'):
        raise ValueError('Must compare matching target scales')
    em={r['rotation']:r for r in e['records']};om={r['rotation']:r for r in o['records']}
    keys=[k for k in em if k in om and em[k].get(metric) is not None and om[k].get(metric) is not None]
    if not keys:raise ValueError(f'No comparable scores for {metric}')
    x=np.array([em[k][metric] for k in keys],dtype=float)
    y=np.array([om[k][metric] for k in keys],dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):raise ValueError('Nonfinite loss')
    ai=int(np.argmin(x));bi=int(np.argmin(y))
    angle_e=em[keys[ai]]['angle_deg'];angle_o=om[keys[bi]]['angle_deg']
    result={'metric':metric,'n_rotations':len(keys),'estimated_best':keys[ai],
            'oracle_best':keys[bi],'oracle_regret':float(y[ai]-y[bi]),
            'angle_error_deg':abs(angle_e-angle_o) if angle_e is not None and angle_o is not None else None,
            'curve_spearman':float(spearmanr(x,y).statistic) if np.std(x)>1e-12 and np.std(y)>1e-12 else None}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    angle=np.array([em[k]['angle_deg'] if em[k]['angle_deg'] is not None else np.nan for k in keys])
    ok=np.isfinite(angle)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(angle[ok],x[ok],label='WY estimated');ax.plot(angle[ok],y[ok],label='Oracle')
    ax.set(xlabel='Rotation angle (deg)',ylabel=metric)
    ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/'compare.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--estimated',required=True);p.add_argument('--oracle',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--metric',default='zx_effective_degree')
    a=p.parse_args();compare(a.estimated,a.oracle,a.output,a.metric)
