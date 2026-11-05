"""Plane sweeps and plots; they never alter or learn the rotation objective."""
import csv
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from .optimizer import as_rotation
from .metrics import diagnostic_metrics


def sweep(proxy,objective,best,config):
    k=proxy.k
    pairs=list(combinations(range(k),2))[:int(config['sweep_pairs'])]
    values=[]
    steps=int(config['sweep_steps'])
    if steps<3:raise ValueError('sweep_steps must be >=3')
    for i,j in pairs:
        for angle in np.linspace(-90.,90.,steps):
            theta=np.deg2rad(angle)
            B=np.eye(k)
            B[i,i]=np.cos(theta);B[j,j]=np.cos(theta)
            B[i,j]=-np.sin(theta);B[j,i]=np.sin(theta)
            for label,A in [('identity',B),('best',B@best)]:
                c=objective.components(A)
                score=objective.value(A)
                values.append(dict(pair=f'u{i+1}-u{j+1}',frame=label,
                                   angle_deg=float(angle),joint_loss=float(score),
                                   **{x:float(c[x]) for x in objective.keys}))
    return values


def save_postfit(folder,proxy,objective,best,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder)
    reports={}
    for name,R in [('identity',np.eye(proxy.k)),('best',best)]:
        r,details=diagnostic_metrics(proxy,R,config)
        reports[name]={'metrics':r,'details':details}
    (folder/'functional__20_metrics.json').write_text(json.dumps(reports,indent=2,allow_nan=False),encoding='utf8')
    rows=sweep(proxy,objective,best,config)
    if rows:
        with (folder/'functional__plane_curves.csv').open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if not config.get('make_plots',True):return
    pairs=sorted({r['pair'] for r in rows})
    if pairs:
        fig,axes=plt.subplots(len(pairs),1,figsize=(9,max(3.0,2.4*len(pairs))),squeeze=False)
        for ax,pair in zip(axes[:,0],pairs):
            for frame in ('identity','best'):
                subset=[x for x in rows if x['pair']==pair and x['frame']==frame]
                ax.plot([x['angle_deg'] for x in subset],[x['joint_loss'] for x in subset],label=frame)
            ax.set_title(pair);ax.set_xlabel('Incremental rotation (degrees)');ax.set_ylabel('Joint structural loss');ax.legend()
        fig.tight_layout();fig.savefig(folder/'functional__plane_curves.png',dpi=170);plt.close(fig)
    before=reports['identity']['metrics'];after=reports['best']['metrics']
    fig,ax=plt.subplots(figsize=(13,5))
    keys=[x for x in before if before[x] is not None and after[x] is not None]
    positions=np.arange(len(keys))
    # Relative change is displayed instead of mixing arbitrary metric units.
    vals=np.asarray([(after[x]-before[x])/max(abs(before[x]),1e-8) for x in keys])
    ax.bar(positions,vals)
    ax.set_xticks(positions,[x.split('_')[0] for x in keys])
    ax.set_ylabel('Relative change vs identity')
    ax.set_xlabel('Metric ID (see functional__20_metrics.json)')
    ax.axhline(0,color='black',lw=.8)
    ax.set_title('20 independent diagnostics; not all are rotation targets')
    fig.tight_layout();fig.savefig(folder/'functional__metric_changes.png',dpi=170);plt.close(fig)