"""One-shot go/no-go: oracle Gate-2 structural validity and blind Gate-3 fidelity."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from core import signed_permutation_angle_error


def go(est_path,ora_path,output,target=None,tol=10.,min_oracle_gain=.03,min_corr=.8):
    a=json.loads(Path(est_path).read_text(encoding='utf8'))
    b=json.loads(Path(ora_path).read_text(encoding='utf8'))
    for key in ('model','method','axis_pair','standardized_latents','seed','z_delta_fraction','x_delta_fraction'):
        if a[key]!=b[key]:raise ValueError(f'Mismatched experiment: {key}')
    if a['source']=='oracle' or b['source']!='oracle':raise ValueError('Estimated / oracle files reversed')
    if a['function_scale']!=b['function_scale']:
        raise ValueError('Output scale differs; link robustness is separate from estimator recovery')
    x=np.array([r['angle_deg'] for r in a['records']]);y=np.array([r['angle_deg'] for r in b['records']])
    if x.shape!=y.shape or not np.allclose(x,y):raise ValueError('Mismatched rotation grids')
    if target is None:
        target = {'toy_a':0.,'toy_b':45.}.get(a['model'])
    report={'model':a['model'],'method':a['method'],'pred_source':a['source'],
            'label':a['label'],'target_angle_mod_90':target,
            'required_oracle_gain':min_oracle_gain,'min_curve_spearman':min_corr,
            'max_angle_error':tol,'scores':{}}
    for key in ('effective_axes','l1'):
        e=np.asarray([np.nan if r[key] is None else r[key] for r in a['records']],dtype=float)
        o=np.asarray([np.nan if r[key] is None else r[key] for r in b['records']],dtype=float)
        mask=np.isfinite(e)&np.isfinite(o)
        if mask.sum()<5:
            report['scores'][key]={'gate2':'FAIL','gate3':'FAIL','reason':'fewer than 5 common finite points'}
            continue
        ix=np.flatnonzero(mask); i_est=ix[np.argmin(e[mask])];i_oracle=ix[np.argmin(o[mask])]
        control = (45. if signed_permutation_angle_error(target,0.)<1e-8 else 0.) if target is not None else None
        baseline_i=int(np.argmin(abs(x-control))) if control is not None else None
        gain=float((o[baseline_i]-o[i_oracle])/max(abs(o[baseline_i]),1e-12)) if control is not None else None
        oe=signed_permutation_angle_error(float(x[i_oracle]),target) if target is not None else None
        ee=signed_permutation_angle_error(float(x[i_est]),target) if target is not None else None
        err_between=signed_permutation_angle_error(float(x[i_est]),float(x[i_oracle]))
        corr=float(spearmanr(e[mask],o[mask]).statistic) if (np.ptp(e[mask])>1e-10 and np.ptp(o[mask])>1e-10) else None
        gate2=(oe<=tol and gain>=min_oracle_gain) if target is not None else None
        gate3=bool((gate2 is not False) and (ee is None or ee<=tol) and
                   err_between<=tol and corr is not None and corr>=min_corr)
        report['scores'][key]={
            'gate2':('PASS' if gate2 else 'FAIL') if gate2 is not None else 'N/A',
            'gate3':'PASS' if gate3 else 'FAIL',
            'oracle_min_angle':float(x[i_oracle]),'estimated_min_angle':float(x[i_est]),
            'oracle_target_equiv_angle_error':oe,
            'estimated_target_equiv_angle_error':ee,
            'estimated_vs_oracle_equiv_angle_error':err_between,
            'oracle_gain_over_control_angle':gain,
            'control_angle':control,
            'curve_spearman':corr,
            'warning':'Target angle not guaranteed to minimize all structural notions.'}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'gate_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for record,label in ((b,'oracle'),(a,'WY')):
        v=np.array([r['effective_axes'] if r['effective_axes'] is not None else np.nan for r in record['records']])
        ax.plot(x,v,label=label,marker='.',lw=1.4)
    ax.set(title=f'{a["model"]} {a["method"]}: effective dimensionality',xlabel='rotation angle',ylabel='effective axes')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--estimated',required=True);p.add_argument('--oracle',required=True)
    p.add_argument('--output',required=True);p.add_argument('--target',type=float)
    p.add_argument('--tol',type=float,default=10.)
    p.add_argument('--min-oracle-gain',type=float,default=.03)
    p.add_argument('--min-corr',type=float,default=.8)
    a=p.parse_args();go(a.estimated,a.oracle,a.output,a.target,a.tol,a.min_oracle_gain,a.min_corr)
