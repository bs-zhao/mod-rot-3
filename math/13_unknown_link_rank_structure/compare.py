"""No-retuning Gate 2/3 report for unknown-link structured regression.

One-pass screening thresholds are declared in code and recorded in JSON.
CR3 has no proven true optimal angle; test ORACLE non-flatness then WY agreement.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def angle_error(a,b):
    return abs(((a-b+45.)%90.)-45.)


def compare(estimated,oracle,output,min_gain=.03,min_corr=.8,max_angle_error=10.,
            min_oracle_r2=.90,min_test_rank_accuracy=.75):
    e=json.loads(Path(estimated).read_text(encoding='utf8'))
    o=json.loads(Path(oracle).read_text(encoding='utf8'))
    if e['source']!='wy_only' or o['source']!='oracle':
        raise ValueError('First result must be WY; second must be Oracle')
    for key in ('model','label','axis_pair','step','z_degree','x_degree','x_basis_terms'):
        if e[key]!=o[key]:raise ValueError(f'Incompatible experiments: {key}')
    a=np.asarray([r['angle_deg'] for r in e['records']])
    b=np.asarray([r['angle_deg'] for r in o['records']])
    if not np.array_equal(a,b):raise ValueError('Different angle grids')
    ef=Path(estimated).parent.parent/'fit.json'
    of=Path(oracle).parent.parent/'fit.json'
    em=json.loads(ef.read_text(encoding='utf8'))
    om=json.loads(of.read_text(encoding='utf8'))
    if em['basis']['zmean']!=om['basis']['zmean'] or em['basis']['zsd']!=om['basis']['zsd']:
        raise ValueError('Oracle/WY dictionaries were standardized on different data')
    ref_r2=float(om['test_r2'])
    model=e['model']
    output_info={'model':model,'wy_method':em['method'],'wy_label_source':em['response_source'],
                 'oracle_projection_r2':ref_r2,'test_rank_accuracy_on_effective_Y':em['test_rank_accuracy'],
                 'thresholds':{'min_oracle_projection_r2':min_oracle_r2,'min_oracle_gain':min_gain,
                               'min_curve_spearman':min_corr,'max_angle_error':max_angle_error,
                               'min_rank_accuracy':min_test_rank_accuracy},'scores':{}}
    for key in ('normalized_l1','effective_partner_sum'):
        E=np.asarray([r[key] for r in e['records']],dtype=float)
        O=np.asarray([r[key] for r in o['records']],dtype=float)
        if not np.isfinite(E).all() or not np.isfinite(O).all():raise ValueError('Nonfinite curve')
        ei=int(np.argmin(E));oi=int(np.argmin(O));i0=int(np.argmin(abs(a)));i45=int(np.argmin(abs(a-45)))
        corr=float(spearmanr(E,O).statistic) if np.ptp(E)>1e-10 and np.ptp(O)>1e-10 else None
        # For C1-C3, 0 is the predefined true coefficient coordinate. CR3 is
        # not generally guaranteed to be the sparsest rotation of its nonlinear
        # CPT formula, so do NOT invent a target-angle Gate 2.
        target=0. if model in ('C1','C2','C3') else None
        if target is not None:
            control=float(O[i45]);best=float(O[i0]); gain=(control-best)/max(abs(control),1e-12)
            oracle_angle_ok=angle_error(a[oi],0.)<=max_angle_error
            estimated_angle_ok=angle_error(a[ei],0.)<=max_angle_error
        else:
            gain=float((np.max(O)-np.min(O))/max(np.max(abs(O)),1e-12))
            oracle_angle_ok=True
            estimated_angle_ok=True
        gate2=bool(ref_r2>=min_oracle_r2 and oracle_angle_ok and gain>=min_gain)
        oracle_est_error=angle_error(a[ei],a[oi])
        gate3=bool(gate2 and oracle_est_error<=max_angle_error and estimated_angle_ok and
                   corr is not None and corr>=min_corr and
                   em['test_rank_accuracy']>=min_test_rank_accuracy)
        output_info['scores'][key]={'gate2':'PASS' if gate2 else 'FAIL',
                 'gate3':'PASS' if gate3 else 'FAIL',
                 'oracle_min_angle':float(a[oi]),'estimated_min_angle':float(a[ei]),
                 'oracle_estimate_angle_error':float(oracle_est_error),
                 'target_angle':target,'oracle_improvement':float(gain),
                 'curve_spearman':corr,
                 'caveat':'CR3 target angle is undefined; Gate2 tests structure signal and oracle projection validity' if target is None else None}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'gate_report.json').write_text(json.dumps(output_info,indent=2),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key in ('normalized_l1','effective_partner_sum'):
        fig,ax=plt.subplots(figsize=(8,4))
        ax.plot(a,[r[key] for r in o['records']],label='Oracle')
        ax.plot(a,[r[key] for r in e['records']],label='WY')
        ax.set(xlabel='Z rotation angle',ylabel=key,title=f"{model} unknown-link {key}")
        ax.legend();ax.grid(alpha=.25);fig.tight_layout()
        fig.savefig(out/f'compare_{key}.png',dpi=150);plt.close(fig)
    print(json.dumps(output_info,indent=2))
    return output_info

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--estimated',required=True);p.add_argument('--oracle',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();compare(args.estimated,args.oracle,args.output)
