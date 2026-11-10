"""Strict, predeclared Gate 2 / Gate 3 outcomes for standalone model 14."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scan import REV


def angle_distance(a, b):
    return abs(((float(a)-float(b)+45.) % 90.) - 45.)


def evaluate(oracle, output, estimated=None, min_r2=.90, min_gain=.03,
             min_corr=.90, max_angle_error=10.):
    ora=json.loads((Path(oracle)/'rotation_curve.json').read_text(encoding='utf8'))
    if ora.get('source')!='oracle' or ora.get('rotation_matrix_type')!=REV:
        raise ValueError('Oracle must come from method 14, with corrected orthogonal rotations')
    vals=[r for r in ora['records'] if r['eligible']]
    if not vals:
        report={'model':ora['model'],'metric':'module_l21','gate2':'FAIL',
                'reason':'No converged informative Oracle orientations'}
        if estimated:report['gate3']='FAIL'
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        (output/'gate_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        print(json.dumps(report,indent=2));return report
    best=min(vals,key=lambda r:r['module_l21'])
    models=('C1','C2','C3')
    angles=np.asarray([r['angle_deg'] for r in ora['records']]);scores=np.asarray([r['module_l21'] for r in ora['records']])
    if ora['model'] in models:
        idx0=int(np.argmin(abs(angles)))
        matches=np.flatnonzero(np.isclose(angles,45.,atol=1e-10))
        if len(matches)!=1:raise ValueError("Oracle C1/C2/C3 Gate 2 requires a 45-degree control; use --step 15 or 5")
        idx45=int(matches[0])
        gain=(scores[idx45]-scores[idx0])/max(abs(scores[idx45]),1e-12)
        axis_error=angle_distance(best['angle_deg'],0)
        axis_ok=axis_error <= max_angle_error
        target=0.
    elif ora['model']=='CR3':
        gain=(max(r['module_l21'] for r in vals)-best['module_l21'])/max(max(r['module_l21'] for r in vals),1e-12)
        axis_error=None;axis_ok=True;target=None
    else:
        raise ValueError('Preregistered gates support C1/C2/C3/CR3')
    all_oracle_converged=all(r['converged'] for r in ora['records'])
    oracle_r2=float(best['test_target_r2'])
    gate2=bool(oracle_r2>=min_r2 and gain>=min_gain and axis_ok and all_oracle_converged)
    report={'model':ora['model'],'metric':'module_l21',
            'gate2':'PASS' if gate2 else 'FAIL',
            'oracle_angle':best['angle_deg'],'original_axis_target':target,
            'oracle_axis_error':axis_error,'oracle_gain':float(gain),
            'oracle_winner_test_r2':oracle_r2,
            'oracle_converged':all_oracle_converged,
            'prespecified_thresholds':{'oracle_r2':min_r2,'oracle_gain':min_gain,
                                       'curve_pearson':min_corr,'angle_error':max_angle_error},
            'note':('CR3 uses informativeness and oracle-vs-WY recovery; it does not '
                    'claim 0 degrees is a uniquely true rotation.')}
    if estimated:
        est=json.loads((Path(estimated)/'rotation_curve.json').read_text(encoding='utf8'))
        keys=['rotation_matrix_type','model','label','latent_source','axis_pair',
              'step','zdegree','xdegree','alpha','l1_ratio','seed','val_fidelity_tolerance']
        for key in keys:
            if ora.get(key)!=est.get(key):raise ValueError(f'Mismatched scans: {key}')
        if est.get('source')!='wy' or ora['source']!='oracle':raise ValueError('Expected WY and Oracle')
        if [r['angle_deg'] for r in est['records']] != [r['angle_deg'] for r in ora['records']]:
            raise ValueError('Rotation grid mismatch')
        ev=[r for r in est['records'] if r['eligible']]
        if not ev:
            report.update({'gate3':'FAIL','reason':'No converged informative WY orientations'})
            output=Path(output);output.mkdir(parents=True,exist_ok=True)
            (output/'gate_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
            print(json.dumps(report,indent=2));return report
        ebest=min(ev,key=lambda r:r['module_l21'])
        oo=np.asarray([r['module_l21'] for r in ora['records']]);ee=np.asarray([r['module_l21'] for r in est['records']])
        corr=float(pearsonr(oo,ee).statistic) if (np.ptp(oo)>1e-9 and np.ptp(ee)>1e-9) else None
        rank_corr=float(spearmanr(oo,ee).statistic) if (np.ptp(oo)>1e-9 and np.ptp(ee)>1e-9) else None
        err=angle_distance(ebest['angle_deg'],best['angle_deg'])
        report.update({'gate3':'PASS' if (gate2 and corr is not None and corr>=min_corr and err<=max_angle_error and all(r['converged'] for r in est['records'])) else 'FAIL',
                       'wy_angle':ebest['angle_deg'],'angle_error':err,'curve_pearson':corr,'curve_spearman_diagnostic':rank_corr,
                       'wy_winner_test_target_r2':ebest['test_target_r2'],
                       'wy_all_converged':all(r['converged'] for r in est['records'])})
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'gate_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--oracle',required=True)
    p.add_argument('--estimated');p.add_argument('--output',required=True)
    a=p.parse_args();evaluate(a.oracle,a.output,a.estimated)
