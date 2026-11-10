"""Gate 2 independent of WY fit: validate oracle projection and landscape first."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from compare import angle_error
from scan import ROTATION_MATRIX_TYPE


def evaluate(fit_folder,output,min_gain=.03,min_r2=.90,max_angle_error=10.):
    d=Path(fit_folder)
    meta=json.loads((d/'fit.json').read_text(encoding='utf8'))
    curve=json.loads((d/'scan'/'rotation_curve.json').read_text(encoding='utf8'))
    if curve.get('rotation_matrix_type')!=ROTATION_MATRIX_TYPE:
        raise ValueError('Old/nonorthogonal rotation curve: rerun scan.py with corrected code')
    if meta['source']!='oracle' or curve['source']!='oracle':
        raise ValueError('Gate 2 must use independently projected true index')
    x=np.asarray([r['angle_deg'] for r in curve['records']])
    report={'model':meta['model'],'oracle_projection_test_r2':meta['test_r2'],
            'min_projection_r2':min_r2,'min_complexity_gain':min_gain,
            'max_angle_error':max_angle_error,'scores':{}}
    for k in ('normalized_l1','effective_partner_sum'):
        y=np.asarray([r[k] for r in curve['records']])
        best=float(x[np.argmin(y)])
        if meta['model'] in ('C1','C2','C3'):
            score0=float(y[np.argmin(abs(x))]);score45=float(y[np.argmin(abs(x-45))]);
            gain=(score45-score0)/max(abs(score45),1e-12)
            angle_ok=angle_error(best,0)<=max_angle_error
            target=0.
        elif meta['model']=='CR3':
            gain=float((np.max(y)-np.min(y))/max(np.max(abs(y)),1e-12))
            angle_ok=True;target=None
        else:
            raise ValueError('Predeclared gates apply only to C1/C2/C3/CR3')
        ok=bool(meta['test_r2']>=min_r2 and gain>=min_gain and angle_ok)
        report['scores'][k]={'gate2':'PASS' if ok else 'FAIL',
                             'optimal_angle':best,'target_angle':target,
                             'oracle_gain':float(gain),
                             'sieve_adequate':bool(meta['test_r2']>=min_r2),
                             'axes_correct_if_identifiable':bool(angle_ok)}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'gate2.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--oracle-fit',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();evaluate(a.oracle_fit,a.output)
