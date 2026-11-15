"""Read-only integrity and reporting consistency audit; never opens raw truth."""
import json
import numpy as np
import pandas as pd
import core as c
from exp13_reproduce.run import audit

def main():
    audit();rows=[];incomplete=[]
    for p in c.HERE.glob('*/results/*/*/*/config.json'):
        d=p.parent;cp=d/'COMPLETE.json'
        if not cp.exists():incomplete.append(str(d.relative_to(c.HERE)));continue
        complete=json.loads(cp.read_text());rp=d/'rotations.npz'
        assert c.sha(rp)==complete['rotations_sha256'],str(rp)
        with np.load(rp) as z:
            best=int(z['best_index']);assert best==int(z['loss'].argmin())
            starts=len(z['R'])
        m=pd.read_csv(d/'metrics.csv');s=m[m.selected]
        assert len(m)==starts and len(s)==1 and int(s.iloc[0].start)==best,str(d)
        with np.load(d/'correlations.npz') as z:M=abs(z['signed_matched'])
        diag=M.diagonal();off=M[~np.eye(len(M),dtype=bool)]
        observed=[diag.mean(),diag.min(),off.mean(),off.max()]
        expected=s.iloc[0][['mcc','worst_matched','leakage','max_leakage']].to_numpy(float)
        np.testing.assert_allclose(observed,expected,atol=1e-12,rtol=1e-12,err_msg=str(d))
        rows.append(dict(result=str(d.relative_to(c.HERE)),starts=starts,rotation_sha256=c.sha(rp),metrics_consistent=True))
    c.dump(c.HERE/'reports/final_integrity.json',dict(passed=True,completed_candidates=len(rows),
        total_start_endpoints=sum(r['starts'] for r in rows),incomplete_candidates_retained=incomplete,
        historical_sources_unchanged=True,raw_truth_loaded_by_this_audit=False,results=rows))
    print('FINAL INTEGRITY PASS',len(rows),'candidates;',sum(r['starts'] for r in rows),'start endpoints;',len(incomplete),'incomplete retained')

if __name__=='__main__':main()
