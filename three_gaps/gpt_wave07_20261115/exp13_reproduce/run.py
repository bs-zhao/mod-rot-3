"""Read-only archive audit and freshly recomputed EXP13 reproduction."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import numpy as np
import torch
import pandas as pd
import subprocess

def audit():
    import json
    out=c.HERE/'exp13_reproduce/results';out.mkdir(parents=True,exist_ok=True)
    manifest=c.ROOT/'three_gaps/post_exp13_20261114/baseline/results/archive_manifest.json'
    m=json.loads(manifest.read_text());rows=[]
    for f in m['files']:
        rows.append(dict(source=f['source'],archive=f['archive'],bytes=f['bytes'],
            source_ok=Path(f['source']).exists() and c.sha(f['source'])==f['sha256'],
            archive_ok=Path(f['archive']).exists() and c.sha(f['archive'])==f['sha256']))
    pd.DataFrame(rows).to_csv(out/'archive_check.csv',index=False)
    paths=['three_gaps/exp13_pure_simplicity','three_gaps/common','three_gaps/exp11_cr3_search/harness.py']
    diff=subprocess.check_output(['git','diff','exp13-frozen-doc05-20261114','--',*paths],cwd=c.ROOT,text=True)
    (out/'frozen_diff.txt').write_text(diff or 'No differences versus frozen tag.\n')
    if diff or not all(x['source_ok'] and x['archive_ok'] for x in rows): raise RuntimeError('frozen audit failed; original files not modified')
    c.dump(out/'audit.json',dict(files=len(rows),bytes=sum(x['bytes'] for x in rows),manifest_sha256=c.sha(manifest),frozen_code_unchanged=True))
    print('AUDIT PASS',len(rows),flush=True)

class HistoricalLoss(c.Loss):
    """Use the exact frozen arithmetic and effective_A, including SD on V."""
    def __init__(self,sx,t4,V):
        self.V=torch.tensor(V,dtype=torch.float32)
        super().__init__(sx,t4,c.cov(V))
    def energies(self,R):
        from common.rotlib import effective_A
        sys.path.insert(0,str(c.ROOT/'three_gaps/exp13_pure_simplicity'))
        import losses2
        batch=R.ndim==3
        if not batch: return losses2.energies(effective_A(R,self.V),self.SX,self.T4)
        e=[losses2.energies(effective_A(q,self.V),self.SX,self.T4) for q in R]
        return tuple(torch.stack([x[k] for x in e]) for k in (0,1))

def main(a):
    audit();st=c.Setting(a)
    dest=c.OUT/'exp13_reproduce'/st.key/a.tag
    sx,t4=c.stream(st,st.V[st.ids0],dest/'fresh300')
    # Only the separately invoked audit process may deserialize an old cache
    # (historical caches also contain truth). The optimizer process never does.
    old=c.ROOT/'outputs/three_gaps/exp13'/f'cache2_{a.model}_{st.cond}_{a.channel}.pt'
    import json
    if a.audit_cache_only:
        hist=torch.load(old,weights_only=False,map_location='cpu')
        assert np.array_equal(hist['ids'],st.ids0)
        assert np.array_equal(hist['X'].numpy(),st.X)
        so,to=c.moments(hist['Htt'],hist['Htx'])
        errors=dict(SX_relative=float(np.linalg.norm(sx-so)/np.linalg.norm(so)),T4_relative=float(np.linalg.norm(t4-to)/np.linalg.norm(to)),old_cache_sha256=c.sha(old))
        if max(errors['SX_relative'],errors['T4_relative'])>1e-5:raise AssertionError(errors)
        c.dump(dest/'cache_agreement.json',errors);return
    subprocess.run([sys.executable,__file__,*sys.argv[1:],'--audit-cache-only'],cwd=c.ROOT,check=True)
    errors=json.loads((dest/'cache_agreement.json').read_text())
    # Exact original float32 cast of aggregated moments, Haar, SD and sequential starts.
    ml=HistoricalLoss(sx.mean(0),t4.mean(0),st.V[st.ids0])
    small=c.HERE/'exp13_reproduce/results'/st.key/a.tag/a.run_name
    if (small/'COMPLETE.json').exists():
        if a.resume:return
        raise FileExistsError(small)
    cfg=st.config();cfg.update(historical_cache_access='separate_audit_process_only',truth_loaded_in_optimizer=False,
        C_ref=c.cov(st.V[st.ids0]),subject_ids=st.ids0.tolist(),point_count=len(st.ids0)*len(st.X),name=a.run_name)
    c.dump(small/'config.json',cfg);c.save_npz(dest/'moments.npz',SX=sx.mean(0),T4=t4.mean(0),C=c.cov(st.V[st.ids0]))
    import time
    t=time.perf_counter();R,L,Q0,trace=c.search(ml,a.starts,a.steps,serial=True)
    c.save_npz(small/'rotations.npz',R=R,loss=L,initial_R=Q0,best_index=int(L.argmin()),C_ref=c.cov(st.V[st.ids0]),trace=trace)
    # Separate fast arithmetic audit: compare batched objective at identical R.
    fast=c.Loss(sx.mean(0),t4.mean(0),c.cov(st.V[st.ids0]))
    errors.update(loss_relative=float(np.max(np.abs(fast.total(torch.tensor(R)).detach().numpy()-L))/max(abs(L.min()),1)),seconds=time.perf_counter()-t,haar_scales=ml.scales.tolist())
    c.dump(small/'diagnostics.json',errors)
    c.dump(small/'COMPLETE.json',dict(rotations_sha256=c.sha(small/'rotations.npz'),**errors))
    print('FRESH EXP13 COMPLETE',st.key,errors,flush=True)

if __name__=='__main__':
    p=c.args(__doc__);p.add_argument('--audit-cache-only',action='store_true');p.add_argument('--run-name',default='A0')
    main(p.parse_args())
