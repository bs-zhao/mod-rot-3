"""64x64 finite-query scan with frozen 300-subject axis metric; no Oracle search."""
import itertools
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as C
from run import Finite

def main(a):
    for rep in (1,2):
        out=C.result_dir('exp15_finite_scale',a.tag,a.model,rep,a.channel)
        if (out/'COMPLETE.json').exists(): continue
        start=time.perf_counter(); c=C.load(a.model,rep,a.channel,64,64)
        full=C.load(a.model,rep,a.channel); fd=Finite(c); fd.cov=C.cov(full['Vtrain'])
        def raw(R):
            return torch.cat([fd.raw(R[i:i+a.rotation_batch]) for i in range(0,len(R),a.rotation_batch)])
        cal=C.rotations(fd.d,200,1); null=raw(cal); scale=null.std(0,unbiased=False).clamp_min(1e-9)
        random=C.rotations(fd.d,a.random_n,91); rv=raw(random)
        O=C.oracle(c); planeR=[]; plane=[]
        for i,j in itertools.combinations(range(fd.d),2):
            for angle in range(0,91,5):
                planeR.append(C.givens(fd.d,i,j,np.deg2rad(angle))@O); plane.append(dict(i=i,j=j,angle=angle))
        pv=raw(C.tensor(np.stack(planeR))); ov=fd.raw(C.tensor(O))
        methods={}
        for i,h in enumerate(fd.scales):
            w=torch.zeros(16,dtype=torch.float64); w[4*i:4*i+2]=1; methods[f'h{h}_combo_scan']=w
        w=torch.zeros(16,dtype=torch.float64); w[0::4]=.25; w[1::4]=.25; methods['multiscale_scan']=w
        rows=[]; diagnostics={}; saved={}; pcs=[]
        for name,w in methods.items():
            loss=(rv/scale*w).sum(-1).numpy(); best=int(np.argmin(loss)); R=random[best].numpy()
            saved[name+'__R']=R[None]; saved[name+'__loss']=np.array([loss[best]])
            # The chosen R is from blind Haar only. Plane candidates/Oracle NEVER enter the candidate set.
            met=C.evaluate(c,R); ol=float((ov/scale*w).sum())
            curve=(pv/scale*w).sum(-1).numpy()
            diagnostics[name]=dict(oracle_loss=ol,oracle_z=float((loss.mean()-ol)/(loss.std()+1e-12)),
                                   oracle_rank_fraction=float((loss<=ol).mean()),
                                   plane_min_delta=float(curve.min()-ol),correlation=met['correlation'],
                                   matched_correlations=met['matched_correlations'],
                                   search='lowest loss among independent Haar; no local optimization',
                                   starts=a.random_n,steps=0)
            pcs.extend({**p,'method':name,'loss':float(y)} for p,y in zip(plane,curve))
            rows.append(dict(model=a.model,repeat=rep,channel=a.channel,method=name,centers=4096,
                             metric_subjects=300,starts=a.random_n,steps=0,loss=loss[best],
                             **{k:met[k] for k in ('mcc','leakage','max_leakage')}))
        np.savez_compressed(out/'rotations.npz',**saved); np.savez_compressed(out/'fields_and_landscapes.npz',
                            haar_R=cal.numpy(),haar_raw=null.numpy(),random_R=random.numpy(),random_raw=rv.numpy(),
                            oracle_raw=ov.numpy(),plane_raw=pv.numpy(),scales=scale.numpy())
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False); pd.DataFrame(pcs).to_csv(out/'plane_curves.csv',index=False)
        C.dump(out/'diagnostics.json',diagnostics)
        C.dump(out/'config.json',dict(**vars(a),environment=C.environment(),repeat=rep,centers_subjects=c['ids'],X=c['X'],
                                    metric_subjects=full['ids'],metric_covariance=fd.cov,haar_n=200,haar_seed=1,random_seed=91,
                                    evidence='decoder-query',query_rows=fd.calls,
                                    fixed_pair=fd.pair,support_rule='diagnostic only; no R-dependent removal',
                                    caution='Scan result is not an equal-optimizer comparison to EXP13'))
        C.dump(out/'COMPLETE.json',dict(rows=len(rows),seconds=time.perf_counter()-start,query_rows=fd.calls))
        print(a.model,rep,'64x64 scan complete',diagnostics,flush=True)

if __name__=='__main__':
    p=C.parser(__doc__); p.set_defaults(tag='scan64_v1',random_n=200)
    p.add_argument('--rotation-batch',type=int,default=4); main(p.parse_args())
