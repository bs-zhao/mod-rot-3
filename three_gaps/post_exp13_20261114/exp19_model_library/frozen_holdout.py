"""New formula families evaluated AFTER freezing grammar_v1 learned weights.

No fitting, early stopping or calibration of the meta-score on these families.
Only the per-function WY surrogate uses its own held-out Y, as at application.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as C
import run as L

NEW=['rational','softplus_mix','sine_product','radial_two_blocks']
old_formula=L.formula
def formula(family,t,x,k):
    a,b,c,d=t.T; x0,x1,x2,x3=x.T
    if family=='rational': return a*x0/(1+.15*b*b)+b*x1+c*x2/(1+.1*d*d)+d*x3+.1*a*d*x1
    if family=='softplus_mix': return np.logaddexp(0,a*x0+b*x1)-np.logaddexp(0,c*x2+d*x3)+.2*(a*x2+c*x0)
    if family=='sine_product': return a*np.sin(x0)+b*x1+c*x2+d*x3+.25*np.sin(a*b)*x2+.2*c*d*x0*x1
    if family=='radial_two_blocks': return (a*a+b*b)*x0+(c*c+d*d)*x1
    return old_formula(family,t,x,k)

def main(a):
    checkpoint=C.ROOT/'outputs/three_gaps/post_exp13_20261114/learned_score'/f'{a.frozen_tag}.pt'
    state=torch.load(checkpoint,weights_only=False,map_location='cpu')
    sha=C.io.sha256(checkpoint)
    net=torch.nn.Sequential(torch.nn.Linear(6,24),torch.nn.Tanh(),torch.nn.Linear(24,24),torch.nn.Tanh(),torch.nn.Linear(24,1)).double()
    net.load_state_dict(state['state']); net.eval(); net.requires_grad_(False)
    L.formula=formula; L.FAMILIES['new_test']=NEW
    out=Path(__file__).parent/'results'/a.tag; out.mkdir(parents=True,exist_ok=True)
    C.dump(out/'config.json',dict(**vars(a),environment=C.environment(),frozen_checkpoint_sha256=sha,
                                formula_families=NEW,meta_model_refit=False,
                                development_caveat='grammar_v1 test families were inspected in pilot; this extra holdout uses frozen weights and new formulas',
                                surrogate_selection='held-out Y only'))
    rows=[]; saved={}; manifests=[]
    for family in NEW:
        for i in range(a.formulas_per_family):
            item=L.build(family,i,'new_test',a); item['negative_control']=family=='radial_two_blocks'
            manifests.append({k:v for k,v in item.items() if k not in ('mm','s2','V','features','target')})
            objects={'EXP13':C.Objective(C.tensor(item['V']),lambda A,item=item:C.terms(A,item['mm'])),
                     'learned':L.Learned(item,net,state['mu'],state['sd'])}
            for name,obj in objects.items():
                ll,rr=C.optimize(obj,4,a.starts,a.steps,7)
                key=f'{family}_{i}_{name}'; saved[key+'__R']=rr; saved[key+'__loss']=ll
                rng=np.random.default_rng(1951); t=rng.normal(size=(5000,4)); v=t@item['Q'].T
                met=C.match_report(v@rr[0].T,t)
                near=rr[ll<=ll[0]+max(.01,.01*abs(ll[0]))]
                disagreement=max([L.target(r@rr[0].T,np.eye(4)) for r in near],default=0.)
                abstain=bool(item['estimated_local_jacobian_rank_ratio']<1e-3 or disagreement>.15)
                rows.append(dict(split='new_test',family=family,formula=i,method=name,mcc=met['mcc'],leakage=met['off_mean'],
                                 max_leakage=met['off_max'],axis_error=L.target(rr[0],item['Q']),loss=ll[0],
                                 validation_prediction_mse=item['validation_mse'],response_rank=item['local_rank'],
                                 negative_control=item['negative_control'],abstain=abstain,low_loss_disagreement=disagreement))
                print(rows[-1],flush=True)
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False); np.savez_compressed(out/'rotations.npz',**saved)
    C.dump(out/'model_manifest.json',manifests)
    if C.io.sha256(checkpoint)!=sha: raise RuntimeError('Frozen meta weights changed')
    C.dump(out/'COMPLETE.json',dict(rows=len(rows),frozen_checkpoint_sha256=sha))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--tag',default='frozen_holdout_v2')
    p.add_argument('--frozen-tag',default='grammar_v1'); p.add_argument('--formulas-per-family',type=int,default=3)
    p.add_argument('--fit-rows',type=int,default=4096); p.add_argument('--starts',type=int,default=24)
    p.add_argument('--steps',type=int,default=250); main(p.parse_args())
