"""Oracle-only projection of the TRUE internal index onto the same fixed basis.

This benchmark is intentionally approximate for nonlinear CR3: project the
analytic logit onto a prespecified polynomial sieve, record its held-out R2,
and refuse to label a bad projection as identifiable ground truth.
Never import this module into fit.py or scan.py.
"""
from __future__ import annotations
import argparse,json,importlib.util,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_squared_error
from basis import Basis
ROOT=Path(__file__).resolve().parents[2]


def build(data,output,seed=31,max_train=9000,z_degree=None,x_degree=2):
    d=Path(data);out=Path(output)
    meta=json.loads((d/'meta.json').read_text(encoding='utf8'))
    if meta.get('latent_source')!='oracle_coordinates':
        raise ValueError('True-parameter analytic oracle invalid for inferred NP latent Z')
    if z_degree is None:z_degree=2 if meta['model']=='CR3' else 1
    old_oracle=ROOT/'math'/'11_wy_nonlinear_interactions'/'oracle.py'
    sys.path.insert(0,str(old_oracle.parent))
    spec=importlib.util.spec_from_file_location('wy11_isolated_oracle',old_oracle)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logits=module.logits
    arrays=[]
    for name in ('train','val','test'):
        with np.load(d/(name+'.npz'),allow_pickle=False) as f:
            arrays.append(np.asarray(f['W'],dtype=float))
    tr,val,test=arrays
    if max_train>0 and len(tr)>max_train:
        idx=np.random.default_rng(seed+800).choice(len(tr),max_train,replace=False)
        tr=tr[idx]
    zdim=int(meta['z_dim'])
    basis=Basis(z_degree,x_degree).fit(tr,zdim)
    ytr=np.asarray(logits(meta['model'],tr,zdim),dtype=float).reshape(-1)
    yval=np.asarray(logits(meta['model'],val,zdim),dtype=float).reshape(-1)
    ytest=np.asarray(logits(meta['model'],test,zdim),dtype=float).reshape(-1)
    # The CR3 analytic routine must correspond to the actual generated bank.
    ref=d/'oracle_reference.npz'
    if ref.exists():
        with np.load(ref,allow_pickle=False) as f:
            truth=np.asarray(f['eta'],dtype=float).reshape(-1)
            w=np.asarray(f['W'],dtype=float)
        sample=np.linspace(0,len(w)-1,min(200,len(w)),dtype=int)
        error=float(np.max(np.abs(logits(meta['model'],w[sample],zdim)-truth[sample])))
        if error>1e-4:
            raise ValueError(f'Analytic generator and dataset disagree: max eta error={error:.6g}')
    a=basis.transform(tr);v=basis.transform(val);t=basis.transform(test)
    reg=Ridge(alpha=1e-5,solver='lsqr',tol=1e-9,max_iter=6000)
    reg.fit(a,ytr)
    score={'train_r2':float(r2_score(ytr,reg.predict(a))),
           'val_r2':float(r2_score(yval,reg.predict(v))),
           'test_r2':float(r2_score(ytest,reg.predict(t))),
           'test_rmse':float(np.sqrt(mean_squared_error(ytest,reg.predict(t))))}
    B=basis.coefficient_matrix(reg.coef_)
    anchor_idx=np.random.default_rng(4004).choice(len(test),size=min(200,len(test)),replace=False)
    anchors=test[anchor_idx,:zdim]
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'coefficients.npz',coef_scaled=reg.coef_,B=B,anchors=anchors)
    np.savetxt(out/'zx_coefficients.csv',B,delimiter=',',fmt='%.9g')
    report={'source':'oracle','model':meta['model'],'label':meta['label'],
            'zdim':zdim,'xdim':int(meta['x_dim']),
            'seed':seed,'basis':basis.serialize(),**score,
            'oracle_projection_valid':bool(score['test_r2']>=.90),
            'projection_warning':('This is a polynomial projection of the true internal logit, '
                                  'not its exact decomposition; low test_r2 invalidates the Gate-2 reference.')}
    (out/'fit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='basis'},indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=31);p.add_argument('--max-train',type=int,default=9000)
    p.add_argument('--z-degree',type=int,choices=[1,2,3])
    p.add_argument('--x-degree',type=int,choices=[1,2],default=2)
    args=p.parse_args()
    build(args.data,args.output,args.seed,args.max_train,args.z_degree,args.x_degree)
