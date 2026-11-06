"""Audit old adaptive weights without changing historical methods or outputs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'np'/'src')]
import json
import numpy as np
from scipy.optimize import minimize,linear_sum_assignment
from rotations.stimulus_contrast.method import from_angles
from rotations.stimulus_contrast.data import corr
from rotations.stimulus_contrast.run import write_json,write_csv


def primitives(g,a):
    rotated=np.einsum('ia,nal->nil',a,g,optimize=True)
    c=np.einsum('nil,nim->ilm',rotated,rotated)/len(g)
    # Historical effective_rank squares its input singular values: energy entropy.
    eigen=np.maximum(np.linalg.eigvalsh(c),0)
    total=eigen.sum(1,keepdims=True)
    p=np.divide(eigen,total,out=np.zeros_like(eigen),where=total>1e-14)
    entropy=-np.sum(np.where(eigen>1e-14,p*np.log(np.maximum(p,1e-300)),0),axis=1)
    rank=np.where(total[:,0]>1e-14,np.exp(entropy),0)
    fp=float(np.sum(np.maximum(rank-1,0)))
    moments=np.einsum('nil,njl->lij',g,g)/len(g)
    moments=moments[:12]
    r=np.einsum('ia,lab,jb->lij',a,moments,a,optimize=True)
    off=r.copy();i=np.arange(len(a));off[:,i,i]=0
    norms=np.sum(moments**2,axis=(1,2));valid=norms>1e-20
    jd=float(np.mean(np.sum(off[valid]**2,axis=(1,2))/(norms[valid]+1e-12)))
    return jd,fp


def main():
    for model in ['C1','CR3']:
        base=ROOT/'outputs'/'choice'/model
        old=list((base/'rotations'/'mlp_teacher_rotation'/'beta0p0005_lambda0_ctx0p25to0p8_L8'/'repeat01').glob('run_*'))
        if len(old)!=1:raise RuntimeError('Ambiguous cached run')
        old=old[0];out=base/'rotations'/'stimulus_contrast'/'weight_audit_v2';out.mkdir(parents=True,exist_ok=False)
        meta=json.loads((old/'rotation__fit.json').read_text());cfg=meta['config']
        mat=np.load(old/'rotation__matrix.npz');saved=np.load(old/'rotation__U.npz');arrays=np.load(old/'rotation__method_arrays.npz')
        q=saved['U']@mat['R'];qfit=q[mat['fit_ids']];n=len(qfit)
        fit_ids=np.random.default_rng(cfg['seed']).permutation(n)[:max(16,int(.8*n))]
        rng=np.random.default_rng(cfg['seed']+918)
        qa=qfit[fit_ids][np.sort(rng.choice(len(fit_ids),min(len(fit_ids),cfg['anchor_subjects']),False))]
        co={i:arrays[f'polynomial_tensor_degree{i}'] for i in range(4)}
        g=np.broadcast_to(co[1],(len(qa),*co[1].shape)).copy()
        g+=2*np.einsum('nj,ijl->nil',qa,co[2])+3*np.einsum('nj,nk,ijkl->nil',qa,qa,co[3])
        k=g.shape[1];j0,f0=primitives(g,np.eye(k));rotations={};reports=[]
        expected=meta['objective_fit']['identity_metrics']
        if not np.allclose([j0,f0],[expected['jointdiag13'],expected['fingerprint11']],atol=1e-9):
            raise ValueError('Cached objective reconstruction does not match historical baseline')
        for mode in ['historical_weight','identity_equal_weight']:
            coefficient=f0/j0 if mode=='historical_weight' else j0/f0
            def objective(v):
                j,f=primitives(g,from_angles(v,k));return j+coefficient*f
            opts=[]
            for start in range(3):
                v=np.zeros(k*(k-1)//2) if start==0 else rng.normal(0,.5,k*(k-1)//2)
                r=minimize(objective,v,method='Powell',options={'maxiter':100})
                opts.append((float(r.fun),from_angles(r.x,k)))
            value,a=min(opts,key=lambda x:x[0]);rotations[mode]=a
            reports.append(dict(mode=mode,identity_jd=j0,identity_fp=f0,fp_coefficient=coefficient,
                                fp_to_jd_contribution_at_identity=coefficient*f0/j0,loss=value))
        np.savez_compressed(out/'rotations.npz',**rotations)
        write_json(out/'fit.json',dict(source=str(old),definition='J+(F0/J0)F versus J+(J0/F0)F',fit_rows=reports))
        truth=np.load(old/'rotation__test_truth.npz')['theta_test']
        for row in reports:
            c=corr(truth,(q@rotations[row['mode']].T)[saved['test_ids']]);i,j=linear_sum_assignment(-c*c)
            row['test_optimal_mean_r2']=float(np.mean(c[i,j]**2))
            row['signed_test_corr']=json.dumps(c.tolist())
        write_csv(out/'comparison.csv',reports)
        print(model,json.dumps(reports,indent=2))


if __name__=='__main__':main()
