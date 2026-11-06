"""Hold one polynomial function fixed; change its stimulus-channel basis.

Random orthogonal right transformations preserve function values exactly if
the stimulus basis is transformed inversely, yet channelwise JD changes.
This is an exploratory diagnostic, not a theta-selected rotation method.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'np'/'src')]
import argparse
import json
import time
import numpy as np
from scipy.optimize import minimize,linear_sum_assignment
from rotations.stimulus_contrast.method import from_angles
from rotations.stimulus_contrast.run import write_json,write_csv
from rotations.stimulus_contrast.data import corr


def channel_moments(g):
    return np.einsum('nil,njl->lij',g,g)/len(g)


def channel_loss(a,c):
    r=np.einsum('ia,lab,jb->lij',a,c,a,optimize=True)
    denominator=np.sum(c*c,axis=(1,2))
    off=np.sum(r*r,axis=(1,2))-np.sum(np.diagonal(r,axis1=1,axis2=2)**2,axis=1)
    active=denominator>1e-12
    return float(np.mean(np.maximum(off[active],0)/denominator[active]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['C1','CR3']);args=p.parse_args()
    base=ROOT/'outputs'/'choice'/args.model
    runs=list((base/'rotations'/'mlp_teacher_rotation'/'beta0p0005_lambda0_ctx0p25to0p8_L8'/'repeat01').glob('run_*'))
    if len(runs)!=1:raise RuntimeError('Specify unambiguous historical source before auditing')
    old=runs[0]; out=base/'rotations'/'stimulus_contrast'/'channel_audit_v1';out.mkdir(parents=True,exist_ok=False)
    mat=np.load(old/'rotation__matrix.npz'); arrays=np.load(old/'rotation__method_arrays.npz')
    saved=np.load(old/'rotation__U.npz');q=saved['U']@mat['R']; train=saved['train_ids'];test=saved['test_ids']
    qfit=q[mat['fit_ids']]
    rng=np.random.default_rng(841);qa=qfit[rng.choice(len(qfit),160,False)]
    coef={i:arrays[f'polynomial_tensor_degree{i}'] for i in range(4)}
    g=np.broadcast_to(coef[1],(len(qa),*coef[1].shape)).copy()
    g+=2*np.einsum('nj,ijl->nil',qa,coef[2])+3*np.einsum('nj,nk,ijkl->nil',qa,qa,coef[3])
    channels=g.shape[-1];k=g.shape[1]; rotations={}; fitting=[];start=time.perf_counter()
    for basis_seed in [None,11,23,37,53,71,89]:
        name='original' if basis_seed is None else f'orthogonal_basis_{basis_seed}'
        o=np.eye(channels) if basis_seed is None else np.linalg.qr(np.random.default_rng(basis_seed).normal(size=(channels,channels)))[0]
        transformed=g@o;c=channel_moments(transformed)
        # A full feature basis is used throughout, unlike legacy first-12 truncation.
        fits=[]
        for si in range(3):
            initial=np.zeros(k*(k-1)//2) if si==0 else rng.normal(0,.6,k*(k-1)//2)
            fit=minimize(lambda v:channel_loss(from_angles(v,k),c),initial,method='BFGS',options={'maxiter':200,'gtol':1e-7})
            fits.append((float(fit.fun),from_angles(fit.x,k)))
        value,a=min(fits,key=lambda r:r[0]);rotations[name]=a
        fitting.append(dict(basis=name,identity_loss=channel_loss(np.eye(k),c),optimized_loss=value,
                            orthogonal_channel_roundtrip_max_abs=float(np.max(np.abs((transformed@o.T)-g))),
                            gradient_total_norm_relative_error=float(abs(np.sum(transformed**2)/np.sum(g**2)-1))))
    np.savez_compressed(out/'rotations.npz',**rotations)
    write_json(out/'fit.json',dict(source=str(old),fit_rows=fitting,full_channel_count=channels,elapsed_seconds=time.perf_counter()-start,
                                 convention='G_new=G O, phi_new=O^T phi. Same function; full basis, no truncation.'))
    # Post-fit theta read, no best candidate or basis selected.
    truth=np.load(old/'rotation__test_truth.npz')['theta_test']
    results=[]
    for item in fitting:
        a=rotations[item['basis']];c=corr(truth,(q@a.T)[test]);i,j=linear_sum_assignment(-c*c)
        off=np.ones(c.shape,bool);off[i,j]=False
        results.append(dict(**item,test_optimal_mean_r2=float(np.mean(c[i,j]**2)),
                            offtarget_mean_r2=float(np.mean(c[off]**2)),signed_test_corr=json.dumps(c.tolist())))
    write_csv(out/'comparison.csv',results);write_json(out/'complete.json',dict(exploratory=True))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
