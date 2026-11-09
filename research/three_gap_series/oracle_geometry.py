"""Separate supervised TRAIN-only R+D/OLS ceiling; never imported by discovery."""
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from core import load,DIMS,dump,correspondence,corr
from run import rotation

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--model',choices=DIMS,required=True)
    p.add_argument('--beta',type=float,required=True);p.add_argument('--zdim',type=int,required=True)
    p.add_argument('--pca',type=int,choices=[0,1],required=True)
    p.add_argument('--starts',type=int,default=32)
    a=p.parse_args();c=load(a);v=c['v'];d=c['d'];ids=c['splits']
    with np.load(c['latent_path']) as f: t=f['theta_true']
    tm=t[ids['train']].mean(0);ts=t[ids['train']].std(0);target=(t-tm)/ts
    z=v[ids['train']];y=target[ids['train']]
    cov=z.T@z/len(z);cross=z.T@y/len(z)
    def objective(angles):
        r=rotation(angles,d);num=np.diag(r@cross);var=np.diag(r@cov@r.T)
        return float(1-np.mean(num*num/var))
    rng=np.random.default_rng(717);solutions=[]
    for k in range(a.starts):
        fit=minimize(objective,rng.uniform(-np.pi,np.pi,d*(d-1)//2),method='BFGS',options=dict(maxiter=500,gtol=1e-8))
        solutions.append((float(fit.fun),rotation(fit.x,d),bool(fit.success)))
    best=min(solutions,key=lambda x:x[0]);r=best[1]
    ols=np.linalg.lstsq(z,y,rcond=None)[0]
    test=ids['test'];oracle=correspondence(v[test]@r.T,t[test]);olse=correspondence(v[test]@ols,t[test])
    olsnorm=ols.T/np.linalg.norm(ols.T,axis=1,keepdims=True)
    result=dict(model=a.model,pca=a.pca,hashes=c['hashes'],train_mse=best[0],rotation=r,
                rotation_test=oracle,ols_test=olse,ols_axis_gram=olsnorm@olsnorm.T,
                z_cov=cov,z_eigenvalues=np.linalg.eigvalsh(cov),converged_starts=sum(x[2] for x in solutions),
                disclaimer='Supervised geometric diagnostic only; not an unsupervised result')
    out=Path(__file__).resolve().parent/'results/oracles';out.mkdir(parents=True,exist_ok=True)
    dump(out/f'{a.model}_beta{a.beta:g}_zdim{a.zdim}_pca{a.pca}.json',result)
    print(a.model,'R+D',oracle['matched_mean'],oracle['matched_min'],'OLS',olse['matched_mean'],olse['matched_min'])

if __name__=='__main__': main()
