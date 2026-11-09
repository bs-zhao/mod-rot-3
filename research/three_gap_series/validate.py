"""Check formula values, network coordinate equivalence and F2 numerical errors."""
import argparse
from pathlib import Path
import numpy as np
import torch
from core import load, DIMS, formula, decoder, derivatives, dump
from run import prepare, numerical_fields, rotation

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--model',choices=DIMS,required=True)
    p.add_argument('--beta',type=float,required=True); p.add_argument('--zdim',type=int,required=True)
    p.add_argument('--pca',type=int,choices=[0,1],required=True)
    a=p.parse_args();a.metric='uu_off_rms'; torch.set_num_threads(2); c=load(a)
    rng=np.random.default_rng(13); rows=rng.choice(len(c['data']['x']),1000,replace=False)
    data=c['data']; t=torch.tensor(data['parameters'][data['subject_index'][rows]])
    x=torch.tensor(data['x'][rows]); eta=torch.vmap(lambda t,x:formula(a.model,t,x))(t,x)
    probability=torch.sigmoid(eta).numpy()
    formula_error=float(abs(probability-data['predictive_true'][rows].reshape(-1)).max())
    if formula_error>1e-9: raise ValueError(f'Formula disagrees with original data: {formula_error}')
    net=decoder(c); d=c['d']; v=c['v'][data['subject_index'][rows[:32]]]
    w=np.concatenate((v,data['x'][rows[:32]]),axis=1)
    wt=torch.tensor(w); exact=torch.vmap(net)(wt).detach().numpy()
    r=rotation(rng.uniform(-3,3,d*(d-1)//2),d)
    u=v@r.T; reconstructed=np.concatenate((u@r,data['x'][rows[:32]]),axis=1)
    rebuilt=torch.vmap(net)(torch.tensor(reconstructed)).detach().numpy()
    inverse_error=float(abs(exact-rebuilt).max())
    if inverse_error>1e-10: raise ValueError('Decoder inverse-coordinate mismatch')
    audit=[]
    for source in ['formula','decoder']:
        a.source=source;a.channel='logit';a.seed=131;a.samples=32
        fn,w,_,_,_,_=prepare(a,c)
        truth=derivatives(fn,w,d)
        for h in [.02,.01,.005]:
            est=numerical_fields(fn,w,d,h)
            audit.append(dict(source=source,step=h,
                gradient_relative=float(np.linalg.norm(est['g']-truth['g'])/max(np.linalg.norm(truth['g']),1e-20)),
                hessian_relative=float(np.linalg.norm(est['h']-truth['h'])/max(np.linalg.norm(truth['h']),1e-20))))
    out=Path(__file__).resolve().parent/'results/validation';out.mkdir(parents=True,exist_ok=True)
    result=dict(model=a.model,pca=a.pca,hashes=c['hashes'],formula_probability_max_error=formula_error,
                decoder_inverse_max_error=inverse_error,finite_difference=audit)
    dump(out/f'{a.model}_pca{a.pca}.json',result);print(result)

if __name__=='__main__': main()
