"""Independent new-subject confirmation after a discovery rotation is frozen."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.stats import truncnorm
from core import load,DIMS,sha,formula,dump,correspondence

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--model',choices=DIMS,required=True)
    p.add_argument('--beta',type=float,required=True);p.add_argument('--zdim',type=int,required=True)
    p.add_argument('--pca',type=int,choices=[0,1],required=True)
    p.add_argument('--result',required=True,help='Completed summary.json; its R is locked before new data are drawn')
    p.add_argument('--subjects',type=int,default=1000);p.add_argument('--trials',type=int,default=500)
    p.add_argument('--seed',type=int,default=20261110);p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(2);c=load(a)
    result=json.loads(Path(a.result).read_text());r=np.asarray(result['best']['r'])
    if result['args']['source']!='decoder' or result['args']['model']!=a.model or result['args']['pca']!=a.pca:
        raise ValueError('Rotation is not a matching completed unsupervised decoder result')
    if result['hashes']!=c['hashes']: raise ValueError('Frozen source mismatch')
    locked_hash=sha(a.result);rng=np.random.default_rng(a.seed);n=a.subjects;trials=a.trials
    oracle_path=Path(__file__).resolve().parent/'results/oracles'/f'{a.model}_beta{a.beta:g}_zdim{a.zdim}_pca{a.pca}.json'
    oracle=json.loads(oracle_path.read_text()) if oracle_path.exists() else None
    if oracle is not None and oracle['hashes']!=c['hashes']: raise ValueError('Oracle source mismatch')
    if a.model=='CR3':
        if trials!=500: raise ValueError('CR3 must retain its 500-stimulus bank')
        t=np.column_stack((rng.beta(12,3,n),rng.gamma(9,.22,n),rng.beta(9,4,n),truncnorm.rvs((.5-2.2)/.55,(4-2.2)/.55,loc=2.2,scale=.55,size=n,random_state=rng)))
        bad=(t[:,1]<.3)|(t[:,1]>5)
        while bad.any():
            t[bad,1]=rng.gamma(9,.22,bad.sum());bad=(t[:,1]<.3)|(t[:,1]>5)
        idx=np.flatnonzero(c['data']['subject_index']==0);idx=idx[np.argsort(c['data']['trial_index'][idx])]
        bank=c['data']['x'][idx];x=np.broadcast_to(bank,(n,500,6)).copy()
    else:
        t=rng.normal(0,.5,(n,c['d']));xdim=c['cp']['model_config']['x_dim']
        x=rng.uniform(-1,1,(n,trials,xdim))
    probability=[]
    with torch.no_grad():
        for start in range(0,n,32):
            tt=np.repeat(t[start:start+32],trials,axis=0);xx=x[start:start+32].reshape(-1,x.shape[-1])
            eta=torch.vmap(lambda t,x:formula(a.model,t,x))(torch.tensor(tt),torch.tensor(xx))
            probability.append(torch.sigmoid(eta).numpy())
    probability=np.concatenate(probability).reshape(n,trials)
    y=rng.binomial(1,probability)
    sys.path.insert(0,str(Path(a.repo)/'np/src'))
    from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig,DiscreteNeuralProcess
    net=DiscreteNeuralProcess(DiscreteNPConfig(**c['cp']['model_config']))
    net.load_state_dict(c['cp']['model_state_dict']);net.to(a.device).eval()
    prep=c['cp']['preprocessing'];mean=prep['encoder_mean'].to(a.device);sd=prep['encoder_std'].to(a.device)
    zs=[]
    with torch.no_grad():
        for start in range(0,n,16):
            xx=torch.tensor(x[start:start+16],dtype=torch.float32,device=a.device)
            yy=torch.tensor(y[start:start+16,:,None],dtype=torch.float32,device=a.device)
            obs=(torch.cat((xx,yy),dim=-1)-mean)/sd
            keep=torch.ones(obs.shape[:2],dtype=torch.bool,device=a.device)
            z,_=net.encode_context(obs[...,:-1],obs[...,-1:],keep);zs.append(z.cpu().numpy())
    z=np.concatenate(zs);v=(z-c['mean'])@c['components'].T
    evaluation=correspondence(v@r.T,t)
    out=Path(__file__).resolve().parent/'results/fresh_cohort'/a.model
    out.mkdir(parents=True,exist_ok=True)
    metadata=dict(args=vars(a),locked_summary_sha256=locked_hash,source_hashes=c['hashes'],
                  fresh_T_sha256=__import__('hashlib').sha256(t.tobytes()).hexdigest(),
                  evaluation=evaluation,rotation=r,protocol='New independent subjects and Bernoulli responses; frozen original encoder; no refit/selection')
    metadata['supervised_oracle_fresh']=correspondence(v@np.asarray(oracle['rotation']).T,t) if oracle else None
    dump(out/f'seed{a.seed}_{result["args"]["metric"]}_{result["args"]["channel"]}_{locked_hash[:8]}.json',metadata)
    print(a.model,evaluation['matched_mean'],evaluation['matched_min'],evaluation['off_max'],flush=True)

if __name__=='__main__': main()
