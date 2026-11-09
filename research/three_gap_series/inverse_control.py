"""MODEL-INFORMED CR3 inversion control; NOT an unknown-formula gap success.

Fits cognitive parameters to frozen decoder probabilities without true T.
Then fits R+D/OLS to the pseudo-parameters on TRAIN. True T is accessed only
after these maps are frozen. This diagnoses decoder vs metric limitations.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from core import load,dump,correspondence,sha
from run import rotation

def design_tensors(x,device,dtype=torch.float32):
    from scipy.special import logit
    out=[]
    for k in [0,3]:
        gain,loss,p=x[:,k:k+3].T
        out.append(tuple(torch.tensor(a,device=device,dtype=dtype) for a in
            [np.log(np.maximum(gain/100,1e-30)),np.log(np.maximum(loss/100,1e-30)),
             (gain>0).astype(float),(loss>0).astype(float),
             logit(np.clip(p,1e-12,1-1e-12)),logit(np.clip(1-p,1e-12,1-1e-12))]))
    return out

def eta_from_parameters(t,design):
    a,lam,gamma,tau=[t[:,i,None] for i in range(4)]
    value=0
    for sign,(lg,ll,mg,ml,lp,lq) in zip([1.,-1.],design):
        value=value+sign*(torch.sigmoid(gamma*lp)*torch.exp(a*lg)*mg-lam*torch.sigmoid(gamma*lq)*torch.exp(a*ll)*ml)
    return tau*value

def decoder_network(context,device):
    cfg=context['cp']['model_config'];dims=[cfg['x_dim']+cfg['latent_dim'],*cfg['decoder_hidden_dims'],2]
    layers=[]
    for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
        layers.append(torch.nn.Linear(a,b))
        if i<len(dims)-2:layers.append(torch.nn.GELU())
    net=torch.nn.Sequential(*layers)
    net.load_state_dict({k.removeprefix('decoder.'):v for k,v in context['cp']['model_state_dict'].items() if k.startswith('decoder.')})
    return net.to(device).eval().requires_grad_(False)

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--model',choices=['CR3'],required=True)
    p.add_argument('--beta',type=float,required=True);p.add_argument('--zdim',type=int,required=True)
    p.add_argument('--pca',type=int,choices=[0,1],required=True)
    p.add_argument('--seed',type=int,default=20261114);p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--restarts',type=int,default=2);p.add_argument('--starts',type=int,default=64)
    p.add_argument('--device',default='cuda');p.add_argument('--objective',choices=['probability','logit'],default='probability')
    p.add_argument('--label',default='round27_inverse_control')
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed);c=load(args)
    data=c['data'];ix=np.flatnonzero(data['subject_index']==0);ix=ix[np.argsort(data['trial_index'][ix])]
    bank=data['x'][ix];ids=c['splits']['train'];d=c['d'];net=decoder_network(c,args.device)
    x=torch.tensor(bank,device=args.device,dtype=torch.float32);targets=[]
    with torch.no_grad():
        for chunk in np.array_split(ids,max(1,int(np.ceil(len(ids)/32)))):
            z=torch.tensor(c['z'][chunk],device=args.device,dtype=torch.float32)
            inputs=torch.cat((x[None].expand(len(chunk),-1,-1),z[:,None].expand(-1,500,-1)),dim=-1)
            logits=net(inputs);targets.append(logits[...,1]-logits[...,0])
    target=torch.cat(targets);probability=torch.sigmoid(target);design=design_tensors(bank,args.device)
    lower=torch.tensor([.05,.05,.05,.1],device=args.device);upper=torch.tensor([1.5,8.,1.8,8.],device=args.device)
    initial=torch.tensor([.8,2.,.7,2.2],device=args.device)
    raw0=torch.logit((initial-lower)/(upper-lower))
    bestloss=torch.full((len(ids),),float('inf'),device=args.device);bestt=torch.empty((len(ids),4),device=args.device)
    history=[]
    for restart in range(args.restarts):
        raw=torch.nn.Parameter(raw0[None].expand(len(ids),-1).clone()+(.15*torch.randn(len(ids),4,device=args.device) if restart else 0.))
        optimizer=torch.optim.Adam([raw],lr=.04)
        for step in range(args.steps):
            if step==args.steps*2//3:
                for group in optimizer.param_groups:group['lr']=.01
            physical=lower+(upper-lower)*torch.sigmoid(raw);eta=eta_from_parameters(physical,design)
            per_subject=(torch.nn.functional.binary_cross_entropy_with_logits(eta,probability,reduction='none').mean(1)
                         if args.objective=='probability' else ((eta-target)**2).mean(1))
            with torch.no_grad():
                improved=per_subject<bestloss;bestloss[improved]=per_subject[improved];bestt[improved]=physical[improved]
            optimizer.zero_grad();per_subject.sum().backward();optimizer.step()
            if step%250==0 or step==args.steps-1:
                record=dict(restart=restart,step=step,mean_loss=float(per_subject.mean().detach()))
                history.append(record);print(record,flush=True)
    pseudo=bestt.detach().cpu().numpy();v=c['v'];z=v[ids]
    pm=pseudo.mean(0);ps=pseudo.std(0)
    if np.any(ps<1e-8):raise ValueError('Pseudo-parameter collapsed')
    y=(pseudo-pm)/ps;cov=z.T@z/len(z);cross=z.T@y/len(z)
    def objective(angles):
        r=rotation(angles,d);num=np.diag(r@cross);var=np.diag(r@cov@r.T)
        return float(1-np.mean(num*num/var))
    rng=np.random.default_rng(args.seed);solutions=[]
    for _ in range(args.starts):
        fit=minimize(objective,rng.uniform(-np.pi,np.pi,6),method='BFGS',options=dict(maxiter=500,gtol=1e-8))
        solutions.append((float(fit.fun),rotation(fit.x,d)))
    best=min(solutions,key=lambda item:item[0]);r=best[1];ols=np.linalg.lstsq(z,y,rcond=None)[0]
    out=Path(__file__).resolve().parent/'results'/args.label/f'CR3_beta{args.beta:g}_zdim{args.zdim}_pca{args.pca}'/args.objective
    out.mkdir(parents=True,exist_ok=True);np.save(out/'rotation.npy',r)
    # First true-parameter access: all parameter fits and maps are locked.
    with np.load(c['latent_path']) as saved:t=saved['theta_true']
    test=c['splits']['test'];evaluation=correspondence(v[test]@r.T,t[test]);olse=correspondence(v[test]@ols,t[test])
    with torch.no_grad():
        fitted_eta=eta_from_parameters(bestt,design)
        entropy=torch.nn.functional.binary_cross_entropy_with_logits(target,probability,reduction='none').mean()
        kl=torch.nn.functional.binary_cross_entropy_with_logits(fitted_eta,probability,reduction='none').mean()-entropy
        rmse=torch.sqrt(torch.mean((fitted_eta-target)**2))
    result=dict(args=dict(**vars(args),source='decoder',metric='formula_aware_control',channel=args.objective,estimator='known-model inversion'),
                hashes=c['hashes'],best=dict(r=r,loss=best[0]),evaluation=evaluation,ols_evaluation=olse,
                pseudo_parameter_mean=pm,pseudo_parameter_sd=ps,train_pseudo_parameters=pseudo,train_ids=ids,
                fit_history=history,train_decoder_to_formula_kl=float(kl),train_decoder_to_formula_logit_rmse=float(rmse),
                code_sha256=sha(__file__),
                warning='MODEL-INFORMED control uses the known CR3 formula. No true T labels in fitting. Not a success under the unknown-formula WY protocol.')
    dump(out/'control.json',result)
    print('INFORMED CONTROL',args.objective,evaluation, 'KL',float(kl),'logit RMSE',float(rmse),flush=True)

if __name__=='__main__':main()
