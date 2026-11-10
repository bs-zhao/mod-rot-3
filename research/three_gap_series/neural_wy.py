"""Generic cubic-plus-smooth-neural regression from numerical WY only."""
import copy
import math
import time
import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures


class NeuralWY:
    def __init__(self,d,width=128,epochs=200,seed=20261128,device=None):
        self.d=d;self.width=width;self.epochs=epochs;self.seed=seed
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def polynomial(self,w):
        scaled=(w-self.mean)/self.scale/np.sqrt(w.shape[1])
        return self.basis.transform(scaled)*self.weights

    def baseline(self,w):
        return np.concatenate([self.polynomial(b)@self.coefficient for b in np.array_split(w,max(1,int(np.ceil(len(w)/8192))))])

    def fit(self,w,y,validation,validation_y):
        started=time.time();rng=np.random.default_rng(self.seed);torch.manual_seed(self.seed)
        self.mean=w.mean(0);self.scale=w.std(0).clip(1e-8)
        self.scale[:self.d]=np.sqrt(np.mean(np.var(w[:,:self.d],axis=0)))
        self.basis=PolynomialFeatures(3).fit(w[:1]);self.powers=self.basis.powers_
        self.weights=np.array([math.sqrt(math.factorial(int(a.sum()))/math.prod(math.factorial(int(j)) for j in a)) for a in self.powers])
        ids=rng.choice(len(w),min(32768,len(w)),replace=False);phi=self.polynomial(w[ids])
        ridge=np.eye(phi.shape[1])*1e-10;ridge[0,0]=0
        self.coefficient=np.linalg.solve(phi.T@phi/len(phi)+ridge,phi.T@y[ids]/len(phi))
        residual=y-self.baseline(w);val_residual=validation_y-self.baseline(validation)
        self.residual_scale=max(float(np.std(residual)),1e-8)
        signal=max(float(np.std(y)),1e-8)
        self.net=None;best_epoch=0;best_mse=float(np.mean(val_residual**2));history=[]
        if self.residual_scale>1e-6*signal:
            self.net=torch.nn.Sequential(torch.nn.Linear(w.shape[1],self.width),torch.nn.GELU(),
                torch.nn.Linear(self.width,self.width),torch.nn.GELU(),torch.nn.Linear(self.width,1)).to(self.device)
            torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
            xx=torch.tensor((w-self.mean)/self.scale,dtype=torch.float32,device=self.device)
            yy=torch.tensor(residual/self.residual_scale,dtype=torch.float32,device=self.device)
            vx=torch.tensor((validation-self.mean)/self.scale,dtype=torch.float32,device=self.device)
            vy=torch.tensor(val_residual/self.residual_scale,dtype=torch.float32,device=self.device)
            optimizer=torch.optim.AdamW(self.net.parameters(),lr=1e-3,weight_decay=1e-5)
            schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,self.epochs)
            best_state=copy.deepcopy(self.net.state_dict());best_normalized=float(np.mean((val_residual/self.residual_scale)**2))
            for epoch in range(self.epochs):
                permutation=torch.randperm(len(xx),device=self.device);self.net.train()
                for batch in permutation.split(4096):
                    optimizer.zero_grad(set_to_none=True);pred=self.net(xx[batch]).squeeze(-1)
                    loss=torch.mean((pred-yy[batch])**2);loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(),10.);optimizer.step()
                schedule.step()
                if (epoch+1)%5==0 or epoch==0:
                    self.net.eval()
                    with torch.no_grad():
                        errors=[torch.sum((self.net(a).squeeze(-1)-b)**2) for a,b in zip(vx.split(8192),vy.split(8192))]
                        mse=float(torch.stack(errors).sum()/len(vx))
                    history.append(dict(epoch=epoch+1,validation_rmse=np.sqrt(mse)*self.residual_scale))
                    if mse<best_normalized:
                        best_normalized=mse;best_epoch=epoch+1;best_state=copy.deepcopy(self.net.state_dict())
                    print('NeuralWY epoch',epoch+1,'validation RMSE',round(np.sqrt(mse)*self.residual_scale,6),flush=True)
                    if epoch+1-best_epoch>=40 and epoch>=40:break
            self.net.load_state_dict(best_state);self.net=self.net.cpu().double().eval();self.net.requires_grad_(False)
            best_mse=best_normalized*self.residual_scale**2
            del xx,yy,vx,vy,optimizer
            if torch.cuda.is_available():torch.cuda.empty_cache()
        self.tensors=tuple(torch.tensor(a,dtype=torch.float64) for a in [self.mean,self.scale,self.weights,self.coefficient])
        self.info=dict(protocol='E12 generic complete cubic baseline plus independently fitted two-layer GELU residual network',
            model_identity_used=False,teacher_derivatives_used_for_fit=False,train_rows=len(w),validation_rows=len(validation),
            polynomial_features=len(self.powers),width=self.width,max_epochs=self.epochs,best_epoch=best_epoch,
            device=self.device,validation_rmse=np.sqrt(best_mse),history=history,
            skipped_neural_residual=self.net is None,seconds=time.time()-started,
            warning='Value-only fitting supplies no guarantee of derivative accuracy; exact derivative comparisons are post-fit diagnostics')
        return self

    def value(self,w):
        mean,scale,weights,coefficient=self.tensors;s=(w-mean)/scale
        v=s/np.sqrt(len(s));phi=torch.ones(len(weights),dtype=w.dtype,device=w.device)
        for j in range(len(v)):
            powers=torch.stack((torch.ones_like(v[j]),v[j],v[j]*v[j],v[j]*v[j]*v[j]))
            phi=phi*powers[torch.tensor(self.powers[:,j],dtype=torch.long)]
        result=(phi*weights)@coefficient
        if self.net is not None:result=result+self.residual_scale*self.net(s).squeeze(-1)
        return result

    def save(self,path):
        torch.save(dict(mean=self.mean,scale=self.scale,powers=self.powers,weights=self.weights,coefficient=self.coefficient,
            residual_scale=self.residual_scale,state=None if self.net is None else self.net.state_dict(),info=self.info),path)
