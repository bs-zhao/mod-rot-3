"""Rotation-equivariant complete polynomial kernel approximation for a global GP link."""
import numpy as np
import torch

import run as wave


class Dictionary:
    uses_centers=False
    description='complete joint-W polynomial kernel, degrees0..4, multinomial weights divided by dimension^(degree/2)'

    def __init__(self,mean,scale,centers,d,device,degree=4):
        self.device,self.d,self.degree=device,d,degree
        self.mean=torch.tensor(mean,dtype=torch.float64,device=device)
        self.scale=torch.tensor(scale,dtype=torch.float64,device=device)
        self.powers=wave.sur.powers_for(len(mean),degree)
        self.weights=wave.sur.weights_for(self.powers)/(len(mean)**(self.powers.sum(1)/2))
        self.power_tensor=torch.tensor(self.powers,dtype=torch.long,device=device)
        self.weight_tensor=torch.tensor(self.weights,dtype=torch.float64,device=device)
        self.width=len(self.powers)

    @torch.no_grad()
    def __call__(self,w):
        z=(torch.tensor(np.asarray(w),dtype=torch.float64,device=self.device)-self.mean)/self.scale
        phi=self.weight_tensor[None].expand(len(z),-1).clone()
        for j in range(z.shape[1]):
            one=torch.ones(len(z),dtype=torch.float64,device=self.device)
            values=[one]
            for k in range(1,self.degree+1):values.append(values[-1]*z[:,j])
            lookup=torch.stack(values,dim=1)
            phi*=lookup[:,self.power_tensor[:,j]]
        return phi

    def artifacts(self):return dict(polynomial_powers=self.powers,polynomial_weights=self.weights)
