"""E03: local complete-polynomial Hessian inference from numerical WY only.

No access to a formula, model identity, true coordinates or derivative labels.
Isotropic distances and multinomial coefficient penalty preserve rotations.
The response channel/link is a declared statistical target, not a model prior.
"""
import itertools
import math
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit
from sklearn.preprocessing import PolynomialFeatures


class LocalPolynomial:
    def __init__(self,degree=3,neighbors=2048,bandwidth=1.,ridge=1e-6,channel='logit'):
        if degree not in [2,3]:raise ValueError('Local polynomial supports degree 2 or 3')
        self.degree=degree;self.neighbors=neighbors;self.bandwidth=bandwidth
        self.ridge=ridge;self.channel=channel

    def fit(self,w,y):
        self.w=np.asarray(w,dtype=float);self.y=np.asarray(y,dtype=float)
        self.tree=cKDTree(self.w);self.basis=PolynomialFeatures(self.degree).fit(self.w[:1])
        self.powers=self.basis.powers_
        self.weights=np.array([math.sqrt(math.factorial(int(a.sum()))/math.prod(math.factorial(int(j)) for j in a)) for a in self.powers])
        if min(self.neighbors,len(w))<2*len(self.powers):raise ValueError('Too few neighbors for a stable complete local basis')
        self.info=dict(protocol='E03 original-WY complete local polynomial; no formula or model identity',
            degree=self.degree,neighbors=self.neighbors,bandwidth=self.bandwidth,ridge=self.ridge,
            features=len(self.powers),train_rows=len(w),capabilities=['F0','F1','F2','F3' if self.degree==3 else 'F2'],
            assumptions=['Local smoothness','Euclidean neighborhoods','Complete polynomial approximation within the bandwidth'],
            warning='Fixed stimulus banks still require smooth interpolation assumptions; neighbor rows are not independent subjects')
        return self

    def fields(self,w,third=False,structure_link='native'):
        if third and self.degree<3:raise ValueError('F3 unavailable for a quadratic local fit')
        w=np.asarray(w);n,width=w.shape
        distance,index=self.tree.query(w,k=min(self.neighbors,len(self.w)),workers=2)
        output=dict(value=np.empty(n),g=np.empty((n,width)),h=np.empty((n,width,width)))
        if third:output['third']=np.empty((n,width,width,width))
        local_scales=[];conditions=[];convergence=[]
        for j in range(n):
            h=max(float(distance[j,-1])*self.bandwidth/np.sqrt(width),1e-8)
            delta=(self.w[index[j]]-w[j])/h
            kernel=np.exp(-.5*np.sum(delta*delta,axis=1));kernel/=kernel.sum()
            design=self.basis.transform(delta)*self.weights
            target=self.y[index[j]].copy()
            if self.channel=='probability' and structure_link=='log_odds':target=np.log(np.clip(target,1e-12,1-1e-12)/np.clip(1-target,1e-12,1-1e-12))
            penalty=np.full(len(self.powers),self.ridge);penalty[0]=0
            gram=design.T@(kernel[:,None]*design)+np.diag(penalty)
            if self.channel=='binary':
                beta=np.zeros(len(self.powers));m=np.clip(np.mean(target),1e-5,1-1e-5);beta[0]=np.log(m/(1-m))
                converged=False
                for _ in range(40):
                    eta=design@beta;p=expit(eta)
                    gradient=design.T@(kernel*(p-target))+penalty*beta
                    curvature=design.T@((kernel*(p*(1-p)).clip(1e-7))[:,None]*design)+np.diag(penalty+1e-9)
                    step=np.linalg.solve(curvature,gradient)
                    old=float(np.sum(kernel*(np.logaddexp(0,eta)-target*eta))+.5*np.sum(penalty*beta*beta))
                    rate=1.
                    while rate>1e-5:
                        trial=beta-rate*step;trialeta=design@trial
                        value=float(np.sum(kernel*(np.logaddexp(0,trialeta)-target*trialeta))+.5*np.sum(penalty*trial*trial))
                        if value<=old:break
                        rate*=.5
                    beta=trial
                    if np.max(abs(rate*step))<1e-7:converged=True;break
                convergence.append(converged)
            else:beta=np.linalg.solve(gram,design.T@(kernel*target))
            coefficient=beta*self.weights
            def derivative(indices):
                power=np.bincount(indices,minlength=width)
                match=np.all(self.powers==power,axis=1)
                return float(coefficient[match].sum()*math.prod(math.factorial(int(k)) for k in power)/h**len(indices))
            output['value'][j]=derivative(())
            for a in range(width):output['g'][j,a]=derivative((a,))
            for a,b in itertools.combinations_with_replacement(range(width),2):
                output['h'][j,a,b]=output['h'][j,b,a]=derivative((a,b))
            if third:
                for abc in itertools.combinations_with_replacement(range(width),3):
                    value=derivative(abc)
                    for abc2 in set(itertools.permutations(abc)):output['third'][(j,*abc2)]=value
            local_scales.append(h)
            # Diagonal spread is a cheap conditioning warning, not a condition number.
            diag=np.diag(gram);conditions.append(float(diag.max()/max(diag.min(),1e-30)))
        if self.channel=='binary' and structure_link=='native':
            eta=output['value'];g=output['g'].copy();hh=output['h'].copy()
            p=expit(eta);s1=p*(1-p);s2=s1*(1-2*p);s3=s1*(1-6*p+6*p*p)
            if third:
                output['third']=(s1[:,None,None,None]*output['third']+s2[:,None,None,None]*(hh[:,:,:,None]*g[:,None,None,:]+hh[:,:,None,:]*g[:,None,:,None]+hh[:,None,:,:]*g[:,:,None,None])+s3[:,None,None,None]*g[:,:,None,None]*g[:,None,:,None]*g[:,None,None,:])
            output['h']=s1[:,None,None]*hh+s2[:,None,None]*g[:,:,None]*g[:,None,:]
            output['g']=s1[:,None]*g;output['value']=p
        self.info.update(local_scale_min=float(min(local_scales)),local_scale_max=float(max(local_scales)),
            design_diagonal_spread_max=float(max(conditions)),binary_converged_fraction=float(np.mean(convergence)) if convergence else None)
        return output
