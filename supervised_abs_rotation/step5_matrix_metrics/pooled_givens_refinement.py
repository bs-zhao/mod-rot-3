"""Exact Givens line evaluations for a fixed pooled soft-support matrix score."""
import numpy as np
from scipy.optimize import minimize_scalar

import pooled_support_certificate as certificate


def line_scores(h,q,i,j,angles,eps=.03):
    transformed=np.einsum('ai,xab,bj->xij',q,h,q,optimize=True)
    diagonal=np.maximum(np.diagonal(transformed,axis1=1,axis2=2),0.)
    f=lambda e:e/(e+eps**2)
    base=f(diagonal).sum(1)-f(diagonal[:,i])-f(diagonal[:,j])
    c,s=np.cos(np.asarray(angles))[:,None],np.sin(np.asarray(angles))[:,None]
    a,b,cross=diagonal[:,i],diagonal[:,j],transformed[:,i,j]
    left=np.maximum(c*c*a+s*s*b+2*c*s*cross,0.)
    right=np.maximum(s*s*a+c*c*b-2*c*s*cross,0.)
    return np.max(base[None]+f(left)+f(right),axis=1)


def refine(h,initial,eps=.03,passes=12):
    q=np.array(initial,copy=True);value=certificate.score(h,q,eps);history=[]
    grid=np.linspace(-np.pi/4,np.pi/4,361);d=len(q)
    for cycle in range(passes):
        before=value
        for i in range(d):
            for j in range(i+1,d):
                values=line_scores(h,q,i,j,grid,eps);index=int(np.argmin(values))
                theta=float(grid[index]);candidate_value=float(values[index])
                left=max(-np.pi/4,theta-np.pi/360);right=min(np.pi/4,theta+np.pi/360)
                result=minimize_scalar(lambda t:float(line_scores(h,q,i,j,[t],eps)[0]),bounds=(left,right),
                    method='bounded',options=dict(xatol=1e-9,maxiter=80))
                if result.fun<candidate_value:theta,candidate_value=float(result.x),float(result.fun)
                if candidate_value<value-1e-12:
                    change=np.eye(d);c,s=np.cos(theta),np.sin(theta)
                    change[i,i]=change[j,j]=c;change[i,j]=-s;change[j,i]=s
                    q=q@change;value=certificate.score(h,q,eps)
                else:theta=0.
                history.append(dict(cycle=cycle,U_i=i+1,U_j=j+1,step_angle_deg=float(np.degrees(theta)),loss=value))
        if before-value<1e-10:break
    return q,value,history


def best_line(h,q,i,j,eps=.03):
    transformed=np.einsum('ai,xab,bj->xij',q,h,q,optimize=True)
    diagonal=np.maximum(np.diagonal(transformed,axis1=1,axis2=2),0.)
    f=lambda e:e/(e+eps**2)
    base=f(diagonal).sum(1)-f(diagonal[:,i])-f(diagonal[:,j])
    a,b,cross=diagonal[:,i],diagonal[:,j],transformed[:,i,j]
    def evaluate(angles):
        c,s=np.cos(np.asarray(angles))[:,None],np.sin(np.asarray(angles))[:,None]
        left=np.maximum(c*c*a+s*s*b+2*c*s*cross,0.)
        right=np.maximum(s*s*a+c*c*b-2*c*s*cross,0.)
        return np.max(base[None]+f(left)+f(right),axis=1)
    grid=np.linspace(-np.pi/4,np.pi/4,361);values=evaluate(grid);index=int(np.argmin(values))
    theta,value=float(grid[index]),float(values[index])
    result=minimize_scalar(lambda t:float(evaluate([t])[0]),
        bounds=(max(-np.pi/4,theta-np.pi/360),min(np.pi/4,theta+np.pi/360)),
        method='bounded',options=dict(xatol=1e-9,maxiter=80))
    return (float(result.x),float(result.fun)) if result.fun<value else (theta,value)


def refine_best_decrease(h,initial,eps=.03,steps=60):
    """Pick the largest available pairwise decrease, instead of axis-order greed."""
    q=np.array(initial,copy=True);value=certificate.score(h,q,eps);history=[];d=len(q)
    for step in range(steps):
        candidates=[]
        for i in range(d):
            for j in range(i+1,d):
                angle,loss=best_line(h,q,i,j,eps);candidates.append((loss,i,j,angle))
        loss,i,j,angle=min(candidates)
        if value-loss<1e-10:break
        change=np.eye(d);c,s=np.cos(angle),np.sin(angle)
        change[i,i]=change[j,j]=c;change[i,j]=-s;change[j,i]=s
        q=q@change;value=certificate.score(h,q,eps)
        history.append(dict(cycle=step,U_i=i+1,U_j=j+1,step_angle_deg=float(np.degrees(angle)),loss=value))
    return q,value,history
