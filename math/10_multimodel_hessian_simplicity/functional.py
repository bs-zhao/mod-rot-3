"""Controlled oracle function-value effects complementary to Hessian metrics.

Finite 3-point self-axis differences and 4-corner mixed differences.
These are LOCAL contrasts, NOT population Sobol or Friedman H indices.
"""
from __future__ import annotations
import numpy as np
from oracle import values, valid_x_mask
from metrics import vector_scores, EPS


def controlled_effects(t, x, model, R, st, sx, step=.04, anchors=24,
                       threshold=.10):
    """Return separate scalar numeric scores for logit and probability.

    Rotate U=R*((T-center)/st); physical T displacement for U_i is
    st*R[i]*step.  Stimuli X_j displacement is sx_j*step.
    Same chosen W anchors across all angles and channels.
    """
    if step<=0 or anchors<2:
        raise ValueError("Step and anchor count must be positive")
    n=len(t)
    chosen=np.linspace(0,n-1,min(anchors,n),dtype=int)
    t=np.asarray(t,float)[chosen];x=np.asarray(x,float)[chosen]
    if model=="CR3":
        # Require ALL +/- X coordinate perturbations to stay in the
        # differentiable interior. This defines a fixed common support.
        good=np.ones(len(x),bool)
        for j in (0,1,3,4):
            good &= x[:,j]>1.05*step*sx[j]
        for j in (2,5):
            good &= (x[:,j]>1.05*step*sx[j]) & (x[:,j]<1-1.05*step*sx[j])
        t=t[good];x=x[good]
    if len(t)<2:
        return {"logit":{},"probability":{}},int(len(t))
    d=len(st);p=len(sx)
    n=len(t)
    base=np.stack(values(t,x,model),axis=1)
    own=np.zeros((2,n,d))
    uu=np.zeros((2,n,d,d))
    ux=np.zeros((2,n,d,p))
    for i in range(d):
        dt=step*st*R[i,:]
        plus=np.stack(values(t+dt,x,model),axis=1)
        minus=np.stack(values(t-dt,x,model),axis=1)
        own[:,:,i]=(plus-2*base+minus).T/(step*step)
        for j in range(i+1,d):
            dt2=step*st*R[j,:]
            pp=np.stack(values(t+dt+dt2,x,model),axis=1)
            pm=np.stack(values(t+dt-dt2,x,model),axis=1)
            mp=np.stack(values(t-dt+dt2,x,model),axis=1)
            mm=np.stack(values(t-dt-dt2,x,model),axis=1)
            value=(pp-pm-mp+mm).T/(4*step*step)
            uu[:,:,i,j]=uu[:,:,j,i]=value
        for j in range(p):
            dx=np.zeros(p);dx[j]=step*sx[j]
            pp=np.stack(values(t+dt,x+dx,model),axis=1)
            pm=np.stack(values(t+dt,x-dx,model),axis=1)
            mp=np.stack(values(t-dt,x+dx,model),axis=1)
            mm=np.stack(values(t-dt,x-dx,model),axis=1)
            ux[:,:,i,j]=(pp-pm-mp+mm).T/(4*step*step)
    result={}
    for c,ch in enumerate(("logit","probability")):
        diag=own[c]
        iu=np.triu_indices(d,k=1)
        pair=uu[c][:,iu[0],iu[1]]
        vzx=ux[c]
        rows={
            "value_self_difference_rms_loss":float(np.sqrt(np.mean(diag*diag))),
            "value_self_difference_mean_abs_loss":float(np.mean(np.abs(diag))),
            "value_uu_mixed_difference_rms_loss":float(np.sqrt(np.mean(pair*pair))),
            "value_uu_mixed_difference_mean_abs_loss":float(np.mean(np.abs(pair))),
            "value_ux_mixed_difference_rms_loss":float(np.sqrt(np.mean(vzx*vzx))),
            "value_ux_mixed_difference_mean_abs_loss":float(np.mean(np.abs(vzx))),
        }
        for typ,field in (("ux",vzx),("uu",np.stack([
                uu[c,:,i,np.arange(d)!=i] for i in range(d)],axis=1))):
            strength=np.sqrt(np.mean(field*field,axis=0))
            scale=max(float(np.sqrt(np.mean(field*field))),EPS)
            # For each U_i: sparsity of its function-value interaction partners.
            scores=[vector_scores(v,scale,relative_threshold=threshold)
                    for v in strength]
            for key in (scores[0] if scores else {}):
                rows[f"value_{typ}_row_{key}"]=float(np.mean([s[key] for s in scores]))
        result[ch]=rows
    return result,int(len(t))
