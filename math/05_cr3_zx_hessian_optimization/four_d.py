"""Independent 4D ZZ-only estimators on the subject-wise 500-trial mean Y.

KRR and cubic tensor B-splines are evaluated alongside 10D tensor
polynomials. Their target is mean_X E[Y|Z,X]; therefore they estimate
ONLY the 4x4 mean ZZ block and MUST NOT be reported as ZX/XX methods.

Code is based on the earlier math/02 method, with Z-adaptive spline
knots (the old hard-coded CR3 T bounds are invalid for learned Z).
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from legacy_krr_spline import fit_krr,SplineDesign,SplineModel


def run_krr(Z,Y,train_ids,val_ids,test_ids,rank=128,seed=20261106,
            lengthscales=(.5,1.,2.,4.),lambdas=(1e-7,1e-5,1e-3)):
    channels={f"y{j}":Y[:,:,j].mean(1) for j in range(Y.shape[2])}
    split=dict(train=np.asarray(train_ids,dtype=int),
               val=np.asarray(val_ids,dtype=int),
               test=np.asarray(test_ids,dtype=int))
    models,selected=fit_krr(Z,channels,split,rank=int(rank),
                             lengthscales=tuple(lengthscales),
                             lambdas=tuple(lambdas),seed=int(seed))
    values=np.full((4,4,len(channels)),np.nan)
    reports=[]
    for j,key in enumerate(channels):
        model=models[key]
        values[:,:,j]=np.mean(model.hessian(Z),axis=0)
        ytest=channels[key][split["test"]]
        pred=model.predict(Z[split["test"]])
        reports.append(dict(validation_rmse=selected[key]["validation_rmse"],
                            test_rmse=float(np.sqrt(np.mean((pred-ytest)**2))),
                            selected_hyperparameters=selected[key]))
    return values,reports


def run_splines(Z,Y,train_ids,val_ids,test_ids,n_basis=5,
                lambdas=(1e-8,1e-6,1e-4)):
    Z=np.asarray(Z,dtype=float)
    Ymean=np.asarray(Y,dtype=float).mean(axis=1)
    tr=np.asarray(train_ids,dtype=int)
    va=np.asarray(val_ids,dtype=int)
    te=np.asarray(test_ids,dtype=int)
    ranges=np.stack((Z[tr].min(0),Z[tr].max(0)),axis=1)
    if np.any(ranges[:,1]<=ranges[:,0]):raise ValueError("Constant Z dimension")
    design=SplineDesign(n_basis=int(n_basis),degree=3,bounds=ranges)
    basis_train=design.features(Z[tr])
    basis_val=design.features(Z[va])
    m=basis_train.shape[1]
    gram=(basis_train.T@basis_train)/len(tr)
    rhs=(basis_train.T@Ymean[tr])/len(tr)
    penalty=design.roughness()
    penalty.flat[::m+1]+=1e-9
    best=np.full(Ymean.shape[1],np.inf)
    selected=[None]*Ymean.shape[1]
    coefs=np.zeros((m,Ymean.shape[1]))
    for lam in lambdas:
        reg=gram+float(lam)*penalty
        reg.flat[::m+1]+=1e-9
        try:
            C=cho_solve(cho_factor(reg,lower=True,check_finite=False),
                        rhs,check_finite=False)
        except np.linalg.LinAlgError:
            continue
        pred=basis_val@C
        rmse=np.sqrt(np.mean((pred-Ymean[va])**2,axis=0))
        for j,error in enumerate(rmse):
            if error<best[j]:
                best[j]=float(error)
                coefs[:,j]=C[:,j]
                selected[j]=dict(n_basis=int(n_basis),smoothing_penalty=float(lam),
                                 validation_rmse=float(error))
    if any(s is None for s in selected):
        raise RuntimeError("Spline failed for every smoothing parameter")
    # Derivative features are evaluated directly in original Z coordinates.
    H=np.zeros((4,4,Ymean.shape[1]),dtype=float)
    for i in range(4):
        for j in range(i,4):
            dmean=np.zeros(m)
            for start in range(0,len(Z),128):
                end=min(start+128,len(Z))
                dmean+=np.sum(design.features(Z[start:end],(i,j)),axis=0)
            dmean/=len(Z)
            H[i,j]=H[j,i]=dmean@coefs
    test_basis=design.features(Z[te])
    testpred=test_basis@coefs
    reports=[]
    for j,sel in enumerate(selected):
        reports.append(dict(validation_rmse=best[j],
                            test_rmse=float(np.sqrt(np.mean((testpred[:,j]-Ymean[te,j])**2))),
                            selected_hyperparameters=sel,
                            n_total_spline_features=m))
    return H,reports
