"""Replay a frozen affine-plus-kernel regression and audit fixed-hyperparameter PRESS."""
import torch

import conditional_projection_profiles as projection


@torch.no_grad()
def audit(design,coefficient,target,linear_count,ridge):
    affine,nonlinear=design[:,:linear_count],design[:,linear_count:]
    inverse=torch.linalg.pinv(affine,rtol=1e-10)
    residual=nonlinear-affine@(inverse@nonlinear)
    left,singular,_=torch.linalg.svd(residual,full_matrices=False)
    scale=max(float(residual.square().sum())/max(residual.shape[1],1),1e-12)
    tau=ridge*scale;weights=singular.square()/(singular.square()+tau)
    lower=projection.basis(affine,tolerance=1e-10)
    prediction=lower@(lower.T@target)+left@(weights[:,None]*(left.T@target))
    frozen_prediction=design@coefficient
    replay=float((prediction-frozen_prediction).square().sum()/target.square().sum().clamp_min(1e-30))
    if replay>1e-10:raise ArithmeticError('Frozen conditional coefficients disagree with their ridge hat matrix')
    _,PRESS,leverage=projection.press_residuals(lower,left,target,weights=weights)
    training=target-frozen_prediction
    return training,PRESS,dict(relative_squared_prediction_replay_error=replay,ridge=ridge,ridge_matrix_penalty=tau,
        **leverage,validation_scope='PRESS conditional on frozen W-only features, response basis and selected ridge; selection is not nested')
