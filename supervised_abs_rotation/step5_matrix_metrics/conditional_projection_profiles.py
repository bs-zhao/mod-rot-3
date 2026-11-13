"""Empirical conditional innovation spaces, independent of feature coordinates."""
import numpy as np
import torch


@torch.no_grad()
def basis(matrix,reference_scale=None,tolerance=1e-8):
    if matrix.shape[1]==0:return matrix
    left,values,_=torch.linalg.svd(matrix,full_matrices=False)
    scale=float(torch.linalg.matrix_norm(matrix)) if reference_scale is None else reference_scale
    keep=values>max(scale*tolerance,1e-12)
    return left[:,keep]


@torch.no_grad()
def innovation_basis(feature,context,tolerance=1e-8):
    reference=float(torch.linalg.matrix_norm(feature));lower=basis(context,tolerance=tolerance)
    residual=feature-lower@(lower.T@feature)
    residual-=lower@(lower.T@residual)
    result=basis(residual,reference_scale=reference,tolerance=tolerance)
    orthogonality=float(torch.max(torch.abs(lower.T@result))) if result.shape[1] and lower.shape[1] else 0.
    gram_error=float(torch.max(torch.abs(result.T@result-torch.eye(result.shape[1],dtype=result.dtype,device=result.device)))) if result.shape[1] else 0.
    if orthogonality>1e-6 or gram_error>1e-8:raise ArithmeticError('Conditional innovation space lost orthogonality')
    return result,dict(feature_columns=feature.shape[1],context_columns=context.shape[1],context_rank=lower.shape[1],
        innovation_rank=result.shape[1],maximum_context_inner_product=orthogonality,maximum_basis_Gram_error=gram_error,
        numerical_singular_cutoff=max(reference*tolerance,1e-12),reference_scale='unprojected feature Frobenius norm')


@torch.no_grad()
def mode_energies(orthogonal_basis,target,projections):
    coefficients=orthogonal_basis.T@target/np.sqrt(len(target));values=[];audits=[]
    for j,projection in projections:
        matrix=coefficients@projection;eigen=torch.linalg.svdvals(matrix).square()
        original=float((target@projection).square().sum()/len(target));captured=float(matrix.square().sum())
        error=abs(float(eigen.sum())-captured)/max(captured,1e-30)
        if error>1e-8 or captured>original*(1+1e-8)+1e-12:raise ArithmeticError('Orthogonal projection failed its energy bound or SVD trace')
        values.append((j,eigen.cpu().numpy()))
        audits.append(dict(X_column=j,captured_energy=captured,original_energy=original,
            captured_energy_fraction=captured/max(original,1e-30),relative_SVD_trace_error=error))
    return values,audits


@torch.no_grad()
def press_residuals(context_basis,innovation,target,weights=None):
    """Exact leave-one-row-out errors conditional on the frozen W-only spaces."""
    context_prediction=context_basis@(context_basis.T@target)
    if weights is None:weights=torch.ones(innovation.shape[1],dtype=innovation.dtype,device=innovation.device)
    additional=innovation@(weights[:,None]*(innovation.T@target))
    h0=context_basis.square().sum(1);h1=h0+(innovation.square()*weights[None]).sum(1)
    if float(torch.min(1-h1))<=1e-10:raise ValueError('PRESS is undefined for a saturated observed-row projection')
    lower=(target-context_prediction)/(1-h0[:,None])
    full=(target-context_prediction-additional)/(1-h1[:,None])
    return lower,full,dict(maximum_lower_leverage=float(h0.max()),maximum_full_leverage=float(h1.max()),
        minimum_full_PRESS_denominator=float((1-h1).min()),fixed_feature_map_conditioning=True,
        effective_degrees_of_freedom=float(h1.sum()))


@torch.no_grad()
def press_mode_gains(context_basis,innovation,target,projections):
    lower,full,audit=press_residuals(context_basis,innovation,target);values=[];checks=[]
    for j,p in projections:
        a,b=lower@p,full@p
        matrix=(a.T@a-b.T@b)/len(target);matrix=(matrix+matrix.T)/2
        eigen=torch.linalg.eigvalsh(matrix);positive=torch.clamp(eigen,min=0.).flip(0)
        lower_error=float(a.square().sum()/len(target));full_error=float(b.square().sum()/len(target))
        trace_error=abs(float(eigen.sum())-(lower_error-full_error))/max(lower_error+full_error,1e-30)
        if trace_error>1e-8:raise ArithmeticError('Signed PRESS improvement spectrum did not replay prediction-risk change')
        original=float((target@p).square().sum()/len(target))
        values.append((j,positive.cpu().numpy()))
        checks.append(dict(X_column=j,lower_PRESS_relative_MSE=lower_error/max(original,1e-30),
            full_PRESS_relative_MSE=full_error/max(original,1e-30),signed_risk_improvement=lower_error-full_error,
            positive_gain_energy=float(positive.sum()),negative_gain_energy=float(torch.clamp(-eigen,min=0.).sum()),
            minimum_signed_eigenvalue=float(eigen.min()),maximum_signed_eigenvalue=float(eigen.max()),
            relative_signed_trace_error=trace_error))
    return values,dict(leverage=audit,X_columns=checks,
        interpretation='positive part of fixed-basis leave-one-out error improvement; signed negative spectrum retained in audits')
