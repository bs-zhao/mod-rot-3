"""Trace-preserving profile lift: sign reversals add functional modes.

For derivative response profile g at each observed U, Psi(g)=(g,||g||)/sqrt(2).
Its squared singular spectrum is invariant to a global axis sign and any
orthogonal output-mode basis. Rank one means all nonzero derivative profiles
lie on the same positive ray. This is not a monotone component-count theorem
for ranks greater than one.
"""
import numpy as np
import torch


@torch.no_grad()
def spectra(gradient,directions,chunk=4):
    """gradient[N,U,mode], directions[K,U]; return two K-by-(mode+1) arrays."""
    if gradient.ndim!=3 or np.asarray(directions).shape[1]!=gradient.shape[1]:
        raise ValueError('Expected observed-subject gradient profiles and matching directions')
    result={key:np.zeros((len(directions),gradient.shape[-1]+1)) for key in ('ordinary','norm_lift')}
    errors=[]
    for a in range(0,len(directions),chunk):
        axes=torch.tensor(np.asarray(directions)[a:a+chunk],dtype=gradient.dtype,device=gradient.device)
        profiles=torch.einsum('nar,ka->knr',gradient,axes)
        lifted=torch.cat((profiles,torch.linalg.vector_norm(profiles,dim=-1,keepdim=True)),dim=-1)/np.sqrt(2.)
        energy=profiles.square().sum((1,2))/len(gradient)
        for key,values in (('ordinary',profiles),('norm_lift',lifted)):
            eigen=torch.linalg.svdvals(values).square()/len(gradient)
            result[key][a:a+len(axes),:eigen.shape[1]]=eigen.cpu().numpy()
            errors.append(float(torch.max(torch.abs(eigen.sum(1)-energy)/energy.clamp_min(1e-30))))
    maximum=max(errors,default=0.)
    if maximum>1e-8:raise ArithmeticError('Profile lift SVD did not preserve ordinary gradient energy')
    return result,maximum
