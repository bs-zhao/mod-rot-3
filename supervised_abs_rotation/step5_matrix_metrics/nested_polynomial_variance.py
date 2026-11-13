"""Coherent nested polynomial-kernel explained-covariance Shapley profiles.

All subspaces share full-input degree-block scales and one absolute ridge.
Nested tensor feature kernels imply H_S <= H_T for S contained in T.
This is a regularized feature-space diagnostic, not oracle Sobol variance.
"""
import hashlib
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import run as wave
from adaptive_coalition_models import subspace_key


class NestedPolynomialVariance:
    def __init__(self,z,target,output,manifest,device='cuda',degrees=(2,3,5),cache_limit=None):
        self.z,self.target,self.output,self.device=z,target,Path(output),device
        self.output.mkdir(parents=True,exist_ok=True);self.models=OrderedDict();self.audits=[];self.cache_limit=cache_limit
        n=len(z);maximum=max(degrees);radius=np.sum(z*z,axis=1)
        self.scales={k:max(float(np.mean(radius**k))**.5,1e-12) for k in range(1,maximum+1)}
        self.observed=target.T@target/n
        self.target_variance=float(torch.trace(self.observed));options=[];fits=[]
        for degree in degrees:
            design=self.design(np.eye(z.shape[1]),degree)
            gram=design.T@design;cross=design.T@target
            eigen,axes=torch.linalg.eigh((gram+gram.T)/2);eigen=eigen.clamp_min(0.)
            rhs=axes.T@cross;scale=float(torch.trace(gram))/max(len(gram),1)
            for relative_ridge in (1e-8,1e-6,1e-4,.01,.1):
                tau=max(relative_ridge*scale,1e-15)
                coefficient=axes@(rhs/(eigen[:,None]+tau))
                prediction=design@coefficient
                error=float((prediction-target).square().sum())/n
                df=float((eigen/(eigen+tau)).sum())
                options.append(dict(degree=degree,relative_ridge=relative_ridge,absolute_ridge=tau,
                    effective_degrees_of_freedom=df,training_MSE_relative=error/max(self.target_variance,1e-30),
                    GCV_relative=error/max(self.target_variance,1e-30)/max(1-df/n,1e-8)**2))
                fits.append((coefficient.cpu().numpy(),
                    (rhs.T@(rhs/(eigen[:,None]+tau))/n).cpu().numpy()))
        selected=int(np.argmin([row['GCV_relative'] for row in options]))
        self.degree=options[selected]['degree'];self.ridge=options[selected]['absolute_ridge']
        self.full=torch.tensor(fits[selected][1],dtype=torch.float64,device=device)
        self.unassigned=(self.observed-self.full);self.unassigned=(self.unassigned+self.unassigned.T)/2
        full_design=self.design(np.eye(z.shape[1]),self.degree)
        coefficient=torch.tensor(fits[selected][0],dtype=torch.float64,device=device)
        inverse=torch.linalg.solve(full_design.T@full_design+self.ridge*torch.eye(full_design.shape[1],device=device,dtype=torch.float64),full_design.T)
        leverage=(full_design*inverse.T).sum(1)
        self.training_residual=target-full_design@coefficient
        self.full_PRESS=self.training_residual/(1-leverage)[:,None]
        np.savez_compressed(self.output/'full_input_model.npz',coefficient=fits[selected][0],degree=self.degree,
            absolute_ridge=self.ridge,degree_block_scales=np.array([self.scales[k] for k in range(1,maximum+1)]))
        self.full_quality=dict(options=options,selected_index=selected,
            fixed_hyperparameter_PRESS_relative_MSE=float(self.full_PRESS.square().sum())/max(float(target.square().sum()),1e-30),
            maximum_leverage=float(leverage.max()))
        wave.json_file(self.output/'full_input_model_quality.json',self.full_quality)
        self.metric_prefix='nested_polynomial_variance';self.spectrum_key='variance_eigenvalues'
        self.protocol_info=dict(source_matrix_manifest_sha256=manifest,degree=self.degree,radial_centers=0,
            decomposition='exact Shapley allocation of Y.T H_S Y/N; separate Y.T (I-H_full) Y/N unassigned covariance',
            value_function='regularized explained covariance Y.T H_S Y/N for every input coalition; zero for empty coalition',
            interpretation='regularized nested feature-space explained covariance, not calibrated population Shapley or Sobol effects',
            conditional_model='one isotropic polynomial kernel; full-input GCV chooses degree and ridge once',
            degree_options=list(degrees),full_input_degree_block_scales=self.scales,absolute_ridge=self.ridge,
            per_coalition_hyperparameter_selection=False,independence_of_inputs_assumed=False,
            unresolved_covariance_retained=True,observed_covariance_conserved=True,
            numerical_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    @torch.no_grad()
    def response_audits(self,projections,tasks,channels):
        rows=[]
        for p,(c,j) in zip(projections,tasks):
            total=float((self.target@p).square().sum())
            rows.append(dict(channel=channels[c],X_column=j,
                full_model_training_relative_MSE=float((self.training_residual@p).square().sum())/max(total,1e-30),
                full_model_fixed_hyperparameter_PRESS_relative_MSE=float((self.full_PRESS@p).square().sum())/max(total,1e-30),
                unassigned_regularized_covariance_trace_fraction=float(torch.trace(p.T@self.unassigned@p))*len(self.z)/max(total,1e-30)))
        return rows

    def design(self,basis,degree):
        rest=self.z@basis;powers=wave.sur.powers_for(rest.shape[1],degree)[1:]
        x=wave.sur.monomials(rest,powers,wave.sur.weights_for(powers))
        x/=np.array([self.scales[int(order)] for order in powers.sum(1)])[None]
        x-=x.mean(0)
        return torch.tensor(x,dtype=torch.float64,device=self.device)

    @torch.no_grad()
    def covariance(self,axes):
        basis=np.linalg.qr(axes)[0][:,:axes.shape[1]];key=subspace_key(basis)
        if key not in self.models:
            x=self.design(basis,self.degree);gram=x.T@x;cross=x.T@self.target
            coefficient=torch.linalg.solve((gram+gram.T)/2+self.ridge*torch.eye(len(gram),dtype=torch.float64,device=self.device),cross)
            covariance=cross.T@coefficient/len(self.z);covariance=(covariance+covariance.T)/2
            self.models[key]=covariance
            self.audits.append(dict(input_dimension=key[0],projector_key=key[1],features=x.shape[1],
                explained_trace=float(torch.trace(covariance)),absolute_ridge=self.ridge))
            if self.cache_limit is not None:
                while len(self.models)>self.cache_limit:self.models.popitem(last=False)
        else:self.models.move_to_end(key)
        return self.models[key]

    @torch.no_grad()
    def frame_covariances(self,q):
        d=q.shape[1];values={0:torch.zeros_like(self.full),2**d-1:self.full}
        for mask in range(1,2**d-1):
            values[mask]=self.covariance(q[:,[i for i in range(d) if mask&(1<<i)]])
        effects=torch.zeros((d+1,*self.full.shape),dtype=torch.float64,device=self.device)
        smallest=0.
        for i in range(d):
            for mask in range(2**d):
                if mask&(1<<i):continue
                increment=values[mask|(1<<i)]-values[mask]
                smallest=min(smallest,float(torch.linalg.eigvalsh(increment).min()))
                effects[i]+=increment/(d*math.comb(d-1,mask.bit_count()))
        effects[-1]=self.unassigned
        error=float((effects.sum(0)-self.observed).square().sum()/self.observed.square().sum().clamp_min(1e-30))
        negative=min(smallest,float(torch.linalg.eigvalsh(effects[-1]).min()))
        if negative < -1e-8*max(self.target_variance,1e-30) or error>1e-18:
            raise ArithmeticError('Nested covariance allocation lost PSD monotonicity or conservation')
        return effects,dict(relative_squared_covariance_conservation_error=error,
            smallest_coalition_increment_eigenvalue=smallest,smallest_unassigned_eigenvalue=float(torch.linalg.eigvalsh(self.unassigned).min()))

    @torch.no_grad()
    def frame_spectra(self,q,projections,tasks,channels,dx):
        effects,audit=self.frame_covariances(q);width=max(p.shape[1] for p in projections)
        spectra=np.zeros((channels,len(effects),dx,width));negative_mass=0.
        for p,(c,j) in zip(projections,tasks):
            covariance=p.T@effects@p;covariance=(covariance+covariance.transpose(-1,-2))/2
            eigen=torch.linalg.eigvalsh(covariance).flip(-1)
            negative_mass+=float((-eigen.clamp_max(0.)).sum())
            spectra[c,:,j,:eigen.shape[1]]=eigen.clamp_min(0.).cpu().numpy()
        audit.update(clipped_roundoff_negative_mass=negative_mass,cached_covariance_subspaces=len(self.models))
        return spectra,audit

    def finalize(self):
        return dict(cached_covariance_subspaces=len(self.models),covariance_fit_evaluations=len(self.audits),full_input_model_quality=self.full_quality)
