"""M1 summaries and M3 losses; source and estimator agnostic."""
import numpy as np
import itertools
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

METRICS = ('uu_off_rms', 'uu_self_rms', 'uu_self_variance', 'uu_self_meanabs',
           'uu_off_q99', 'ux_soft_count', 'ux_row_entropy', 'ux_column_entropy',
           'ux_mean_l1', 'signature_l1', 'signature_plain_l1', 'signature_column_entropy', 'zzx_mixed_rms', 'conditional_soft_count','conditional_ratios','conditional_anchor','conditional_bilinear','conditional_bilinear_hrow','conditional_relative')

COMBINATIONS={f'combo_{stat}_t{tau:g}_w{weight:g}':(stat,tau,weight)
              for stat in ['rms','sd'] for tau in [.03,.1,.3,1.] for weight in [.25,1.,4.]}
UX_VARIANTS={f'ux_{stat}_{score}':(stat,score) for stat in ['rms','sd','absmean']
             for score in ['l1','gini','column_entropy','row_entropy']}
METRICS=METRICS+tuple(COMBINATIONS)+tuple(UX_VARIANTS)
METRICS=METRICS+('conditional_relative_directions','conditional_relative_dual','conditional_relative_cumulant')
METRICS=METRICS+('transport_profile','transport_concentration','transport_dual','transport_dual_cosine')
METRICS=METRICS+('conditional_fingerprint',)

def entropy(v, axis):
    total=v.sum(axis=axis,keepdims=True)
    p=np.divide(v,total,out=np.zeros_like(v),where=total>1e-28)
    return float(-np.sum(p*np.log(np.maximum(p,1e-30))))

class Loss:
    def __init__(self, fields, covariance, d, metric):
        self.fields=fields; self.cov=covariance; self.d=d; self.metric=metric
        self.has_hessian='h' in fields
        if not self.has_hessian and not metric.startswith('transport_'):raise ValueError('This loss requires F2')
        raw_h=fields['h'] if self.has_hessian else np.zeros((len(fields['g']),d,d))
        weights=fields.get('weights',np.ones(len(raw_h)))
        weights=weights/weights.sum()
        h=raw_h[:,:d,:d].reshape(len(raw_h),d*d)
        self.hm=(weights@h).reshape(d,d)
        self.hh=h.T@(h*weights[:,None])
        ux=raw_h[:,:d,d:]
        self.ux_moment=np.einsum('n,nap,nbp->pab',weights,ux,ux)
        self.uxmean=np.einsum('n,nap->ap',weights,ux)
        sig=[(weights@fields['g'][:,:d])[:,None], self.uxmean]
        if 'third' in fields:
            sig.append(np.einsum('n,nabc->abc',weights,fields['third'][:,:d,d:,d:]).reshape(d,-1))
        self.signature=np.concatenate(sig,axis=1)
        if 'groups' in fields:
            self.group_gram=np.array([fields['g'][fields['groups']==k,:d].T@(fields['g'][fields['groups']==k,:d]*weights[fields['groups']==k,None])/weights[fields['groups']==k].sum()
                                      for k in np.unique(fields['groups'])])
            self.gradient_gram=fields['g'][:,:d].T@(fields['g'][:,:d]*weights[:,None])
        if metric in ['conditional_relative_directions','conditional_relative_dual','conditional_relative_cumulant']:
            def null_direction(matrix,reference):
                ridge=1e-8*max(np.trace(reference)/d,1e-30)
                values,vectors=eigh(matrix,reference+ridge*np.eye(d))
                vector=vectors[:,0];return vector/np.linalg.norm(vector),values
            lam,el=null_direction(self.group_gram[0],self.gradient_gram)
            gamma,eg=null_direction(self.group_gram[1],self.gradient_gram)
            tau,et=null_direction(fields['relative_cov'],fields['relative_gram'])
            forward=np.column_stack((lam,gamma,tau))
            u,_,_=np.linalg.svd(forward,full_matrices=True)
            alpha_covector=u[:,-1]
            # EXTRA assumption: independent underlying factors and covariance
            # completion. This is not a pure decoder-Hessian loss.
            alpha_forward=self.cov@alpha_covector
            mixing=np.column_stack((alpha_forward,lam,gamma,tau))
            mixing/=np.linalg.norm(mixing,axis=0)
            if np.linalg.cond(mixing)>1e8:raise ValueError('Inferred factor directions are degenerate')
            dual=np.linalg.inv(mixing);dual/=np.linalg.norm(dual,axis=1,keepdims=True)
            self.direction_targets=forward.T if metric.endswith('directions') else dual
            self.direction_diagnostics=dict(forward_directions=forward,completed_mixing=mixing,
                dual_covectors=dual,absence_gain_eigenvalues=el,absence_neutral_eigenvalues=eg,relative_eigenvalues=et,
                assumption='Covariance completion assumes independent factors; additional geometric prior, no T labels')
            if metric=='conditional_relative_cumulant':
                from cumulants import three_skew_directions
                non_gaussian,info=three_skew_directions(fields['latent_train'])
                mixing=np.column_stack((non_gaussian,tau))
                if np.linalg.cond(mixing)>1e8:raise ValueError('Skew directions and scale direction are degenerate')
                dual=np.linalg.inv(mixing);dual/=np.linalg.norm(dual,axis=1,keepdims=True)
                self.direction_targets=dual
                self.direction_diagnostics=dict(cumulant_control=info,completed_mixing=mixing,
                                                dual_covectors=dual,relative_eigenvalues=et)
        if metric in ['transport_dual','transport_dual_cosine']:
            energy,vectors=np.linalg.eigh(fields['transport_vector_grams'])
            mixing=vectors[:,:,-1].T
            if np.linalg.cond(mixing)>1e8:raise ValueError('Transport directions do not span the latent space')
            dual=np.linalg.inv(mixing);dual/=np.linalg.norm(dual,axis=1,keepdims=True)
            self.transport_dual=dual
            self.direction_diagnostics=dict(transport_forward_directions=mixing,
                transport_dual_covectors=dual,condition_number=np.linalg.cond(mixing),
                direction_rank_one_fraction=energy[:,-1]/np.maximum(energy.sum(1),1e-30),
                assumption='All four input-symmetry transport directions estimated, then inverted; no independence/covariance completion prior')

    def describe(self,r):
        d=self.d;sd=np.sqrt(np.diag(r@self.cov@r.T));a=sd[:,None]*np.linalg.inv(r).T
        k=np.einsum('ia,jb->ijab',a,a).reshape(d*d,d*d)
        square=np.maximum(np.einsum('ia,ab,ib->i',k,self.hh,k),0).reshape(d,d)
        mean=a@self.hm@a.T
        result=dict(candidate_sd=sd,zz_signed_mean=mean,zz_rms=np.sqrt(square),
                    zz_variance=np.maximum(square-mean*mean,0),self_rms=np.sqrt(np.diag(square)),
                    ux_signed_mean=a@self.uxmean,
                    ux_rms=np.sqrt(np.maximum(np.einsum('ia,pab,ib->ip',a,self.ux_moment,a,optimize=True),0)),
                    signed_effect_signature=a@self.signature)
        if hasattr(self,'group_gram'):
            result['group_gradient_mean_square']=np.maximum(np.einsum('ia,kab,ib->ki',a,self.group_gram,a,optimize=True),0)
        if 'relative_cov' in self.fields:
            result['relative_gradient_covariance']=a@self.fields['relative_cov']@a.T
            result['relative_gradient_gram']=a@self.fields['relative_gram']@a.T
        if hasattr(self,'direction_diagnostics'):result['inferred_direction_diagnostics']=self.direction_diagnostics
        if not self.has_hessian:
            for key in ['zz_signed_mean','zz_rms','zz_variance','self_rms','ux_signed_mean','ux_rms']:result.pop(key)
            result['capabilities']='F1 only; Hessian blocks unavailable, not zero'
        if self.metric.startswith('transport_'):
            result['transport_vector_grams']=self.fields['transport_vector_grams']
        if self.metric=='conditional_fingerprint':
            cross=r@self.fields['fingerprint_cross']
            result['algebraic_fingerprint_correlation']=cross/np.sqrt(np.diag(r@self.cov@r.T)[:,None]*self.fields['fingerprint_variance'][None])
        return result

    def __call__(self, r):
        d=self.d
        # r can also be D R or a signed permutation, for invariance tests.
        sd=np.sqrt(np.diag(r@self.cov@r.T))
        if sd.min() < 1e-14: raise ValueError('Collapsed candidate axis')
        # derivatives in v=r z: h_v = inv(r).T h_z inv(r)
        a=sd[:,None]*np.linalg.inv(r).T
        kind=self.metric
        if kind in ['transport_dual','transport_dual_cosine']:
            if kind=='transport_dual_cosine':
                normalized=r/np.linalg.norm(r,axis=1,keepdims=True)
                similarity=(normalized@self.transport_dual.T)**2
            else:
                cross=r@self.cov@self.transport_dual.T
                v1=np.diag(r@self.cov@r.T);v2=np.diag(self.transport_dual@self.cov@self.transport_dual.T)
                similarity=np.clip(cross*cross/np.maximum(v1[:,None]*v2[None,:],1e-30),0,1)
            i,j=linear_sum_assignment(-similarity)
            return float(len(j)-similarity[i,j].sum())
        if kind=='conditional_fingerprint':
            cross=r@self.fields['fingerprint_cross'];axis=np.diag(r@self.cov@r.T)
            corr2=np.clip(cross*cross/np.maximum(axis[:,None]*self.fields['fingerprint_variance'][None],1e-30),0,1)
            absence=np.einsum('ia,ab,ib->i',a,self.group_gram[1],a)/np.maximum(np.einsum('ia,ab,ib->i',a,self.gradient_gram,a),1e-30)
            similarity=np.column_stack((corr2,1-absence))
            i,j=linear_sum_assignment(-similarity)
            return float(len(j)-similarity[i,j].sum())
        if kind=='transport_profile':
            axis_energy=np.einsum('ia,nab,ib->ni',a,self.fields['transport_gram'],a,optimize=True)
            cross=np.einsum('ia,nak->nik',a,self.fields['transport_cross'],optimize=True)
            similarity=np.mean(np.clip(cross*cross/np.maximum(axis_energy[:,:,None]*self.fields['transport_energy'][:,None,:],1e-30),0,1),axis=0)
            i,j=linear_sum_assignment(-similarity)
            return float(len(j)-similarity[i,j].sum())
        if kind=='transport_concentration':
            av=r/sd[:,None]
            energy=np.maximum(np.einsum('ia,kab,ib->ik',av,self.fields['transport_vector_grams'],av,optimize=True),0)
            similarity=energy/np.maximum(energy.sum(0,keepdims=True),1e-30)
            i,j=linear_sum_assignment(-similarity)
            return float(len(j)-similarity[i,j].sum())
        if kind in ['conditional_relative_directions','conditional_relative_dual','conditional_relative_cumulant']:
            normalized=a/np.linalg.norm(a,axis=1,keepdims=True)
            similarity=(normalized@self.direction_targets.T)**2
            i,j=linear_sum_assignment(-similarity)
            return float(len(j)-similarity[i,j].sum())
        if kind=='conditional_relative':
            energy=np.maximum(np.einsum('ia,kab,ib->ki',a,self.group_gram,a,optimize=True),0)
            total=np.maximum(np.einsum('ia,ab,ib->i',a,self.gradient_gram,a),1e-30)
            cv=np.maximum(np.einsum('ia,ab,ib->i',a,self.fields['relative_cov'],a),0)
            gram=np.maximum(np.einsum('ia,ab,ib->i',a,self.fields['relative_gram'],a),1e-30)
            return float(min(energy[0,i]/total[i]+energy[1,j]/total[j]+cv[k]/gram[k]
                             for i,j,k in itertools.permutations(range(d),3)))
        if kind in COMBINATIONS or kind in UX_VARIANTS:
            raw=np.maximum(np.einsum('ia,pab,ib->ip',a,self.ux_moment,a,optimize=True),0)
            stat=(COMBINATIONS.get(kind) or UX_VARIANTS[kind])[0]
            if stat=='sd': raw=np.maximum(raw-(a@self.uxmean)**2,0)
            if stat=='absmean':raw=(a@self.uxmean)**2
            if kind in COMBINATIONS:
                _,tau,weight=COMBINATIONS[kind]
                k=np.einsum('ia,ib->iab',a,a).reshape(d,d*d)
                diagonal=np.maximum(np.einsum('ia,ab,ib->i',k,self.hh,k),0)
                if stat=='sd':diagonal=np.maximum(diagonal-np.diag(a@self.hm@a.T)**2,0)
                def count(v): return float(np.mean(v/(v+tau*tau*max(v.mean(),1e-30))))
                return count(raw)+weight*count(diagonal)
            score=UX_VARIANTS[kind][1]
            if score=='l1':return float(np.sqrt(raw).sum())
            if score=='row_entropy':return entropy(raw,1)
            if score=='column_entropy':return entropy(raw,0)
            total=raw.sum(0,keepdims=True)
            prob=np.divide(raw,total,out=np.zeros_like(raw),where=total>1e-28)
            return float(np.sum(1-np.sum(prob*prob,axis=0)))
        if kind=='conditional_anchor':
            energy=np.maximum(np.einsum('ia,kab,ib->ki',a,self.group_gram,a,optimize=True),0)
            energy=energy/np.maximum(energy.sum(axis=1,keepdims=True),1e-30)
            return float(min(4-energy[0,tau]-(energy[1,tau]+energy[1,lam])-(energy[2,tau]+energy[2,alpha])-(energy[3,tau]+energy[3,gamma])
                             for tau,lam,alpha,gamma in itertools.permutations(range(d))))
        if kind in ['conditional_ratios','conditional_bilinear','conditional_bilinear_hrow']:
            energy=np.maximum(np.einsum('ia,kab,ib->ki',a,self.group_gram,a,optimize=True),0)
            total=np.maximum(np.einsum('ia,ab,ib->i',a,self.gradient_gram,a),1e-30)
            k=np.einsum('ia,ib->iab',a,a).reshape(d,d*d)
            self_energy=np.maximum(np.einsum('ia,ab,ib->i',k,self.hh,k),0)
            absence_gain=energy[0]/total
            absence_neutral=energy[1]/total
            linearity=self_energy/total
            if kind=='conditional_bilinear_hrow':
                kk=np.einsum('ia,jb->ijab',a,a).reshape(d*d,d*d)
                all_energy=np.maximum(np.einsum('ia,ab,ib->i',kk,self.hh,kk),0).reshape(d,d)
                linearity=self_energy/np.maximum(all_energy.sum(1),1e-30)
            if kind.startswith('conditional_bilinear'):
                return float(min(absence_gain[i]+absence_neutral[j]+linearity[i]+linearity[k] for i,j,k in itertools.permutations(range(d),3)))
            return float(min(absence_gain[i]+absence_neutral[j]+linearity[k] for i,j,k in itertools.permutations(range(d),3)))
        if kind=='conditional_soft_count':
            energies=np.maximum(np.einsum('ia,kab,ib->ki',a,self.group_gram,a,optimize=True),0)
            score=np.sum(energies/(energies+1e-3*energies.mean(axis=1,keepdims=True)+1e-30))
            k=np.einsum('ia,ib->iab',a,a).reshape(d,d*d)
            diag=np.maximum(np.einsum('ia,ab,ib->i',k,self.hh,k),0)
            return float(score+np.sum(diag/(diag+1e-3*diag.mean()+1e-30)))
        if kind.startswith('uu_'):
            k=np.einsum('ia,jb->ijab',a,a).reshape(d*d,d*d)
            energy=np.maximum(np.einsum('ia,ab,ib->i',k,self.hh,k),0).reshape(d,d)
            diag=np.diag(energy)
            off=~np.eye(d,dtype=bool)
            if kind=='uu_off_rms': return float(np.sqrt(energy[off]).sum())
            if kind=='uu_self_rms': return float(np.sqrt(diag).sum())
            if kind=='uu_self_variance':
                means=np.diag(a@self.hm@a.T)
                return float(np.maximum(diag-means**2,0).sum())
            local=np.einsum('ia,nab,jb->nij',a,self.fields['h'][:,:d,:d],a,optimize=True)
            if kind=='uu_self_meanabs': return float(np.abs(np.diagonal(local,axis1=1,axis2=2)).mean(0).sum())
            return float(np.quantile(abs(local[:,off]),.99,axis=0).sum())
        if kind.startswith('ux_'):
            energy=np.maximum(np.einsum('ia,pab,ib->ip',a,self.ux_moment,a,optimize=True),0)
            if kind=='ux_row_entropy': return entropy(energy,1)
            if kind=='ux_column_entropy': return entropy(energy,0)
            if kind=='ux_mean_l1': return float(abs(a@self.uxmean).sum())
            scale=energy.mean()
            return float((energy/(energy+.01*scale+1e-30)).sum())
        if kind.startswith('signature_'):
            v=a@self.signature
            if kind=='signature_plain_l1': return float(abs(v).sum())
            # Each effect column carries equal weight; all-zero columns excluded.
            norm=np.linalg.norm(v,axis=0)
            # Avoid assigning full weight to noise in mathematically zero effects.
            floor=.01*max(norm.max(),1e-20)
            v=v/np.maximum(norm,floor)
            if kind=='signature_l1': return float(abs(v).sum())
            prob=np.divide(v*v,np.sum(v*v,axis=0,keepdims=True),out=np.zeros_like(v),where=np.sum(v*v,axis=0,keepdims=True)>1e-28)
            return float(-np.sum(prob*np.log(np.maximum(prob,1e-30))*np.minimum(norm/floor,1.)[None,:]**2))
        if kind=='zzx_mixed_rms':
            h=self.fields['third'][:,:d,:d,d:]
            u=np.einsum('ia,nabp,jb->nijp',a,h,a,optimize=True)
            return float(np.sqrt(np.mean(u*u,axis=0))[~np.eye(d,dtype=bool)].sum())
        raise ValueError(kind)
