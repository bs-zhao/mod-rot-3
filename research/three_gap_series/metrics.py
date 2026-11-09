"""M1 summaries and M3 losses; source and estimator agnostic."""
import numpy as np
import itertools

METRICS = ('uu_off_rms', 'uu_self_rms', 'uu_self_variance', 'uu_self_meanabs',
           'uu_off_q99', 'ux_soft_count', 'ux_row_entropy', 'ux_column_entropy',
           'ux_mean_l1', 'signature_l1', 'signature_plain_l1', 'signature_column_entropy', 'zzx_mixed_rms', 'conditional_soft_count','conditional_ratios','conditional_anchor','conditional_bilinear','conditional_bilinear_hrow')

def entropy(v, axis):
    total=v.sum(axis=axis,keepdims=True)
    p=np.divide(v,total,out=np.zeros_like(v),where=total>1e-28)
    return float(-np.sum(p*np.log(np.maximum(p,1e-30))))

class Loss:
    def __init__(self, fields, covariance, d, metric):
        self.fields=fields; self.cov=covariance; self.d=d; self.metric=metric
        h=fields['h'][:,:d,:d].reshape(len(fields['h']),d*d)
        self.hm=h.mean(0).reshape(d,d)
        self.hh=h.T@h/len(h)
        ux=fields['h'][:,:d,d:]
        self.ux_moment=np.einsum('nap,nbp->pab',ux,ux)/len(h)
        self.uxmean=ux.mean(0)
        sig=[fields['g'][:,:d].mean(0)[:,None], self.uxmean]
        if 'third' in fields:
            sig.append(fields['third'][:,:d,d:,d:].mean(0).reshape(d,-1))
        self.signature=np.concatenate(sig,axis=1)
        if 'groups' in fields:
            self.group_gram=np.array([fields['g'][fields['groups']==k,:d].T@fields['g'][fields['groups']==k,:d]/np.sum(fields['groups']==k)
                                      for k in np.unique(fields['groups'])])
            self.gradient_gram=fields['g'][:,:d].T@fields['g'][:,:d]/len(fields['g'])

    def __call__(self, r):
        d=self.d
        # r can also be D R or a signed permutation, for invariance tests.
        sd=np.sqrt(np.diag(r@self.cov@r.T))
        if sd.min() < 1e-14: raise ValueError('Collapsed candidate axis')
        # derivatives in v=r z: h_v = inv(r).T h_z inv(r)
        a=sd[:,None]*np.linalg.inv(r).T
        kind=self.metric
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
