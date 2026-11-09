"""M1 summaries and M3 losses; source and estimator agnostic."""
import numpy as np

METRICS = ('uu_off_rms', 'uu_self_rms', 'uu_self_variance', 'uu_self_meanabs',
           'uu_off_q99', 'ux_soft_count', 'ux_row_entropy', 'ux_column_entropy',
           'ux_mean_l1', 'signature_l1', 'signature_column_entropy', 'zzx_mixed_rms')

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

    def __call__(self, r):
        d=self.d
        # r can also be D R or a signed permutation, for invariance tests.
        sd=np.sqrt(np.diag(r@self.cov@r.T))
        if sd.min() < 1e-14: raise ValueError('Collapsed candidate axis')
        # derivatives in v=r z: h_v = inv(r).T h_z inv(r)
        a=sd[:,None]*np.linalg.inv(r).T
        kind=self.metric
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
            # Each effect column carries equal weight; all-zero columns excluded.
            norm=np.linalg.norm(v,axis=0)
            v=v[:,norm>1e-10]/norm[norm>1e-10]
            if kind=='signature_l1': return float(abs(v).sum())
            return entropy(v*v,0)
        if kind=='zzx_mixed_rms':
            h=self.fields['third'][:,:d,:d,d:]
            u=np.einsum('ia,nabp,jb->nijp',a,h,a,optimize=True)
            return float(np.sqrt(np.mean(u*u,axis=0))[~np.eye(d,dtype=bool)].sum())
        raise ValueError(kind)
