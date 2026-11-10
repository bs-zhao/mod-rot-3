"""Model-blind M1/M3 catalogue: Hessian blocks and their variation only.

No model names, labels, cognitive templates, gradients or response values.
Every candidate ID has the same formula and constants in every dimension.
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

CATALOG={}


def register(name,**spec):
    CATALOG['hs_'+name]=dict(order=2,**spec)


for stat in ['rms','sd','signed','local']:
    for axis in ['row','column']:
        for k in [1,2]:
            for score in ['tail1','tail2sq','relative_tail2']:
                register(f'ux_{stat}_{axis}_{score}_k{k}',family='ux',stat=stat,axis=axis,score=score,k=k)
    for score in ['l1','lp05','soft003','soft01','cap01','log01','entropy']:
        register(f'ux_{stat}_{score}',family='ux',stat=stat,axis='all',score=score)
for part in ['self','off','all']:
    for stat in ['rms','sd','signed','local']:
        for score in ['l1','lp05','soft01','log01']:
            register(f'uu_{part}_{stat}_{score}',family='uu',part=part,stat=stat,score=score)
for degree in [1,2]:
    for score in ['l1','lp05','soft003','soft01','log01','tail1']:
        register(f'ux_profile{degree}_{score}',family='profile',degree=degree,score=score,axis='column',k=1)
for ux in ['rms','signed','profile1','profile2']:
    for part in ['self','off']:
        for weight in [.01,.1,1.,10.,100.]:
            register(f'joint_{ux}_{part}_w{weight:g}',family='joint',ux=ux,part=part,weight=weight,score='l1')
for ux in ['rms','profile1','profile2']:
    for kind in ['self','mixed']:
        for weight in [.01,.1,1.,10.]:
            register(f'joint_{ux}_third{kind}_w{weight:g}',family='third_joint',ux=ux,kind=kind,weight=weight,score='l1')
            CATALOG['hs_'+f'joint_{ux}_third{kind}_w{weight:g}']['order']=3
for ux in ['rms','profile1','profile2']:
    for part in ['self','off','all']:
        for score in ['soft003','soft01','log01','lp05']:
            for weight in [.1,1.,10.]:
                register(f'joint_{ux}_localuu_{part}_{score}_w{weight:g}',family='joint_sparse',ux=ux,part=part,weight=weight,score=score)
for degree in [1,2]:
    for strength in [0.,1.]:
        for self_score in ['l1','soft003']:
            for weight in [0.,.1,1.,10.]:
                register(f'jd_profile{degree}_strength{strength:g}_self{self_score}_w{weight:g}',
                    family='jd',degree=degree,strength=strength,self_score=self_score,weight=weight)
for kind in ['self','mixed','repeat']:
    for score in ['l1','lp05','soft01']:
        name=f'uu_third_{kind}_{score}'
        register(name,family='third',kind=kind,score=score)
        CATALOG['hs_'+name]['order']=3
# Added only after h04 single-loss complementarity and a feasible common
# analytic weight interval were measured; this is not another weight grid.
register('jd_profile1_strength0_selfl1_w8',family='jd',degree=1,strength=0.,self_score='l1',weight=8.)
# h12 single-loss evidence: a common sufficient plane interval [81.82,98.49]
# supports its rounded geometric midpoint, 90, before the new joint G1 test.
register('jd_profile1_strength0_repeat_w90',family='jd',degree=1,strength=0.,
         self_score='l1',weight=90.,curvature_component='repeat')
CATALOG['hs_jd_profile1_strength0_repeat_w90']['order']=3
for degree in [1,2]:
    for strength in [0.,1.]:
        register(f'jd_mean_profile{degree}_strength{strength:g}',family='jd',degree=degree,
            strength=strength,self_score='l1',weight=0.,aggregation='mean_then_outer')
for component in ['self','modulation']:
    for score in ['soft','cap','log']:
        for threshold in [.03,.1]:
            name=f'nonlinearity_{component}_{score}_t{threshold:g}'
            register(name,family='curvature_support',component=component,score=score,threshold=threshold)
            if component=='modulation':CATALOG['hs_'+name]['order']=3
# Evidence from h16: these capped supports have exactly the missing five
# plane preferences and no alpha/gamma preference. Coefficients are twice
# the sufficient common lower bounds from the single-loss curves, rounded.
register('jd_profile1_strength0_selfcap03_w9',family='jd',degree=1,strength=0.,self_score='cap',
    weight=9.,curvature_component='support_self',curvature_threshold=.03)
register('jd_profile1_strength0_modcap03_w144',family='jd',degree=1,strength=0.,self_score='cap',
    weight=144.,curvature_component='support_modulation',curvature_threshold=.03)
CATALOG['hs_jd_profile1_strength0_modcap03_w144']['order']=3
register('jd_profile1_strength0_selfcap03_w20',family='jd',degree=1,strength=0.,self_score='cap',
    weight=20.,curvature_component='support_self',curvature_threshold=.03)
for part in ['entries','rows']:
    for score in ['cap','soft','log']:
        for threshold in [.03,.1]:
            register(f'uu_graph_{part}_{score}_t{threshold:g}',family='uu_graph',part=part,score=score,threshold=threshold)
# Single penalties on the U-related Hessian row matrix [H_UU | H_UX].
# There are no separately weighted component losses in this family.
for stat in ['signed','rms','sd','local']:
    for score in ['l1','lp05','soft003','soft01','cap01','log01','entropy']:
        register(f'full_{stat}_{score}',family='full',stat=stat,axis='row',score=score)
    for axis in ['row','column']:
        for k in [1,2]:
            for score in ['tail1','tail2sq','relative_tail2']:
                register(f'full_{stat}_{axis}_{score}_k{k}',family='full',stat=stat,axis=axis,k=k,score=score)
for part in ['self','off','all']:
    for stat in ['signed','rms','sd']:
        register(f'uu_{part}_{stat}_l2sq',family='uu',part=part,stat=stat,score='l2sq',
            control=bool(part=='all' and stat=='rms'))
for stat in ['signed','rms','sd']:
    for part in ['self','off']:
        register(f'uu_{part}_{stat}_energy_fraction',family='uu_fraction',part=part,stat=stat)
for family in ['ux','full']:
    register(f'{family}_rms_l2sq',family=family,stat='rms',axis='row',score='l2sq',control=True)
for stat in ['meanabs','median','q90','q99']:
    for score in ['l1','lp05','log01']:
        register(f'ux_{stat}_{score}',family='ux',stat=stat,axis='all',score=score)
        register(f'full_{stat}_{score}',family='full',stat=stat,axis='row',score=score)
        for part in ['self','off']:
            register(f'uu_{part}_{stat}_{score}',family='uu',part=part,stat=stat,score=score)

# B01: single-objective breadth, declared before examining this batch.
for family in ['ux','full']:
    for axis in ['global','row','column']:
        for score in ['hoyer','gini','kurtosis']:
            register(f'b01_{family}_rms_{axis}_{score}',family=family,stat='rms',axis=axis,score=score)
for threshold in [.003,.01,.02,.05,.07]:
    for normalization in ['reference','ux','column']:
        register(f'b01_ux_rms_soft_t{threshold:g}_{normalization}',family='ux',stat='rms',axis='row',
            score=f'soft@{threshold:g}',normalization=normalization)
for degree in [1,2]:
    for axis in ['global','row','column']:
        for score in ['l1','lp05','soft003','hoyer','gini','kurtosis']:
            register(f'b01_functional{degree}_{axis}_{score}',family='functional',degree=degree,axis=axis,score=score)
for normalization in ['reference','ux','column']:
    register(f'b01_ux_rms_soft_t0.03_{normalization}',family='ux',stat='rms',axis='row',
        score='soft003',normalization=normalization)
for statistic in ['signed','rms']:
    for normalized in [False,True]:
        register(f'b01_ux_gram_{statistic}_corr{int(normalized)}',family='ux_gram',stat=statistic,normalized=normalized)
# B02: an undirected Hessian graph counts H_ij and H_ji as one edge.
for score in ['l1','lp05','lp@0.25','lp@0.75','log01','cap01','entropy','hoyer','gini','kurtosis']+[
    f'soft@{t:g}' for t in [.001,.003,.01,.02,.03,.05,.07,.1,.2]]:
    register('b02_unique_edges_'+score.replace('@','_t'),family='edge',stat='rms',axis='global',score=score)


def is_combination(metric):
    spec=CATALOG[metric]
    return spec['family'] in ['joint','third_joint','joint_sparse'] or spec['family']=='jd' and spec['weight']!=0


def summarize_absolute(values,stat):
    if stat=='meanabs':return np.mean(values,axis=0)
    if stat=='median':return np.median(values,axis=0)
    if stat=='q90':return np.quantile(values,.9,axis=0)
    if stat=='q99':return np.quantile(values,.99,axis=0)
    raise ValueError(stat)


def penalty(values,score,axis=-1,k=1):
    v=np.abs(values)
    if '@' in score:
        kind,number=score.split('@');number=float(number)
        if kind=='soft':return float(np.mean(v*v/(v*v+number**2)))
        if kind=='lp':return float(np.mean((v+1e-16)**number))
        raise ValueError(score)
    if score in ['hoyer','gini','kurtosis']:
        count=v.shape[axis];total=v.sum(axis=axis);energy=np.sum(v*v,axis=axis)
        if score=='hoyer':result=total/np.sqrt(np.maximum(count*energy,1e-30))
        elif score=='kurtosis':result=-np.sum(v**4,axis=axis)/np.maximum(energy**2,1e-30)
        else:
            sorted_v=np.sort(v,axis=axis)
            shape=[1]*v.ndim;shape[axis]=count
            weights=np.arange(count,0,-1).reshape(shape)
            result=2*np.sum(weights*sorted_v,axis=axis)/np.maximum(count*total,1e-30)-1/count
            result=np.where(total>1e-15,result,0.)
        return float(np.mean(result))
    if score=='l1':return float(v.mean())
    if score=='l2sq':return float(np.mean(v*v))
    if score=='lp05':return float(np.sqrt(v+1e-16).mean())
    if score=='soft003':return float(np.mean(v*v/(v*v+.03**2)))
    if score=='soft01':return float(np.mean(v*v/(v*v+.1**2)))
    if score=='cap01':return float(np.mean(np.minimum(v/.1,1)))
    if score=='log01':return float(np.mean(np.log1p(v/.1)))
    if score=='entropy':
        prob=v/np.maximum(v.sum(axis=axis,keepdims=True),1e-30)
        return float(np.mean(-np.sum(prob*np.log(np.maximum(prob,1e-30)),axis=axis)))
    sorted_values=np.sort(v,axis=axis)
    size=v.shape[axis]
    if k>=size:raise ValueError('Tail k must leave at least one residual in every model')
    tail=np.take(sorted_values,np.arange(size-k),axis=axis)
    if score=='tail1':return float(np.sum(tail,axis=axis).mean())
    energy=np.sum(tail*tail,axis=axis)
    if score=='tail2sq':return float(energy.mean())
    if score=='relative_tail2':return float(np.sqrt(energy/np.maximum(np.sum(v*v,axis=axis),1e-30)).mean())
    raise ValueError(score)


class HessianLoss:
    def __init__(self,fields,covariance,d,metric):
        if 'h' not in fields:raise ValueError('Hessian required, missing F2 cannot be treated as zero')
        self.spec=CATALOG[metric];self.metric=metric;self.d=d;self.cov=covariance
        h=fields['h'];self.n=len(h);self.uu=fields.get('uu_h',h[:,:d,:d]);self.ux=h[:,:d,d:]
        self.common_uu=h[:,:d,:d]
        self.common_uu_mean=self.common_uu.mean(0)
        common_flat=self.common_uu.reshape(self.n,-1)
        self.common_uu_moment=common_flat.T@common_flat/self.n
        self.um=self.uu.mean(0);self.xm=self.ux.mean(0)
        flat=self.uu.reshape(len(self.uu),-1);self.uu_moment=flat.T@flat/len(self.uu)
        self.ux_moment=np.einsum('nap,nbp->pab',self.ux,self.ux,optimize=True)/self.n
        sd=np.sqrt(np.diag(covariance))
        u=self.uu*sd[None,:,None]*sd[None,None,:];x=self.ux*sd[None,:,None]
        self.uu_energy=float(np.mean(np.sum(u*u,axis=(1,2))));self.ux_energy=float(np.mean(np.sum(x*x,axis=(1,2))))
        self.scale=max(float(np.sqrt(self.uu_energy+self.ux_energy)),1e-12)
        common_standard=self.common_uu*sd[None,:,None]*sd[None,None,:]
        self.common_scale=max(float(np.sqrt(np.mean(np.sum(common_standard*common_standard,axis=(1,2)))+self.ux_energy)),1e-12)
        self.third=fields.get('uu_t',fields.get('third'))
        self.third_is_zero=self.third is not None and not np.any(self.third)
        self.third_energy=0.
        if self.third is not None:
            t=self.third[:,:d,:d,:d]*sd[None,:,None,None]*sd[None,None,:,None]*sd[None,None,None,:]
            self.third_energy=float(np.mean(np.sum(t*t,axis=(1,2,3))))
        if self.spec['order']==3 and self.third is None:raise ValueError('Third derivative of Hessian required')
        self.profiles={}
        for degree in [1,2]:
            phi=PolynomialFeatures(degree).fit_transform(fields['strict_x'])
            ridge=np.eye(phi.shape[1])*1e-8;ridge[0,0]=0
            coeff=np.linalg.solve(phi.T@phi/self.n+ridge,phi.T@self.ux.reshape(self.n,-1)/self.n)
            self.profiles[degree]=coeff.reshape(-1,d,self.ux.shape[2]).transpose(1,0,2).reshape(d,-1)
        self.joint_operators=None
        self.functional_operators=None
        if self.spec['family']=='functional':
            if 'strict_subject_groups' not in fields:raise ValueError('Functional Hessian needs product grid groups')
            groups=fields['strict_subject_groups'];ids=np.unique(groups);coefs=[]
            phi_all=PolynomialFeatures(self.spec['degree']).fit_transform(fields['strict_x'])
            phi_scale=np.maximum(np.sqrt(np.mean(phi_all**2,axis=0)),1e-12)
            phi_all=phi_all/phi_scale
            for sid in ids:
                mask=groups==sid;phi=phi_all[mask]
                projection=np.linalg.solve(phi.T@phi/mask.sum()+1e-5*np.eye(phi.shape[1]),phi.T/mask.sum())
                cp=projection@self.ux[mask].reshape(mask.sum(),-1)
                coefs.append(cp.reshape(-1,d,self.ux.shape[2]).transpose(1,0,2).reshape(d,-1))
            coefficients=np.asarray(coefs)
            self.functional_operators=np.einsum('saq,sbq->qab',coefficients,coefficients)/len(ids)
            self.functional_scale=max(float(np.sqrt(np.einsum('i,qii,i->',sd,self.functional_operators,sd))),1e-12)
        if self.spec['family']=='jd':
            if 'strict_subject_groups' not in fields:raise ValueError('Hessian profile JD requires a grouped subject x stimulus grid')
            groups=fields['strict_subject_groups'];ids=np.unique(groups);coefs=[]
            for sid in ids:
                mask=groups==sid;phi=PolynomialFeatures(self.spec['degree']).fit_transform(fields['strict_x'][mask])
                gram=phi.T@phi/mask.sum()+1e-6*np.eye(phi.shape[1])
                cp=np.linalg.solve(gram,phi.T@self.ux[mask].reshape(mask.sum(),-1)/mask.sum())
                coefs.append(cp.reshape(-1,d,self.ux.shape[2]).transpose(1,0,2).reshape(d,-1))
            coefficients=np.asarray(coefs)
            if self.spec.get('aggregation')=='mean_then_outer':
                average=coefficients.mean(0)
                operators=np.einsum('aq,bq->qab',average,average)
            else:operators=np.einsum('saq,sbq->qab',coefficients,coefficients)/len(ids)
            trace=np.trace(operators,axis1=1,axis2=2)
            active=trace>1e-3*max(float(trace.max()),1e-30)
            if not active.any():raise ValueError('No nonzero Hessian-profile operators')
            self.joint_operators=operators[active]/trace[active,None,None]
            weights=trace[active]**self.spec['strength'];self.jd_weights=weights/weights.sum()

    def _ux(self,a,stat):
        if stat.startswith('profile'):return (a@self.profiles[int(stat[-1])])/self.scale
        if stat=='local':return np.einsum('ia,nap->nip',a,self.ux,optimize=True)/self.scale
        if stat in ['meanabs','median','q90','q99']:
            value=np.abs(np.einsum('ia,nap->nip',a,self.ux,optimize=True))/self.scale
            return summarize_absolute(value,stat)
        mean=a@self.xm
        if stat=='signed':return mean/self.scale
        square=np.maximum(np.einsum('ia,pab,ib->ip',a,self.ux_moment,a,optimize=True),0)
        if stat=='sd':square=np.maximum(square-mean*mean,0)
        return np.sqrt(square)/self.scale

    def _uu(self,a,stat,part):
        if stat=='local' or stat in ['meanabs','median','q90','q99']:
            mat=np.einsum('ia,nab,jb->nij',a,self.uu,a,optimize=True)/self.scale
            if stat!='local':mat=summarize_absolute(abs(mat),stat)
        else:
            mean=a@self.um@a.T
            if stat=='signed':mat=mean/self.scale
            else:
                transform=np.einsum('ia,jb->ijab',a,a).reshape(self.d*self.d,-1)
                square=np.maximum(np.einsum('ia,ab,ib->i',transform,self.uu_moment,transform,optimize=True),0).reshape(self.d,self.d)
                if stat=='sd':square=np.maximum(square-mean*mean,0)
                mat=np.sqrt(square)/self.scale
        if part=='self':return np.diagonal(mat,axis1=-2,axis2=-1)
        if part=='off':return mat[...,~np.eye(self.d,dtype=bool)]
        return mat

    def _full(self,a,stat):
        if stat=='local' or stat in ['meanabs','median','q90','q99']:
            uu=np.einsum('ia,nab,jb->nij',a,self.common_uu,a,optimize=True)
            ux=np.einsum('ia,nap->nip',a,self.ux,optimize=True)
            values=np.concatenate((uu,ux),axis=-1)/self.common_scale
            return values if stat=='local' else summarize_absolute(abs(values),stat)
        mean=a@self.common_uu_mean@a.T
        if stat=='signed':uu=mean
        else:
            transform=np.einsum('ia,jb->ijab',a,a).reshape(self.d*self.d,-1)
            square=np.maximum(np.einsum('ia,ab,ib->i',transform,self.common_uu_moment,transform,optimize=True),0).reshape(self.d,self.d)
            if stat=='sd':square=np.maximum(square-mean*mean,0)
            uu=np.sqrt(square)
        ux=self._ux(a,stat)*self.scale
        return np.concatenate((uu,ux),axis=-1)/self.common_scale

    def _functional(self,a):
        return np.sqrt(np.maximum(np.einsum('ia,qab,ib->iq',a,self.functional_operators,a,optimize=True),0))/self.functional_scale

    def __call__(self,r):
        sd=np.sqrt(np.diag(r@self.cov@r.T))
        a=sd[:,None]*np.linalg.inv(r).T
        s=self.spec;family=s['family']
        if family in ['full','functional']:
            values=self._full(a,s['stat']) if family=='full' else self._functional(a)
            if s['axis']=='global':values=values.reshape(-1)
            return penalty(values,s['score'],-2 if s['axis']=='column' else -1,s.get('k',1))
        if family=='edge':
            matrix=self._full(a,'rms');uu=matrix[:,:self.d]
            values=np.r_[uu[np.triu_indices(self.d)],matrix[:,self.d:].reshape(-1)]
            return penalty(values,s['score'])
        if family=='ux':
            v=self._ux(a,s['stat']);axis=-1 if s['axis']!='column' else -2
            normalization=s.get('normalization','reference')
            if normalization=='ux':v=v*self.scale/max(np.sqrt(self.ux_energy),1e-12)
            if normalization=='column':
                original_sd=np.sqrt(np.diag(self.cov))
                norm=np.sqrt(np.maximum(np.einsum('a,paa,a->p',original_sd,self.ux_moment,original_sd),0))
                v=v*self.scale/np.maximum(norm,1e-12)
            if s['axis']=='global':v=v.reshape(-1)
            return penalty(v,s['score'],axis,s.get('k',1))
        if family=='ux_gram':
            if s['stat']=='signed':vectors=a@self.xm;gram=vectors@vectors.T
            else:gram=np.einsum('ia,pab,jb->ij',a,self.ux_moment,a,optimize=True)
            if s['normalized']:
                norms=np.sqrt(np.maximum(np.diag(gram),0));gram=gram/np.maximum(norms[:,None]*norms[None,:],1e-20)
            else:gram=gram/self.scale**2
            return float(np.mean(gram[~np.eye(self.d,dtype=bool)]**2))
        if family=='profile':return penalty(self._ux(a,f'profile{s["degree"]}'),s['score'],-2,s.get('k',1))
        if family=='uu':return penalty(self._uu(a,s['stat'],s['part']),s['score'])
        if family=='uu_fraction':
            values=self._uu(a,s['stat'],'all')**2;total=float(values.sum())
            own=float(np.trace(values))
            return (own if s['part']=='self' else total-own)/max(total,1e-30)
        if family=='curvature_support':
            return self._support_value(a,s['component'],s['score'],s['threshold'])
        if family=='uu_graph':
            if self.uu_energy==0:return 0.
            matrix=self._uu(a,'local','all')*self.scale
            amplitude=np.sqrt(np.sum(matrix*matrix,axis=-1)) if s['part']=='rows' else abs(matrix)
            ratio=amplitude/max(s['threshold']*np.sqrt(self.uu_energy),1e-30)
            if s['score']=='cap':score=np.minimum(ratio,1).mean()
            elif s['score']=='soft':score=(ratio*ratio/(1+ratio*ratio)).mean()
            else:score=np.log1p(ratio).mean()
            return float(self.uu_energy/max(self.uu_energy+self.ux_energy,1e-30)*score)
        if family=='jd':
            transformed=np.einsum('ia,kab,jb->kij',a,self.joint_operators,a,optimize=True)
            off=transformed[:,~np.eye(self.d,dtype=bool)]
            joint=float(self.jd_weights@np.sum(off*off,axis=1))
            if s['weight']==0:return joint
            if s.get('curvature_component','').startswith('support_'):
                component=s['curvature_component'].removeprefix('support_')
                return joint+s['weight']*self._support_value(a,component,s['self_score'],s['curvature_threshold'])
            curvature=self._third_values(a,'repeat') if s.get('curvature_component')=='repeat' else self._uu(a,'local','self')
            return joint+s['weight']*penalty(curvature,s['self_score'])
        if family=='third':
            return penalty(self._third_values(a,s['kind']),s['score'])
        ux=penalty(self._ux(a,s['ux']),'l1')
        if family=='joint_sparse':return ux+s['weight']*penalty(self._uu(a,'local',s['part']),s['score'])
        if family=='joint':return ux+s['weight']*penalty(self._uu(a,'rms',s['part']),'l1')
        if family=='third_joint':
            t=self.third[:,:self.d,:self.d,:self.d]
            transformed=np.einsum('ia,jb,kc,nabc->nijk',a,a,a,t,optimize=True)/self.scale
            if s['kind']=='self':v=transformed[:,np.arange(self.d),np.arange(self.d),np.arange(self.d)]
            else:
                mask=np.ones((self.d,)*3,dtype=bool);mask[np.arange(self.d),np.arange(self.d),np.arange(self.d)]=False
                v=transformed[:,mask]
            return ux+s['weight']*penalty(v,'l1')
        raise ValueError(family)

    def _support_value(self,a,component,score,threshold):
        if component=='self':
            energy=self.uu_energy;curvature=self._uu(a,'rms','self')*self.scale
        else:
            energy=self.third_energy;value=self._third_values(a,'repeat')*self.scale
            curvature=np.sqrt(np.mean(value*value,axis=(0,2)))
        denominator=energy+self.uu_energy+self.ux_energy if component=='modulation' else self.uu_energy+self.ux_energy
        strength=energy/max(denominator,1e-30)
        ratio=curvature/max(threshold*np.sqrt(energy),1e-30)
        if score=='soft':value=np.mean(ratio*ratio/(1+ratio*ratio))
        elif score=='cap':value=np.mean(np.minimum(ratio,1))
        else:value=np.mean(np.log1p(ratio))
        return float(strength*value)

    def _third_values(self,a,kind):
        if self.third_is_zero:return np.zeros((1,self.d,self.d))
        t=self.third[:,:self.d,:self.d,:self.d]
        if kind=='repeat':
            # Contract only T_iij, not the unused full rotated cubic tensor.
            # Algebraically identical, substantially cheaper for multistart.
            square=np.einsum('ia,ib->iab',a,a)
            contracted=np.einsum('iab,nabc->nic',square,t,optimize=True)
            return np.einsum('nic,jc->nij',contracted,a,optimize=True)/self.scale
        if kind=='self':return np.einsum('ia,ib,ic,nabc->ni',a,a,a,t,optimize=True)/self.scale
        transformed=np.einsum('ia,jb,kc,nabc->nijk',a,a,a,t,optimize=True)/self.scale
        mask=np.ones((self.d,)*3,dtype=bool);mask[np.arange(self.d),np.arange(self.d),np.arange(self.d)]=False
        return transformed[:,mask]

    def describe(self,r):
        sd=np.sqrt(np.diag(r@self.cov@r.T));a=sd[:,None]*np.linalg.inv(r).T
        parts=None
        if self.spec['family']=='jd':
            transformed=np.einsum('ia,kab,jb->kij',a,self.joint_operators,a,optimize=True)
            joint=float(self.jd_weights@np.sum(transformed[:,~np.eye(self.d,dtype=bool)]**2,axis=1))
            if self.spec.get('curvature_component','').startswith('support_'):
                raw=self._support_value(a,self.spec['curvature_component'].removeprefix('support_'),self.spec['self_score'],self.spec['curvature_threshold'])
            else:
                curvature=self._third_values(a,'repeat') if self.spec.get('curvature_component')=='repeat' else self._uu(a,'local','self')
                raw=penalty(curvature,self.spec['self_score'])
            parts=dict(hessian_profile_jd=joint,curvature_raw=raw,curvature_weight=self.spec['weight'],curvature_weighted=self.spec['weight']*raw,total=joint+self.spec['weight']*raw)
        return dict(category='Hessian-only, model-blind M1/M3',specification=self.spec,loss_components=parts,
                    block_energy_reference=dict(uu=self.uu_energy,ux=self.ux_energy,third=self.third_energy),
                    full_matrix_domain='Same differentiable W points for UU and UX; excludes undefined X derivatives',full_matrix_scale=self.common_scale,
                    joint_operators=self.joint_operators,
                    fixed_block_scale=self.scale,uu_diag_rms=self._uu(a,'rms','self'),
                    uu_off_rms=self._uu(a,'rms','off'),ux_rms=self._ux(a,'rms'),
                    ux_signed=self._ux(a,'signed'),loss=self(r))
