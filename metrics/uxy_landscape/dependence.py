"""Sample dependence estimators. Inputs are numerical U/X/Y, never theta or a decoder.

No p-values are claimed. CODEC uses exact rank/min-NN formulas; pdCor is
U-centered distance residual correlation, not conditional independence.
Continuous kNN CMI is deliberately unavailable for binary Y.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import rankdata, kendalltau


def center(a):return a-a.mean(0)


def correlation(a,b):
    a=center(np.asarray(a,float));b=center(np.asarray(b,float))
    return (a.T@b)/np.maximum(np.linalg.norm(a,axis=0)[:,None]*np.linalg.norm(b,axis=0)[None,:],1e-20)


def double_center(d):return d-d.mean(0)[None,:]-d.mean(1)[:,None]+d.mean()


def ucenter(d):
    n=len(d);s=d.sum(0)
    a=d-s[None,:]/(n-2)-s[:,None]/(n-2)+d.sum()/((n-1)*(n-2))
    np.fill_diagonal(a,0.)
    return a


def rbf(d2):
    scale=np.median(d2[d2>1e-14]) if np.any(d2>1e-14) else 1.
    return np.exp(-d2/max(scale,1e-12))


def matrix_corr(a,bank):
    numerator=np.einsum('ij,yij->y',a,bank)
    return numerator/np.maximum(np.sqrt(np.sum(a*a)*np.sum(bank*bank,axis=(1,2))),1e-20)


def residual(y,c):
    design=np.column_stack([np.ones(len(c)),c])
    return y-design@np.linalg.lstsq(design,y,rcond=1e-10)[0]


def nearest(d2,seed=31):
    # Uniform random tie order, held fixed across all five responses.
    rng=np.random.default_rng(seed);order=rng.permutation(len(d2))
    a=d2[:,order].copy();a[np.arange(len(a)),np.argsort(order)]=np.inf
    return order[np.argmin(a,axis=1)]


def codec(ranks,nn_condition,nn_joint):
    base=np.minimum(ranks,ranks[nn_condition])
    denominator=np.sum(ranks-base,axis=0)
    numerator=np.sum(np.minimum(ranks,ranks[nn_joint])-base,axis=0)
    return np.divide(numerator,denominator,out=np.full(ranks.shape[1],np.nan),where=denominator>0)


def xi(x,y):
    rng=np.random.default_rng(193);permutation=rng.permutation(len(x))
    order=permutation[np.argsort(x[permutation],kind='stable')]
    n=len(x);r=rankdata(y,method='max');l=n-rankdata(y,method='min')+1
    denominator=2*np.sum(l*(n-l))
    return np.nan if denominator==0 else 1-n*np.abs(np.diff(r[order])).sum()/denominator


def bins(a,n_bins):
    a=np.asarray(a)
    if len(np.unique(a))<=n_bins:
        return np.unique(a,return_inverse=True)[1]
    return np.searchsorted(np.unique(np.quantile(a,np.linspace(0,1,n_bins+1)[1:-1])),a,side='right')


def discrete_mi(x,y):
    x=np.unique(x,return_inverse=True)[1];y=np.unique(y,return_inverse=True)[1]
    table=np.zeros((x.max()+1,y.max()+1));np.add.at(table,(x,y),1)
    p=table/table.sum();den=p.sum(1)[:,None]*p.sum(0)[None,:];good=p>0
    return float(np.sum(p[good]*np.log(p[good]/den[good])))


def discrete_cmi(x,y,c):
    value=0.
    for block in np.unique(c):
        ix=c==block
        if ix.sum()>1:value+=ix.mean()*discrete_mi(x[ix],y[ix])
    return value


def entropy_discrete(x):
    p=np.unique(x,return_counts=True)[1]/len(x)
    return -np.sum(p*np.log(p))


def rdc_features(x,n_features,seed):
    x=np.atleast_2d(x).T if np.asarray(x).ndim==1 else np.asarray(x)
    ranks=np.column_stack([rankdata(x[:,j],method='average')/(len(x)+1) for j in range(x.shape[1])])
    rng=np.random.default_rng(seed)
    w=rng.normal(size=(ranks.shape[1]+1,n_features))
    return center(np.sin(np.column_stack([ranks,np.ones(len(x))])@w/6.))


def invsqrt(a):
    d,v=np.linalg.eigh(a)
    return (v/np.sqrt(np.maximum(d,1e-8)))@v.T


def rdc(a,b):
    # Regularized CCA prevents singular empirical feature covariance.
    aa=a.T@a+1e-4*np.eye(a.shape[1]);bb=b.T@b+1e-4*np.eye(b.shape[1])
    return float(np.linalg.svd(invsqrt(aa)@(a.T@b)@invsqrt(bb),compute_uv=False)[0])


def cmi_knn(x,y,c,k=5):
    x=np.asarray(x).reshape(len(x),-1);y=np.asarray(y).reshape(len(y),-1)
    c=np.asarray(c).reshape(len(x),-1)
    xyz=np.column_stack([x,y,c])
    # Strict radii and count conventions are kept explicit.
    radius=cKDTree(xyz).query(xyz,k=k+1,p=np.inf)[0][:,-1]
    if np.any(radius<=0):return np.nan
    radius=np.nextafter(radius,0.)
    def counts(a):return cKDTree(a).query_ball_point(a,radius,p=np.inf,return_length=True)-1
    nxz=counts(np.column_stack([x,c]));nyz=counts(np.column_stack([y,c]));nz=counts(c)
    return float(digamma(k)+np.mean(digamma(nz+1)-digamma(nxz+1)-digamma(nyz+1)))


class Dependence:
    def __init__(self,x,y,cfg):
        self.x,self.y,self.cfg=x,y,cfg
        self.yz=(y-y.mean(0))/np.maximum(y.std(0),1e-12)
        self.ranks=np.column_stack([rankdata(y[:,j],method='max') for j in range(y.shape[1])])
        self.rank_y=np.column_stack([rankdata(y[:,j]) for j in range(y.shape[1])])
        self.dy=np.stack([double_center(cdist(y[:,j,None],y[:,j,None])) for j in range(y.shape[1])])
        self.uy=np.stack([ucenter(cdist(y[:,j,None],y[:,j,None])) for j in range(y.shape[1])])
        self.ky=np.stack([double_center(rbf(cdist(y[:,j,None],y[:,j,None])**2)) for j in range(y.shape[1])])
        self.dx2=cdist(x,x)**2;self.cx=ucenter(np.sqrt(self.dx2));self.nn_x=nearest(self.dx2)
        self.y_res_x=residual(y,x)
        self.y_rff=[rdc_features(y[:,j],cfg['rdc_features'],37) for j in range(y.shape[1])]
        self.by=[bins(y[:,j],cfg['association_bins']) for j in range(y.shape[1])]
        self.entropy=np.array([entropy_discrete(v) for v in self.by])
        # Quantile strata of first two fixed X PCs: explicitly coarsened conditioning.
        _,_,v=np.linalg.svd(center(x),full_matrices=False)
        pc=x@v[:min(2,len(v))].T
        self.strata=sum(bins(pc[:,j],3)*(3**j) for j in range(pc.shape[1]))

    def compute(self,u):
        y=self.y;k=u.shape[1];ny=y.shape[1];n=len(u)
        names=['pearson','spearman','kendall','chatterjee_xi','rdc','distance_correlation_squared',
               'normalized_hsic','binned_eta_squared','binned_mi_nats','adaptive_grid_nmi_proxy',
               'partial_pearson_given_X','partial_pearson_given_otherU_X','partial_spearman_given_X',
               'pdcor_given_X','pdcor_given_otherU_X','codec_given_X','codec_given_otherU_X',
               'cmi_knn_given_X','cmi_knn_given_otherU_X','coarsened_X_cmi_nats','UX_interaction_information_proxy']
        values={name:np.full((k,ny),np.nan) for name in names}
        values['pearson']=correlation(u,y)
        ru=np.column_stack([rankdata(u[:,i]) for i in range(k)])
        values['spearman']=correlation(ru,self.rank_y)
        du2=(u[:,None,:]-u[None,:,:])**2
        all_u2=du2.sum(-1)
        ranks_x=np.column_stack([rankdata(self.x[:,i]) for i in range(self.x.shape[1])])
        ry_rank=residual(self.rank_y,ranks_x)
        for i in range(k):
            a=u[:,i];others=np.delete(u,i,axis=1);condition=np.column_stack([self.x,others])
            d=np.sqrt(du2[:,:,i]);dc=double_center(d);uc=ucenter(d)
            kc=double_center(rbf(du2[:,:,i]))
            values['distance_correlation_squared'][i]=np.maximum(matrix_corr(dc,self.dy),0.)
            values['normalized_hsic'][i]=matrix_corr(kc,self.ky)
            values['partial_pearson_given_X'][i]=correlation(residual(a[:,None],self.x),self.y_res_x)[0]
            values['partial_pearson_given_otherU_X'][i]=correlation(residual(a[:,None],condition),residual(y,condition))[0]
            values['partial_spearman_given_X'][i]=correlation(residual(ru[:,i,None],ranks_x),ry_rank)[0]
            conditional_d2=self.dx2+np.maximum(all_u2-du2[:,:,i],0.)
            for label,c in [('X',self.cx),('otherU_X',ucenter(np.sqrt(conditional_d2)))]:
                c_norm=max(np.sum(c*c),1e-20)
                ar=uc-c*np.sum(uc*c)/c_norm
                yr=self.uy-c[None]*np.einsum('yij,ij->y',self.uy,c)[:,None,None]/c_norm
                values['pdcor_given_'+label][i]=matrix_corr(ar,yr)
            values['codec_given_X'][i]=codec(self.ranks,self.nn_x,nearest(self.dx2+du2[:,:,i]))
            values['codec_given_otherU_X'][i]=codec(self.ranks,nearest(conditional_d2),nearest(self.dx2+all_u2))
            fa=rdc_features(a,self.cfg['rdc_features'],17)
            bx=bins(a,self.cfg['association_bins'])
            for j in range(ny):
                values['kendall'][i,j]=kendalltau(a,y[:,j]).statistic
                values['chatterjee_xi'][i,j]=xi(a,y[:,j])
                values['rdc'][i,j]=rdc(fa,self.y_rff[j])
                group_means=np.array([y[bx==v,j].mean() for v in np.unique(bx)])
                group_weights=np.array([np.mean(bx==v) for v in np.unique(bx)])
                values['binned_eta_squared'][i,j]=np.sum(group_weights*(group_means-y[:,j].mean())**2)/max(np.var(y[:,j]),1e-20)
                values['binned_mi_nats'][i,j]=discrete_mi(bx,self.by[j])
                scores=[]
                for count in [2,3,4,6]:
                    bb=bins(a,count);cc=bins(y[:,j],count)
                    denom=max(min(entropy_discrete(bb),entropy_discrete(cc)),1e-20)
                    scores.append(discrete_mi(bb,cc)/denom)
                values['adaptive_grid_nmi_proxy'][i,j]=max(scores)
                values['coarsened_X_cmi_nats'][i,j]=discrete_cmi(bx,self.by[j],self.strata)
                # Interaction information with coarsened joint X; signed, not PID synergy.
                joint=bx+(self.cfg['association_bins']+1)*self.strata
                values['UX_interaction_information_proxy'][i,j]=discrete_mi(joint,self.by[j])-discrete_mi(bx,self.by[j])-discrete_mi(self.strata,self.by[j])
                if len(np.unique(y[:,j]))>2:
                    values['cmi_knn_given_X'][i,j]=cmi_knn(a,self.yz[:,j],self.x,self.cfg['knn_k'])
                    values['cmi_knn_given_otherU_X'][i,j]=cmi_knn(a,self.yz[:,j],condition,self.cfg['knn_k'])
        whole=double_center(np.sqrt(all_u2))
        controls={'full_U_dcor_squared_control':matrix_corr(whole,self.dy)}
        scalar={}
        for name,value in values.items():
            valid=np.isfinite(value).any(0)
            v=np.where(np.isfinite(value),value,0.)
            scalar[name+'__axis_rms']=np.where(valid,np.sqrt(np.mean(v*v,axis=0)),np.nan)
            weights=v*v;weights/=np.maximum(weights.sum(0),1e-30)
            scalar[name+'__axis_entropy']=np.where(valid,-np.sum(weights*np.log(weights+1e-30),axis=0)/np.log(k),np.nan)
        return scalar|controls,values
