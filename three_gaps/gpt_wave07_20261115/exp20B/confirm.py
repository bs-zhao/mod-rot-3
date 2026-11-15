"""Four-model P0/P1/P3 confirmation; same T-free quadrature rule as CR3.

RQMC4 scrambles; 1024/2048/4096 prefixes. Extend to 8192 only if moment
error >=1% or common landscape Spearman <=.99. No TEST criterion is read.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import nodes
import numpy as np
import torch
from scipy.stats import spearmanr

def main(a):
    st=c.Setting(a);out=c.OUT/'exp20B_confirm'/st.key/a.tag
    H=torch.tensor(c.haar_set(st.d,1000,11),dtype=torch.float32)
    rows=[]
    for measure in ['P1','P3']:
        means=[]
        for seed in range(4):
            e,_,_=nodes.gaussian(8192,st.d,20261115+seed);V=nodes.physical(e,st.C,st.mu,measure)
            sx,t4=c.stream(st,V[:4096],out/f'{measure}_s{seed}_prefix4096')
            old=None;err=None
            for n in [1024,2048,4096]:
                s,t=sx[:n].mean(0),t4[:n].mean(0);obj=c.Loss(s,t,st.C)
                with torch.no_grad():land=obj.total(H).numpy()
                if old:err=[float(np.linalg.norm(s-old[0])/np.linalg.norm(s)),float(np.linalg.norm(t-old[1])/np.linalg.norm(t)),float(spearmanr(land,old[2]).statistic)]
                c.run_loss(st,f'{measure}_qmc_s{seed}_N{n}',s,t,experiment='exp20B_confirm',extra=dict(N=n,scramble=seed,measure=measure,convergence_vs_previous=err))
                old=(s,t,land)
            if max(err[:2])>=.01 or err[2]<=.99:
                sr,tr=c.stream(st,V[4096:],out/f'{measure}_s{seed}_suffix8192')
                s=np.concatenate([sx,sr]).mean(0);t=np.concatenate([t4,tr]).mean(0);obj=c.Loss(s,t,st.C)
                with torch.no_grad():land=obj.total(H).numpy()
                err=[float(np.linalg.norm(s-old[0])/np.linalg.norm(s)),float(np.linalg.norm(t-old[1])/np.linalg.norm(t)),float(spearmanr(land,old[2]).statistic)]
                c.run_loss(st,f'{measure}_qmc_s{seed}_N8192',s,t,experiment='exp20B_confirm',extra=dict(N=8192,scramble=seed,measure=measure,convergence_vs_previous=err))
            means.append((s,t));rows.append(dict(measure=measure,scramble=seed,error=err,converged=max(err[:2])<.01 and err[2]>.99))
        c.run_loss(st,f'{measure}_4scramble_mean',np.mean([s for s,t in means],0),np.mean([t for s,t in means],0),experiment='exp20B_confirm')
    sx,t4=c.stream(st,st.V[st.tr],out/'P0_TRAIN')
    c.run_loss(st,'P0_full_train',sx.mean(0),t4.mean(0),experiment='exp20B_confirm')
    c.dump(c.HERE/'exp20B_confirm/results'/st.key/a.tag/'convergence.json',rows)

if __name__=='__main__':main(c.args(__doc__).parse_args())
