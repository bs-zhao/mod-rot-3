"""No-T integration convergence audit, including fixed Haar-SD comparison."""
import sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import nodes
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(a):
    st=c.Setting(a);source=c.OUT/'exp20B'/st.key/a.tag
    q=[]
    for seed in range(4):
        p=source/f'P1_qmc_{seed}'/'moments.npz'
        if not p.exists():raise FileNotFoundError('all QMC scrambles must finish before defining reference')
        with np.load(p) as z:q.append((z['SX'].mean(0),z['T4'].mean(0)))
    sr=np.mean([s for s,t in q],0);tr=np.mean([t for s,t in q],0)
    ref=c.Loss(sr,tr,st.C);H=torch.tensor(c.haar_set(st.d,1000,11),dtype=torch.float32)
    with torch.no_grad():Lr=ref.total(H).numpy()
    rows=[]
    for p in source.glob('P1_*/moments.npz'):
        name=p.parent.name
        if not re.fullmatch(r'P1_(?:qmc_\d+|mc_\d+|gh_m\d+)',name):continue
        with np.load(p) as z:sx,t4=z['SX'],z['T4']
        method=name.split('_')[1]
        if method=='gh':budgets=[len(sx)]
        else:budgets=[n for n in [256,300,512,1024,2048,4096,8192] if n<=len(sx) and (method=='mc' or n!=300)]
        for n in budgets:
            if method=='gh':
                m=int(name.split('_m')[-1]);_,w,_=nodes.gaussian(0,st.d,0,'gh',order=m)
                s=np.einsum('n,n...->...',w,sx);t=np.einsum('n,n...->...',w,t4)
            else:s=sx[:n].mean(0);t=t4[:n].mean(0)
            obj=c.Loss(s,t,st.C,scales=ref.scales)
            with torch.no_grad():L=obj.total(H).numpy()
            rows.append(dict(method=method,source=name,N=n,SX_relative=float(np.linalg.norm(s-sr)/np.linalg.norm(sr)),
                T4_relative=float(np.linalg.norm(t-tr)/np.linalg.norm(tr)),landscape_spearman=float(spearmanr(L,Lr).statistic),
                common_SD_loss_RMSE=float(np.sqrt(((L-Lr)**2).mean())),fixed_reference_SD=ref.scales.tolist()))
    df=pd.DataFrame(rows);small=c.HERE/'exp20B/results'/st.key/a.tag
    df.to_csv(small/'integration_error.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(13,3.6))
    for method in ['mc','qmc','gh']:
        x=df[df.method==method].groupby('N')[['SX_relative','T4_relative','common_SD_loss_RMSE']].agg(['mean','std'])
        for ax,m in zip(axs,['SX_relative','T4_relative','common_SD_loss_RMSE']):
            ax.errorbar(x.index,x[(m,'mean')],yerr=x[(m,'std')].fillna(0),fmt='o-',label=method,ms=3)
            ax.set(xlabel='Gaussian nodes',ylabel=m,xscale='log',yscale='log');ax.grid(alpha=.2)
    axs[0].legend();fig.suptitle(f'{st.a.model} rep{st.a.repeat}: error to combined 4-scramble QMC reference')
    fig.tight_layout();fig.savefig(small/'integration_convergence.png',dpi=150);plt.close(fig)
    print(df.groupby('method').tail(1)[['method','N','SX_relative','T4_relative','landscape_spearman']].to_string(index=False))

if __name__=='__main__':main(c.args(__doc__).parse_args())
