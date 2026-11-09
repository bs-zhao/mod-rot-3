"""Read-only scientific figures for single-loss complementarity and G2 errors."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'results'


def get(label,model,metric=None):
    found=[]
    for path in (ROOT/label).rglob('summary.json'):
        j=json.loads(path.read_text(encoding='utf8'))
        if j['args']['model']==model and (metric is None or j['args']['metric']==metric):found.append((j,path))
    if len(found)!=1:raise ValueError((label,model,len(found)))
    return found[0]


def planes(path):
    with (path.parent/'planes.csv').open(newline='') as f:return list(csv.DictReader(f))


def main():
    out=ROOT/'hessian_search';out.mkdir(exist_ok=True)
    a,ap=get('h04_single_components','CR3','hs_jd_profile1_strength0_selfl1_w0')
    b,bp=get('h04_single_components','CR3','hs_uu_self_local_l1')
    aa,bb=planes(ap),planes(bp);names=[r'$\alpha$',r'$\lambda$',r'$\gamma$',r'$\tau$']
    fig,axs=plt.subplots(2,3,figsize=(12,6.7),sharex=True)
    for ax,(i,j) in zip(axs.flat,[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]):
        ids=[k for k,r in enumerate(aa) if (int(r['i']),int(r['j']))==(i,j)]
        angle=np.array([float(aa[k]['angle']) for k in ids]);u=np.array([float(aa[k]['loss']) for k in ids]);s=np.array([float(bb[k]['loss']) for k in ids])
        for curve,label in [(u,'UX Hessian profile JD'),(s,'UU self mean absolute'),(u+8*s,'Same fixed combination: JD + 8 self')]:
            scaled=(curve-curve.min())/max(np.ptp(curve),1e-30)
            ax.plot(angle,scaled,label=label,lw=1.7)
        ax.axvspan(0,6,color='grey',alpha=.15);ax.axvspan(84,90,color='grey',alpha=.15)
        ax.set(title=f'{names[i]} / {names[j]}',xlabel='Plane rotation (degrees)',ylabel='Each curve rescaled to [0,1]')
    axs[0,0].legend(fontsize=7)
    fig.suptitle('CR3: observed single-loss complementarity (same fields and Q; h04)',fontsize=12)
    fig.tight_layout();fig.savefig(out/'complementarity_cr3.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,3.8));models=['C1','C2','C3','CR3'];angles=[]
    for m in models:
        j,_=get('h05_complementarity_confirm',m);angles.append(j['worst_axis_error_degrees'])
    ax.bar(models,angles);ax.axhline(6,color='red',ls='--',label='User-accepted screening tolerance')
    ax.set(ylabel='Worst full-space axis error (degrees)',title='G1: new seed, larger Q, 64 starts')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'g1_angles.png',dpi=170);plt.close(fig)
    exact,ep=get('h05_complementarity_confirm','CR3');ee=planes(ep)
    variants=[]
    for label in ['h07_g2_tensor','h08_g2_resolution','h09_g2_resolution','h10_g2_local']:
        try:j,p=get(label,'CR3')
        except ValueError:continue
        variants.append((label,j,p,planes(p)))
    if variants:
        fig,axs=plt.subplots(2,3,figsize=(12,7.2),sharex=True)
        for ax,(i,j) in zip(axs.flat,[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]):
            ids=[k for k,r in enumerate(ee) if (int(r['i']),int(r['j']))==(i,j)]
            angle=[float(ee[k]['angle']) for k in ids]
            ax.plot(angle,[float(ee[k]['loss']) for k in ids],color='black',lw=2,label='Exact Hessian')
            for label,record,path,pp in variants:
                ax.plot(angle,[float(pp[k]['loss']) for k in ids],lw=1.3,ls='--',label=label)
            ax.axvspan(0,6,color='grey',alpha=.12);ax.axvspan(84,90,color='grey',alpha=.12)
            ax.set(title=f'{names[i]} / {names[j]}',xlabel='Plane rotation (degrees)',ylabel='Same combined Hessian loss')
        axs[0,0].legend(fontsize=7)
        fig.suptitle('CR3 G2: same W, metric and weight; only WY inference changes',fontsize=12)
        fig.tight_layout();fig.savefig(out/'gap2_cr3.png',dpi=170);plt.close(fig)
    print('Saved Hessian research figures to',out)


if __name__=='__main__':main()
