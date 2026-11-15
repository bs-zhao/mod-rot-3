"""Fresh TRAIN-only per-subject streaming, five nested subsets, native/fixed C."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import numpy as np
from scipy.stats import pearsonr,spearmanr
import torch

def main(a):
    st=c.Setting(a);dest=c.OUT/'exp20A'/st.key/a.tag
    sx,t4=c.stream(st,st.V[st.tr],dest/'train_stream')
    ix={int(s):i for i,s in enumerate(st.tr)}
    full=(sx.mean(0),t4.mean(0));h0=np.array([ix[int(i)] for i in st.ids0])
    base=c.Loss(sx[h0].mean(0),t4[h0].mean(0),c.cov(st.V[st.ids0]))
    H=torch.tensor(c.haar_set(st.d,1000,11),dtype=torch.float32)
    target=c.Loss(*full,st.C)
    with torch.no_grad():Lt=target.total(H).numpy()
    c.run_loss(st,'full_train',*full,experiment='exp20A',extra=dict(N=len(st.tr),train_only=True))
    c.run_loss(st,'full_train_histSD',*full,scales=base.scales,experiment='exp20A',extra=dict(N=len(st.tr),fixed_historical_scales=True))
    for name,idx,ids in [('A0',h0,st.ids0)]+[(f'N{n}_seed{s}',np.random.default_rng(1000+s).permutation(len(st.tr))[:n],None)
            for n in [100,300,500,1000,2000] if n<len(st.tr) for s in range(5)]:
        ids=st.tr[idx] if ids is None else ids
        ss,tt=sx[idx].mean(0),t4[idx].mean(0)
        for mode,C in [('native',c.cov(st.V[ids])),('fixed',st.C)]:
            ml=c.Loss(ss,tt,C)
            with torch.no_grad():L=ml.total(H).numpy()
            extra=dict(N=len(ids),subject_ids=ids.tolist(),C_mode=mode,
                SX_relative=float(np.linalg.norm(ss-full[0])/np.linalg.norm(full[0])),
                T4_relative=float(np.linalg.norm(tt-full[1])/np.linalg.norm(full[1])),
                landscape_pearson=float(pearsonr(L,Lt).statistic),landscape_spearman=float(spearmanr(L,Lt).statistic))
            c.run_loss(st,name+'_'+mode,ss,tt,C,extra=extra,experiment='exp20A')
    energy=sx.trace(axis1=-2,axis2=-1).sum(-1)+t4.reshape(len(t4),st.d**2,st.d**2).trace(axis1=-2,axis2=-1)
    order=np.argsort(energy)[::-1]
    c.dump(c.HERE/'exp20A/results'/st.key/a.tag/'influence.json',dict(energy_quantiles=np.quantile(energy,[0,.5,.9,.95,.99,1]),
        top_1pct_share=float(energy[order[:max(1,len(energy)//100)]].sum()/energy.sum()),subjects=len(st.tr)))

if __name__=='__main__':main(c.args(__doc__).parse_args())
