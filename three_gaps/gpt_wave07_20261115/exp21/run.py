"""Data-calibrated halfwidths and matched blind searches, never tensor-rotate FD."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import numpy as np
import torch
import pandas as pd
from scales import resolution,support,chord_design
from stencils import Field,FDLoss
from scipy.stats import spearmanr

def main(a):
    if a.channel!='prob':raise ValueError('finite-scale probability protocol only; logit is not silently substituted')
    st=c.Setting(a);rng=np.random.default_rng(21);ids=st.ids0[:a.subjects]
    xi=np.sort(rng.choice(len(st.X),a.trials,False));X=st.X[xi];V=st.V[ids]
    small=c.HERE/'exp21/results'/st.key/a.tag;small.mkdir(parents=True,exist_ok=True)
    r=resolution(st.X);c.dump(small/'resolution.json',r)
    fd=Field(st,V,X,torch.float64 if a.decoder_dtype=='float64' else torch.float32)
    # Explicit float64 audit copy; production defaults to the original dtype.
    fd64=Field(st,V,X,torch.float64)
    from common.derivs import base_derivatives
    b=base_derivatives(fd64.scalar,torch.tensor(V,device=st.dev),torch.tensor(X,device=st.dev),which=('Htt','Htx'),chunk=a.chunk,out_dtype=torch.float64)
    sx,t4=c.moments(b['Htt'],b['Htx']);ml=c.Loss(sx.mean(0),t4.mean(0),st.C)
    c.run_loss(st,f'autograd64_{len(V)}x{len(X)}',sx.mean(0),t4.mean(0),experiment='exp21',extra=dict(decoder='float64_copy',subjects=ids.tolist(),X_indices=xi.tolist()))
    original=c.fields(st,V,X);fsx,ft4=c.moments(original['Htt'],original['Htx'])
    c.run_loss(st,f'autograd32_{len(V)}x{len(X)}',fsx.mean(0),ft4.mean(0),experiment='exp21',extra=dict(decoder='original_float32',subjects=ids.tolist(),X_indices=xi.tolist()))
    H=torch.tensor(c.haar_set(st.d,1000,11),dtype=torch.float32)
    if a.decoder_dtype=='float32':ml=c.Loss(fsx.mean(0),ft4.mean(0),st.C)
    with torch.no_grad():La=ml.total(H).numpy()
    gap=r['gap_quantiles'][:,1]
    configs=[('hybrid',h,None,f'hybrid_h{h:g}') for h in [.05,.1,.25,.5,1]]
    configs += [('ambient',.25,fac*gap,f'ambient_gap{fac:g}') for fac in [.5,1,2,4]]
    try:
        _,_,_,rank,cond=chord_design(st.X)
        c.dump(small/'chord_design.json',dict(rank=rank,condition=cond,ridge=0,units='direction and norm in X/SD; reconstructed UX converted to raw X units'))
        configs += [('chord',.25,None,'chord_h0.25')]
    except ValueError as e:
        c.dump(small/'chord_REJECTED.json',dict(reason=str(e),full_UX_claim=False))
    for mode,h,hX,name in configs:
        d=small/name
        if (d/'COMPLETE.json').exists():
            if a.resume:continue
            raise FileExistsError(d)
        cfg=st.config();cfg.update(mode=mode,h_U_halfwidth=h,h_X_halfwidth=hX,end_to_end_X=None if hX is None else 2*hX,
            decoder=a.decoder_dtype,subjects=ids.tolist(),X_indices=xi.tolist(),matched_budget=dict(starts=a.starts,steps=a.steps))
        c.dump(d/'config.json',cfg);t=time.perf_counter();before=fd.query_rows
        fl=FDLoss(fd,h,mode,hX)
        # Every trial R changes physical query points; no precomputed finite tensor.
        R,L,Q0,trace=c.search(fl,a.starts,a.steps,serial=True);best=int(L.argmin())
        c.save_npz(d/'rotations.npz',R=R,loss=L,initial_R=Q0,best_index=best,C_ref=st.C,trace=trace)
        with torch.no_grad():Lc=fl.total(H).detach().cpu().numpy()
        out=c.OUT/'exp21'/st.key/a.tag/name
        c.save_npz(out/'random_landscapes.npz',R=H.numpy(),loss=Lc,matched_autograd_loss=La)
        planes=[]
        with torch.no_grad():
            for i in range(st.d):
                for j in range(i+1,st.d):
                    for degree in range(0,91,5):
                        q=c.givens(st.d,i,j,np.deg2rad(degree))@R[best]
                        planes.append((i,j,degree,float(fl.total(torch.tensor(q,dtype=torch.float32)))))
        pd.DataFrame(planes,columns=['i','j','degrees','loss']).to_csv(d/'plane_curves.csv',index=False)
        with torch.no_grad():
            eu,ee=fl.field.energies(torch.tensor(R[best]),h,mode,hX)
        c.save_npz(out/'selected_energies.npz',UX=eu.cpu().numpy(),UU=ee.cpu().numpy())
        A=c.effective_A(torch.tensor(R[best],dtype=torch.float64),torch.tensor(st.C)).numpy()
        Vq=np.concatenate([V+sg*h*A[i] for i in range(st.d) for sg in [-1,1]])
        invC=np.linalg.inv(st.C);trainR=np.einsum('na,ab,nb->n',st.V[st.tr],invC,st.V[st.tr]);queryR=np.einsum('na,ab,nb->n',Vq,invC,Vq)
        diag=dict(seconds=time.perf_counter()-t,forward_query_rows=fd.query_rows-before,haar_scales=fl.scales,
            landscape_spearman=float(spearmanr(Lc,La).statistic),support=None if hX is None else support(st.X,hX),
            U_axial_within_train_mahalanobis99=float((queryR<=np.quantile(trainR,.99)).mean()),
            U_query_mahalanobis_quantiles=np.quantile(queryR,[.5,.9,.99]),
            chord_relative_residual=fd.last_chord_residual if mode=='chord' else None,
            selected_by='lowest_loss',truth_loaded=False,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        c.dump(d/'diagnostics.json',diag);c.dump(d/'COMPLETE.json',dict(rotations_sha256=c.sha(d/'rotations.npz'),**diag))
        print(name,diag['seconds'],diag['landscape_spearman'],flush=True)

if __name__=='__main__':
    p=c.args(__doc__);p.set_defaults(starts=8,steps=120)
    p.add_argument('--decoder-dtype',choices=['float32','float64'],default='float32')
    p.add_argument('--subjects',type=int,default=64);p.add_argument('--trials',type=int,default=64);main(p.parse_args())
