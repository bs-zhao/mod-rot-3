"""Post-hoc only: requires completed rotations before reading truth / Oracle."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core as c
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from common.rotlib import corr_matrix,match_report,givens
import torch

def angles(R,Q):
    R=np.asarray(R,dtype=np.float64);Q=np.asarray(Q,dtype=np.float64)
    R=R/np.linalg.norm(R,axis=1,keepdims=True);Q=Q/np.linalg.norm(Q,axis=1,keepdims=True)
    dots=abs(R@Q.T);r,col=linear_sum_assignment(-dots)
    return np.rad2deg(np.arccos(np.clip(dots[r,col],0,1)))

def main(a):
    st=c.Setting(a);paths=list(c.HERE.glob(f'*/results/{st.key}/{a.tag}/*/COMPLETE.json'))
    if a.experiment:paths=[p for p in paths if p.parents[4].name==a.experiment]
    if not paths:raise RuntimeError('no completed blind rotations; truth access refused')
    # First load every R and verify immutable completion hash, THEN read theta.
    loaded=[]
    import json
    for cp in paths:
        d=cp.parent;complete=json.loads(cp.read_text(encoding='utf-8'));rp=d/'rotations.npz'
        if c.sha(rp)!=complete['rotations_sha256']:raise ValueError('rotation changed after completion')
        with np.load(rp) as z:loaded.append((d,{k:z[k] for k in z.files}))
    with np.load(st.paths['latent'],allow_pickle=False) as z:T=z['theta_true']
    oracle=c.ROOT/'outputs/three_gaps/exp00'/f'{a.model}_{st.cond}_pca0'/'R_oracle_raw.npy'
    Q=np.load(oracle)@st.Minv if oracle.exists() else None
    historical=c.HERE/'exp13_reproduce/results'/st.key/a.baseline_tag/'A0_strict/rotations.npz'
    if not historical.exists():historical=historical.parent.parent/'A0/rotations.npz'
    R0=None
    if historical.exists():
        with np.load(historical) as z:R0=z['R'][int(z['best_index'])]
    fullpath=c.HERE/'exp20A/results'/st.key/a.baseline_tag/'full_train/rotations.npz'
    Rfull=None
    if fullpath.exists():
        with np.load(fullpath) as z:Rfull=z['R'][int(z['best_index'])]
    rows=[]
    for d,z in loaded:
        R=z['R'];loss=z['loss'];best=int(z['best_index']);metrics=[]
        for k,q in enumerate(R):
            m=match_report(st.V[st.te]@q.T,T[st.te]);ao=angles(q,Q) if Q is not None else np.full(st.d,np.nan)
            ab=angles(q,R0) if R0 is not None else np.full(st.d,np.nan)
            af=angles(q,Rfull) if Rfull is not None else np.full(st.d,np.nan)
            r=dict(experiment=d.parents[3].name,model=a.model,repeat=a.repeat,channel=a.channel,name=d.name,
                start=k,selected=k==best,loss=float(loss[k]),mcc=m['mcc'],worst_matched=m['worst'],leakage=m['off_mean'],
                max_leakage=m['off_max'],oracle_angle_max=float(ao.max()),oracle_angle_mean=float(ao.mean()),baseline_angle_max=float(ab.max()),
                baseline_angle_mean=float(ab.mean()),full_train_angle_max=float(af.max()),full_train_angle_mean=float(af.mean()))
            metrics.append(r)
        pd.DataFrame(metrics).to_csv(d/'metrics.csv',index=False);rows.append(metrics[best])
        raw=corr_matrix(st.V[st.te]@R[best].T,T[st.te]);rr,cc=linear_sum_assignment(-abs(raw));matched=raw[:,cc]
        c.save_npz(d/'correlations.npz',signed_raw=raw,signed_matched=matched,permutation=cc,T_corr=corr_matrix(T[st.te],T[st.te]),
            oracle_angles=angles(R[best],Q) if Q is not None else [],baseline_angles=angles(R[best],R0) if R0 is not None else [])
        # Oracle is opened only here, after all rotations have been sealed.
        experiment=d.parents[3].name
        mp=c.OUT/experiment/st.key/a.tag/d.name/'moments.npz'
        if experiment=='exp13_reproduce':mp=c.OUT/experiment/st.key/a.tag/'moments.npz'
        if mp.exists() and Q is not None:
            diag=json.loads((d/'diagnostics.json').read_text(encoding='utf-8'))
            with np.load(mp) as m:obj=c.Loss(m['SX'],m['T4'],m['C'],scales=diag.get('haar_scales'))
            if d.name.endswith('C_loss'):
                sys.path.insert(0,str(c.HERE/'exp23'))
                from run import AverageLoss
                cfg=json.loads((d/'config.json').read_text(encoding='utf-8'))['extra'];n=cfg['inner']
                with np.load(cfg['field_source']) as m:
                    obj=AverageLoss([c.Loss(m[f's{s:g}_n{n}_SX_s'].mean(0),m[f's{s:g}_n{n}_T4_s'].mean(0),st.C) for s in cfg['scales']])
            curves=[]
            with torch.no_grad():
                for i in range(st.d):
                    for j in range(i+1,st.d):
                        for deg in range(0,91,5):
                            M=givens(st.d,i,j,np.deg2rad(deg))@Q
                            curves.append((i,j,deg,float(obj.total(torch.tensor(M,dtype=torch.float32)))))
            curve=pd.DataFrame(curves,columns=['i','j','degrees_from_oracle','loss'])
            curve.to_csv(d/'oracle_plane_curves.csv',index=False)
            curve.loc[curve.groupby(['i','j']).loss.idxmin()].to_csv(d/'oracle_plane_minima.csv',index=False)
            with torch.no_grad():
                Lh=obj.total(torch.tensor(c.haar_set(st.d,1000,11),dtype=torch.float32)).numpy()
                Lo=float(obj.total(torch.tensor(Q,dtype=torch.float32)));Lf=float(obj.total(torch.tensor(R[best])))
            c.dump(d/'oracle_posthoc.json',dict(loss_oracle=Lo,loss_found=Lf,oracle_minus_found_Haar_SD=float((Lo-Lf)/Lh.std()),
                oracle_initialization_used=False,per_axis_angles=angles(R[best],Q)))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(8,3.6))
        for ax,M,title in zip(axs,[raw,matched],['Signed U-T','Matched |U-T|']):
            show=M if ax==axs[0] else abs(M);im=ax.imshow(show,vmin=-1 if ax==axs[0] else 0,vmax=1,cmap='coolwarm' if ax==axs[0] else 'viridis')
            ax.set_title(title);ax.set_xlabel('T');ax.set_ylabel('U')
            for i in range(st.d):
                for j in range(st.d):ax.text(j,i,f'{show[i,j]:.2f}',ha='center',va='center',fontsize=8,color='black')
            fig.colorbar(im,ax=ax,fraction=.045)
        fig.suptitle(d.name,fontsize=9);fig.tight_layout();fig.savefig(d/'UT_heatmaps.png',dpi=140);plt.close(fig)
    out=c.HERE/'reports';out.mkdir(exist_ok=True)
    suffix='' if not a.experiment else '_'+a.experiment
    pd.DataFrame(rows).to_csv(out/f'{st.key}_{a.tag}{suffix}.csv',index=False)
    hist=pd.read_csv(c.ROOT/'three_gaps/post_exp13_20261114/baseline/results/metrics_all.csv')
    h=hist[(hist.model==a.model)&(hist.repeat==a.repeat)&(hist.channel==a.channel)].iloc[0]
    bas=[r for r in rows if r['name']=='A0_strict'] or [r for r in rows if r['name']=='A0']
    if bas:
        delta={k:bas[0][k]-float(h[k]) for k in ['mcc','leakage','max_leakage']}
        c.dump(out/f'{st.key}_reproduction.json',dict(deltas=delta,tolerance=1e-4,passed=max(abs(v) for v in delta.values())<1e-4))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    p=c.args(__doc__);p.add_argument('--experiment',help='Optional completed experiment folder filter; no candidate selection')
    main(p.parse_args())
