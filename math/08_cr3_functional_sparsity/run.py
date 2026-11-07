"""08: Standalone CR3 oracle functional-sparsity rotation experiment.

Run from repository root: python math/08_cr3_functional_sparsity/run.py
No imports from other math/ experiments. No NP checkpoint or decoder required.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from oracle import PARAM_NAMES, X_NAMES, sample_prior, outputs, derivatives

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BANK = ROOT / 'outputs/choice/CR3/s0_design/CR3_4P_fixed500.npz'
DEFAULT_OUT = ROOT / 'outputs/math/08_cr3_functional_sparsity'
CHANNELS = ('logit', 'probability')


def rotation(i, j, angle):
    r = np.eye(4)
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def parse_angles(spec):
    if ':' in spec:
        a, b, step = [float(v) for v in spec.split(':')]
        if step <= 0 or b < a: raise ValueError('Invalid angle range')
        values = np.arange(a, b+step*1e-5, step)
    else:
        values = np.asarray([float(t) for t in spec.split(',')])
    if len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError('Angles must be finite and nonempty')
    return values


def soft_l0(m, relative):
    """Saturating support count with an orientation-invariant Frobenius scale
    for the analytic ZX matrix (same dimensions and shared samples).
    """
    m = np.asarray(m, float)
    scale = np.sqrt(np.mean(m*m))
    eps = max(relative*scale, 1e-14)
    soft = float(np.sum(m*m/(m*m+eps*eps)))
    hard = int(np.count_nonzero(m > eps))
    return soft, hard, float(eps)


def load_bank(path, synthetic, rng):
    if synthetic:
        # DEBUG ONLY: not the actual CR3 fixed500 design.
        amounts = np.array([10,20,30,50,80,100,125,150],float)
        probs = np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9])
        x = np.empty((500,6))
        for k in (0,1,3,4): x[:,k]=rng.choice(amounts,500)
        for k in (2,5): x[:,k]=rng.choice(probs,500)
        return x, 'SYNTHETIC_DEBUG_BANK_NOT_CR3'
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Official CR3 bank not found: {path}. Run np/s1_generate.py for CR3 first, or use --synthetic-bank for a debug-only smoke test.')
    with np.load(path,allow_pickle=False) as d: x=np.asarray(d['X'],float)
    if x.shape != (500,6) or np.any(~np.isfinite(x)):
        raise ValueError('Official CR3 bank must be finite and have shape (500,6)')
    return x, hashlib.sha256(path.read_bytes()).hexdigest()


def pick_x(x, sx, step, max_x, rng):
    # Strict differentiability and symmetric finite contrast support.
    valid = np.all(x[:,[0,1,3,4]] > 1.1*step*sx[[0,1,3,4]],axis=1)
    valid &= np.all(x[:,[2,5]] > 1.1*step*sx[[2,5]],axis=1)
    valid &= np.all(x[:,[2,5]] < 1-1.1*step*sx[[2,5]],axis=1)
    indices=np.flatnonzero(valid)
    if len(indices)==0: raise ValueError('No X trials support interior Hessian and symmetric contrasts at this --step')
    selected=np.sort(rng.choice(indices,size=min(max_x,len(indices)),replace=False))
    return x[selected],int(len(indices)), selected


def matrix_rms(zz_raw, zx_raw, r, st, sx):
    # Baseline coordinates v_k=(T_k-center_k)/sd_k; U=R v.
    zz = zz_raw*st[None,None,:,None]*st[None,None,None,:]
    zx = zx_raw*st[None,None,:,None]*sx[None,None,None,:]
    zzr = np.einsum('ia,ncab,jb->ncij',r,zz,r,optimize=True)
    zxr = np.einsum('ia,ncak->ncik',r,zx,optimize=True)
    return np.sqrt(np.mean(zzr**2,axis=0)),np.sqrt(np.mean(zxr**2,axis=0))


def finite_effects(t, x, r, st, sx, step):
    """Paired base samples, manipulate one standardized U and physical X at a time.

    3x3 factorial double-centering measures NONADDITIVE FUNCTION-VALUE
    contrast (NOT a Sobol index). The 3-point self contrast measures
    non-affinity without differentiating a surrogate.
    """
    n=len(t)
    base=outputs(t,x)
    pair=np.zeros((2,4,6)); selfv=np.zeros((2,4))
    offsets=(-1.,0.,1.)
    for i in range(4):
        dt=step*st*r[i]
        one=np.stack([outputs(t+v*dt,x) for v in offsets],axis=0)  # 3,n,2
        selfv[:,i]=np.sqrt(np.mean((one[0]-2*one[1]+one[2])**2,axis=0))
        for j in range(6):
            dx=np.zeros(6); dx[j]=step*sx[j]
            cube=np.stack([
                np.stack([outputs(t+v*dt,x+q*dx) for q in offsets],axis=0)
                for v in offsets],axis=0)   # 3,3,n,2
            residual=(cube-cube.mean(axis=1,keepdims=True)
                       -cube.mean(axis=0,keepdims=True)
                       +cube.mean(axis=(0,1),keepdims=True))
            pair[:,i,j]=np.sqrt(np.mean(residual**2,axis=(0,1,2)))
    return selfv,pair


def score_angle(zz, zx, selfv, pair, relative):
    rows=[]
    for c, channel in enumerate(CHANNELS):
        diag=np.diag(zz[c]); off=zz[c].copy(); np.fill_diagonal(off,0.)
        hzx,hcount,heps=soft_l0(zx[c],relative)
        ascore,acount,aeps=soft_l0(pair[c],relative)
        dscore,dcount,deps=soft_l0(diag,relative)
        rows.append(dict(channel=channel,
            zx_soft_l0=hzx,zx_hard_support=hcount,zx_threshold=heps,
            zx_frobenius=float(np.linalg.norm(zx[c])),
            zx_nonzero_min=float(np.min(zx[c])),
            zz_self_soft_l0=dscore,zz_self_hard_support=dcount,
            zz_self_rms=float(np.linalg.norm(diag)),
            zz_offdiag_rms=float(np.linalg.norm(off)/np.sqrt(2)),
            zz_total_frobenius=float(np.linalg.norm(zz[c])),
            value_pair_soft_l0=ascore,value_pair_hard_support=acount,
            value_pair_threshold=aeps,
            value_pair_frobenius=float(np.linalg.norm(pair[c])),
            value_self_frobenius=float(np.linalg.norm(selfv[c])),
            value_vs_hessian_zx_corr=(float(np.corrcoef(zx[c].ravel(),pair[c].ravel())[0,1])
                if np.std(pair[c])>1e-13 and np.std(zx[c])>1e-13 else None),
        ))
    return rows


def save_csv(path, records):
    if not records: return
    with Path(path).open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=list(records[0]),extrasaction='ignore')
        writer.writeheader();writer.writerows(records)


def plots(out, plane, angles, curve, hzx, apair):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    for c,channel in enumerate(CHANNELS):
        sub=[r for r in curve if r['channel']==channel]
        fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
        for name,label,ax in [
            ('zx_soft_l0','Hessian ZX soft support',axs[0,0]),
            ('value_pair_soft_l0','3x3 function-value ANOVA soft support',axs[0,1]),
            ('zz_self_rms','Hessian ZZ diagonal RMS sum',axs[1,0]),
            ('zz_offdiag_rms','Hessian ZZ off-diagonal RMS',axs[1,1])]:
            ax.plot(angles,[r[name] for r in sub],marker='.',ms=3)
            ax.axvline(0,color='gray',linestyle=':',lw=1)
            ax.axvline(90,color='gray',linestyle=':',lw=1)
            ax.set(xlabel='Rotation angle (degrees)',ylabel=label,title=label)
            ax.grid(alpha=.2)
        fig.suptitle(f'CR3 {plane} / {channel}: analytic function, no NP')
        fig.savefig(out/f'curves__{plane}__{channel}.png',dpi=150);plt.close(fig)
        chosen=sorted(set([0,len(angles)//2,len(angles)-1]))
        vmax_h=max(float(hzx[i,c].max()) for i in chosen)
        vmax_a=max(float(apair[i,c].max()) for i in chosen)
        fig,axs=plt.subplots(2,len(chosen),figsize=(4*len(chosen),7),constrained_layout=True)
        for ci,k in enumerate(chosen):
            for ri,(m,vmax,label) in enumerate(((hzx[k,c],vmax_h,'Hessian RMS'),(apair[k,c],vmax_a,'3x3 ANOVA gap'))):
                ax=axs[ri,ci]
                im=ax.imshow(m,aspect='auto',vmin=0,vmax=max(vmax,1e-12))
                ax.set_xticks(range(6),X_NAMES,rotation=55,ha='right',fontsize=8)
                ax.set_yticks(range(4),PARAM_NAMES)
                ax.set_title(f'{angles[k]:g}°: {label}')
                fig.colorbar(im,ax=ax,shrink=.65)
        fig.savefig(out/f'heatmaps__{plane}__{channel}.png',dpi=140);plt.close(fig)


def audit(curves, angles, plane, weight):
    out=[]
    for channel in CHANNELS:
        sub=[r for r in curves if r['channel']==channel]
        for name in ('zx_soft_l0','value_pair_soft_l0','zz_self_rms',
                     'zz_offdiag_rms','combined_zx_plus_self'):
            if name=='combined_zx_plus_self':
                vals=np.asarray([r['zx_soft_l0']+weight*r['zz_self_soft_l0'] for r in sub])
            else: vals=np.asarray([r[name] for r in sub])
            best=int(np.argmin(vals)); endpoints=min(vals[0],vals[-1]); span=float(np.ptp(vals))
            out.append(dict(plane=plane,channel=channel,metric=name,
                best_angle=float(angles[best]),best_value=float(vals[best]),
                zero_value=float(vals[0]),ninety_value=float(vals[-1]),
                endpoint_is_min=bool(endpoints <= vals[best]+1e-8*max(1,abs(vals[best]))),
                flat=bool(span < 1e-5*max(1.,abs(float(vals.mean())))),
                span=span,
                note='No guarantee that truth is minimum; endpoint 90 degrees is signed permutation'))
    return out


def run(args):
    rng=np.random.default_rng(args.seed)
    x_full,bank_identity=load_bank(args.bank,args.synthetic_bank,rng)
    st_ref=sample_prior(args.scale_reference,rng)
    center=st_ref.mean(axis=0); st=st_ref.std(axis=0)
    sx=x_full.std(axis=0)
    if np.any(st<=0) or np.any(sx<=0): raise ValueError('Zero input scale')
    xs,n_valid,x_indices=pick_x(x_full,sx,args.step,args.max_x,rng)
    T=sample_prior(args.subjects,rng)
    pair_t=np.repeat(T,len(xs),axis=0)
    pair_x=np.tile(xs,(len(T),1))
    if args.step>0.0:
        if np.any(pair_t[:,[0,2]] < args.step*np.max(st)):
            raise ValueError('Some sampled alpha/gamma too close to zero for contrast: reduce --step')
    zz_raw,zx_raw=derivatives(pair_t,pair_x)
    angles=parse_angles(args.angles)
    planes=[(i,j) for i in range(4) for j in range(i+1,4)]
    if args.planes!='all':
        selected=set(args.planes.split(','));planes=[(i,j) for i,j in planes if f'{i}{j}' in selected]
        if not planes: raise ValueError('No valid planes; use all or comma-separated 01,02,03,12,13,23')
    out=Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{out} is nonempty; use --overwrite or --output NEW_DIR')
    out.mkdir(parents=True,exist_ok=True)
    all_curves=[];audit_rows=[]
    frob_zx_reference=np.sqrt(np.mean(zx_raw**2,axis=0))
    for i,j in planes:
        plane=f'{i}{j}_{PARAM_NAMES[i]}_{PARAM_NAMES[j]}'
        curve=[];zh=[];zp=[];vself=[];ap=[]
        for angle in angles:
            r=rotation(i,j,angle)
            zz,zx=matrix_rms(zz_raw,zx_raw,r,st,sx)
            selfv,pair=finite_effects(pair_t,pair_x,r,st,sx,args.step)
            for item in score_angle(zz,zx,selfv,pair,args.relative_threshold):
                curve.append(dict(plane=plane,angle=float(angle),**item))
            zh.append(zz);zp.append(zx);vself.append(selfv);ap.append(pair)
        zh=np.asarray(zh);zp=np.asarray(zp);vself=np.asarray(vself);ap=np.asarray(ap)
        # Orthogonal rotation must preserve full Hessian block Frobenius energy,
        # checked on local tensor before RMS; squaring and averaging commutes.
        for c in range(2):
            norms=np.sum(zp[:,c]**2,axis=(1,2))
            if not np.allclose(norms,norms[0],rtol=2e-9,atol=1e-12):
                raise AssertionError(f'Orthogonal ZX energy not conserved in {plane}/{CHANNELS[c]}')
        np.savez_compressed(out/f'matrices__{plane}.npz',angles=angles,
            hessian_zz_rms=zh,hessian_zx_rms=zp,
            value_self_contrast_rms=vself,value_pair_anova_gap_rms=ap)
        save_csv(out/f'curves__{plane}.csv',curve)
        if not args.no_plots: plots(out/'plots',plane,angles,curve,zp,ap)
        audit_rows+=audit(curve,angles,plane,args.self_weight)
        all_curves+=curve
        print('COMPLETE',plane,'angles=',len(angles),flush=True)
    save_csv(out/'all_curves.csv',all_curves)
    save_csv(out/'best_angles.csv',audit_rows)
    metadata=dict(model='CR3',source='exact generating equation (no fitted model)',
        input_bank=str(args.bank),bank_identity=bank_identity,
        synthetic_debug_bank=bool(args.synthetic_bank),
        full_x_count=int(len(x_full)),eligible_x_count=n_valid,
        chosen_x_original_indices=x_indices.tolist(),
        subjects=len(T),subject_x_pairs=len(pair_t),
        seed=args.seed,center_T=center.tolist(),std_T=st.tolist(),std_X=sx.tolist(),
        angles=angles.tolist(),planes=[f'{i}{j}' for i,j in planes],
        derivative_coordinates='U=R*((T-center_T)/std_T), X standardized by std_X for derivatives',
        finite_difference_step_in_standardized_coordinates=args.step,
        threshold_relative_to_matrix_rms=args.relative_threshold,
        self_weight=args.self_weight,choice_note='Binary choice not computed; condition mean is probability',
        sobol_note='Local 3x3 two-way contrasts are NOT classical Sobol indices. Raw X bank is dependent/fixed.',
        boundary_note='Only X with positive gains/losses and interior p, and symmetric perturbation space, are used.',
        qualitative_note='An endpoint minimum is not assumed; see best_angles.csv. No true-theta recovery claim.')
    (out/'summary.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Saved',out.resolve())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank',default=str(DEFAULT_BANK))
    ap.add_argument('--output',default=str(DEFAULT_OUT))
    ap.add_argument('--subjects',type=int,default=48)
    ap.add_argument('--max-x',type=int,default=30)
    ap.add_argument('--scale-reference',type=int,default=10000)
    ap.add_argument('--angles',default='0:90:5')
    ap.add_argument('--planes',default='all')
    ap.add_argument('--step',type=float,default=.04)
    ap.add_argument('--relative-threshold',type=float,default=.12)
    ap.add_argument('--self-weight',type=float,default=.25)
    ap.add_argument('--seed',type=int,default=20261107)
    ap.add_argument('--synthetic-bank',action='store_true',help='Debug only, not the official CR3 X design')
    ap.add_argument('--no-plots',action='store_true')
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    if args.subjects<3 or args.max_x<3 or args.scale_reference<20 or args.step<=0 or args.relative_threshold<=0:
        ap.error('Subjects>=3, max-x>=3, reference>=20, step>0 and relative-threshold>0 required')
    run(args)

if __name__=='__main__': main()
