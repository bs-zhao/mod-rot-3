"""One-shot analytical rotation of a FIXED inferred structured index.

No generator calls and no model retraining: d[g(R^-1 U,X)]/dU = R d[g(Z,X)]/dZ.
Valid for nonlinear polynomial-Z models as well as C1 linear parameters.
X is never perturbed, which allows the discrete valid CR3 lottery bank.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from basis import Basis

ROTATION_MATRIX_TYPE = 'orthogonal_givens_v2'


def givens(n,i,j,deg):
    if not 0<=i<n or not 0<=j<n or i==j:
        raise ValueError('Expected distinct valid latent axes')
    a=np.deg2rad(deg);r=np.eye(n)
    r[i,i]=r[j,j]=np.cos(a)
    r[i,j]=np.sin(a)
    r[j,i]=-np.sin(a)
    return r


def complexity(s):
    s=np.asarray(s,dtype=float)
    denom=float(np.sqrt(np.sum(s*s)))
    if denom<1e-10:raise ValueError('No measurable structural sensitivity')
    a=s/denom
    weights=a*a
    # each X basis term gets effective number of latent partners
    col=weights.sum(axis=0)
    prob=np.divide(weights,col[None,:],out=np.zeros_like(weights),where=col[None,:]>1e-10)
    entropy=-np.sum(prob*np.log(np.maximum(prob,1e-20)),axis=0)
    eff=np.where(col>1e-10,np.exp(entropy),0)
    return {'normalized_l1':float(np.sum(a)),
            'effective_partner_sum':float(np.sum(eff)),
            'per_latent_strength':np.linalg.norm(s,axis=1).tolist(),
            'per_x_effective_partners':eff.tolist(),
            'matrix':s.tolist()}


def scan(folder,output,axis_i=1,axis_j=2,step=5.):
    folder=Path(folder);out=Path(output)
    fit=json.loads((folder/'fit.json').read_text(encoding='utf8'))
    with np.load(folder/'coefficients.npz',allow_pickle=False) as f:
        coef=np.asarray(f['coef_scaled'],dtype=float)
        anchors=np.asarray(f['anchors'],dtype=float)
    basis=Basis.from_json(fit['basis']);zdim=basis.zdim
    c=basis.coefficient_matrix(coef)
    if not 0<=axis_i<zdim or not 0<=axis_j<zdim or axis_i==axis_j:
        raise ValueError('Invalid latent plane')
    if step<=0 or abs(90./step-round(90./step))>1e-10:
        raise ValueError('Step must divide 90 (e.g. 5 or 10)')
    # The only Z interventions are symbolic coordinate transformations of a
    # learned polynomial index. No X derivative, no synthetic lottery design.
    J=basis.jacobian_z(anchors)
    G=np.einsum('nit,tk->nik',J,c,optimize=True)
    degrees=np.linspace(-90.,90.,2*int(round(90/step))+1)
    records=[]
    for deg in degrees:
        r=givens(zdim,axis_i,axis_j,deg)
        if not np.allclose(r.T @ r,np.eye(zdim),rtol=0,atol=1e-12) or not np.isclose(np.linalg.det(r),1.,atol=1e-12):
            raise AssertionError('Rotation matrix is not an element of SO(d)')
        rotated=np.einsum('ij,njk->nik',r,G,optimize=True)
        rms=np.sqrt(np.mean(rotated**2,axis=0))
        metrics=complexity(rms)
        records.append({'angle_deg':float(deg),**metrics})
    out.mkdir(parents=True,exist_ok=True)
    result={'rotation_matrix_type':ROTATION_MATRIX_TYPE,
            'source':fit['source'],'model':fit['model'],'label':fit['label'],
            'method':fit.get('method','oracle_polynomial_projection'),
            'axis_pair':[axis_i,axis_j], 'step':float(step),
            'z_degree':basis.z_degree,'x_degree':basis.x_degree,
            'x_basis_terms':fit['basis']['x_terms'],
            'n_anchor_subject_rows':int(len(anchors)),
            'explanation':'Fixed-index rotation; Z derivatives analytic, no X differentiation or lottery feature perturbation.',
            'records':records}
    (out/'rotation_curve.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    with (out/'rotation_curve.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['angle_deg','normalized_l1','effective_partner_sum'])
        for rec in records:w.writerow([rec['angle_deg'],rec['normalized_l1'],rec['effective_partner_sum']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('normalized_l1','effective_partner_sum'):
        fig,ax=plt.subplots(figsize=(8,4))
        ax.plot(degrees,[r[metric] for r in records])
        ax.axvline(0,ls='--',c='gray',alpha=.7)
        ax.set(xlabel='Z rotation angle (degrees)',ylabel=metric,
               title=f"{fit['model']} | {fit['source']} | {metric}")
        ax.grid(alpha=.25);fig.tight_layout()
        fig.savefig(out/f'{metric}.png',dpi=150);plt.close(fig)
    best=min(records,key=lambda r:r['normalized_l1'])
    zero=min(records,key=lambda r:abs(r['angle_deg']))
    info={'model':fit['model'],'source':fit['source'],
          'optimal_angle':best['angle_deg'],'normalized_l1_min':best['normalized_l1'],
          'normalized_l1_at_zero':zero['normalized_l1'],
          'n_angles':len(records), 'z_degree':basis.z_degree,
          'valid_CR3_X_treatment':'no X derivatives, valid observed stimuli only' if fit['model']=='CR3' else None}
    (out/'summary.json').write_text(json.dumps(info,indent=2),encoding='utf8')
    print(json.dumps(info,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fit',required=True,help='Oracle projection folder OR WY rank-estimator folder')
    p.add_argument('--output',required=True)
    p.add_argument('--axis-i',type=int,default=1);p.add_argument('--axis-j',type=int,default=2)
    p.add_argument('--step',type=float,default=5)
    args=p.parse_args();scan(args.fit,args.output,args.axis_i,args.axis_j,args.step)
