"""Audit the same raw-stimulus function and gradients, without oracle transport.

Reconstruct the historical stimulus map, keeping the saved cubic coefficients.
High coefficient-profile R² need not imply low trialwise/gradient error.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'np'/'src')]
import argparse
import json
from types import SimpleNamespace
import numpy as np
from scipy.special import expit
from rotations.stimulus_contrast.data import load_frozen
from rotations.stimulus_contrast.method import matrices,loss,from_angles
from rotations.stimulus_contrast.run import write_json
from rotations.mlp_teacher_rotation.surrogate import fit_stimulus_map


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    run=Path(args.run);meta=json.loads((run/'fit.json').read_text());base=ROOT/'outputs'/'choice'/meta['model']
    old=list((base/'rotations'/'mlp_teacher_rotation'/meta['condition']/f"repeat{meta['repeat']:02d}").glob('run_*'))
    if len(old)!=1:raise RuntimeError('Ambiguous historical profile source')
    old=old[0];oldmeta=json.loads((old/'rotation__fit.json').read_text());cfg=oldmeta['config']
    source=load_frozen(SimpleNamespace(model=meta['model'],condition=meta['condition'],repeat=meta['repeat'],
                                     whiten_floor=meta['config']['whiten_floor'],seed=cfg['seed'],fit_max_subjects=2500))
    oldmap=np.load(old/'rotation__matrix.npz')
    if not np.allclose(oldmap['R'].T@oldmap['A'],source['W'],atol=1e-10):raise ValueError('Profile and teacher PCA maps differ')
    n=len(source['chosen']);fit_ids=np.random.default_rng(cfg['seed']).permutation(n)[:max(16,int(.8*n))]
    stimulus=fit_stimulus_map(source['x_fit'][fit_ids],source['mask_fit'][fit_ids],cfg['stimulus_features'])
    coeff=np.load(old/'rotation__method_arrays.npz');co={i:coeff[f'polynomial_tensor_degree{i}'] for i in range(4)}
    direct=np.load(run/'gradients.npz');x=direct['common_x'];phi=stimulus.transform(x).astype(float)
    q=(source['z']-source['pca'].mean_)@source['W'].T
    teacher=np.load(run/'oracle_structure_audit'/'arrays.npz')
    # Read teacher arrays only, not oracle gradients or parameters.
    reports={};arrays={}
    for split in ['FIT_anchors','TEST_anchors']:
        ids=teacher[split+'_ids'];qa=q[ids]
        h=co[0]+qa@co[1]+np.einsum('ni,nj,ijl->nl',qa,qa,co[2])+np.einsum('ni,nj,nk,ijkl->nl',qa,qa,qa,co[3])
        gh=np.broadcast_to(co[1],(len(qa),*co[1].shape)).copy()
        gh+=2*np.einsum('nj,ijl->nil',qa,co[2])+3*np.einsum('nj,nk,ijkl->nil',qa,qa,co[3])
        prediction=h@phi.T;gradient=np.einsum('nil,xl->nix',gh,phi)
        teacher_probability=expit(teacher[split+'_teacher_logit'])
        exact_gradient=teacher[split+'_teacher_gradient']*teacher_probability[:,None,:]*(1-teacher_probability[:,None,:])
        cp=matrices(gradient);ct=matrices(exact_gradient);rng=np.random.default_rng(67)
        angles=np.vstack((np.zeros((1,qa.shape[1]*(qa.shape[1]-1)//2)),rng.normal(0,.7,(63,qa.shape[1]*(qa.shape[1]-1)//2))))
        loss_pairs=np.array([(loss(from_angles(a,qa.shape[1]),cp),loss(from_angles(a,qa.shape[1]),ct)) for a in angles])
        reports[split]=dict(probability_rmse=float(np.sqrt(np.mean((prediction-teacher_probability)**2))),
            probability_r2=float(1-np.sum((prediction-teacher_probability)**2)/np.sum((teacher_probability-teacher_probability.mean())**2)),
            predicted_outside_unit_interval_fraction=float(np.mean((prediction<0)|(prediction>1))),
            gradient_relative_frobenius_error=float(np.linalg.norm(gradient-exact_gradient)/np.linalg.norm(exact_gradient)),
            gradient_cosine=float(np.sum(gradient*exact_gradient)/(np.linalg.norm(gradient)*np.linalg.norm(exact_gradient))),
            raw_stimulus_contrast_loss_correlation_64_rotations=float(np.corrcoef(loss_pairs.T)[0,1]),
            raw_stimulus_contrast_loss_rmse=float(np.sqrt(np.mean((loss_pairs[:,0]-loss_pairs[:,1])**2))))
        arrays[split+'_loss_pairs']=loss_pairs
    write_json(run/'profile_fidelity_audit.json',dict(historical_heldout_coefficient_profile_r2=oldmeta['objective_fit']['response_profile_fidelity']['heldout_profile_r2'],
                source_profile=str(old),reports=reports,
                note='No refit, clipping, theta transport, or best-TEST candidate selection. Compare identical shared raw X anchors.'))
    np.savez_compressed(run/'profile_fidelity_loss_pairs.npz',**arrays)
    print(json.dumps(dict(model=meta['model'],reports=reports),indent=2))


if __name__=='__main__':main()
