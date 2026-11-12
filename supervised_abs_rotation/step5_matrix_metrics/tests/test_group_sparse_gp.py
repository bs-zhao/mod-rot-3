import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import group_sparse_gp as gp


def test_group_GP_system_matches_dense_function_ensemble_evidence_and_posterior():
    rng=np.random.default_rng(771);x=rng.normal(size=(22,7));y=rng.normal(size=(2,22,4))
    groups=torch.tensor([0,0,1,1,1,2,2]);a=torch.tensor([[.2,.6,1.],[.7,.1,.5]])
    noise=torch.tensor([.03,.12]);gram=torch.tensor(x.T@x);cross=torch.tensor(np.einsum('np,tnr->tpr',x,y))
    yy=torch.tensor(np.sum(y*y,axis=(1,2)))
    loss,mean,cov=gp.system(gram,cross,yy,groups,a,noise,len(x))
    for t in range(2):
        prior=np.diag(a[t].numpy()[groups.numpy()]);c=x@prior@x.T+float(noise[t])*np.eye(len(x))
        expected=.5*(np.linalg.slogdet(c)[1]+np.sum(y[t]*np.linalg.solve(c,y[t])))
        np.testing.assert_allclose(float(loss[t]),expected,rtol=1e-6)
        np.testing.assert_allclose(mean[t],prior@x.T@np.linalg.solve(c,y[t]),rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(cov[t],prior-prior@x.T@np.linalg.solve(c,x)@prior,rtol=1e-6,atol=1e-6)


def test_group_EM_lowers_ensemble_density_and_component_energy_matches_actual_functions():
    rng=np.random.default_rng(772);z=rng.normal(size=(140,4));radial=np.exp(-.5*(z[:,:1]-z[None,:8,0])**2)
    x,groups,masks,audit=gp.dictionary(z,3,radial)
    y=np.column_stack((z[:,0]+z[:,0]*z[:,1],.3*z[:,0]**2));y/=np.sqrt(np.mean(np.sum(y*y,axis=1)))
    g=torch.tensor(x.T@x);c=torch.tensor((x.T@y)[None]);yy=torch.tensor([np.sum(y*y)])
    gi=torch.tensor(groups)
    with threadpool_limits(2):state,best,history=gp.fit(g,c,yy,gi,len(z),steps=30)
    assert history[-1]['mean_nll_per_subject']<history[0]['mean_nll_per_subject']
    loss,mean,cov=gp.system(g,c,yy,gi,*state,len(z))
    point,posterior,components=gp.component_energies(g,mean,cov,gi,masks,len(z),4)
    assert np.all(posterior>=point-1e-12)
    for j in range(len(masks)):
        ids=groups==j;f=x[:,ids]@mean[0,ids].numpy()
        np.testing.assert_allclose(components[0,j],np.mean(np.sum(f*f,axis=1)),atol=1e-10)
        np.testing.assert_allclose(np.mean(np.sum(x[:,ids]**2,axis=1)),1.,atol=1e-12)


def test_complete_group_GP_pipeline_keeps_all_subjects_and_exact_endpoint_cache(tmp_path):
    import csv
    import json
    import matrix_function_profiles as profiles
    import run_conditional_kernel_moments as kernel
    import run_group_sparse_gp as runner
    from test_matrix_metrics import make_cases
    source=tmp_path/'source';make_cases(source)
    folder,original,output=tmp_path/'bank',tmp_path/'original',tmp_path/'out'
    torch.set_num_threads(2)
    with threadpool_limits(2):
        profiles.prepare(source,folder)
        kernel.analyze(source,original,folder,centers=8,device='cpu')
        runner.analyze(source,original,folder,output,degrees=(2,),radial_centers=8,
                       training_steps=2,refit_steps=3,device='cpu')
    with (output/'metric_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert rows and max(float(r['endpoint_error']) for r in rows)<1e-12
    protocol=json.loads((output/'protocol.json').read_text())
    assert protocol['status']=='COMPLETE' and protocol['all_subjects']==32 and protocol['all_original_rows']==160
    with (output/'function_quality.csv').open() as stream:quality=list(csv.DictReader(stream))
    assert all(np.isfinite(float(r['subject_validation_nmse'])) for r in quality)
