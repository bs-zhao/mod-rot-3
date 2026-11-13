import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adaptive_coalition_models import AdaptiveCoalitionModels,ResidualAwareAdaptiveModels


@pytest.fixture(scope='module')
def sources(tmp_path_factory):
    import matrix_function_profiles as profiles
    import run_conditional_kernel_moments as kernel
    import run_shapley_profile_support as runner
    from test_matrix_metrics import make_cases
    folder=tmp_path_factory.mktemp('adaptive_sources');source=folder/'source';make_cases(source)
    bank,original=folder/'bank',folder/'kernel';torch.set_num_threads(2)
    with threadpool_limits(2):
        profiles.prepare(source,bank);kernel.analyze(source,original,bank,centers=4,device='cpu')
        for tag,degree in (('base',2),('radial',1)):
            runner.analyze(source,original,bank,folder/tag,degree=degree,centers=4,device='cpu')
    data=profiles.MatrixBank(source,bank).data;z=data.u-data.u.mean(0);z/=np.sqrt(np.var(data.u,axis=0).mean())
    with np.load(folder/'base'/'shared_response_basis.npz') as saved:target=torch.tensor(saved['latent'])
    return folder,source,bank,original,z,target


def test_data_selected_models_reconstruct_and_fit_unseen_subspaces_without_mutating_sources(sources,tmp_path):
    folder,source,bank,original,z,target=sources
    paths={tag:folder/tag for tag in ('base','radial')}
    before={tag:hashlib.sha256((path/'blind_scores.csv').read_bytes()).hexdigest() for tag,path in paths.items()}
    with threadpool_limits(2):
        model=AdaptiveCoalitionModels(z,target,tmp_path/'adaptive',paths,device='cpu')
        first,error=model.frame_shares(np.eye(4));assert error<1e-24
        for record in model.audits:
            assert record['selected_relative_GCV']==min(c['relative_GCV'] for c in record['candidate_scores'])
        q=np.linalg.qr(np.random.default_rng(921).normal(size=(4,4)))[0]
        cold,error=model.frame_shares(q);assert error<1e-24
        perm=[2,0,3,1];other,_=model.frame_shares(q[:,perm]*np.array([-1.,1.,-1.,1.]))
        np.testing.assert_allclose(other,cold[perm],atol=1e-12)
        summary=model.finalize()
        import run_shapley_profile_support as runner
        def factory(z,target,output,**kwargs):
            return AdaptiveCoalitionModels(z,target,output,paths,device=kwargs['device'])
        runner.analyze(source,original,bank,tmp_path/'pipeline',device='cpu',model_factory=factory)
    assert sum(summary['new_candidate_fit_counts'].values())>0
    protocol=json.loads((tmp_path/'pipeline'/'protocol.json').read_text())
    assert protocol['adaptive_GCV'] and protocol['all_subjects']==32 and protocol['degree']=='selected_by_GCV'
    for tag,path in paths.items():assert hashlib.sha256((path/'blind_scores.csv').read_bytes()).hexdigest()==before[tag]


def test_adaptive_emulators_align_an_equivalent_rotated_output_basis(sources,tmp_path):
    folder,source,bank,original,z,target=sources
    changed=tmp_path/'rotated_source';shutil.copytree(folder/'base',changed)
    rotation=np.linalg.qr(np.random.default_rng(922).normal(size=(target.shape[1],target.shape[1])))[0]
    file=changed/'shared_response_basis.npz'
    with np.load(file) as saved:values={name:saved[name].copy() for name in saved.files}
    values['latent']=values['latent']@rotation;values['projection']=rotation.T@values['projection'];np.savez_compressed(file,**values)
    for file in (changed/'conditional_models').glob('model*.npz'):
        with np.load(file) as saved:values={name:saved[name].copy() for name in saved.files}
        values['coefficient']=values['coefficient']@rotation;np.savez_compressed(file,**values)
    with threadpool_limits(2):
        first=AdaptiveCoalitionModels(z,target,tmp_path/'first',{'base':folder/'base'},device='cpu')
        second=AdaptiveCoalitionModels(z,target,tmp_path/'second',{'rotated':changed},device='cpu')
        expected,_=first.frame_shares(np.eye(4));actual,_=second.frame_shares(np.eye(4))
    relative=float((actual-expected).square().sum()/target.square().sum())
    assert relative<1e-16


def test_residual_separation_preserves_all_observations_and_uses_a_fixed_full_model(sources,tmp_path):
    folder,source,bank,original,z,target=sources;paths={tag:folder/tag for tag in ('base','radial')}
    with threadpool_limits(2):
        observed=AdaptiveCoalitionModels(z,target,tmp_path/'observed',paths,device='cpu')
        resolved=ResidualAwareAdaptiveModels(z,target,tmp_path/'resolved',paths,device='cpu')
        before,_=observed.frame_shares(np.eye(4));after,error=resolved.frame_shares(np.eye(4))
        assert after.shape[0]==5 and error<1e-24
        np.testing.assert_allclose(after.sum(0),target,atol=1e-12)
        np.testing.assert_allclose(after[:-1].sum(0),resolved.fitted_full,atol=1e-12)
        np.testing.assert_allclose(after[:-1],before-resolved.unassigned[None]/4,atol=1e-12)
        q=np.linalg.qr(np.random.default_rng(923).normal(size=(4,4)))[0]
        other,_=resolved.frame_shares(q)
        np.testing.assert_array_equal(other[-1],after[-1])
        import run_shapley_profile_support as runner
        def factory(z,target,output,**kwargs):
            return ResidualAwareAdaptiveModels(z,target,output,paths,device=kwargs['device'])
        runner.analyze(source,original,bank,tmp_path/'pipeline',device='cpu',model_factory=factory)
    protocol=json.loads((tmp_path/'pipeline'/'protocol.json').read_text())
    assert protocol['residual_separated'] and protocol['effect_group_count']==5
    with np.load(tmp_path/'pipeline'/'frame000_profile_spectra.npz') as saved:
        assert saved['squared_singular_values'].shape[1]==5


def test_PRESS_variance_pipeline_freezes_nonnegative_conserved_UX_allocations(sources,tmp_path):
    folder,source,bank,original,z,target=sources
    import run_predictive_shapley_variance as runner
    with threadpool_limits(2):
        runner.analyze(source,tmp_path/'predictive',device='cpu',original=folder/'base',bank_folder=bank,
            sources={tag:folder/tag for tag in ('base','radial')})
    protocol=json.loads((tmp_path/'predictive'/'protocol.json').read_text())
    assert protocol['status']=='COMPLETE' and protocol['all_subjects']==32
    assert protocol['unassigned_risk_included_in_primary_support']
    for file in (tmp_path/'predictive').glob('frame*_predictive_variance.npz'):
        with np.load(file) as saved:effects=saved['UX_variance_allocation']
        assert np.min(effects)>=0
        np.testing.assert_allclose(effects.sum(0),1,atol=1e-12)
    from predictive_variance_blind import Scorer
    with threadpool_limits(2):
        scorer=Scorer(source,tmp_path/'predictive',tmp_path/'blind_predictive',device='cpu',bank_folder=bank)
        with np.load(tmp_path/'predictive'/'W_only_geometry.npz') as saved:
            q=saved['directions'][saved['frame_axes'][saved['case_frame_index'][0]]].T
        import csv
        with (tmp_path/'predictive'/'blind_scores.csv').open() as stream:
            expected=next(float(r['loss']) for r in csv.DictReader(stream) if r['anonymous_case']=='case_000'
                and r['channel']=='logit_nn' and r['metric']=='PRESS_variance_UX_columns_worst_soft_count_eps003')
        assert np.isclose(scorer.score(q,force=True),expected,rtol=1e-10)
        assert scorer.score(q[:,[2,0,3,1]]*np.array([-1,1,-1,1]))==scorer.score(q)


def test_nested_covariance_pipeline_records_variance_spectra_and_full_model_quality(sources,tmp_path):
    folder,source,bank,original,z,target=sources
    import run_shapley_profile_support as runner
    from nested_polynomial_variance import NestedPolynomialVariance
    manifest=json.loads((folder/'base'/'protocol.json').read_text())['source_matrix_manifest_sha256']
    def factory(z,target,output,**kwargs):
        return NestedPolynomialVariance(z,target,output,manifest,device=kwargs['device'],degrees=(2,3))
    with threadpool_limits(2):
        runner.analyze(source,original,bank,tmp_path/'nested',device='cpu',model_factory=factory)
    protocol=json.loads((tmp_path/'nested'/'protocol.json').read_text())
    assert protocol['status']=='COMPLETE' and protocol['observed_covariance_conserved']
    assert not protocol['per_coalition_hyperparameter_selection']
    assert protocol['full_input_model_quality']['fixed_hyperparameter_PRESS_relative_MSE']>=0
    with np.load(tmp_path/'nested'/'frame000_profile_spectra.npz') as saved:
        assert saved['variance_eigenvalues'].shape[1]==5
    from nested_variance_blind import Scorer
    with threadpool_limits(2):
        scorer=Scorer(source,tmp_path/'nested',tmp_path/'blind',device='cpu',bank_folder=bank)
        with np.load(tmp_path/'nested'/'W_only_geometry.npz') as saved:
            q=saved['directions'][saved['frame_axes'][saved['case_frame_index'][0]]].T
        import csv
        with (tmp_path/'nested'/'blind_scores.csv').open() as stream:
            expected=next(float(r['loss']) for r in csv.DictReader(stream) if r['anonymous_case']=='case_000'
                and r['channel']=='p_nn' and r['metric']=='nested_polynomial_variance_profile_UX_operator_columns_mean_soft_count_eps003')
        assert np.isclose(scorer.score(q,force=True),expected,rtol=1e-10)
        assert scorer.score(q[:,[2,0,3,1]]*np.array([-1,1,-1,1]))==scorer.score(q)
