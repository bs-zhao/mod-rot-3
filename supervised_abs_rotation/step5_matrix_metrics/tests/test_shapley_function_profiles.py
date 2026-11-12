import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import shapley_function_profiles as shapley


def test_function_shares_reconstruct_full_response_even_for_arbitrary_coalition_models():
    rng=np.random.default_rng(871)
    for d in (4,6):
        values={mask:rng.normal(size=(17,5)) for mask in range(1,2**d)}
        shares=shapley.shares_from_coalitions(values,d)
        np.testing.assert_allclose(shares.sum(0),values[2**d-1],atol=1e-14)


def test_additive_coalition_functions_return_the_individual_profiles_exactly():
    rng=np.random.default_rng(872);parts=rng.normal(size=(4,30,7))
    values={mask:sum(parts[i] for i in range(4) if mask&(1<<i)) for mask in range(1,16)}
    np.testing.assert_allclose(shapley.shares_from_coalitions(values,4),parts,atol=1e-14)


def test_conditional_models_use_all_rows_and_cache_rotated_bases_of_the_same_subspace(tmp_path):
    rng=np.random.default_rng(873);z=rng.normal(size=(100,4));target=np.column_stack((2*z[:,0],z[:,1]**2,z[:,2]))
    torch.set_num_threads(2)
    with threadpool_limits(2):
        model=shapley.CoalitionModels(z,torch.tensor(target),tmp_path/'models',centers=8,device='cpu')
        basis=np.eye(4)[:,:2];q=np.linalg.qr(rng.normal(size=(2,2)))[0]
        first=model.predict(basis).numpy();second=model.predict(basis@q).numpy()
        assert len(model.models)==1
        np.testing.assert_allclose(first,second,atol=1e-12)
        # Independent fits must respect the subspace, not merely the cache key.
        independent=shapley.CoalitionModels(z,torch.tensor(target),tmp_path/'rotated_basis',centers=8,device='cpu')
        rotated=independent.predict(basis@q).numpy()
        np.testing.assert_allclose(first,rotated,rtol=1e-7,atol=1e-8)
        assert np.mean((first[:,0]-target[:,0])**2)<1e-20
        shares,error=model.frame_shares(np.eye(4))
    assert error<1e-25
    np.testing.assert_allclose(shares.sum(0).numpy(),target,atol=1e-12)


def test_complete_observed_matrix_shapley_pipeline_preserves_profiles_and_endpoints(tmp_path):
    import csv
    import json
    import matrix_function_profiles as profiles
    import run_conditional_kernel_moments as kernel
    import run_shapley_profile_support as runner
    from test_matrix_metrics import make_cases
    source=tmp_path/'source';make_cases(source)
    folder,original,output=tmp_path/'bank',tmp_path/'original',tmp_path/'out'
    torch.set_num_threads(2)
    with threadpool_limits(2):
        profiles.prepare(source,folder)
        kernel.analyze(source,original,folder,centers=8,device='cpu')
        runner.analyze(source,original,folder,output,centers=8,degree=2,device='cpu')
    with (output/'metric_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert rows and max(float(r['endpoint_error']) for r in rows)<1e-12
    protocol=json.loads((output/'protocol.json').read_text())
    assert protocol['all_subjects']==32 and protocol['all_original_rows']==160 and protocol['status']=='COMPLETE'
    errors=json.loads((output/'numerical_audits.json').read_text())
    assert max(r['relative_squared_efficiency_error'] for r in errors)<1e-24
    from shapley_profile_blind import Scorer
    with threadpool_limits(2):
        scorer=Scorer(source,output,tmp_path/'blind',device='cpu',bank_folder=folder)
        with np.load(output/'W_only_geometry.npz') as saved:
            axes=saved['frame_axes'][saved['case_frame_index'][0]];q=saved['directions'][axes].T
        score=scorer.score(q,force=True)
    with (output/'blind_scores.csv').open() as stream:
        expected=next(float(r['loss']) for r in csv.DictReader(stream) if r['anonymous_case']=='case_000'
            and r['channel']=='p_nn' and r['metric']=='shapley_profile_UX_operator_columns_mean_soft_count_eps003')
    np.testing.assert_allclose(score,expected,atol=1e-10)
    import run_mobius_function_profiles as dividend_runner
    with threadpool_limits(2):
        dividend_runner.analyze(source,output,tmp_path/'dividends',device='cpu',bank_folder=folder)
    audits=json.loads((tmp_path/'dividends'/'allocation_and_trace_audits.json').read_text())
    assert max(r['relative_squared_efficiency_error'] for r in audits)<1e-24
    assert sum(r['new_conditional_models'] for r in audits)==0
    import run_tree_haar_profiles as haar_runner
    with threadpool_limits(2):
        haar_runner.analyze(source,output,tmp_path/'haar',bank_folder=folder)
    haar_audit=json.loads((tmp_path/'haar'/'energy_and_fit_audits.json').read_text())
    assert max(a['relative_Parseval_error'] for r in haar_audit for a in r['X_columns'])<1e-12
    assert json.loads((tmp_path/'haar'/'protocol.json').read_text())['all_subjects']==32
