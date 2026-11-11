import sys
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rank_coordinates as ranks
import prepare_response_bank as prep
import run_profile_polynomial_complexity as runner
from test_matrix_metrics import make_cases


@pytest.mark.parametrize('method', ['normal_rank', 'uniform_rank'])
def test_marginal_coordinates_are_invariant_to_monotone_units_and_equivariant_to_signed_permutations(method):
    rng = np.random.default_rng(601); z = rng.normal(size=(120, 4))
    actual = ranks.transform(z, method)
    np.testing.assert_allclose(ranks.transform(np.exp(.3*z), method), actual, atol=1e-12)
    order = [2, 0, 3, 1]; signs = np.array([-1, 1, 1, -1])
    np.testing.assert_allclose(ranks.transform(z[:, order]*signs, method), actual[:, order]*signs, atol=1e-12)


def test_rank_coordinate_profile_pipeline_retains_all_subjects_and_endpoint_symmetry(tmp_path):
    import json
    source = tmp_path/'source'; make_cases(source)
    folder, output = tmp_path/'bank', tmp_path/'out'
    with threadpool_limits(limits=2):
        prep.prepare(source, folder, estimators=('poly3',), centers=10)
        rows = runner.analyze(source, output, folder, estimators=('poly3',), degrees=(3,), maximum=16,
                              input_transform='normal_rank')
    assert rows and max(r['endpoint_error'] for r in rows) < 1e-12
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['status'] == 'COMPLETE' and protocol['input_transform'] == 'normal_rank'
    assert protocol['scalar_input_independence_assumed'] is False
    assert json.loads((output/'read_audit.json').read_text())['all_subjects_in_compression'] == 32
