import csv
import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matrix_function_profiles as profiles
import run_conditional_kernel_moments as kernel
from conditional_kernel_blind import Scorer
from test_matrix_metrics import make_cases


def test_exact_blind_kernel_matches_frozen_scores_and_signs_with_finite_proxy(tmp_path):
    source = tmp_path/'source'; make_cases(source)
    folder, original = tmp_path/'bank', tmp_path/'original'
    torch.set_num_threads(2)
    with threadpool_limits(2):
        profiles.prepare(source, folder)
        kernel.analyze(source, original, folder, centers=8, device='cpu')
        bank = profiles.MatrixBank(source, folder)
        scorer = Scorer(bank, original, device='cpu')
        assert scorer.centers == 8
        v, expected = scorer.exact_cache[0]
        np.testing.assert_allclose(scorer.direction(v, force=True), expected, atol=1e-12)
        with np.load(original/'W_only_geometry.npz') as saved:
            q = saved['directions'][saved['membership'][0]].T
        value = scorer.score(q)
        with (original/'blind_scores.csv').open() as stream:
            row = next(r for r in csv.DictReader(stream) if r['anonymous_case']=='case_000'
                and r['channel']=='logit_nn'
                and r['metric']=='Ui_innovation_with_context_raw_UX_columns_worst_soft_count_eps010')
        np.testing.assert_allclose(value, float(row['loss']), atol=1e-12)
        np.testing.assert_allclose(scorer.score(q*np.array([-1., 1., -1., 1.])), value, atol=1e-12)
        assert np.isfinite(scorer.score(q, exact=False))


def test_entire_frozen_frame_family_reproduces_original_anonymous_case(tmp_path):
    import hashlib
    import json
    import audit_kernel_counterexample_family as family
    import run as wave
    source = tmp_path/'source'; make_cases(source)
    folder, original, counter = tmp_path/'bank', tmp_path/'original', tmp_path/'counter'
    counter.mkdir()
    with threadpool_limits(2):
        profiles.prepare(source, folder)
        kernel.analyze(source, original, folder, centers=8, device='cpu')
        with np.load(original/'W_only_geometry.npz', allow_pickle=False) as saved:
            q = saved['directions'][saved['membership'][0]].T
        np.savez_compressed(counter/'blind_rotations.npz', optimized=q)
        wave.json_file(counter/'blind_freeze.json', dict(rotations_sha256=
            hashlib.sha256((counter/'blind_rotations.npz').read_bytes()).hexdigest()))
        output = tmp_path/'audit'
        family.analyze(source, original, folder, counter, output, device='cpu')
    with (original/'blind_scores.csv').open() as stream:
        expected = {(r['channel'],r['metric']):float(r['loss']) for r in csv.DictReader(stream)
                    if r['anonymous_case']=='case_000'}
    with (output/'assessment.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows)==len(expected)
    for row in rows:
        np.testing.assert_allclose(float(row['loss']), expected[row['channel'],row['metric']], atol=1e-12)
    assert json.loads((output/'protocol.json').read_text())['all_original_rows']==160
