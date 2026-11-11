import hashlib
import json
import sys
from pathlib import Path

from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_bank as jb
import run_predictive_compression as rm
import run_joint_separation as rn
import run_local_joint as rr
import run_all_trial_support as rt
from test_matrix_metrics import make_cases


def test_both_empirical_compression_and_mixed_spectra_complete_from_wy_only(tmp_path):
    source = tmp_path/'source'; make_cases(source)
    bank = tmp_path/'bank'
    with threadpool_limits(limits=2):
        jb.prepare(source, bank, degrees=(2,), device='numpy')
        for name, analyze in [('compression', rm.analyze), ('spectra', rn.analyze)]:
            output = tmp_path/name
            rows = analyze(source, output, bank)
            assert rows
            protocol = json.loads((output/'protocol.json').read_text())
            assert protocol['status'] == 'COMPLETE'
            assert protocol['scores_sha256'] == hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
            audit = json.loads((output/'read_audit.json').read_text())
            assert audit['all_subjects'] == 32
            assert audit['all_trial_rows'] == 160
            assert audit['forbidden_files_read'] == []
            assert not any('R_best' in f or 'Zstd' in f for f in audit['files'])


def test_actual_ux_derivative_pipeline_uses_every_subject(tmp_path):
    source = tmp_path/'source'; make_cases(source)
    bank = tmp_path/'bank'; output = tmp_path/'local_joint'
    with threadpool_limits(limits=2):
        jb.prepare(source, bank, degrees=(2,), device='numpy')
        rows = rr.analyze(source, output, bank, per_subject=2)
    assert any(row['metric'].startswith('UX_entries_') for row in rows)
    assert json.loads((output/'protocol.json').read_text())['status'] == 'COMPLETE'
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['all_subjects'] == 32
    assert audit['evaluated_rows'] == 64
    assert audit['forbidden_files_read'] == []


def test_complete_all_trial_covariance_pipeline_from_original_matrices(tmp_path):
    source = tmp_path/'source'; make_cases(source)
    bank = tmp_path/'bank'; output = tmp_path/'all_trials'
    with threadpool_limits(limits=2):
        jb.prepare(source, bank, degrees=(2,), device='numpy')
        rows = rt.analyze(source, output, bank)
    assert rows
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['status'] == 'COMPLETE'
    assert protocol['all_original_rows_used_in_metric'] is True
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['all_original_rows_in_fit_and_metric'] == 160
    assert audit['forbidden_files_read'] == []
