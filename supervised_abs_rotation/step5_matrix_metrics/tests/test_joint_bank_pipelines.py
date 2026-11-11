import hashlib
import json
import sys
from pathlib import Path

from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_bank as jb
import run_predictive_compression as rm
import run_joint_separation as rn
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
