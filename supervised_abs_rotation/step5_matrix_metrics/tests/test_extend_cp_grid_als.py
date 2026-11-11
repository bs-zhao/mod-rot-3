import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_bank
import run_cp_grid_als as grid
import extend_cp_grid_als as extend
from test_approximate_ux_function import shared_fixture


def test_rank_extension_reproduces_geometry_and_preserves_all_original_fit_losses(tmp_path):
    import json
    source = tmp_path/'step2'; shared_fixture(source); torch.set_num_threads(2)
    bank = tmp_path/'step5_matrix_metrics/joint_bank_seed391'
    base, output = tmp_path/'base', tmp_path/'extended'
    with threadpool_limits(limits=2):
        joint_bank.prepare(source, bank, degrees=(4,), device='numpy')
        grid.analyze(source, base, ranks=(1, 2), sweeps=1, starts=1, device='cpu')
        rows = extend.analyze(base, output, ranks=(3,), sweeps=1, device='cpu')
    assert rows
    with np.load(base/'anonymous_fit_losses.npz') as a, np.load(output/'anonymous_fit_losses.npz') as b:
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name])
        assert 'rank3' in b.files
    assert json.loads((output/'resume_audit.json').read_text())['original_low_rank_fits_unchanged']
    assert json.loads((output/'protocol.json').read_text())['status'] == 'COMPLETE'
