import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_warp_source
import matrix_function_profiles as raw
import learned_response_profiles as learned
from test_approximate_ux_function import shared_fixture


def test_learned_link_cache_uses_transformed_actual_rows_and_identity_control_is_affine(tmp_path):
    source = tmp_path/'source'; shared_fixture(source)
    warp_folder, raw_folder = tmp_path/'warp', tmp_path/'raw'
    torch.set_num_threads(2)
    with threadpool_limits(2):
        joint_warp_source.analyze(source, warp_folder, centers=8, knot_count=9, steps=2, chunk=71, device='cpu')
        raw.prepare(source, raw_folder)
        learned.prepare(source, tmp_path/'learned', raw_folder, warp_folder)
        learned.prepare(source, tmp_path/'identity', raw_folder, warp_folder, kind='identity')
        bank = learned.MatrixBank(source, tmp_path/'learned')
        identity = learned.MatrixBank(source, tmp_path/'identity')
    _, _, links = learned.load_links(warp_folder)
    np.testing.assert_array_equal(bank.responses.transpose(0, 2, 1).reshape(160, 2), learned.forward(bank.data.y, links))
    a = identity.responses.transpose(0, 2, 1).reshape(160, 2)
    for c in range(2):
        coef = np.polyfit(identity.data.y[:, c], a[:, c], 1)
        assert coef[0] > 0.
        np.testing.assert_allclose(a[:, c], np.polyval(coef, identity.data.y[:, c]), atol=1e-12)
