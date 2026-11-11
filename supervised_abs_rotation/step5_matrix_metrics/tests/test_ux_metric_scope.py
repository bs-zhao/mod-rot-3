import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import subset_ux
import ux_metric_scope as scope


def test_every_primary_subset_score_respects_the_user_sparsity_example():
    def curve(amplitudes):
        energy = np.sort(np.asarray(amplitudes)**2)[::-1]
        return np.tile(np.r_[1., 1-np.cumsum(energy)/energy.sum()][:, None], (1, 6))
    a = subset_ux.scores(curve([2., 2., .01, .01]))
    b = subset_ux.scores(curve([1., .5, .1, .1]))
    for metric in a:
        if scope.primary(metric):
            assert a[metric] <= b[metric]
    assert a['UX_minimum_U_subset_tail2_mean_l2'] < b['UX_minimum_U_subset_tail2_mean_l2']
    assert a['UX_subset_count_eps100_mean'] < b['UX_subset_count_eps100_mean']
    # Area rewards a dominant first variable in this example. Keep it as
    # a diagnostic instead of silently treating it as sparse support.
    assert a['UX_subset_budget_area_mean'] > b['UX_subset_budget_area_mean']
    assert not scope.primary('UX_subset_budget_area_mean')
    assert not scope.primary('rank1_UX_prediction_tail_l2')
    assert scope.primary('rank2_UX_prediction_tail_l2')
