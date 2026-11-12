import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import continue_tree_function_wave as continuation


def test_prediction_error_does_not_veto_a_grid_candidate_but_controls_and_failed_planes_do():
    rows = [dict(channel='a', metric='sparse', adequate=False, all_planes_pass=True, is_control=False),
            dict(channel='a', metric='failed_plane', adequate=True, all_planes_pass=False, is_control=False),
            dict(channel='a', metric='dominance', adequate=True, all_planes_pass=True, is_control=True),
            dict(channel='b', metric='serialized', adequate='False', all_planes_pass='True', is_control='False')]
    assert continuation.candidate_keys(rows) == {('a', 'sparse'), ('b', 'serialized')}
