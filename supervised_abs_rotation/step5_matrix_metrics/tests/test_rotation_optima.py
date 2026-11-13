import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from record_rotation_optima import positions


def test_15_degree_minimum_is_retained_with_angle_and_score_gap_as_different_units():
    cases=[dict(id='baseline',angle_deg=0)]
    losses={'baseline':10.}
    for a in (15,30,45,60,75,90):
        key=f'angle{a}';cases.append(dict(id=key,angle_deg=a,pair_1based=[1,2]))
        losses[key]=9.98 if a==15 else 10. if a==90 else 10.1
    result=positions(losses,cases)
    assert result['minimum_grid_angle_deg']==15 and result['best_interior_axis_deviation_deg']==15
    assert abs(result['minimum_grid_relative_change_percent']+.2)<1e-12
    assert result['baseline_rank']==2
    import json
    bands=json.loads(result['near_optimal_grid_bands_json'])
    assert bands[0]['baseline_included'] and bands[0]['maximum_axis_deviation_deg']==15
    assert bands[0]['baseline_and_interior_points']==2  # 90-degree endpoint excluded
    assert bands[1]['maximum_axis_deviation_deg']==45
    assert result['minimum_grid_axis_deviation_deg']==15
    assert result['grid_points']==7
    assert abs(result['baseline_strictly_better_other_grid_points_percent']-100/6)<1e-12
    assert abs(result['baseline_regret_relative_grid_span_percent']-100*.02/.12)<1e-10


def test_exact_90_degree_tie_is_reported_as_reference_axes_with_all_ties_retained():
    cases=[dict(id='baseline',angle_deg=0)]
    losses={'angle90':10.-1e-12,'baseline':10.}
    for a in (15,30,45,60,75,90):
        key=f'angle{a}';cases.append(dict(id=key,angle_deg=a,pair_1based=[1,2]))
        losses[key]=10.-1e-12 if a==90 else 10.1
    result=positions(losses,cases)
    assert result['minimum_grid_angle_deg']==0
    assert 'angle90' in result['near_minimum_cases_json']
    assert result['minimum_grid_relative_change_percent']==0.
    assert result['raw_minimum_grid_relative_change_percent']<0.


def test_blind_axis_positions_distinguish_15_degree_turn_from_sign_permutation():
    import numpy as np
    from record_blind_axis_positions import axis_angles
    a=np.deg2rad(15);q=np.eye(4)
    q[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]
    result=axis_angles(np.eye(4),q[:,[2,0,3,1]]*np.array([-1.,1.,1.,-1.]))
    np.testing.assert_allclose(result['matched_maximum_angle_deg'],15.,atol=1e-10)
    np.testing.assert_allclose(result['matched_mean_angle_deg'],7.5,atol=1e-10)
