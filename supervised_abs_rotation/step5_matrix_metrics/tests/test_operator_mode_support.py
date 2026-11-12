import json
import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import operator_mode_support as modes
import conditional_ux
import matrix_function_profiles as profiles
import run_conditional_kernel_moments as kernel
import run_operator_mode_support as runner
from test_matrix_metrics import make_cases


def test_rank_one_modes_reduce_to_UX_edge_support_and_preserve_multiple_strong_entries():
    rng=np.random.default_rng(851);energy=rng.uniform(.001,1.,size=(4,3))
    actual=modes.scores(energy[:,:,None]);expected=conditional_ux.sparsity(energy)
    for key,value in actual.items():
        if key.startswith('UX_operator_columns_'):
            np.testing.assert_allclose(value,expected[key.replace('UX_operator_','UX_')],atol=1e-12)
    sparse=np.array([2.,2.,.01,.01])**2;dense=np.array([1.,.5,.1,.1])**2
    assert modes.scores(sparse[:,None,None])['UX_operator_columns_worst_soft_count_eps010']<modes.scores(dense[:,None,None])['UX_operator_columns_worst_soft_count_eps010']


def test_same_edge_energy_with_two_independent_function_modes_costs_more_than_one():
    rank_one=np.zeros((2,1,2));rank_one[0,0,0]=1.
    rank_two=np.zeros_like(rank_one);rank_two[0,0]=[.5,.5]
    key='UX_operator_columns_worst_fractional_count_eps003'
    assert modes.scores(rank_two)[key]>modes.scores(rank_one)[key]


def test_actual_matrix_operator_pipeline_preserves_trace_endpoints_and_marks_scope(tmp_path):
    source=tmp_path/'source';make_cases(source)
    folder,original,output=tmp_path/'bank',tmp_path/'original',tmp_path/'out';torch.set_num_threads(2)
    with threadpool_limits(2):
        profiles.prepare(source,folder)
        kernel.analyze(source,original,folder,centers=32,device='cpu')
        runner.analyze(source,original,folder,output,device='cpu',extra_frames=[np.eye(4)])
    import csv
    with (output/'metric_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert rows and max(float(r['endpoint_error']) for r in rows)<1e-12
    assert all(r['is_control']=='True' for r in rows if r['metric'].startswith('Ui_innovation_') and not r['metric'].startswith('Ui_innovation_with_context_'))
    assert json.loads((output/'numerical_audit.json').read_text())['maximum_relative_SVD_trace_error']<1e-12
    assert (output/'frozen_frame_assessment.csv').exists()
