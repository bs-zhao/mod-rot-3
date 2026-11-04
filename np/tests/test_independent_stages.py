"""S1-S7 each independently executable; no multi-stage dispatcher."""
from pathlib import Path
import runpy
from modrot3.cli import parse_args


STEPS = (
    ('s1_generate.py', 'generate'),
    ('s2_precheck.py', 'precheck'),
    ('s3_train.py', 'train'),
    ('s4_evaluate.py', 'report'),
    ('s5_extract_z.py', 'extract'),
    ('s6_pca.py', 'pca_report'),
    ('s7_regression.py', 'regression_report'),
)


def test_each_step_contains_real_stage_implementation():
    root = Path(__file__).resolve().parents[1]
    assert not (root/'run.py').exists()
    for filename, implementation in STEPS:
        p = root/filename
        assert p.is_file()
        namespace = runpy.run_path(str(p))  # __main__ guard does NOT execute stage
        assert callable(namespace[implementation]), p
        assert 'all' not in namespace, p


def test_training_condition_has_same_defaults_as_original():
    args = parse_args('test', ['--model', 'CR3', '--beta', '0', '--latent-dim', '8'])
    assert (args.model,args.beta,args.latent_dim,args.context_min,args.context_max) == ('CR3',0,8,.25,.8)
    assert args.data_seed == 20261028