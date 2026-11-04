from pathlib import Path
import sys
# Support direct invocation as: python np/sN_*.py ...
NP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NP_ROOT / 'src'))
"""S2: Original full generator precheck, including CR3 fixed-X provenance/FOSD."""
import json
import numpy as np
from modrot3.cli import parse_args
from modrot3.pipeline import ROOT, dataset_path, precheck_path, sha256, _cr3_bank, _run_legacy
from modlearn_rot.generators.registry import get_model
from modlearn_rot.generators.dataset import load_dataset
from modlearn_rot.diagnostics.precheck import PrecheckSuite
from modlearn_rot.utils.io import save_json

def _verify_cr3(data):
    bank = _cr3_bank()
    if not bank.is_file():
        raise FileNotFoundError(f'Missing original CR3 fixed-X bank: {bank}')
    with np.load(bank, allow_pickle=False) as d:
        x = np.asarray(d['X'], dtype=np.float64)
    if x.shape != (500, 6):
        raise ValueError(f'CR3 bank must be (500,6), got {x.shape}')
    # Import the exact FOSD and categorization functions from frozen reference.
    sys.path.insert(0, str(ROOT / 'np' / 'reference' / 'cr3'))
    from s0_s1_cr3 import dominance_audit, condition_counts, COUNTS
    if dominance_audit(x)['total']:
        raise ValueError('CR3 stimulus bank contains strictly dominant trials')
    observed = condition_counts(x)
    expected = dict(gain_only=COUNTS['gain_only'], loss_only=COUNTS['loss_only'],
                    gamma_neutral=COUNTS['gamma_neutral'])
    if any(observed[k] != v for k, v in expected.items()):
        raise ValueError(f'CR3 bank composition changed: {observed}')
    manifest_path = dataset_path('CR3').with_name('s1_generate__manifest.json')
    if not manifest_path.exists():
        raise FileNotFoundError('Missing CR3 dataset provenance manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['bank_sha256'] != sha256(bank):
        raise RuntimeError('CR3 dataset originated from a different stimulus bank')
    if manifest['dataset_sha256'] != sha256(dataset_path('CR3')):
        raise RuntimeError('CR3 dataset checksum mismatch')
    n = len(data.parameters)
    if data.parameters.shape != (n, 4) or not np.all(data.trial_counts == 500):
        raise ValueError('CR3 must have four parameters and 500 trials per subject')
    if data.x.shape != (500 * n, 6):
        raise ValueError('CR3 observed X geometry changed')
    if not np.array_equal(data.x.reshape(n, 500, 6)[0], x):
        raise ValueError('CR3 trial order no longer matches frozen bank')
    if not np.array_equal(data.x.reshape(n, 500, 6), np.broadcast_to(x, (n, 500, 6))):
        raise ValueError('CR3 subjects do not share the frozen 500-trial bank')
    return x

def precheck(args):
    target = dataset_path(args.model)
    if not target.is_file():
        raise FileNotFoundError(f'Generate first: {target}')
    data = load_dataset(target)
    if args.model == 'CR3':
        x = _verify_cr3(data)
        probability = np.asarray(data.predictive_true, dtype=float).reshape(len(data.parameters), 500)
        y = np.asarray(data.y, dtype=int).reshape(len(data.parameters), 500)
        qualified = bool(np.isfinite(probability).all() and
                         ((probability > 0) & (probability < 1)).all() and
                         np.isin(y, [0, 1]).all())
        report = dict(model_id='CR3', model_version='4P_fixed500',
                      qualified=qualified, subjects=len(data.parameters), trials=500,
                      parameter_names=['alpha', 'lambda_loss', 'gamma', 'tau'],
                      mean_accuracy_bayes=float(np.maximum(probability, 1-probability).mean()),
                      response_rate=float(y.mean()),
                      true_p_std=float(probability.std()),
                      parameter_correlation=np.corrcoef(data.parameters, rowvar=False).tolist(),
                      bank_sha256=sha256(_cr3_bank()),
                      dataset_sha256=sha256(target),
                      checks={'structure_and_provenance':{'passed':True},
                              'no_strict_FOSD':{'passed':True},
                              'binary_probability':{'passed':qualified}},
                      note='Exact original CR3 s2 structural checks plus bank provenance/FOSD audit. Optional original oracle MLE is available separately.')
        path = precheck_path(args.model)
        save_json(report, path)
        # Preserve old CR3 S2 report structure for provenance.
        print('CR3 original structural/FOSD precheck:', qualified)
    else:
        model = get_model(args.model)
        report = PrecheckSuite(model, data, args.precheck_seed).run(save_outputs=True)
        print(args.model, 'original full precheck qualified:', report['qualified'])
    if args.model == 'CR3' and args.oracle_precheck:
        _run_legacy(ROOT/'reference'/'cr3'/'s2_oracle_cr3.py',
                    ['--stage', 'mle', '--mle-subjects', str(args.oracle_subjects),
                     '--mle-starts', str(args.oracle_starts)])
    if not report['qualified']:
        raise RuntimeError('Precheck failed; NN training blocked')
    return report

if __name__ == '__main__':
    precheck(parse_args('S2: original simulator precheck'))