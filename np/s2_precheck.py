from pathlib import Path
import sys
# Support direct invocation as: python np/sN_*.py ...
NP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NP_ROOT / 'src'))
"""S2: Original full generator precheck, including CR3 fixed-X provenance/FOSD."""
import json
import numpy as np
from modrot3.cli import parse_args
from modrot3.pipeline import (ROOT, dataset_path, precheck_path, sha256,
                              _cr3_bank, _cr4_bank, _cr4_oracle_ref, _cr5_bank,
                              _cr5_oracle_ref, _run_legacy)
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


def _verify_cr4(data):
    """Exact 3P shared bank, positive amounts, interior probabilities, hashes."""
    bank = _cr4_bank()
    if not bank.is_file():
        raise FileNotFoundError(f'Missing CR4 bank: {bank}')
    with np.load(bank, allow_pickle=False) as stored:
        x = np.asarray(stored['X'], dtype=np.float64)
    design_file = bank.parent / 'CR4_3P_design.json'
    design = json.loads(design_file.read_text(encoding='utf-8'))
    sys.path.insert(0, str(ROOT / 'np' / 'reference' / 'cr4'))
    from s0_s1_cr4 import verify_bank, logits
    if not verify_bank(x, design['amount_range'], design['probability_range']):
        raise ValueError('CR4 bank invalid: expected 500 independent-design interior rows')
    manifest_file = dataset_path('CR4').with_name('s1_generate__manifest.json')
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest['bank_sha256'] != sha256(bank) or design['bank_sha256'] != sha256(bank):
        raise RuntimeError('CR4 dataset/design bank SHA mismatch')
    if manifest['dataset_sha256'] != sha256(dataset_path('CR4')):
        raise RuntimeError('CR4 dataset SHA mismatch')
    n = len(data.parameters)
    if data.parameters.shape != (n,3) or not np.all(data.trial_counts == 500):
        raise ValueError('CR4 must have exactly 3 parameters, 500 trials per subject')
    if data.x.shape != (500*n, 6):
        raise ValueError('CR4 X trial dimension mismatch')
    if not np.array_equal(data.x.reshape(n,500,6), np.broadcast_to(x, (n,500,6))):
        raise ValueError('CR4 subject X differs from frozen continuous-random bank')
    # Audit numerical generator equivalence without scanning all 2.5M records.
    from scipy.special import expit
    ids = sorted(set((0, n//2, n-1)))
    max_error = max(float(np.max(np.abs(
        expit(logits(x[None], data.parameters[s:s+1])) -
        np.asarray(data.predictive_true).reshape(n,500)[s:s+1]
    ))) for s in ids)
    if max_error > 1e-12:
        raise RuntimeError(f'CR4 generator fidelity error {max_error}')
    return x, manifest, max_error

def _verify_cr5(data):
    """Verify exact CR3 trial bank/codes, 3P CR4 generator and strong provenance."""
    sys.path.insert(0,str(ROOT/'np'/'reference'/'cr5'))
    from s0_s1_cr5 import validate_immutable_bank,logits
    bank,codes,names,design=validate_immutable_bank()
    n=len(data.parameters)
    if data.parameters.shape != (n,3) or not np.all(np.asarray(data.trial_counts)==500):
        raise ValueError('CR5 must have exactly 3 parameters and 500 trials per subject')
    if data.x.shape != (n*500,6):
        raise ValueError('CR5 observed X dimension mismatch')
    if not np.array_equal(
        np.asarray(data.x).reshape(n,500,6),
        np.broadcast_to(bank,(n,500,6))
    ):
        raise RuntimeError('CR5 raw X does not equal CR3 bank in every subject')
    manifest_path=dataset_path('CR5').with_name('s1_generate__manifest.json')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['bank_sha256']!=sha256(_cr5_bank()) or
        manifest['dataset_sha256']!=sha256(dataset_path('CR5')) or
        manifest['cr3_source_sha256']!=design['cr3_source_sha256']):
        raise RuntimeError('CR5 dataset or CR3 bank provenance mismatch')
    from scipy.special import expit
    p=np.asarray(data.predictive_true).reshape(n,500)
    check_ids=sorted({0,n//2,n-1})
    error=max(float(np.max(np.abs(
        expit(logits(bank[None],data.parameters[s:s+1]))-p[s:s+1]
    ))) for s in check_ids)
    if error>1e-12:
        raise RuntimeError(f'CR5 3P CPT generator fidelity error {error}')
    return bank,codes,names,design,error


def precheck(args):
    target = dataset_path(args.model)
    if not target.is_file():
        raise FileNotFoundError(f'Generate first: {target}')
    data = load_dataset(target)
    if args.model == 'CR5':
        x,codes,names,design,logit_prob_error=_verify_cr5(data)
        p=np.asarray(data.predictive_true,dtype=float)
        y=np.asarray(data.y,dtype=np.int64)
        qualified=bool(np.isfinite(p).all() and np.all((p>0)&(p<1)) and
                       np.isin(y,[0,1]).all())
        group_counts={str(name):int((codes==j).sum()) for j,name in enumerate(names)}
        report=dict(
            model_id='CR5',model_version='3P_exact_CR3_bank500_alpha1',
            qualified=qualified,subjects=len(data.parameters),trials=500,
            parameter_names=['lambda_loss','gamma','tau'],alpha_fixed=1.0,
            original_cr3_stimulus_bank=True,original_cr3_trial_order=True,
            original_cr3_condition_counts=group_counts,
            source_cr3_bank_sha256=design['cr3_source_sha256'],
            bank_sha256=sha256(_cr5_bank()),dataset_sha256=sha256(target),
            mean_accuracy_bayes=float(np.maximum(p,1-p).mean()),
            response_rate=float(y.mean()),true_p_std=float(p.std()),
            parameter_correlation=np.corrcoef(data.parameters,rowvar=False).tolist(),
            checks={
                'exact_cr3_X_and_native_labels':{'passed':True,'counts':group_counts},
                'no_strict_FOSD':{'passed':True},
                'original_cr3_X_zero_amounts_preserved':{'passed':bool(np.any(x[:,[0,1,3,4]]==0))},
                'finite_CPT_alpha1_generator':{'passed':logit_prob_error<=1e-12,
                                               'max_abs_error':logit_prob_error},
                'valid_binary_choices':{'passed':qualified},
            },
            note='Exact CR3 stimulus-design / CR4 three-parameter function controlled contrast. Optional S2 oracle MLE+Fisher uses unchanged CR4 estimator.',
        )
        save_json(report,precheck_path(args.model))
        print('CR5 exact CR3 bank / 3P CPT precheck:',qualified)
    elif args.model == 'CR4':
        x, manifest, logit_prob_error = _verify_cr4(data)
        prob = np.asarray(data.predictive_true,dtype=float)
        y = np.asarray(data.y,dtype=int)
        qualified = bool(np.isfinite(prob).all() and
                         np.all((prob >= 0) & (prob <= 1)) and
                         np.isin(y,[0,1]).all())
        report = dict(
            model_id='CR4', model_version='3P_random500_alpha1',
            qualified=qualified, subjects=len(data.parameters), trials=500,
            parameter_names=['lambda_loss','gamma','tau'], alpha_fixed=1.0,
            mean_accuracy_bayes=float(np.maximum(prob,1-prob).mean()),
            response_rate=float(y.mean()), true_p_std=float(prob.std()),
            parameter_correlation=np.corrcoef(data.parameters,rowvar=False).tolist(),
            bank_sha256=sha256(_cr4_bank()), dataset_sha256=sha256(target),
            checks={
                'structure_provenance':{'passed':True},
                'six_independent_random_X_design':{'passed':True,
                    'note':'Unconstrained draws; independence is a sampling policy, not a finite-sample hypothesis test'},
                'no_zero_amounts_or_probability_boundaries':{'passed':True},
                'generator_matches_CR2_alpha1':{'passed':logit_prob_error<=1e-12,
                    'max_abs_error':logit_prob_error},
                'valid_binary_probabilities':{'passed':qualified},
            },
            note='CR4 adds to existing S1-S7. S2 Oracle MLE/Fisher optional via --oracle-precheck; no special trial classes.',
        )
        save_json(report, precheck_path(args.model))
        print('CR4 random-bank structural precheck:',qualified)
    elif args.model == 'CR3':
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
    if args.model == 'CR4' and args.oracle_precheck:
        _run_legacy(_cr4_oracle_ref(), ['--subjects',str(args.oracle_subjects),
                 '--starts',str(args.oracle_starts),
                 '--seed',str(args.precheck_seed)])
    if args.model == 'CR5' and args.oracle_precheck:
        _run_legacy(_cr5_oracle_ref(), ['--subjects',str(args.oracle_subjects),
                 '--starts',str(args.oracle_starts),
                 '--seed',str(args.precheck_seed)])
    if not report['qualified']:
        raise RuntimeError('Precheck failed; NN training blocked')
    return report

if __name__ == '__main__':
    precheck(parse_args('S2: original simulator precheck'))