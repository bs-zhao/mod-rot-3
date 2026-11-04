"""Binary-choice baseline: exact source generators and NP, clean orchestration.

Four models only: C1/C2/C3 and the exact CR3 four-parameter 500-trial bank.
No rotation objective and no true theta is used in training or z extraction.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from modlearn_rot.generators.registry import get_model
from modlearn_rot.generators.dataset import (
    load_dataset, generate_dataset, save_dataset, summarize_dataset,
)
from modlearn_rot.task_specs import get_task_spec_for_model
from modlearn_rot.utils.io import save_json
from modlearn_rot.utils.paths import stage_file
from modlearn_rot.diagnostics.precheck import PrecheckSuite
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig, DiscreteNeuralProcess
from modlearn_rot.training.np.point.discrete import (
    DiscreteData, DiscreteNPTrainer, DiscreteNPTrainingConfig,
)
from modlearn_rot.training.continuous import (
    split_subject_indices, make_fixed_context_masks, stable_run_seed,
    training_condition_tag, save_checkpoint, save_history_csv, load_checkpoint,
)

ROOT = Path(__file__).resolve().parents[3]
MODELS = ('C1', 'C2', 'C3', 'CR3')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def out_dir(model: str) -> Path:
    return ROOT / 'outputs' / 'choice' / model


def dataset_path(model: str) -> Path:
    return out_dir(model) / 's1_generate' / 's1_generate__dataset.npz'


def precheck_path(model: str) -> Path:
    return out_dir(model) / 's2_precheck' / 's2_precheck__report.json'


def condition(args) -> str:
    return training_condition_tag(args.beta, args.lambda_sparse,
                                  args.context_min, args.context_max,
                                  args.latent_dim)


def stage_folder(args, stage: str) -> Path:
    return out_dir(args.model) / stage / condition(args) / f'repeat{args.repeat:02d}'


def train_path(args) -> Path:
    return stage_folder(args, 's3_train') / 's3_train__checkpoint.pt'


def latent_path(args) -> Path:
    return stage_folder(args, 's4_evaluate') / 's4_evaluate__latent.npz'


def _run_legacy(script: Path, argv: list[str]):
    if not script.exists():
        raise FileNotFoundError(script)
    cmd = [sys.executable, str(script), *argv]
    print('> ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _cr3_refs():
    return (ROOT / 'np' / 'reference' / 's0_cr2_design_recovery.py',
            ROOT / 'np' / 'reference' / 'cr3' / 's0_s1_cr3.py')


def _cr3_bank() -> Path:
    return out_dir('CR3') / 's0_design' / 'CR3_4P_fixed500.npz'


def _cr2_bank() -> Path:
    return out_dir('CR2') / 's0_design_recovery' / 'CR2_4P_seed20261023_fixed500.npz'


def generate(args):
    target = dataset_path(args.model)
    if target.exists():
        if args.overwrite:
            raise RuntimeError('Never overwrite a generated dataset in place; archive its output directory first.')
        print('Existing dataset preserved:', target)
        return
    if args.model == 'CR3':
        cr2, cr3 = _cr3_refs()
        # CR3 uses the original CR2-4P Fisher-optimized 500-item bank.
        if not _cr2_bank().is_file():
            _run_legacy(cr2, ['--mode', 'design', '--variants', '4',
                              '--seed', str(args.design_seed)])
        if not _cr3_bank().is_file():
            _run_legacy(cr3, ['--stage', 'design', '--seed', str(args.design_seed)])
        _run_legacy(cr3, ['--stage', 'dataset', '--subjects', str(args.subjects),
                          '--data-seed', str(args.data_seed)])
        return
    model = get_model(args.model)
    data = generate_dataset(model, args.data_seed)
    save_dataset(data, target)
    task = get_task_spec_for_model(model)
    save_json(dict(model=model.metadata(), generator=model.manifest_extras(),
                   task_spec=task.__dict__, data_seed=args.data_seed),
              target.with_name('s1_generate__manifest.json'))
    save_json(dict(model_id=model.model_id, response_family=model.response_family,
                   has_x=True, module=task.module, **summarize_dataset(data)),
              target.with_name('s1_generate__summary.json'))
    print(f'Generated original {args.model} binary-choice data:', target)


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


def _discrete_data(args, data):
    if args.model == 'CR3':
        # Exactly the original CR3 fast reshape; same trial and choice ordering.
        n = len(data.trial_counts)
        return DiscreteData(x=torch.from_numpy(np.asarray(data.x, np.float32).reshape(n,500,6).copy()),
                            choice=torch.from_numpy(np.asarray(data.y, np.int64).reshape(n,500).copy()),
                            trial_mask=torch.ones((n,500), dtype=torch.bool))
    # Fast contiguous fill equivalent to upstream DiscreteData.from_generated.
    # The original C-series generator emits each subject's trials contiguously.
    n = len(data.trial_counts)
    counts = np.asarray(data.trial_counts,dtype=int)
    offsets = np.r_[0,np.cumsum(counts)]
    if offsets[-1] != len(data.subject_index):
        raise ValueError('Trial counts do not sum to observations')
    if not np.array_equal(data.subject_index,np.repeat(np.arange(n),counts)):
        return DiscreteData.from_generated(data)  # original ordering fallback
    t = int(counts.max()); d = int(data.x.shape[-1])
    x = torch.zeros((n,t,d),dtype=torch.float32)
    choice = torch.zeros((n,t),dtype=torch.long)
    mask = torch.zeros((n,t),dtype=torch.bool)
    for i in range(n):
        a,b=int(offsets[i]),int(offsets[i+1])
        x[i,:counts[i]]=torch.as_tensor(np.asarray(data.x[a:b],np.float32))
        choice[i,:counts[i]]=torch.as_tensor(np.asarray(data.y[a:b]).reshape(-1),dtype=torch.long)
        mask[i,:counts[i]]=True
    return DiscreteData(x=x,choice=choice,trial_mask=mask)


def _require_qualified(args):
    path = precheck_path(args.model)
    if not path.is_file():
        raise FileNotFoundError(f'Run precheck first: {path}')
    report = json.loads(path.read_text(encoding='utf-8'))
    if not report.get('qualified'):
        raise RuntimeError('Dataset was not qualified by precheck')
    if args.model == 'CR3' and report.get('dataset_sha256') != sha256(dataset_path(args.model)):
        raise RuntimeError('CR3 precheck belongs to a different dataset')


def _run_seed(args):
    if args.model == 'CR3':
        # Exact original s1_s4_cr2.py seed policy for CR3: beta-independent.
        return stable_run_seed(args.train_seed, 'CR2_4P', 0.0, args.repeat)
    return stable_run_seed(args.train_seed, args.model, args.beta, args.repeat)


def train(args):
    _require_qualified(args)
    ckpt = train_path(args)
    if ckpt.exists() and not args.overwrite:
        old = load_checkpoint(ckpt)
        source = old.get('extra_state', {})
        if source.get('dataset_sha256') != sha256(dataset_path(args.model)):
            raise RuntimeError('Existing checkpoint belongs to a different dataset')
        if (int(old['model_config']['latent_dim']) != args.latent_dim or
            abs(float(old['training_config']['beta']) - args.beta)>1e-12 or
            abs(float(old['training_config']['lambda_sparse'])-args.lambda_sparse)>1e-12):
            raise RuntimeError('Existing checkpoint has mismatched beta/lambda/dimension')
        expected={'batch_size':args.batch_size,'learning_rate':args.learning_rate,
                  'weight_decay':args.weight_decay,'max_epochs':args.max_epochs,
                  'patience':args.patience,'context_min_fraction':args.context_min,
                  'context_max_fraction':args.context_max}
        for name,value in expected.items():
            if abs(float(old['training_config'][name])-float(value))>1e-10:
                raise RuntimeError(f'Existing checkpoint differs in {name}: {old["training_config"][name]} vs {value}. Use a different --repeat, not silent reuse.')
        print('Preserved existing NN checkpoint:', ckpt)
        return
    if ckpt.exists() and args.overwrite:
        raise RuntimeError('In-place checkpoint overwrite prohibited; change --repeat or archive old run.')
    generated = load_dataset(dataset_path(args.model))
    data = _discrete_data(args, generated)
    model_cfg = DiscreteNPConfig(x_dim=data.x_dim, latent_dim=args.latent_dim, num_choices=2)
    config = DiscreteNPTrainingConfig(beta=args.beta, lambda_sparse=args.lambda_sparse,
                                     learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                                     batch_size=args.batch_size, max_epochs=args.max_epochs,
                                     patience=args.patience, min_delta=args.min_delta,
                                     context_min_fraction=args.context_min,
                                     context_max_fraction=args.context_max,
                                     seed=_run_seed(args))
    splits = split_subject_indices(data.n_subjects, config.train_fraction,
                                   config.validation_fraction, args.split_seed)
    masks = make_fixed_context_masks(data, splits, args.context_min,
                                     args.context_max, args.split_seed)
    trainer = DiscreteNPTrainer(model_cfg, config, args.device)
    def progress(epoch, tr, val):
        if epoch == 0 or (epoch+1) % args.log_every == 0:
            print(f'{args.model} epoch={epoch+1} trainCE={tr["reconstruction"]:.5f} '
                  f'valCE={val["reconstruction"]:.5f} valACC={val["accuracy"]:.4f} '
                  f'betaKL={args.beta*tr["kl"]:.5f}', flush=True)
    result = trainer.fit(data, subject_splits=splits,
                         fixed_context_masks=masks, epoch_callback=progress)
    save_checkpoint(ckpt, result, model_cfg, config, splits, masks,
                    extra_state=dict(model_id=args.model, repeat=args.repeat,
                                     run_seed=_run_seed(args), condition=condition(args),
                                     dataset_sha256=sha256(dataset_path(args.model)),
                                     source='same original DiscreteNeuralProcess and DiscreteNPTrainer'))
    save_history_csv(result['history'], ckpt.parent/'s3_train__history.csv')
    save_json(dict(model=args.model, condition=condition(args), repeat=args.repeat,
                   model_config=model_cfg.to_dict(), training_config=config.to_dict(),
                   best_epoch=int(result['best_epoch'])+1,
                   validation_metrics=result['validation_metrics'],
                   test_metrics=result['test_metrics'],
                   dataset_sha256=sha256(dataset_path(args.model))),
              ckpt.parent/'s3_train__summary.json')
    print('Saved original NP architecture checkpoint:', ckpt)


def extract(args):
    _require_qualified(args)
    source = train_path(args)
    if not source.is_file():
        raise FileNotFoundError(f'Train first: {source}')
    ck = load_checkpoint(source)
    if ck.get('extra_state', {}).get('dataset_sha256') != sha256(dataset_path(args.model)):
        raise RuntimeError('Checkpoint/dataset mismatch')
    target = latent_path(args)
    if target.exists() and not args.overwrite:
        with np.load(target, allow_pickle=False) as old:
            if str(old['checkpoint_sha256']) != sha256(source):
                raise RuntimeError('Saved latent belongs to a different checkpoint; archive before re-extraction')
        print('Preserved existing all-trial Z:', target)
        return
    if target.exists():
        raise RuntimeError('Do not overwrite Z in place; archive the old S4 outputs first')
    generated = load_dataset(dataset_path(args.model))
    data = _discrete_data(args, generated)
    model = DiscreteNeuralProcess(DiscreteNPConfig(**ck['model_config']))
    model.load_state_dict(ck['model_state_dict'])
    device = torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else
                          'cpu' if args.device=='auto' else args.device)
    model = model.to(device).eval()
    mean = ck['preprocessing']['encoder_mean'].to(device).reshape(-1)
    std = ck['preprocessing']['encoder_std'].to(device).reshape(-1)
    z = np.zeros((data.n_subjects, args.latent_dim), dtype=np.float32)
    z_logvar = np.zeros_like(z)
    ids = {k:np.asarray(v, dtype=np.int64) for k,v in ck['subject_splits'].items()}
    with torch.no_grad():
        for split in ('train','validation','test'):
            subjects = ids[split]
            for offset in range(0, len(subjects), args.inference_batch):
                sub = subjects[offset:offset+args.inference_batch]
                x = data.x[sub].to(device)
                y = data.choice[sub].float().to(device).unsqueeze(-1)
                mask = data.trial_mask[sub].to(device)
                if not bool(mask.any(dim=1).all()):
                    raise ValueError('Every subject must have at least one observed trial')
                obs = (torch.cat((x,y), dim=-1)-mean)/std
                mu, lv = model.encode_context(obs[...,:data.x_dim],
                                               obs[...,data.x_dim:],mask)
                z[sub], z_logvar[sub] = mu.cpu().numpy(), lv.cpu().numpy()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, z=z, z_logvar=z_logvar,
                        theta_true=np.asarray(generated.parameters),
                        train_ids=ids['train'],validation_ids=ids['validation'],test_ids=ids['test'],
                        checkpoint_sha256=np.array(sha256(source)),
                        dataset_sha256=np.array(sha256(dataset_path(args.model))),
                        context_mode=np.array('all_observed_trials'),
                        context_trials_per_subject=np.asarray(generated.trial_counts))
    print(f'Saved ALL subjects Z using ALL observed trials: {target} shape={z.shape}')


def evaluate_np(args):
    from .nn_eval import report
    report(args)


def analyze(args):
    from .analysis import report
    report(args)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Binary-choice generator / precheck / NP / Z / PCA & Ridge')
    ap.add_argument('stage', choices=('generate','precheck','train','evaluate-np','extract','analyze','all','selftest'))
    ap.add_argument('--model', choices=MODELS, default='CR3')
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--lambda-sparse', type=float, default=0.0)
    ap.add_argument('--latent-dim', type=int, default=8)
    ap.add_argument('--context-min', type=float, default=.25)
    ap.add_argument('--context-max', type=float, default=.80)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--learning-rate', type=float, default=5e-4)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--max-epochs', type=int, default=500)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--min-delta', type=float, default=0.0)
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--train-seed', type=int, default=31)
    ap.add_argument('--split-seed', type=int, default=31)
    ap.add_argument('--data-seed', type=int, default=None)
    ap.add_argument('--design-seed', type=int, default=20261023)
    ap.add_argument('--precheck-seed', type=int, default=20261006)
    ap.add_argument('--subjects', type=int, default=5000)
    ap.add_argument('--inference-batch', type=int, default=128)
    ap.add_argument('--log-every', type=int, default=10)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--oracle-precheck', action='store_true', help='CR3: optional original Oracle MLE precheck')
    ap.add_argument('--oracle-subjects', type=int, default=100)
    ap.add_argument('--oracle-starts', type=int, default=4)
    args = ap.parse_args(argv)
    if args.data_seed is None:
        args.data_seed = 20261028 if args.model=='CR3' else 20261006
    if (args.latent_dim<1 or args.batch_size<1 or args.inference_batch<1 or
        args.beta<0 or args.lambda_sparse<0 or args.repeat<1 or args.max_epochs<1 or
        not 0<args.context_min<=args.context_max<=1):
        ap.error('Invalid hyperparameters')
    if args.stage == 'selftest':
        from .sanity import selftest
        selftest()
        return
    commands={'generate':generate,'precheck':precheck,'train':train,
              'evaluate-np':evaluate_np,'extract':extract,'analyze':analyze}
    if args.stage == 'all':
        for stage in ('generate','precheck','train','evaluate-np','extract','analyze'):
            print('\n===',args.model, stage,'===',flush=True)
            commands[stage](args)
    else:
        commands[args.stage](args)


if __name__ == '__main__':
    main()