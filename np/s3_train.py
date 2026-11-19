from pathlib import Path
import sys
# Support direct invocation as: python np/sN_*.py ...
NP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NP_ROOT / 'src'))
"""S3: Train the original discrete-choice Neural Process with configurable hyperparameters."""
import numpy as np
from modrot3.cli import parse_args
from modrot3.pipeline import (dataset_path, train_path, latent_path, condition, sha256,
                               _require_qualified, _discrete_data, _run_seed,
                               cr3_input_scaler, _cr3_bank,
                               cr4_input_scaler,
                               cr5_input_scaler, _cr5_bank)
from modlearn_rot.generators.dataset import load_dataset
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig
from modlearn_rot.training.np.point.discrete import DiscreteNPTrainer, DiscreteNPTrainingConfig
from modlearn_rot.training.continuous import (load_checkpoint, save_checkpoint,
    save_history_csv, split_subject_indices, make_fixed_context_masks)
from modlearn_rot.utils.io import save_json

def train(args):
    _require_qualified(args)
    ckpt = train_path(args)
    if ckpt.exists() and not args.overwrite:
        old = load_checkpoint(ckpt)
        source = old.get('extra_state', {})
        if args.model in ('CR3', 'CR4', 'CR5'):
            pre = source.get('decoder_x_standardization')
            if not isinstance(pre, dict):
                raise RuntimeError('Checkpoint missing decoder X scaler')
            if args.model == 'CR4':
                mu, sd = cr4_input_scaler()
                if (pre.get('policy') != 'CR4:iid_uniform_theoretical_moments_v1' or
                    pre.get('dataset_sha256') != sha256(dataset_path('CR4')) or
                    not np.allclose(pre.get('mean', []), mu) or
                    not np.allclose(pre.get('sd', []), sd)):
                    raise RuntimeError('Existing CR4 checkpoint does not use current IID-X scaler')
            else:
                bank = {'CR3': _cr3_bank, 'CR5': _cr5_bank}[args.model]()
                if pre.get('bank_sha256') != sha256(bank):
                    raise RuntimeError('Existing checkpoint has missing/mismatched bank X scaler')
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
    input_transform = {}
    if args.model == 'CR4':
        x_mu, x_sd = cr4_input_scaler()
        input_transform = dict(decoder_x_standardization=dict(
            policy='CR4:iid_uniform_theoretical_moments_v1',
            mean=x_mu.tolist(), sd=x_sd.tolist(),
            dataset_sha256=sha256(dataset_path('CR4'))))
    elif args.model in ('CR3', 'CR5'):
        scaler={'CR3': cr3_input_scaler,
                'CR5': cr5_input_scaler}[args.model]
        bank={'CR3': _cr3_bank, 'CR5': _cr5_bank}[args.model]()
        x_mu, x_sd = scaler()
        input_transform = dict(
            decoder_x_standardization=dict(
                policy=f'{args.model}: physical X -> (X - bank_mean)/bank_sd',
                mean=x_mu.tolist(), sd=x_sd.tolist(),
                bank_sha256=sha256(bank)))
    save_checkpoint(ckpt, result, model_cfg, config, splits, masks,
                    extra_state=dict(model_id=args.model, repeat=args.repeat,
                                     run_seed=_run_seed(args), condition=condition(args),
                                     dataset_sha256=sha256(dataset_path(args.model)),
                                     source='same original DiscreteNeuralProcess and DiscreteNPTrainer',
                                     **input_transform))
    save_history_csv(result['history'], ckpt.parent/'s3_train__history.csv')
    save_json(dict(model=args.model, condition=condition(args), repeat=args.repeat,
                   model_config=model_cfg.to_dict(), training_config=config.to_dict(),
                   best_epoch=int(result['best_epoch'])+1,
                   validation_metrics=result['validation_metrics'],
                   test_metrics=result['test_metrics'],
                   dataset_sha256=sha256(dataset_path(args.model)),
                   **input_transform),
              ckpt.parent/'s3_train__summary.json')
    print('Saved original NP architecture checkpoint:', ckpt)

if __name__ == '__main__':
    train(parse_args('S3: train binary-choice NP from S1/S2 data'))