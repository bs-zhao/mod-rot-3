from pathlib import Path
import sys
# Support direct invocation as: python np/sN_*.py ...
NP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NP_ROOT / 'src'))
"""S3: Train the original discrete-choice Neural Process with configurable hyperparameters."""
import numpy as np
from modrot3.cli import parse_args
from modrot3.pipeline import (dataset_path, train_path, latent_path, condition, sha256,
                               _require_qualified, _discrete_data, _run_seed)
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

if __name__ == '__main__':
    train(parse_args('S3: train binary-choice NP from S1/S2 data'))