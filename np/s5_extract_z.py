from pathlib import Path
import sys
# Support direct invocation as: python np/sN_*.py ...
NP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NP_ROOT / 'src'))
"""S5: Extract and freeze Z for every subject using ALL observed trials."""
import numpy as np
import torch
from modrot3.cli import parse_args
from modrot3.pipeline import _require_qualified, train_path, dataset_path, latent_path, sha256, _discrete_data
from modlearn_rot.generators.dataset import load_dataset
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig, DiscreteNeuralProcess
from modlearn_rot.training.continuous import load_checkpoint

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

if __name__ == '__main__':
    extract(parse_args('S5: extract all-subject all-trial Z using frozen NP'))