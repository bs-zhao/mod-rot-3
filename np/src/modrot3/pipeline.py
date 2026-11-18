"""Shared, stage-independent paths, data adaptation and consistency checks.

Each experiment is independently runnable from np/sN_*.py; there is NO
single-command full pipeline. The old model and training implementations
are preserved in modlearn_rot.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from modlearn_rot.generators.dataset import load_dataset
from modlearn_rot.training.np.point.discrete import DiscreteData
from modlearn_rot.training.continuous import stable_run_seed, training_condition_tag

ROOT = Path(__file__).resolve().parents[3]
MODELS = ('C1','C2','C3','CR3','CR4','CR5')


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
    tag = training_condition_tag(args.beta, args.lambda_sparse,
                                 args.context_min, args.context_max,
                                 args.latent_dim)
    # CR3 now uses the same standardized decoder-X protocol as CR4/CR5.
    # Preserve all existing raw-X CR3 checkpoints, Z and reports in the
    # historical unsuffixed condition directory. New CR3 outputs are
    # isolated and cannot silently load or overwrite old raw-X runs.
    if getattr(args, 'model', None) == 'CR3':
        tag += '_xstd'
    return tag

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

def _cr4_bank() -> Path:
    return out_dir('CR4') / 's0_design' / 'CR4_3P_random500.npz'


def _cr5_bank() -> Path:
    return out_dir('CR5') / 's0_design' / 'CR5_3P_cr3bank500.npz'


def _cr5_ref() -> Path:
    return ROOT / 'np' / 'reference' / 'cr5' / 's0_s1_cr5.py'


def _cr5_oracle_ref() -> Path:
    return ROOT / 'np' / 'reference' / 'cr5' / 's2_oracle_cr5.py'


def _cr4_ref() -> Path:
    return ROOT / 'np' / 'reference' / 'cr4' / 's0_s1_cr4.py'


def _cr4_oracle_ref() -> Path:
    return ROOT / 'np' / 'reference' / 'cr4' / 's2_oracle_cr4.py'


def cr3_input_scaler():
    """CR3 standardized X: same train-equivalent bank moments as CR4/CR5.

    The original physical 500-trial bank remains untouched, including
    its zero amounts, sure outcomes and gamma-neutral conditions. Every
    TRAIN subject has the exact same X bank, so bank moments are identical
    to train-subject X moments without reading TEST choices or theta.
    """
    with np.load(_cr3_bank(), allow_pickle=False) as stored:
        x = np.asarray(stored['X'], dtype=np.float64)
    if x.shape != (500, 6) or not np.isfinite(x).all():
        raise ValueError('Invalid CR3 fixed bank for decoder X standardization')
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    if np.any(sd <= 1e-8):
        raise ValueError('CR3 bank has a near-constant X coordinate')
    return mu, sd


def cr4_input_scaler():
    """Identical bank X for all subjects => bank statistics equal TRAIN-X stats.

    CR4 ONLY. Return values for converting physical lottery X into the
    standardized coordinates used by *both* encoder and decoder. Physical X
    stays unchanged in the S1 dataset and downstream analytic generators.
    """
    with np.load(_cr4_bank(), allow_pickle=False) as stored:
        x = np.asarray(stored['X'], dtype=np.float64)
    if x.shape != (500, 6) or not np.isfinite(x).all():
        raise ValueError('Invalid CR4 bank for X standardization')
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    if np.any(sd <= 1e-8):
        raise ValueError('CR4 bank X has near-constant dimension')
    return mu, sd


def cr5_input_scaler():
    """CR5: preserve RAW CR3 bank in S1, normalize X for BOTH encoder and decoder.

    Since the identical 500-trial CR3 bank appears for every CR5 subject,
    computing these X-only moments on the bank is exactly equivalent to
    computing TRAIN-subject X moments, without accessing TEST theta/choice.
    """
    with np.load(_cr5_bank(),allow_pickle=False) as saved:
        x=np.asarray(saved['X'],dtype=np.float64)
    if x.shape != (500,6) or not np.isfinite(x).all():
        raise ValueError('Invalid CR5 shared X bank for preprocessing')
    mu=x.mean(axis=0)
    sd=x.std(axis=0,ddof=0)
    if np.any(sd <= 1e-8):
        raise ValueError('CR5 bank X has near-constant coordinate')
    return mu,sd


def _discrete_data(args, data):
    if args.model in ('CR3','CR4','CR5'):
        n = len(data.trial_counts)
        if not np.all(np.asarray(data.trial_counts) == 500):
            raise ValueError(f'{args.model} requires 500 trials per subject')
        x = np.asarray(data.x, np.float64).reshape(n, 500, 6)
        # CR3, CR4 and CR5 share the exact same X preprocessing policy:
        # six X coordinates are z-scored BEFORE the unchanged NP receives
        # them, so both context encoder and decoder see scaled X. Retain
        # physical X and all special/boundary trials in S1 on disk.
        mu, sd = {'CR3': cr3_input_scaler,
                  'CR4': cr4_input_scaler,
                  'CR5': cr5_input_scaler}[args.model]()
        x = (x - mu) / sd
        return DiscreteData(
            x=torch.from_numpy(np.asarray(x, np.float32).copy()),
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
    if args.model in ('CR3','CR4','CR5') and report.get('dataset_sha256') != sha256(dataset_path(args.model)):
        raise RuntimeError(f'{args.model} precheck belongs to a different dataset')

def _run_seed(args):
    if args.model == 'CR3':
        # Exact original s1_s4_cr2.py seed policy for CR3: beta-independent.
        return stable_run_seed(args.train_seed, 'CR2_4P', 0.0, args.repeat)
    return stable_run_seed(args.train_seed, args.model, args.beta, args.repeat)