"""Shared argument schema only; each NP experiment has its own runnable .py.

No multi-stage dispatch and no automatic execution of later experiments.
"""
from __future__ import annotations
import argparse

MODELS = ('C1','C2','C3','CR3')


def parse_args(description: str, argv=None, *, configure_parser=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--model', choices=MODELS, default='CR3')
    p.add_argument('--beta', type=float, default=0.0)
    p.add_argument('--lambda-sparse', type=float, default=0.0)
    p.add_argument('--latent-dim', type=int, default=8)
    p.add_argument('--context-min', type=float, default=.25)
    p.add_argument('--context-max', type=float, default=.80)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--learning-rate', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--max-epochs', type=int, default=500)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--min-delta', type=float, default=0.0)
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--device', default='auto')
    p.add_argument('--train-seed', type=int, default=31)
    p.add_argument('--split-seed', type=int, default=31)
    p.add_argument('--data-seed', type=int)
    p.add_argument('--design-seed', type=int, default=20261023)
    p.add_argument('--precheck-seed', type=int, default=20261006)
    p.add_argument('--subjects', type=int, default=5000)
    p.add_argument('--inference-batch', type=int, default=128)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--oracle-precheck', action='store_true')
    p.add_argument('--oracle-subjects', type=int, default=100)
    p.add_argument('--oracle-starts', type=int, default=4)
    if configure_parser is not None:
        configure_parser(p)
    args = p.parse_args(argv)
    if args.data_seed is None:
        args.data_seed = 20261028 if args.model == 'CR3' else 20261006
    if (args.latent_dim < 1 or args.batch_size < 1 or args.inference_batch < 1
        or args.beta < 0 or args.lambda_sparse < 0 or args.repeat < 1
        or args.max_epochs < 1 or args.patience < 1 or args.log_every < 1
        or args.subjects < 1 or not 0 < args.context_min <= args.context_max <= 1):
        p.error('Invalid hyperparameters')
    return args