"""Single-package entrypoint; no global rotation engine or suite registry."""
from __future__ import annotations
from .experiment import cli,execute
from . import fit,postfit


def main(argv=None):
    args,config=cli('functional_composition: one integrated ZXY rotation suite',argv)
    folder=execute(args,config,fit)
    if not args.dry_run:
        # The generic runner can skip identical completed experiments; only
        # produce diagnostics if fitted state exists in this process.
        from . import _LAST_FIT
        if _LAST_FIT is not None and not (folder/'functional__20_metrics.json').is_file():
            postfit(folder,config)
    return folder