"""Package-local runner; no shared rotation dispatch or suite registry."""
from . import fit, postfit
from .experiment import cli, execute


def main(argv=None):
    args,config=cli("Frozen MLP soft-Y and direct-gradient rotation suite",argv)
    folder=execute(args,config,fit)
    from . import _LAST_RUN as current
    if not args.dry_run and current is not None:
        if not (folder/"teacher__summary.json").is_file():
            postfit(folder,config)
    return folder
