"""Launch only the effect_fingerprint suite and save package-local diagnostics."""
from .experiment import cli,execute
from . import fit,postfit,_LAST_FIT


def main(argv=None):
    args,config=cli("Effect-fingerprint rank rotation (independent suite)",argv)
    folder=execute(args,config,fit)
    # execute() persists the FIT R/U and evaluates theta afterwards.
    from . import _LAST_FIT as latest
    if latest is not None and not args.dry_run:
        from pathlib import Path
        if not (Path(folder)/"fingerprint__diagnostics.json").is_file():
            postfit(folder,config)
    return folder
