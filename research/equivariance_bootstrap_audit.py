"""FIT-only subject-and-stimulus bootstrap of the frozen functional detector.

This is uncertainty within a fixed NP; it is never counted as an independent
training replicate and never changes the frozen acceptance thresholds.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'np/src')]
import argparse,json
import numpy as np
import torch
from rotations.differential_program.data import load_frozen
from rotations.differential_program.frozen_decoder import _restore
from rotations.differential_program.symmetries import detect
from rotations.differential_program.run import write_json
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--draws',type=int,default=256);args=p.parse_args()
    path=Path(args.run);fit=json.loads((path/'fit.json').read_text());cfg=fit['config']
    target=path/'bootstrap_symmetries.json'
    if target.exists():raise FileExistsError(target)
    original=[r for r in fit['symmetries'] if r['accepted']]
    if not original:
        write_json(target,dict(status='no accepted original role'));return
    with np.load(fit['jet_cache']) as f:
        z=f['source_z'];x=f['common_x'];g=f['block_1_0'].reshape(len(z),len(x),-1)
    source=dict(z=z,x_fit=x[None],checkpoint_path=ROOT/'outputs/choice'/fit['model']/'s3_train'/fit['condition']/f"repeat{fit['repeat']:02d}"/'s3_train__checkpoint.pt')
    model,device=_restore(source,dict(teacher_device='cuda'));torch.set_num_threads(1)
    with torch.no_grad():
        logits=model.decode(torch.tensor(x,dtype=torch.float32,device=device)[None].expand(len(z),-1,-1),torch.tensor(z,dtype=torch.float32,device=device))
        eta=(logits[...,1]-logits[...,0]).cpu().numpy().astype(float)
    rng=np.random.default_rng(6101);alignments={r['property']:[] for r in original}
    for draw in range(args.draws):
        subjects=rng.integers(0,len(z),len(z));stimuli=rng.integers(0,len(x),len(x))
        _,_,reports=detect(eta[subjects][:,stimuli],g[subjects][:,stimuli],cfg['symmetry_relative_cap'],cfg['symmetry_gap_multiplier'],cfg['symmetry_activity_floor'])
        for role in original:
            match=[r for r in reports if r['property']==role['property'] and r['accepted']]
            if match:alignments[role['property']].append(abs(float(np.dot(match[0]['direction'],role['direction']))))
    summaries={}
    for role,values in alignments.items():
        summaries[role]=dict(accepted_draws=len(values),acceptance_fraction=len(values)/args.draws,
                median_axis_alignment=float(np.median(values)) if values else None,
                p05_axis_alignment=float(np.quantile(values,.05)) if values else None)
    report=dict(draws=args.draws,seed=6101,roles=summaries,no_theta=True,
                protocol='resample FIT subjects and common X independently with replacement; fixed decoder, PCA and detector thresholds; not independent NP validation')
    write_json(target,report);print(fit['model'],json.dumps(report,indent=2))


if __name__=='__main__':main()
