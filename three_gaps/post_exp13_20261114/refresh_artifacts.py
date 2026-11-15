"""Regenerate corrected diagnostic figures and metadata without re-running fits."""
import sys
from pathlib import Path
import numpy as np
import core as C
sys.path.insert(0,str(C.HERE/'baseline'))
from robustness import beta_cache

def main():
    for beta in (.00001,.0001):
        for channel in ('prob','logit'):
            directory=C.HERE/'baseline/results/robustness_sensitivity_v1'/f'CR3_rep1_{channel}_beta{beta:g}'
            if not (directory/'rotations.npz').exists(): continue
            c=beta_cache(beta,channel); R=np.load(directory/'rotations.npz')['frozen_rule__R'][0]
            C.heatmap(c,{'Identity (new NP)':np.eye(4),'EXP13 rule (new NP)':R,'Oracle (new NP)':C.oracle(c)},directory/'UT_heatmaps.png')
            C.dump(directory/'figure_note.json',dict(note='No frozen R from a different NP coordinate system is applied to this beta checkpoint.'))
    from importlib.metadata import version
    C.dump(C.HERE/'environment.json',{**C.environment(),
           'packages':{name:version(name) for name in ('scipy','pandas','scikit-learn','matplotlib')},
           'conda_env':'py11','date_in_user_timezone':'2026-11-14',
           'freeze_tag':'exp13-frozen-doc05-20261114',
           'frozen_source_commit':'bfda311d4baaf3669d774ee336a58ddb100da43d'})

if __name__=='__main__': main()
