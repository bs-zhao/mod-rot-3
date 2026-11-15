"""Postprocess all candidates with a pure validation-Y selection rule.

The selected name and R are persisted BEFORE any TEST metric is read.
This does not convert an exploratory benchmark into a fresh held-out test.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as C

def main():
    rows=[]
    for path in (Path(__file__).parent/'results').glob('*/*/prediction.json'):
        if 'pilot' in path.parts or not (path.parent/'COMPLETE.json').exists(): continue
        fitting=json.loads(path.read_text(encoding='utf8'))
        candidates={name:float(info['validation_mse']) for name,info in fitting.items()}
        chosen=min(candidates,key=lambda name:(candidates[name],name))
        # Pure held-out Y-MSE chooses among seven individually fitted model families/initializations.
        config=dict(rule='lowest held-out-subject Y-MSE across individually fitted candidates; lexical tie-break',
                    chosen=chosen,validation_mse=candidates[chosen],all_validation_mse=candidates,
                    prediction_sha256=C.io.sha256(path),T_used_in_selection=False,
                    exclusion='ensembles have no separate entry in the original prediction.json; not silently added',
                    status='Exploratory benchmark; rule is Y-only, but introduced during implementation after pilot')
        C.dump(path.parent/'Y_ONLY_SELECTION.json',config)
        rotations=np.load(path.parent/'rotations.npz')
        np.savez_compressed(path.parent/'selected_by_Y.npz',R=rotations[chosen+'__R'],
                            loss=rotations[chosen+'__loss'],axis_sd=rotations[chosen+'__sd'])
        # Only now look up the evaluation, after freezing the chosen name and rotations.
        met=pd.read_csv(path.parent/'metrics.csv'); met=met[met.method==chosen].iloc[0].to_dict()
        met.update(selection='validation_Y_MSE',result_path=str(path.parent.relative_to(C.HERE)),
                   validation_mse=candidates[chosen]); rows.append(met)
    pd.DataFrame(rows).to_csv(Path(__file__).parent/'results/Y_selected_metrics.csv',index=False)
    print('Y-only selections:',len(rows))

if __name__=='__main__': main()
