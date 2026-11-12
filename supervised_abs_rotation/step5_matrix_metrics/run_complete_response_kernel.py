"""BK: retain the pure-U response mean alongside every normalized UX component."""
import argparse
import csv
import hashlib
import json

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import run_conditional_kernel_moments as kernel


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--nonaffine-main',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4);common=None;all_rows=[]
        tag='BL' if args.nonaffine_main else 'BK'
        evidence=wave.ROOT/f'outputs/supervised_abs_rotation/wave{tag}_complete_response_four_model_evidence'
        evidence.mkdir(parents=True,exist_ok=True)
        for model in ('CR3','C1','C2','C3'):
            source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
            output=parent/f'wave{tag}_complete_response_kernel_seed430';frame=None
            if model=='CR3':
                counter=parent/'waveBC3_conditional_kernel_blind_seed439'
                frozen=json.loads((counter/'blind_freeze.json').read_text())
                if hashlib.sha256((counter/'blind_rotations.npz').read_bytes()).hexdigest()!=frozen['rotations_sha256']:
                    raise ValueError('Frozen counterexample frame changed')
                with np.load(counter/'blind_rotations.npz',allow_pickle=False) as saved:frame=saved['optimized'].copy()
            if not output.exists():kernel.analyze(source,output,parent/'matrix_response_table_seed383',centers=32,
                                                  include_main=True,audit_frame=frame,main_nonaffine=args.nonaffine_main)
            protocol=json.loads((output/'protocol.json').read_text())
            if protocol['status']!='COMPLETE' or hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
                raise ValueError('Complete response kernel run incomplete or changed')
            with (output/'metric_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
            passed={(r['channel'],r['metric']) for r in rows if r['all_planes_pass']=='True' and r['is_control']!='True'}
            common=passed if common is None else common&passed;all_rows.extend(dict(model=model,**r) for r in rows)
            wave.csv_file(evidence/'all_model_evidence.csv',all_rows)
            wave.json_file(evidence/'policy.json',dict(status='COMPLETE' if model=='C3' else 'RUNNING',
                completed_models=list(dict.fromkeys(r['model'] for r in all_rows)),
                common=[dict(channel=c,metric=m) for c,m in sorted(common)],fixed_MSE_veto=False))
            print(f'[Wave{tag}] {model}: {len(passed)} complete-response grid combinations, {len(common)} common',flush=True)
