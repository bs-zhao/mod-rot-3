"""CD: the same Shapley UX score with GCV-selected generic conditional models."""
import argparse
import csv
import hashlib
import json

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import run_shapley_profile_support as runner
from adaptive_coalition_models import AdaptiveCoalitionModels,ResidualAwareAdaptiveModels
from record_rotation_optima import WAVES
from run_operator_mode_support import audit_frames


def main(models,split_residual=False):
    tag='CE' if split_residual else 'CD'
    common=None;all_rows=[];evidence=wave.ROOT/f'outputs/supervised_abs_rotation/wave{tag}_adaptive_Shapley_four_model_evidence'
    evidence.mkdir(parents=True,exist_ok=True)
    for model in models:
        source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
        sources={tag:parent/WAVES[tag] for tag in ('BU','BU3','CB','CC')}
        output=parent/('waveCE_residual_aware_adaptive_Shapley_profiles_seed471' if split_residual else 'waveCD_adaptive_GCV_Shapley_profiles_seed471')
        if not output.exists():
            frames,descriptions=audit_frames(parent,model)
            refined=parent/'waveBU6_local_Shapley_refinement_seed489'
            protocol=json.loads((refined/'protocol.json').read_text())
            digest=hashlib.sha256((refined/'refined_frames.npz').read_bytes()).hexdigest()
            if protocol['status']!='COMPLETE' or digest!=protocol['frames_sha256']:raise ValueError('Frozen refinement changed')
            if hashlib.sha256((refined/'blind_refined_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
                raise ValueError('Frozen refined scores changed')
            with (refined/'blind_refined_scores.csv').open() as stream:rows=list(csv.DictReader(stream))
            with np.load(refined/'refined_frames.npz',allow_pickle=False) as saved:
                for row,q in zip(rows,saved['frames']):
                    frames.append(q.copy());descriptions.append(dict(source='BU6 '+row['start'],frame_sha256=digest))
            def factory(z,target,folder,**kwargs):
                model_type=ResidualAwareAdaptiveModels if split_residual else AdaptiveCoalitionModels
                return model_type(z,target,folder,sources,device=kwargs['device'])
            runner.analyze(source,parent/'waveBC2_conditional_kernel_centers32_seed430',parent/'matrix_response_table_seed383',
                output,extra_frames=frames,extra_descriptions=descriptions,model_factory=factory,
                protocol_overrides=dict(conditional_model='GCV-selected generic conditional emulators',
                    adaptive_runner_sha256=hashlib.sha256((wave.HERE/'run_adaptive_shapley_profiles.py').read_bytes()).hexdigest()))
        protocol=json.loads((output/'protocol.json').read_text())
        if protocol['status']!='COMPLETE' or hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
            raise ValueError('Adaptive Shapley run incomplete or changed')
        with (output/'metric_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
        passed={(r['channel'],r['metric']) for r in rows if r['all_planes_pass']=='True' and r['is_control']!='True'}
        common=passed if common is None else common&passed;all_rows.extend(dict(model=model,**r) for r in rows)
        wave.csv_file(evidence/'all_model_evidence.csv',all_rows)
        wave.json_file(evidence/'policy.json',dict(status='COMPLETE' if model==models[-1] else 'RUNNING',
            completed_models=list(dict.fromkeys(r['model'] for r in all_rows)),
            common=[dict(channel=c,metric=m) for c,m in sorted(common)],no_error_or_position_veto=True))
        print(f'[Wave{tag}] {model}: {len(passed)} strict grid combinations, {len(common)} common',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=wave.io.DIMS,default='CR3')
    p.add_argument('--batch',action='store_true');p.add_argument('--split-residual',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4);main(('CR3','C1','C2','C3') if args.batch else (args.model,),args.split_residual)
