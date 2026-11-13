"""CH: nonparametric quantile conditional-variance UX complexity."""
import argparse
import hashlib
import json

import torch
from threadpoolctl import threadpool_limits

import run as wave
import run_shapley_profile_support as runner
from quantile_conditional_variance import QuantileConditionalVariance
from pairwise_quantile_variance import PairwiseQuantileVariance
from run_predictive_shapley_variance import frozen_extra_frames


def main(models,occupancy=20,run_tag='CH',audit_latest=False,pairwise=False):
    model_type=PairwiseQuantileVariance if pairwise else QuantileConditionalVariance
    for model in models:
        source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
        output=parent/f'wave{run_tag}_quantile_conditional_variance_seed471'
        if not output.exists():
            tags=('CF2','CG2',('CH3',501)) if pairwise else ('CF2','CG2')
            frames,descriptions=frozen_extra_frames(parent,model,additional_tags=tags if audit_latest else ())
            manifest=json.loads((parent/'matrix_response_table_seed383'/'protocol.json').read_text())['manifest_sha256']
            def factory(z,target,folder,**kwargs):
                return model_type(z,target,folder,manifest,device=kwargs['device'],target_cell_occupancy=occupancy)
            runner.analyze(source,parent/'waveBC2_conditional_kernel_centers32_seed430',parent/'matrix_response_table_seed383',output,
                extra_frames=frames,extra_descriptions=descriptions,model_factory=factory,
                protocol_overrides=dict(latest_frozen_frames_included=audit_latest,conditional_covariance_estimator_class=model_type.__name__))
        protocol=json.loads((output/'protocol.json').read_text())
        if protocol['target_proper_cell_occupancy']!=occupancy:raise ValueError('Existing run uses a different bin-resolution policy')
        if protocol.get('latest_frozen_frames_included',False)!=audit_latest:raise ValueError('Existing run uses a different frozen-frame audit')
        if protocol.get('conditional_covariance_estimator_class','QuantileConditionalVariance')!=model_type.__name__:
            raise ValueError('Existing run uses a different covariance estimator')
        if protocol['status']!='COMPLETE' or hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
            raise ValueError('Quantile conditional-variance experiment incomplete or changed')
        print(f'[Wave{run_tag}] {model} COMPLETE',flush=True)


def parse_arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',action='store_true')
    parser.add_argument('--model',choices=wave.io.DIMS,default='CR3');parser.add_argument('--occupancy',type=int,default=20)
    parser.add_argument('--run-tag',default='CH')
    parser.add_argument('--audit-latest',action='store_true')
    parser.add_argument('--pairwise-correction',action='store_true')
    return parser.parse_args(argv)


if __name__=='__main__':
    args=parse_arguments()
    with threadpool_limits(4):
        torch.set_num_threads(4);main(('CR3','C1','C2','C3') if args.batch else (args.model,),args.occupancy,args.run_tag,args.audit_latest,args.pairwise_correction)
