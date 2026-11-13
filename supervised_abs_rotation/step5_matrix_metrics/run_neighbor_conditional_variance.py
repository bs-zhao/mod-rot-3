"""CJ: all-subject data-only nearest-neighbor conditional variance support."""
import argparse
import hashlib
import json

import torch
from threadpoolctl import threadpool_limits

import run as wave
import run_shapley_profile_support as runner
from neighbor_conditional_variance import NeighborConditionalVariance
from run_predictive_shapley_variance import frozen_extra_frames


def main(models,neighbors=3,run_tag='CJ',local_linear=False,stable_ties=False):
    for model in models:
        source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
        output=parent/f'wave{run_tag}_knn_conditional_variance_seed471'
        if not output.exists():
            frames,descriptions=frozen_extra_frames(parent,model,additional_tags=('CF2','CG2',('CH3',501)))
            manifest=json.loads((parent/'matrix_response_table_seed383'/'protocol.json').read_text())['manifest_sha256']
            def factory(z,target,folder,**kwargs):
                return NeighborConditionalVariance(z,target,folder,manifest,device=kwargs['device'],neighbors=neighbors,
                    local_linear=local_linear,stable_ties=stable_ties)
            runner.analyze(source,parent/'waveBC2_conditional_kernel_centers32_seed430',parent/'matrix_response_table_seed383',output,
                extra_frames=frames,extra_descriptions=descriptions,model_factory=factory)
        protocol=json.loads((output/'protocol.json').read_text())
        if protocol['neighbors']!=neighbors:raise ValueError('Existing run uses a different neighbor count')
        if protocol.get('local_linear_conditional_mean',False)!=local_linear or protocol.get('stable_neighbor_tie_ordering',False)!=stable_ties:
            raise ValueError('Existing run uses different conditional-mean or tie handling settings')
        if protocol['status']!='COMPLETE' or hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
            raise ValueError('Nearest-neighbor conditional variance experiment incomplete or changed')
        print(f'[Wave{run_tag}] {model} COMPLETE',flush=True)


def parse_arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',action='store_true')
    parser.add_argument('--model',choices=wave.io.DIMS,default='CR3');parser.add_argument('--neighbors',type=int,default=3)
    parser.add_argument('--run-tag',default='CJ');parser.add_argument('--local-linear',action='store_true')
    parser.add_argument('--stable-ties',action='store_true');return parser.parse_args(argv)


if __name__=='__main__':
    args=parse_arguments()
    with threadpool_limits(4):
        torch.set_num_threads(4);main(('CR3','C1','C2','C3') if args.batch else (args.model,),args.neighbors,args.run_tag,args.local_linear,args.stable_ties)
