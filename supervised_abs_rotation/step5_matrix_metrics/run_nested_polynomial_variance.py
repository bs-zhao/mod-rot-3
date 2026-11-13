"""CG: coherent polynomial explained-covariance UX profile support."""
import argparse
import hashlib
import json

import torch
from threadpoolctl import threadpool_limits

import run as wave
import run_shapley_profile_support as runner
from nested_polynomial_variance import NestedPolynomialVariance
from run_predictive_shapley_variance import frozen_extra_frames


def main(models):
    for model in models:
        source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
        output=parent/'waveCG_nested_polynomial_variance_seed471'
        if not output.exists():
            frames,descriptions=frozen_extra_frames(parent,model)
            manifest=json.loads((parent/'matrix_response_table_seed383'/'protocol.json').read_text())['manifest_sha256']
            def factory(z,target,folder,**kwargs):
                return NestedPolynomialVariance(z,target,folder,manifest,device=kwargs['device'])
            runner.analyze(source,parent/'waveBC2_conditional_kernel_centers32_seed430',parent/'matrix_response_table_seed383',output,
                extra_frames=frames,extra_descriptions=descriptions,model_factory=factory,
                protocol_overrides=dict(nested_covariance_runner_sha256=hashlib.sha256((wave.HERE/'run_nested_polynomial_variance.py').read_bytes()).hexdigest()))
        protocol=json.loads((output/'protocol.json').read_text())
        if protocol['status']!='COMPLETE' or hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
            raise ValueError('Nested covariance experiment incomplete or changed')
        print(f'[WaveCG] {model} COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',action='store_true')
    parser.add_argument('--model',choices=wave.io.DIMS,default='CR3');args=parser.parse_args()
    with threadpool_limits(4):
        torch.set_num_threads(4);main(('CR3','C1','C2','C3') if args.batch else (args.model,))
