"""CG2: blind all-subject optimization of the CG UX variance-profile score."""
import argparse

import torch
from threadpoolctl import threadpool_limits

import run as wave
from nested_variance_blind import Scorer
from record_rotation_optima import WAVES
from run_shapley_profile_blind import main


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',action='store_true')
    parser.add_argument('--model',choices=wave.io.DIMS,default='CR3');args=parser.parse_args()
    with threadpool_limits(4):
        torch.set_num_threads(4)
        for model in (('CR3','C1','C2','C3') if args.batch else (args.model,)):
            source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
            main(model,'CG','CG2',scorer_factory=Scorer,source_folder=parent/WAVES['CG'],
                 metric='nested_polynomial_variance_profile_UX_operator_columns_mean_soft_count_eps003')
