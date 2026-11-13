"""CF2: all-subject blind search for the common approximate CF logit criterion."""
import argparse

import torch
from threadpoolctl import threadpool_limits

import run as wave
from predictive_variance_blind import Scorer
from record_rotation_optima import WAVES
from run_shapley_profile_blind import main


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',action='store_true')
    parser.add_argument('--model',choices=wave.io.DIMS,default='CR3');args=parser.parse_args()
    with threadpool_limits(4):
        torch.set_num_threads(4)
        for model in (('CR3','C1','C2','C3') if args.batch else (args.model,)):
            source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
            main(model,'CF','CF2',scorer_factory=Scorer,source_folder=parent/WAVES['CF'],channel='logit_nn',
                metric='PRESS_variance_UX_columns_worst_soft_count_eps003',random_proposals=16,optimizer_starts=1,maxfev=40)
