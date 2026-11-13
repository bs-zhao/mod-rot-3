"""CJ2: fresh seed-509 all-subject blind kNN covariance-mode search."""
import argparse
import json

import torch
from threadpoolctl import threadpool_limits

import run as wave
from neighbor_variance_blind import Scorer
from record_rotation_optima import WAVES
from run_shapley_profile_blind import main


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',action='store_true')
    parser.add_argument('--model',choices=wave.io.DIMS,default='CR3');parser.add_argument('--source-tag',choices=('CJ','CK'),default='CJ')
    parser.add_argument('--run-tag');parser.add_argument('--seed',type=int);parser.add_argument('--channel',choices=wave.io.CHANNELS)
    parser.add_argument('--support-key');args=parser.parse_args()
    local=args.source_tag=='CK';channel=args.channel or ('p_nn' if local else 'logit_nn')
    key=args.support_key or ('UX_operator_columns_worst_fractional_count_eps010' if local else 'UX_operator_columns_mean_soft_count_eps010')
    seed=args.seed if args.seed is not None else (521 if local else 509);tag=args.run_tag or args.source_tag+'2'
    with threadpool_limits(4):
        torch.set_num_threads(4)
        for model in (('CR3','C1','C2','C3') if args.batch else (args.model,)):
            source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
            original=parent/WAVES[args.source_tag];protocol=json.loads((original/'protocol.json').read_text())
            prefix='knn_local_linear_cross_covariance' if protocol.get('local_linear_conditional_mean',False) else 'knn_conditional_cross_covariance'
            def factory(source,original,output):return Scorer(source,original,output,channel=channel,metric_key=key)
            main(model,args.source_tag,tag,scorer_factory=factory,source_folder=original,channel=channel,seed=seed,
                metric=prefix+'_profile_'+key)
