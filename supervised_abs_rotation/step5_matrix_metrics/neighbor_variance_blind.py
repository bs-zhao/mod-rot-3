"""Reload the data-only kNN logit covariance-mode objective without labels."""
import json
from pathlib import Path

from neighbor_conditional_variance import NeighborConditionalVariance
from quantile_variance_blind import Scorer as ProfileScorer


class Scorer(ProfileScorer):
    def __init__(self,source,original,output,device='cuda',bank_folder=None,channel='logit_nn',
                 metric_key='UX_operator_columns_mean_soft_count_eps010'):
        protocol=json.loads((Path(original)/'protocol.json').read_text())
        super().__init__(source,original,output,device=device,bank_folder=bank_folder,
            channel=channel,metric_key=metric_key,
            model_factory=NeighborConditionalVariance,
            model_options=dict(neighbors=protocol['neighbors'],seed=protocol['neighbor_seed'],
                local_linear=protocol.get('local_linear_conditional_mean',False),
                stable_ties=protocol.get('stable_neighbor_tie_ordering',False)))
