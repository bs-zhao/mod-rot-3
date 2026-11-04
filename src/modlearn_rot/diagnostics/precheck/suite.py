from __future__ import annotations

import csv
import numpy as np

from ...utils.io import save_json
from ...utils.paths import stage_file
from ...utils.seeds import make_rng_streams
from .validity import run_validity
from .relevance import run_relevance
from .local_identifiability import run_local_identifiability
from .minimality import run_minimality
from .global_identifiability import run_global_identifiability
from .oracle_recovery import run_oracle_recovery
from .information_balance import run_information_balance
from .functional_distinguishability import run_functional_distinguishability
from .design_information import run_design_information
from .predictive_ceiling import run_predictive_ceiling
from .response_balance import run_response_balance


class PrecheckSuite:
    STAGE="s2_precheck"

    def __init__(self,model,dataset,seed=20261006):
        self.model=model; self.dataset=dataset; self.seed=int(seed)

    def run(self,save_outputs=True):
        rng=make_rng_streams(self.seed,(
            "relevance","local_identifiability","minimality",
            "global_identifiability","information_balance",
            "functional_distinguishability",
        ))
        oracle=run_oracle_recovery(self.model,self.dataset)
        checks={
            "validity":run_validity(self.model,self.dataset),
            "relevance":run_relevance(self.model,rng["relevance"]),
            "local_identifiability":run_local_identifiability(self.model,rng["local_identifiability"]),
            "minimality":run_minimality(self.model,rng["minimality"]),
            "global_identifiability":run_global_identifiability(self.model,rng["global_identifiability"]),
            "oracle_recovery":oracle,
            "information_balance":run_information_balance(self.model,rng["information_balance"]),
            "functional_distinguishability":run_functional_distinguishability(self.model,rng["functional_distinguishability"]),
            "design_information":run_design_information(self.model,self.dataset),
            "predictive_ceiling":run_predictive_ceiling(
                self.model,self.dataset,oracle_estimates=oracle["estimates"]
            ),
        }
        if self.model.response_family=="discrete_choice":
            checks["response_balance"]=run_response_balance(
                self.model,self.dataset
            )
        report={
            "model_id":self.model.model_id,
            "model_version":self.model.model_version,
            "precheck_seed":self.seed,
            "qualified":all(bool(v.get("passed",False)) for v in checks.values()),
            "checks":checks,
        }
        if save_outputs:
            self._save(report)
        return report

    def _save(self,report):
        save_json(report,stage_file(self.model,self.STAGE,"report",".json"))
        with stage_file(self.model,self.STAGE,"summary",".csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["check","passed"])
            for name,result in report["checks"].items():
                w.writerow([name,bool(result.get("passed",False))])
            w.writerow(["OVERALL",bool(report["qualified"])])

        for name,result in report["checks"].items():
            arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
            scalars={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
            if arrays:
                np.savez_compressed(stage_file(self.model,self.STAGE,name,".npz"),**arrays)
            if scalars:
                save_json(scalars,stage_file(self.model,self.STAGE,name+"_summary",".json"))

        oracle=report["checks"]["oracle_recovery"]
        est=np.asarray(oracle["estimates"])
        with stage_file(self.model,self.STAGE,"oracle_recovery",".csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            header=["subject_id"]
            for p in self.model.parameter_names:
                header += [p+"_true",p+"_estimate"]
            w.writerow(header)
            for sid in range(est.shape[0]):
                row=[sid]
                for j in range(self.model.n_factors):
                    row += [float(self.dataset.parameters[sid,j]),float(est[sid,j])]
                w.writerow(row)
