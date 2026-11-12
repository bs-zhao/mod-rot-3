"""BJ: group GP support after W-only hierarchy orthogonalization."""
import csv
import hashlib
import json

import torch
from threadpoolctl import threadpool_limits

import run as wave
import run_group_sparse_gp as runner
import hierarchical_group_gp as hierarchy


if __name__=='__main__':
    with threadpool_limits(limits=4):
        torch.set_num_threads(4);all_rows=[];common=None
        evidence=wave.ROOT/'outputs/supervised_abs_rotation/waveBJ_hierarchical_GP_four_model_evidence'
        evidence.mkdir(parents=True,exist_ok=True)
        for model in ('CR3','C1','C2','C3'):
            source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
            output=parent/'waveBJ_hierarchical_group_GP_seed448'
            if not output.exists():
                runner.analyze(source,parent/'waveBC2_conditional_kernel_centers32_seed430',parent/'matrix_response_table_seed383',output,
                    counterexample=parent/'waveBC3_conditional_kernel_blind_seed439' if model=='CR3' else None,
                    dictionary_factory=hierarchy.dictionary)
            protocol=json.loads((output/'protocol.json').read_text())
            if protocol['status']!='COMPLETE' or hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
                raise ValueError('Hierarchical GP experiment incomplete or changed')
            with (output/'metric_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
            passed={(r['channel'],r['metric']) for r in rows if r['all_planes_pass']=='True' and r['is_control']!='True'}
            common=passed if common is None else common&passed;all_rows.extend(dict(model=model,**r) for r in rows)
            wave.csv_file(evidence/'all_model_evidence.csv',all_rows)
            wave.json_file(evidence/'policy.json',dict(status='COMPLETE' if model=='C3' else 'RUNNING',
                completed_models=list(dict.fromkeys(r['model'] for r in all_rows)),
                common=[dict(channel=c,metric=m) for c,m in sorted(common)],fixed_MSE_veto=False))
            print(f'[WaveBJ] {model}: {len(passed)} grid combinations, {len(common)} common',flush=True)
