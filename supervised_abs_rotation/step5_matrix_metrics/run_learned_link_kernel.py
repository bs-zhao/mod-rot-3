"""BF: conditional UX kernel moments after a single globally learned output link."""
import argparse
import csv
import hashlib
import json

import torch
from threadpoolctl import threadpool_limits

import run as wave
import joint_warp_source
import learned_response_profiles as learned
import run_conditional_kernel_moments as kernel


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', action='store_true')
    args = p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        evidence = wave.ROOT/'outputs/supervised_abs_rotation/waveBF_learned_link_four_model_evidence'
        evidence.mkdir(parents=True, exist_ok=True)
        common, all_rows, model_results = None, [], []
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else ('CR3',)):
            source = wave.io.default_step2(wave.ROOT, model)
            parent = source.parent/'step5_matrix_metrics'
            warp_folder = parent/'waveBE_global_warped_GP_source_seed438'
            if not warp_folder.exists():
                joint_warp_source.analyze(source, warp_folder)
            folder = parent/'matrix_response_global_warp_learned_seed438'
            if not folder.exists():
                learned.prepare(source, folder, parent/'matrix_response_table_seed383', warp_folder)
            output = parent/'waveBF_learned_global_link_kernel_seed430'
            if not output.exists():
                kernel.analyze(source, output, folder, centers=32, bank_factory=learned.MatrixBank)
            protocol = json.loads((output/'protocol.json').read_text())
            digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
            if protocol['status']!='COMPLETE' or digest!=protocol['scores_sha256']:
                raise ValueError('Preserve incomplete or changed learned-link experiment')
            with (output/'metric_summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            passed = {(r['channel'],r['metric']) for r in rows if r['all_planes_pass']=='True' and r['is_control']!='True'}
            common = passed if common is None else common & passed
            all_rows.extend(dict(model=model, **r) for r in rows)
            model_results.append(dict(model=model, primary_grid_passes=len(passed), scores_sha256=digest))
            wave.csv_file(evidence/'all_model_evidence.csv', all_rows)
            wave.json_file(evidence/'policy.json', dict(status='COMPLETE' if model==('C3' if args.batch else 'CR3') else 'RUNNING',
                order=['CR3','C1','C2','C3'], unchanged_criterion=True, models=model_results,
                common=[dict(channel=c,metric=m) for c,m in sorted(common)], fixed_prediction_error_veto=False))
            print(f'[WaveBF] {model}: {len(passed)} primary grid combinations, {len(common)} common', flush=True)
