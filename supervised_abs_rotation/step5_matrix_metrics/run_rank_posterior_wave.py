"""Continue the frozen ordinal GP posterior UX family through all four models."""
import csv
import hashlib
import json

import torch
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as raw
import rank_response_profiles as ranked
import run_matrix_gp_ard as gp
import run_gp_posterior_support as posterior


def completed_rows(folder):
    protocol = json.loads((folder/'protocol.json').read_text())
    digest = hashlib.sha256((folder/'blind_scores.csv').read_bytes()).hexdigest()
    if protocol['status'] != 'COMPLETE' or digest != protocol['scores_sha256']:
        raise ValueError(f'Preserve incomplete or altered source wave: {folder}')
    with (folder/'metric_summary.csv').open() as stream:
        return list(csv.DictReader(stream)), digest


def main():
    output = wave.ROOT/'outputs/supervised_abs_rotation/waveBA2_rank_posterior_four_model_continuation'
    output.mkdir(parents=True, exist_ok=True)
    policy = dict(status='RUNNING', order=['CR3', 'C1', 'C2', 'C3'],
        definition='same global-normal-rank target, W-only X witnesses, GP fit and posterior-effect support in each model',
        prediction_quality_policy='continuous diagnostic, no fixed MSE veto', models=[],
        code_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())
    survivors = None; all_rows = []
    for model in policy['order']:
        source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
        raw_folder = parent/'matrix_response_table_seed383'
        if not raw_folder.exists():
            raw.prepare(source, raw_folder)
        folder = parent/'matrix_response_normal_rank_seed383'
        if not folder.exists():
            ranked.prepare(source, folder, raw_folder)
        gp_folder = parent/'waveAZ_normal_rank_conditional_UX_GP_seed416'
        if not gp_folder.exists():
            gp.analyze(source, gp_folder, folder, X_contrast='conditional_witness', bank_factory=ranked.MatrixBank)
        completed_rows(gp_folder)
        final = parent/'waveBA2_rank_GP_posterior_UX_seed416'
        if not final.exists():
            posterior.analyze(source, gp_folder, final)
        rows, digest = completed_rows(final)
        passed = {(r['channel'], r['metric']) for r in rows if r['all_planes_pass']=='True' and r['is_control']!='True'}
        survivors = passed if survivors is None else survivors & passed
        all_rows.extend(dict(model=model, **r) for r in rows)
        policy['models'].append(dict(model=model, source_run=str(final.resolve()), scores_sha256=digest,
            primary_grid_passes=len(passed), common=[dict(channel=c, metric=m) for c, m in sorted(survivors)]))
        wave.json_file(output/'policy.json', policy)
        wave.csv_file(output/'all_model_evidence.csv', all_rows)
        print(f'[RankPosteriorWave] {model}: {len(passed)} primary grid passes, {len(survivors)} common; quality reported', flush=True)
    policy['status'] = 'COMPLETE'
    wave.json_file(output/'policy.json', policy)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        main()
