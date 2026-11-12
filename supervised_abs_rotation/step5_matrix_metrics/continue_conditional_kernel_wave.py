"""Continue the same derivative-free conditional UX kernel criterion in four models."""
import csv
import hashlib
import json

from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import run_conditional_kernel_moments as kernel


def main():
    output = wave.ROOT/'outputs/supervised_abs_rotation/waveBC_conditional_kernel_four_model_continuation'
    output.mkdir(parents=True, exist_ok=True)
    policy = dict(status='RUNNING', order=['CR3', 'C1', 'C2', 'C3'],
        criterion_definition_unchanged=True, Ui_centers=16, seed=430,
        prediction_quality_policy='continuous diagnostic, no fixed veto', models=[],
        code_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())
    common, all_rows = None, []
    for model in policy['order']:
        source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
        folder = parent/'matrix_response_table_seed383'
        if not folder.exists():
            profiles.prepare(source, folder)
        final = parent/'waveBC_conditional_kernel_UX_seed430'
        if not final.exists():
            kernel.analyze(source, final, folder)
        protocol = json.loads((final/'protocol.json').read_text())
        digest = hashlib.sha256((final/'blind_scores.csv').read_bytes()).hexdigest()
        if protocol['status'] != 'COMPLETE' or digest != protocol['scores_sha256']:
            raise ValueError('Retain incomplete or changed conditional moment experiment')
        with (final/'metric_summary.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        passed = {(r['channel'], r['metric']) for r in rows if r['all_planes_pass']=='True' and r['is_control']!='True'}
        common = passed if common is None else common & passed
        all_rows.extend(dict(model=model, **r) for r in rows)
        policy['models'].append(dict(model=model, source_run=str(final.resolve()), scores_sha256=digest,
            primary_grid_passes=len(passed), common=[dict(channel=c, metric=m) for c, m in sorted(common)]))
        wave.json_file(output/'policy.json', policy)
        wave.csv_file(output/'all_model_evidence.csv', all_rows)
        print(f'[ConditionalKernelWave] {model}: {len(passed)} primary grid passes, {len(common)} common', flush=True)
    policy['status'] = 'COMPLETE'
    wave.json_file(output/'policy.json', policy)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
