"""AU2: report prediction error continuously; continue frozen tree grid candidates.

User steering: a .01 heldout-MSE threshold must not veto an otherwise
interesting structural candidate. Numerical-integrity checks remain.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import run_tree_function_complexity as trees


def truth(value):
    return value is True or value == 'True'


def candidate_keys(rows):
    return {(r['channel'], r['metric']) for r in rows
            if truth(r['all_planes_pass']) and not truth(r.get('is_control', False))}


def frozen_rows(source, folder):
    """Re-evaluate only the immutable score file, preserving approximation diagnostics."""
    protocol = json.loads((folder/'protocol.json').read_text())
    digest = hashlib.sha256((folder/'blind_scores.csv').read_bytes()).hexdigest()
    reveal = json.loads((folder/'reveal.json').read_text())
    if protocol['status'] != 'COMPLETE' or digest != protocol['scores_sha256'] or digest != reveal['scores_sha256']:
        raise ValueError('Cannot continue an incomplete or altered tree experiment')
    if protocol['source'] != str(Path(source).resolve()):
        raise ValueError('Frozen experiment belongs to a different matrix source')
    import csv
    with (folder/'blind_scores.csv').open() as stream:
        scores = list(csv.DictReader(stream))
    with (folder/'fit_quality.csv').open() as stream:
        quality = list(csv.DictReader(stream))
    index = json.loads((Path(source)/'index.json').read_text())
    grouped = defaultdict(dict)
    for row in scores:
        grouped[row['channel'], row['metric']][reveal['mapping'][row['anonymous_case']]] = float(row['loss'])
    rows = [dict(channel=c, metric=m, **wave.st.evaluate_curve(values, index['cases']), is_control=False)
            for (c, m), values in grouped.items()]
    return rows, dict(source_run=str(folder.resolve()), scores_sha256=digest,
        original_protocol_sha256=hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest(),
        original_prediction_diagnostics=quality, score_values_modified=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', default='waveAU_conditional_UX_tree_complexity_seed421')
    p.add_argument('--policy-name', default='waveAU2_tree_prediction_quality_report_only')
    args = p.parse_args()
    output = wave.ROOT/'outputs/supervised_abs_rotation'/args.policy_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain earlier continuation policy: {output}')
    output.mkdir(parents=True, exist_ok=True)
    policy = dict(status='RUNNING', model_order=['CR3', 'C1', 'C2', 'C3'],
        quality_policy='Prediction MSE is reported continuously; no fixed MSE veto',
        user_requested_policy_change=True, score_definition_unchanged=True,
        all_subjects_used=True, acceptance='provisional same-rule four-model grid candidate; stability still required',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), models=[])
    wave.json_file(output/'policy.json', policy)
    survivors = None
    with threadpool_limits(limits=4):
        for model in policy['model_order']:
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            folder = parent/args.run_name
            if not folder.exists():
                bank_folder = parent/'matrix_response_table_seed383'
                if not bank_folder.exists():
                    profiles.prepare(source, bank_folder)
                trees.analyze(source, folder, bank_folder)
            rows, audit = frozen_rows(source, folder)
            passed = candidate_keys(rows)
            survivors = passed if survivors is None else survivors & passed
            policy['models'].append(dict(model=model, **audit, grid_candidate_count=len(passed),
                common_candidates=[dict(channel=c, metric=m) for c, m in sorted(survivors)]))
            wave.json_file(output/'policy.json', policy)
            print(f'[WaveAU2] {model}: {len(passed)} provisional grid candidates, {len(survivors)} common; prediction errors reported', flush=True)
            if not survivors:
                policy['status'] = 'COMPLETE_NO_COMMON_GRID_CANDIDATE'
                wave.json_file(output/'policy.json', policy)
                print('[WaveAU2] The identical grid criterion failed; later models skipped', flush=True)
                return
    policy['status'] = 'COMPLETE_PROVISIONAL_FOUR_MODEL_GRID_CANDIDATE'
    wave.json_file(output/'policy.json', policy)
    print('[WaveAU2] Four-model grid candidate requires approximation/resampling/blind-rotation follow-up', flush=True)


if __name__ == '__main__':
    main()
