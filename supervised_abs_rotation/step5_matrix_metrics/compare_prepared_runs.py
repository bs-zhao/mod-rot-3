"""Strict comparison of four frozen scores backed by validated function banks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joint_bank as jb
import run as wave


def compare(paths, destination, bank_name='joint_bank_seed391'):
    if set(paths) != set(wave.io.DIMS):
        raise ValueError('All four datasets are required')
    all_rows, common, bank_common = {}, None, None
    for model, folder in paths.items():
        folder = Path(folder); protocol = json.loads((folder/'protocol.json').read_text())
        if (protocol['status'] != 'COMPLETE' or protocol['model'] != model or
            protocol['n_subjects'] != 5000 or protocol['Z_dim'] != wave.io.DIMS[model]):
            raise ValueError('Each dataset must have its own complete 5000-subject scores')
        digest = hashlib.sha256((folder/'blind_scores.csv').read_bytes()).hexdigest()
        reveal = json.loads((folder/'reveal.json').read_text())
        if digest != protocol['scores_sha256'] or digest != reveal['scores_sha256']:
            raise ValueError('Frozen scores changed')
        source = Path(protocol['source'])
        bank = jb.JointBank(source, source.parent/'step5_matrix_metrics'/bank_name)
        if bank.protocol['manifest_sha256'] != protocol['source_signature_sha256']:
            raise ValueError('Scoring did not use the validated source bank')
        comparable = {k: v for k, v in protocol.items() if k not in
            ('source', 'model', 'n_subjects', 'Z_dim', 'scores_sha256', 'status', 'source_signature_sha256')}
        source_comparable = {k: v for k, v in bank.protocol.items() if k not in
            ('source', 'model', 'n_subjects', 'Z_dim', 'status', 'manifest_sha256', 'device')}
        # auto and cuda are dispatch requests. Compare and verify the actual
        # numeric backend for every fitted degree instead of that request string.
        for degree in bank.protocol['degrees']:
            fit = json.loads((bank.folder/f'fit_poly{degree}.json').read_text())
            if fit['backend'] != bank.protocol['numeric_backend']:
                raise ValueError('Source numeric backend metadata is inconsistent')
        if common is not None and (comparable != common or source_comparable != bank_common):
            raise ValueError('Scoring or source-preparation rules differ across datasets')
        common, bank_common = comparable, source_comparable
        with (folder/'metric_summary.csv').open(newline='') as stream:
            all_rows[model] = {(r['estimator'], r['channel'], r['metric']): r for r in csv.DictReader(stream)}
    keys = set.intersection(*(set(rows) for rows in all_rows.values())); result = []
    for key in sorted(keys):
        row = dict(estimator=key[0], channel=key[1], metric=key[2],
            all_four_grid_pass=all(all_rows[m][key]['all_planes_pass'] == 'True' and
                all_rows[m][key]['adequate'] == 'True' and all_rows[m][key]['is_control'] == 'False'
                for m in wave.io.DIMS), stability_verified=False, blind_optimization_verified=False)
        for model in wave.io.DIMS:
            r = all_rows[model][key]
            row[model+'_rank'] = r['baseline_rank']
            row[model+'_planes'] = r['success_planes']+'/'+r['n_planes']
        result.append(row)
    wave.csv_file(destination, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', default='waveR_local_joint_seed397')
    p.add_argument('--bank-name', default='joint_bank_seed391')
    args = p.parse_args()
    paths = {model: wave.io.default_step2(wave.ROOT, model).parent/'step5_matrix_metrics'/args.run_name
             for model in wave.io.DIMS}
    destination = wave.ROOT/'outputs'/'supervised_abs_rotation'/f'step5_{args.run_name}_four_models.csv'
    rows = compare(paths, destination, args.bank_name)
    print(json.dumps(dict(four_model_grid_passes=sum(r['all_four_grid_pass'] for r in rows),
                          output=str(destination), stability_and_optimization_required=True)))


if __name__ == '__main__':
    main()
