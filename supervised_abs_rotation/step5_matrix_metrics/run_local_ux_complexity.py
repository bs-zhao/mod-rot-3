"""Wave Z2: reference/column-scale-invariant UX support of frozen Z energies."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import fitted_bank
import prepare_response_bank
import run_ux_predictive_complexity as source_runner
import local_ux_complexity as local


def analyze(fields, output, seed=406):
    fields, output = Path(fields), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain prior local UX scores: {output}')
    old = json.loads((fields/'protocol.json').read_text())
    digest = hashlib.sha256((fields/'blind_scores.csv').read_bytes()).hexdigest()
    if old['status'] != 'COMPLETE' or digest != old['scores_sha256']:
        raise ValueError('Source UX energies need a complete frozen experiment')
    raw = (fields/'UX_energy_audit.json').read_bytes()
    audit = json.loads(raw)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=old['model'], n_subjects=old['n_subjects'], Z_dim=old['Z_dim'], source=old['source'],
        seed=seed, source_run=fields.name, source_scores_sha256=digest,
        source_energy_sha256=hashlib.sha256(raw).hexdigest(), source_code_sha256=old['code_sha256'],
        primary_scope='UX support before averaging; each local reference and X column normalized independently',
        all_subjects_used_in_source_fit=True, status='RUNNING', score_sign='lower for every dataset',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_local_ux_complexity.py', 'local_ux_complexity.py', 'tail_support.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    with (fields/'fit_quality.csv').open(newline='') as stream:
        quality = {(r['estimator'], r['channel'], float(r['sigma']), r['X_mode']): r for r in csv.DictReader(stream)}
    blind = []
    for row in audit:
        prefix = f's{row["sigma"]:g}_{row["X_mode"]}_'
        for order, key in (('U_first', 'U_first_energy'), ('U_total', 'U_total_energy')):
            for metric, value in local.scores(row[key]).items():
                blind.append(dict(anonymous_case=row['anonymous_case'], estimator=row['estimator'],
                    channel=row['channel'], metric=prefix+order+'_'+metric, loss=value))
    wave.csv_file(output/'blind_scores.csv', blind)
    new_digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    old_reveal = json.loads((fields/'reveal.json').read_text())
    if old_reveal['scores_sha256'] != digest:
        raise ValueError('Source reveal does not match source scores')
    reveal = old_reveal['mapping']
    wave.json_file(output/'reveal.json', dict(scores_sha256=new_digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    cases = json.loads((Path(old['source'])/'index.json').read_text())['cases']
    summaries = []
    for (name, channel, metric), values in grouped.items():
        parts = metric.split('_'); sigma = float(parts[0][1:]); mode = '_'.join(parts[1:3])
        qr = quality[name, channel, sigma, mode]
        summaries.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, cases), adequate=qr['adequate'] == 'True',
            validation_nmse=qr['validation_nmse'], is_control=False))
    wave.csv_file(output/'metric_summary.csv', summaries)
    wave.csv_file(output/'fit_quality.csv', list(quality.values()))
    wave.json_file(output/'read_audit.json', dict(source_experiment=str(fields.resolve()),
        source_scores_sha256=digest, source_energy_sha256=hashlib.sha256(raw).hexdigest(),
        forbidden_files_read=[], all_subjects_in_source_fit=old['n_subjects'],
        raw_response_or_behavioral_formula_used_by_rescorer=False))
    protocol.update(status='COMPLETE', scores_sha256=new_digest)
    wave.json_file(output/'protocol.json', protocol)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--source-run', default='waveZ_UX_complexity_seed405')
    p.add_argument('--run-name', default='waveZ2_local_UX_support_seed406')
    args = p.parse_args(); survivors = None
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            fields = parent/args.source_run
            if not fields.exists():
                folder = parent/'UX_complexity_source_seed383'
                if not folder.exists():
                    prepare_response_bank.prepare(source, folder, estimators=('poly3', 'poly5', 'rbf1', 'rbf2'))
                source_runner.analyze(source, fields, folder)
            rows = analyze(fields, parent/args.run_name)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows if r['adequate'] and r['all_planes_pass']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveZ2] {model}: {len(passed)} scale-balanced UX passes; {len(survivors)} common criteria', flush=True)
            if args.batch and not survivors:
                print('[WaveZ2] No identical criterion remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
