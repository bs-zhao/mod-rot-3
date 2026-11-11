"""Freeze unchanged UX losses while classifying dominance scores as controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import run as wave
import ux_metric_scope as scope


def analyze(source, output):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain prior scoped scores: {output}')
    old = json.loads((source/'protocol.json').read_text())
    data = (source/'blind_scores.csv').read_bytes(); digest = hashlib.sha256(data).hexdigest()
    reveal = json.loads((source/'reveal.json').read_text())
    if old['status'] != 'COMPLETE' or old['scores_sha256'] != digest or reveal['scores_sha256'] != digest:
        raise ValueError('Source UX experiment is not complete and frozen')
    output.mkdir(parents=True, exist_ok=True)
    (output/'blind_scores.csv').write_bytes(data)
    (output/'reveal.json').write_bytes((source/'reveal.json').read_bytes())
    with (source/'metric_summary.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['is_control'] = row['is_control'] == 'True' or not scope.primary(row['metric'])
    wave.csv_file(output/'metric_summary.csv', rows)
    if (source/'fit_quality.csv').exists():
        (output/'fit_quality.csv').write_bytes((source/'fit_quality.csv').read_bytes())
    protocol = dict(old, scope_source=str(source.resolve()), scope_source_scores_sha256=digest,
        primary_scope='relative tails at budgets >=2 and variable/component counts at 1%,3%,10%; dominance diagnostics are controls',
        scope_code_sha256=hashlib.sha256((wave.HERE/'ux_metric_scope.py').read_bytes()).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    wave.json_file(output/'read_audit.json', dict(source_run=str(source.resolve()),
        unchanged_numerical_scores_sha256=digest, forbidden_files_read=[], raw_WY_reread=False))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--source-run', required=True); p.add_argument('--run-name', required=True)
    args = p.parse_args(); parent = wave.io.default_step2(wave.ROOT, args.model).parent/'step5_matrix_metrics'
    rows = analyze(parent/args.source_run, parent/args.run_name)
    passed = [r for r in rows if str(r['adequate']) == 'True' and str(r['all_planes_pass']) == 'True' and not r['is_control']]
    print(f'[UX scope] {args.model}: {len(passed)} primary sparsity passes; unchanged numerical scores', flush=True)


if __name__ == '__main__':
    main()
