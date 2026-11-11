"""Wave P: input-only identifiability diagnostic, not a WY simplicity claim."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln
from threadpoolctl import threadpool_limits

import run as wave


def entropy(points, k):
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    n, d = points.shape
    distances = cKDTree(points).query(points, k=k+1, p=2, workers=1)[0][:, -1]
    if np.any(distances <= 0):
        raise ValueError('Coincident observations invalidate continuous kNN entropy')
    log_volume = d*.5*np.log(np.pi)-gammaln(d*.5+1.)
    return float(digamma(n)-digamma(k)+log_volume+d*np.mean(np.log(distances)))


def analyze(source, output, seed=395, neighbors=(3, 10, 30, 100)):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain old diagnostic: {output}')
    data = wave.io.MatrixCases(source, ('logit_nn', 'p_nn'), seed)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=data.index['model'], n_subjects=data.n, Z_dim=data.d, source=str(source.resolve()),
        seed=seed, neighbors=list(neighbors), status='RUNNING', all_subjects_used=True,
        score_sign='lower', role='input-only identifiability diagnostic; does not establish response simplicity',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_input_independence.py', 'matrix_io.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    u = data.u-data.u.mean(0); u /= np.sqrt(np.mean(u*u))
    joint = {k: entropy(u, k) for k in neighbors if k < data.n}
    blind, reveal, registration = [], {}, []
    for i, path in enumerate(data.paths):
        q, residual = wave.io.registration(data.u, data.coordinates(path)); registration.append(residual)
        ur = u @ q; reveal[f'case_{i:03d}'] = path.parent.name
        for k, h in joint.items():
            marginal = sum(entropy(ur[:, j], k) for j in range(data.d))
            blind.append(dict(anonymous_case=f'case_{i:03d}', estimator=f'knn{k}', channel='W_input_only',
                              metric='marginal_entropy_sum', loss=marginal))
            blind.append(dict(anonymous_case=f'case_{i:03d}', estimator=f'knn{k}', channel='W_input_only',
                              metric='total_correlation_estimate', loss=marginal-h))
        if i in (0, len(data.paths)-1):
            print(f'[WaveP diagnostic] {data.index["model"]} {i+1}/{len(data.paths)}', flush=True)
    wave.csv_file(output/'blind_scores.csv', blind)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = [dict(estimator=name, channel=channel, metric=metric, adequate=True, is_control=False,
                 diagnostic_only=True, **wave.st.evaluate_curve(values, data.index['cases']))
            for (name, channel, metric), values in grouped.items()]
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_subjects=data.n, all_trial_rows=data.m, registration_max_residual=max(registration),
        response_used_in_scores=False, joint_entropy=joint))
    protocol.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', protocol)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveP_input_diagnostic_seed395')
    args = p.parse_args(); survivors = None
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model)
            output = source.parent/'step5_matrix_metrics'/args.run_name
            rows = analyze(source, output)
            passed = {(r['estimator'], r['metric']) for r in rows if r['all_planes_pass']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveP diagnostic] {model}: {len(passed)} passes, {len(survivors)} common input diagnostics', flush=True)
            if args.batch and not survivors:
                print('[WaveP] No common diagnostic; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
