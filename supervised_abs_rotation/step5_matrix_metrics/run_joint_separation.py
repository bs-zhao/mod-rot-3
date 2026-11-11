"""Wave N: mixed U/X functional-block spectra; screen CR3 first."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import joint_bank as jb
import joint_separation as js
import coefficient_support as cs


def analyze(source, output, bank_folder, seed=393, sigmas=(.5, 1., 2.)):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain existing experiment: {output}')
    bank = jb.JointBank(source, bank_folder); data = bank.data
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(source.resolve()),
        seed=seed, bank_seed=bank.protocol['seed'], degrees=bank.protocol['degrees'], channels=list(data.channels),
        sigmas=list(sigmas), adequacy_validation_nmse=.1, all_subjects_used=True, status='RUNNING',
        reference='isotropic Gaussian U times empirical univariate X marginals; no empirical-independence claim',
        score_sign='lower for all datasets', source_signature_sha256=bank.protocol['manifest_sha256'],
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_joint_separation.py', 'joint_separation.py', 'coefficient_support.py', 'kernel_anova.py',
             'joint_bank.py', 'tail_support.py', 'matrix_io.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', config)
    maps, reveal, residuals = [], {}, []
    for k, path in enumerate(data.paths):
        q, residual = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); residuals.append(residual); reveal[f'case_{k:03d}'] = path.parent.name
    blind, quality, audits = [], [], []
    for degree in bank.protocol['degrees']:
        name = f'poly{degree}'; saved = bank.load(degree)
        model = js.JointSeparation(saved['powers'], saved['weights'], data.d,
                                  saved['g_all']-saved['g_validation'],
                                  int(saved['n_all']-saved['n_validation']))
        structure = cs.Structure(saved['powers'], data.d, seed)
        moments = {sigma: model.moment_map(sigma) for sigma in sigmas}
        for channel in data.channels:
            qr = bank.quality[name, channel]
            quality.append(dict(estimator=name, channel=channel, validation_nmse=float(qr['validation_nmse']),
                                adequate=float(qr['validation_nmse']) <= .1))
        for k, q in enumerate(maps):
            coef = structure.rotate(saved['coef'], q)
            for sigma, moment in moments.items():
                orthogonal = moment.T @ (saved['weights'][:, None]*coef)
                for c, channel in enumerate(data.channels):
                    scores, audit = model.scores(orthogonal[:, c])
                    prefix = f's{sigma:g}_'
                    for metric, loss in scores.items():
                        metric = ('control_'+prefix+metric[8:]) if metric.startswith('control_') else prefix+metric
                        blind.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                                          metric=metric, loss=loss))
                    audits.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                                       sigma=sigma, x_marginal_ranks=model.x_ranks, spectra=audit))
            if k % 12 == 0:
                print(f'[WaveN] {data.index["model"]} {name} anonymous {k+1}/{len(maps)}', flush=True)
        wave.csv_file(output/'blind_scores.csv', blind)
        wave.csv_file(output/'fit_quality.csv', quality)
        wave.json_file(output/'separation_audit.json', audits)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    quality_map = {(r['estimator'], r['channel']): r for r in quality}
    summaries = []
    for (name, channel, metric), values in grouped.items():
        control = metric.startswith('control_')
        if control and np.ptp(list(values.values())) > 1e-7*max(1., max(values.values())):
            raise ArithmeticError('Joint product-reference norm changed under orthogonal U rotation')
        qr = quality_map[name, channel]
        summaries.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index['cases']), adequate=qr['adequate'],
            validation_nmse=qr['validation_nmse'], is_control=control))
    wave.csv_file(output/'metric_summary.csv', summaries)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_subjects=data.n, all_trial_rows=data.m, registration_max_residual=max(residuals)))
    config.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', config)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveN_joint_spectra_seed393')
    p.add_argument('--bank-name', default='joint_bank_seed391')
    p.add_argument('--seed', type=int, default=393)
    args = p.parse_args(); survivors = None; seen = {}
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); folder = source.parent/'step5_matrix_metrics'
            bank_folder, output = folder/args.bank_name, folder/args.run_name
            if not bank_folder.exists():
                jb.prepare(source, bank_folder)
            rows = analyze(source, output, bank_folder, args.seed)
            seen[model] = rows
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows
                      if r['adequate'] and r['all_planes_pass'] and not r['is_control']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveN] {model}: {len(passed)} passes; {len(survivors)} common survivors', flush=True)
            wave.csv_file(wave.ROOT/'outputs'/'supervised_abs_rotation'/f'step5_{args.run_name}_screen.csv',
                [dict(model=m, **r) for m, result in seen.items() for r in result])
            if args.batch and not survivors:
                print('[WaveN] No common candidate remains; later models skipped', flush=True)
                return
        if args.batch:
            print(f'[WaveN] Four-model grid survivors: {len(survivors)}; stability and optimization still required', flush=True)


if __name__ == '__main__':
    main()
