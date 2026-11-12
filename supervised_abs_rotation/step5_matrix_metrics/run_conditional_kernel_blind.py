"""BC3: blind rotations of the frozen derivative-free conditional kernel metric."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import tree_rotation_stability as rotations
from conditional_kernel_blind import Scorer


def main(original_name='waveBC_conditional_kernel_UX_seed430',output_name='waveBC3_conditional_kernel_blind_seed439',seed=439):
    source = wave.io.default_step2(wave.ROOT, 'CR3'); parent = source.parent/'step5_matrix_metrics'
    original = parent/original_name
    output = parent/output_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain earlier kernel blind search: {output}')
    old = json.loads((original/'protocol.json').read_text())
    digest = hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()
    if old['status'] != 'COMPLETE' or digest != old['scores_sha256']:
        raise ValueError('Source conditional kernel wave is incomplete or changed')
    bank = profiles.MatrixBank(source, parent/'matrix_response_table_seed383')
    if bank.protocol['manifest_sha256'] != old['source_matrix_manifest_sha256']:
        raise ValueError('Original matrix profile source changed')
    output.mkdir(parents=True)
    protocol = dict(status='RUNNING', seed=seed, all_subjects=bank.data.n, source_scores_sha256=digest,
        primary='logit, Ui innovation with context, raw moment, worst-X soft count at 10%',
        include_pure_U_main_component=old.get('include_pure_U_main_component',False),
        main_component_nonaffine=old.get('main_component_nonaffine',False),
        proposals='32 random SO frames, 2 KNN-proxy Powell starts with 60 evaluations',
        final_score='original nested W-only polynomial/RBF conditional regression, unchanged BC statistic',
        exact_refinement='Givens coordinate descent at +/- .25/.1 radians, one pass',
        fixed_prediction_MSE_veto=False,
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_conditional_kernel_blind.py', 'conditional_kernel_blind.py', 'conditional_ui_witness.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    scorer = Scorer(bank, original)
    # Independent numerical reproduction, using a W-only cached direction.
    v, expected = scorer.exact_cache[0]
    reproduced = scorer.direction(v, force=True)
    if not np.allclose(expected, reproduced, rtol=1e-7, atol=1e-10):
        raise ArithmeticError('Exact blind scorer did not reproduce the original moment statistic')
    rng = np.random.default_rng(seed); proposals, proxy_rows = [], []
    for k in range(32):
        q = rotations.random_rotation(bank.data.d, rng)
        loss = scorer.score(q, exact=False)
        proposals.append(q); proxy_rows.append(dict(proposal=k, loss=loss))
        if (k+1) % 8 == 0:
            print(f'[WaveBC3] blind proposal frames {k+1}/32', flush=True)
    candidates = [proposals[k] for k in np.argsort([r['loss'] for r in proxy_rows])[:6]]
    proxy_history = []
    for start, q0 in enumerate(candidates[:2]):
        def objective(parameters):
            loss = scorer.score(q0 @ rotations.increment(parameters, bank.data.d), exact=False)
            proxy_history.append(dict(start=start, parameters=np.asarray(parameters).tolist(), loss=loss))
            return loss
        minimize(objective, np.zeros(bank.data.d*(bank.data.d-1)//2), method='Powell',
            bounds=[(-np.pi/3, np.pi/3)]*(bank.data.d*(bank.data.d-1)//2), options=dict(maxfev=60, xtol=.01, ftol=0.))
        best = min((r for r in proxy_history if r['start']==start), key=lambda r:r['loss'])
        candidates.append(q0 @ rotations.increment(best['parameters'], bank.data.d))
    wave.csv_file(output/'proxy_proposals.csv', proxy_rows)
    wave.csv_file(output/'proxy_optimization.csv', proxy_history)
    exact_rows, exact_rotations = [], []
    for k, q in enumerate(candidates):
        loss = scorer.score(q, exact=True)
        exact_rows.append(dict(candidate=k, loss=loss, plane='', step_radians=''))
        exact_rotations.append(q.copy())
        wave.csv_file(output/'blind_exact_scores.csv', exact_rows)
        print(f'[WaveBC3] exact candidate {k+1}/{len(candidates)}: {loss:.8g}', flush=True)
    best_index = int(np.argmin([r['loss'] for r in exact_rows]))
    qbest, value = candidates[best_index].copy(), exact_rows[best_index]['loss']
    for i in range(bank.data.d):
        for j in range(i+1, bank.data.d):
            base_q = qbest.copy(); trial_q, trial_value = qbest, value
            for angle in (-.25, -.1, .1, .25):
                change = np.eye(bank.data.d); c, s = np.cos(angle), np.sin(angle)
                change[i, i] = change[j, j] = c; change[i, j] = -s; change[j, i] = s
                q = base_q @ change; loss = scorer.score(q, exact=True)
                exact_rows.append(dict(candidate=len(exact_rows), loss=loss, plane=[i,j], step_radians=angle))
                exact_rotations.append(q.copy())
                if loss < trial_value:
                    trial_q, trial_value = q, loss
            qbest, value = trial_q, trial_value
            wave.csv_file(output/'blind_exact_scores.csv', exact_rows)
            print(f'[WaveBC3] exact Givens plane {i}/{j}, best {value:.8g}', flush=True)
    np.savez_compressed(output/'blind_rotations.npz', proposals=np.asarray(candidates), optimized=qbest,
        exact_evaluated=np.asarray(exact_rotations),
        evaluated_directions=np.asarray([v for v,e in scorer.exact_cache]),
        directional_moments=np.asarray([e for v,e in scorer.exact_cache]))
    wave.json_file(output/'new_U_feature_audits.json', scorer.audits)
    frozen = hashlib.sha256((output/'blind_exact_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'blind_freeze.json', dict(exact_scores_sha256=frozen,
        rotations_sha256=hashlib.sha256((output/'blind_rotations.npz').read_bytes()).hexdigest()))
    # Labels enter only the assessment of already frozen exact scores.
    reveal = json.loads((original/'reveal.json').read_text())['mapping']
    with np.load(original/'W_only_geometry.npz', allow_pickle=False) as saved:
        axes, membership = saved['directions'], saved['membership']
    k = next(int(k.split('_')[1]) for k,v in reveal.items() if v=='baseline')
    qbase = axes[membership[k]].T
    actual_base = wave.io.registration(bank.data.u, bank.data.coordinates(bank.data.paths[k]))[0]
    if np.max(np.minimum(np.linalg.norm(actual_base-qbase, axis=0), np.linalg.norm(actual_base+qbase, axis=0))) > 1e-6:
        raise ValueError('Actual baseline W no longer matches the frozen source geometry')
    baseline = scorer.score(qbase, exact=True)
    assessment = dict(baseline_loss=baseline, exact_blind_minimum_loss=float(value),
        relative_margin=(value-baseline)/baseline, exact_evaluated_frames=len(exact_rows),
        proxy_scores_not_used_as_final_evidence=True, optimizer_globality_verified=False)
    wave.json_file(output/'assessment.json', assessment)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(bank.data.read_files)), forbidden_files_read=[],
        all_subjects=bank.data.n, all_original_rows=bank.data.m))
    protocol.update(status='COMPLETE', exact_scores_sha256=frozen)
    wave.json_file(output/'protocol.json', protocol)
    print('[WaveBC3]', assessment, flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--complete-response',action='store_true')
    p.add_argument('--nonaffine-main',action='store_true')
    args=p.parse_args()
    with threadpool_limits(limits=4):
        if args.nonaffine_main:
            main('waveBL_complete_response_kernel_seed430','waveBL2_nonaffine_response_blind_seed457',457)
        elif args.complete_response:
            main('waveBK_complete_response_kernel_seed430','waveBK2_complete_response_blind_seed456',456)
        else:main()
