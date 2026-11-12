"""Response tables and spectral targets from W/Y, without a U-function fit."""
import hashlib
import json
from pathlib import Path

import numpy as np

import run as wave
import run_finite


def prepare(source, folder, seed=383):
    source, folder = Path(source), Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'Retain existing matrix response table: {folder}')
    data = wave.io.MatrixCases(source, ('logit_nn', 'p_nn'), seed)
    folder.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=data.index['model'], n_subjects=data.n, source=str(source.resolve()),
        seed=seed, channels=data.channels, all_original_rows=data.m, status='RUNNING',
        U_function_approximation_before_GP=False,
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('matrix_function_profiles.py', 'run_finite.py', 'additive_profiles.py', 'matrix_io.py', 'surrogates.py'))).hexdigest())
    wave.json_file(folder/'protocol.json', protocol)
    responses, x, quality = run_finite.response_bank(data, seed)
    np.savez_compressed(folder/'response_table.npz', responses=responses, x=x)
    wave.json_file(folder/'response_quality.json', quality)
    hashes = {f: hashlib.sha256((folder/f).read_bytes()).hexdigest() for f in ('response_table.npz', 'response_quality.json')}
    wave.json_file(folder/'manifest.json', hashes)
    wave.json_file(folder/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_subjects=data.n, all_original_rows=data.m, original_neural_weights_read=False,
        U_regression_or_U_kernel_teacher_used=False))
    protocol.update(status='COMPLETE', manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest())
    wave.json_file(folder/'protocol.json', protocol)
    return protocol


class MatrixBank:
    def __init__(self, source, folder):
        self.folder = Path(folder)
        self.protocol = json.loads((self.folder/'protocol.json').read_text())
        manifest = (self.folder/'manifest.json').read_bytes()
        if self.protocol['status'] != 'COMPLETE' or hashlib.sha256(manifest).hexdigest() != self.protocol['manifest_sha256']:
            raise ValueError('Matrix response preparation is incomplete or changed')
        for filename, digest in json.loads(manifest).items():
            if Path(filename).name != filename or hashlib.sha256((self.folder/filename).read_bytes()).hexdigest() != digest:
                raise ValueError('Prepared response artifact changed')
        self.data = wave.io.MatrixCases(source, tuple(self.protocol['channels']), self.protocol['seed'])
        if self.protocol['source'] != str(Path(source).resolve()) or self.protocol['n_subjects'] != self.data.n:
            raise ValueError('Response table does not match the actual matrix dataset')
        with np.load(self.folder/'response_table.npz', allow_pickle=False) as saved:
            self.responses, self.x = saved['responses'].copy(), saved['x'].copy()
        self.quality = json.loads((self.folder/'response_quality.json').read_text())
        if self.quality['mode'] == 'numerically verified shared observed X':
            counts = np.diff(self.data.offsets)
            if not np.all(counts == len(self.x)) or not np.array_equal(self.x, self.data.w[:counts[0], self.data.d:]):
                raise ValueError('Shared X declaration no longer matches W')
            for a, b in zip(self.data.offsets[:-1], self.data.offsets[1:]):
                if not np.array_equal(self.data.w[a:b, self.data.d:], self.x):
                    raise ValueError('Shared X did not hold for every original subject row')
            actual = self.responses.transpose(0, 2, 1).reshape(self.data.y.shape)
            if not np.array_equal(actual, self.data.y):
                raise ValueError('The supposedly raw response table differs from actual Y')


def task_modes(responses, operators, tolerance=1e-6):
    """Direct SVD of coupling contrasts; retained and discarded energies audited.

    There is no assumed U-polynomial or U-kernel function in these targets.
    Output modes are only a numerical compression of a known response table.
    """
    f = responses-responses.mean(0, keepdims=True)-responses.mean(2, keepdims=True)+responses.mean((0, 2), keepdims=True)
    targets, tasks, discarded, audits = [], [], [], []
    for c in range(f.shape[1]):
        for j, residual in enumerate(operators):
            values = f[:, c] @ residual.T
            total = float(np.sum(values*values))
            if total == 0.:
                targets.append(np.zeros((len(f), 1))); discarded.append(np.zeros(len(f))); tasks.append((c, j))
                audits.append(dict(channel_index=c, X_column=j, rank=0, original_energy=0., retained_fraction=0.))
                continue
            left, singular, _ = np.linalg.svd(values, full_matrices=False)
            rank = min(int(np.searchsorted(np.cumsum(singular**2)/total, 1-tolerance))+1, len(singular))
            target = left[:, :rank]*singular[:rank]*np.sqrt(len(f)/total)
            row_total = np.sum(values*values, axis=1)*(len(f)/total)
            tail = np.maximum(row_total-np.sum(target*target, axis=1), 0.)
            targets.append(target); discarded.append(tail); tasks.append((c, j))
            audits.append(dict(channel_index=c, X_column=j, rank=rank, original_energy=total/len(f),
                retained_fraction=float(np.sum(singular[:rank]**2)/total),
                relative_squared_reconstruction_error=float(np.sum(singular[rank:]**2)/total)))
    return tasks, targets, np.asarray(discarded), audits


def full_error(retained_error, retained_energy, discarded_energy):
    return (retained_error*retained_energy+discarded_energy)/np.maximum(retained_energy+discarded_energy, 1e-30)
