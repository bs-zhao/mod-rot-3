"""One global monotone rank calibration per original Y channel.

This defines an ordinal-response sensitivity diagnostic. It is not a full
warped-GP likelihood and does not assume a Gaussian joint input/output law.
Calibration uses all actual Y rows before any X-profile interpolation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

import run as wave
import matrix_function_profiles as raw_profiles
import run_finite


def global_scores(y, transform='normal_rank'):
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError('Rank calibration needs finite original response rows')
    p = (rankdata(y, method='average', axis=0)-.5)/len(y)
    if transform == 'normal_rank':
        return ndtri(p)
    if transform == 'uniform_rank':
        return np.sqrt(12.)*(p-.5)
    raise ValueError(f'Unknown global response calibration: {transform}')


def prepare(source, folder, raw_folder, transform='normal_rank'):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'Retain earlier rank-response preparation: {folder}')
    raw = raw_profiles.MatrixBank(source, raw_folder); data = raw.data
    folder.mkdir(parents=True, exist_ok=True)
    protocol = dict(status='RUNNING', source=str(Path(source).resolve()), seed=raw.protocol['seed'],
        model=data.index['model'], n_subjects=data.n, all_original_rows=data.m, channels=data.channels,
        response_transform=transform, calibration='one monotone map per channel over all actual original rows',
        calibration_includes_all_subject_Y=True, raw_folder=str(Path(raw_folder).resolve()),
        raw_manifest_sha256=raw.protocol['manifest_sha256'],
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('rank_response_profiles.py', 'matrix_function_profiles.py', 'run_finite.py', 'additive_profiles.py'))).hexdigest())
    wave.json_file(folder/'protocol.json', protocol)
    original_y = data.y
    try:
        data.y = global_scores(original_y, transform)
        responses, x, quality = run_finite.response_bank(data, protocol['seed'])
    finally:
        data.y = original_y
    quality.update(response_transform=transform, calibration_original_rows=data.m,
        validation_interpretation='GP subject holdout conditional on frozen all-data rank calibration')
    np.savez_compressed(folder/'response_table.npz', responses=responses, x=x)
    wave.json_file(folder/'response_quality.json', quality)
    hashes = {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
              for name in ('response_table.npz', 'response_quality.json')}
    wave.json_file(folder/'manifest.json', hashes)
    protocol.update(status='COMPLETE', manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest())
    wave.json_file(folder/'protocol.json', protocol)


class MatrixBank:
    def __init__(self, source, folder):
        self.folder = Path(folder)
        self.protocol = json.loads((self.folder/'protocol.json').read_text())
        manifest = (self.folder/'manifest.json').read_bytes()
        if self.protocol['status'] != 'COMPLETE' or hashlib.sha256(manifest).hexdigest() != self.protocol['manifest_sha256']:
            raise ValueError('Rank response preparation is incomplete or changed')
        for filename, digest in json.loads(manifest).items():
            if Path(filename).name != filename or hashlib.sha256((self.folder/filename).read_bytes()).hexdigest() != digest:
                raise ValueError('A calibrated response artifact changed')
        raw = raw_profiles.MatrixBank(source, self.protocol['raw_folder']); self.data = raw.data
        if self.protocol['source'] != str(Path(source).resolve()) or raw.protocol['manifest_sha256'] != self.protocol['raw_manifest_sha256']:
            raise ValueError('Rank response preparation belongs to a different raw matrix source')
        with np.load(self.folder/'response_table.npz', allow_pickle=False) as saved:
            self.responses, self.x = saved['responses'].copy(), saved['x'].copy()
        self.quality = json.loads((self.folder/'response_quality.json').read_text())
        if self.quality['mode'] == 'numerically verified shared observed X':
            actual = self.responses.transpose(0, 2, 1).reshape(self.data.y.shape)
            if not np.array_equal(actual, global_scores(self.data.y, self.protocol['response_transform'])):
                raise ValueError('Calibrated table differs from the transformed original Y rows')
            if not np.array_equal(self.x, raw.x):
                raise ValueError('Calibrated X reference changed')
