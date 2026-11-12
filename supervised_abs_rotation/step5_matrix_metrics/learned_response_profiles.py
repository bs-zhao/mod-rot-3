"""Apply a frozen global data-learned output link before X-profile preparation."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import run as wave
import matrix_function_profiles as raw
import run_finite


def load_links(folder, kind='learned'):
    folder = Path(folder)
    protocol = json.loads((folder/'protocol.json').read_text())
    manifest = (folder/'manifest.json').read_bytes()
    if protocol['status'] != 'COMPLETE' or hashlib.sha256(manifest).hexdigest() != protocol['manifest_sha256']:
        raise ValueError('Global learned output link is incomplete or changed')
    for filename, digest in json.loads(manifest).items():
        if Path(filename).name != filename or hashlib.sha256((folder/filename).read_bytes()).hexdigest() != digest:
            raise ValueError('A global-warp source artifact changed')
    with (folder/'original_row_validation.csv').open() as stream:
        quality = list(csv.DictReader(stream))
    links = []
    with np.load(folder/'global_warps.npz', allow_pickle=False) as saved:
        for c in range(len(quality)):
            knots, heights = saved[f'channel{c}_knots'], saved[f'channel{c}_{kind}_heights']
            if not np.all(np.diff(knots)>0.) or not np.all(np.diff(heights)>0.):
                raise ValueError('Global response link is not strictly monotone')
            links.append((knots.copy(), heights.copy()))
    return protocol, quality, links


def forward(y, links):
    result = np.empty_like(y, dtype=float)
    for c, (knots, heights) in enumerate(links):
        bins = np.clip(np.searchsorted(knots, y[:, c], side='right')-1, 0, len(knots)-2)
        result[:, c] = heights[bins]+(y[:, c]-knots[bins])*(heights[bins+1]-heights[bins])/(knots[bins+1]-knots[bins])
    return result


def prepare(source, folder, raw_folder, warp_folder, kind='learned'):
    folder, warp_folder = Path(folder), Path(warp_folder)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'Retain previous learned-link response cache: {folder}')
    wp, wq, links = load_links(warp_folder, kind)
    original = raw.MatrixBank(source, raw_folder); data = original.data
    if wp['source'] != str(Path(source).resolve()) or wp['all_subjects'] != data.n or [q['channel'] for q in wq] != list(data.channels):
        raise ValueError('Learned output link belongs to a different actual matrix source')
    folder.mkdir(parents=True)
    protocol = dict(status='RUNNING', source=str(Path(source).resolve()), n_subjects=data.n,
        all_original_rows=data.m, channels=data.channels, seed=original.protocol['seed'],
        raw_folder=str(Path(raw_folder).resolve()), raw_manifest_sha256=original.protocol['manifest_sha256'],
        warp_folder=str(warp_folder.resolve()), warp_manifest_sha256=wp['manifest_sha256'],
        warp_source_protocol_sha256=hashlib.sha256((warp_folder/'protocol.json').read_bytes()).hexdigest(),
        warp_kind=kind, response_transform='learned_global_monotone_link_'+kind,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    wave.json_file(folder/'protocol.json', protocol)
    original_y = data.y
    try:
        data.y = forward(original_y, links)
        responses, x, quality = run_finite.response_bank(data, protocol['seed'])
    finally:
        data.y = original_y
    quality.update(response_transform=protocol['response_transform'], global_warp_original_row_validation=wq,
                   transform_applied_before_X_interpolation=True)
    np.savez_compressed(folder/'response_table.npz', responses=responses, x=x)
    wave.json_file(folder/'response_quality.json', quality)
    hashes = {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ('response_table.npz','response_quality.json')}
    wave.json_file(folder/'manifest.json', hashes)
    protocol.update(status='COMPLETE', manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest())
    wave.json_file(folder/'protocol.json', protocol)


class MatrixBank:
    def __init__(self, source, folder):
        self.folder = Path(folder); self.protocol = json.loads((self.folder/'protocol.json').read_text())
        manifest = (self.folder/'manifest.json').read_bytes()
        if self.protocol['status']!='COMPLETE' or hashlib.sha256(manifest).hexdigest()!=self.protocol['manifest_sha256']:
            raise ValueError('Learned-link response cache is incomplete or changed')
        for filename, digest in json.loads(manifest).items():
            if Path(filename).name != filename or hashlib.sha256((self.folder/filename).read_bytes()).hexdigest()!=digest:
                raise ValueError('A learned-link response artifact changed')
        original = raw.MatrixBank(source, self.protocol['raw_folder']); self.data = original.data
        wp, _, links = load_links(self.protocol['warp_folder'], self.protocol['warp_kind'])
        if wp['manifest_sha256']!=self.protocol['warp_manifest_sha256'] or original.protocol['manifest_sha256']!=self.protocol['raw_manifest_sha256']:
            raise ValueError('The actual-row or output-link source changed')
        with np.load(self.folder/'response_table.npz', allow_pickle=False) as saved:
            self.responses, self.x = saved['responses'].copy(), saved['x'].copy()
        self.quality = json.loads((self.folder/'response_quality.json').read_text())
        if self.quality['mode']=='numerically verified shared observed X':
            if not np.array_equal(self.responses.transpose(0, 2, 1).reshape(self.data.y.shape), forward(self.data.y, links)):
                raise ValueError('Learned-link table differs from transformed actual Y')
