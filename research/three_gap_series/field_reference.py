"""Validate read-only reuse of fixed M2 fields, independent of a new M1/M3.

Never returns the old rotation, loss evaluation against T, or any labels.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

FIELD_ARGUMENTS=('model','beta','zdim','pca','source','estimator','channel','structure_link',
    'samples','seed','degree','ridge','train_rows','validation_rows','train_subjects','validation_subjects',
    'utility_ridge','rbf_centers','rbf_length','neighbors','bandwidth','step','hessian_grid','grid_stimuli',
    'uu_full_domain','subject_mean','field_mean','center_point','q_weight','fisher_power','stimulus_kernel',
    'latent_kernel','latent_centers','latent_length','neural_width','neural_epochs')


def validate_reference(args,context,protocol,w_shape,full_rows,third,root=None):
    root=Path(root or Path(__file__).resolve().parent).resolve()
    path=Path(args.field_reference).resolve()
    if not path.is_relative_to(root):raise ValueError('Field reference must stay inside the experiment directory')
    source=json.loads(path.read_text(encoding='utf8'))
    if source.get('scope')!='hessian_only':raise ValueError('Only Hessian-only fields may be reused')
    if source['hashes']!=context['hashes']:raise ValueError('Field source checkpoint/latent/dataset mismatch')
    for key in FIELD_ARGUMENTS:
        default={'stimulus_kernel':'gaussian','latent_kernel':'polynomial','latent_centers':256,'latent_length':1.,
            'neural_width':128,'neural_epochs':200}.get(key)
        if source['args'].get(key,default)!=getattr(args,key,default):raise ValueError(f'Incompatible field protocol: {key}')
    for key in ['x_mean','x_sd']:
        if not np.array_equal(np.asarray(source['protocol'][key]),np.asarray(protocol[key])):raise ValueError(f'Field standardization mismatch: {key}')
    if source['protocol'].get('hessian_grid')!=json.loads(json.dumps(protocol.get('hessian_grid'),default=lambda a:a.tolist())):
        raise ValueError('Field product-grid subjects/stimuli mismatch')
    cache=Path(source['field_cache']).resolve()
    if not cache.is_relative_to(root) or not cache.is_file():raise ValueError('Missing local numerical fields; reproduce the referenced M2 first')
    with np.load(cache,allow_pickle=False) as f:
        if str(f['checkpoint_hash'])!=context['hashes']['checkpoint']:raise ValueError('Stale cached checkpoint')
        if f['h'].shape!=(w_shape[0],w_shape[1],w_shape[1]):raise ValueError('Hessian field shape does not match W')
        if full_rows is not None:
            if 'uu_h' not in f or not np.array_equal(f['uu_row_ids'],full_rows):raise ValueError('UU all-row domain mismatch')
            if third and 'uu_t' not in f:raise ValueError('Referenced fields do not supply required latent Hessian variation')
        elif third and 'third' not in f:raise ValueError('Referenced fields do not supply required F3')
    # Deliberately omit source['best'], source['evaluation'] and any T data.
    provenance=dict(reference_summary=str(path),reference_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        field_cache=str(cache),original_fitting_code_hashes=source.get('code_hashes'),
        protocol='Read-only reuse of identical numerical M2 fields; no old rotation or T evaluation consumed')
    return cache,provenance
