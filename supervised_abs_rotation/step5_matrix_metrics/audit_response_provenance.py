"""Audit existing case Y only; never run a decoder or a generating equation."""
import hashlib
import json

import numpy as np
from scipy.special import expit

import run as wave
from validation_response_policy import FINAL_RESPONSE_SOURCES,response_fields


def audit(source):
    source=source.resolve();index=json.loads((source/'index.json').read_text())
    file=source/'Y.npz'
    with file.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=index['source_files_sha256']['Y.npz']:raise ValueError('Case Y changed from its recorded export')
    with np.load(file,allow_pickle=False) as saved:
        fields=saved.files;values={name:saved[name].copy() for name in FINAL_RESPONSE_SOURCES}
        logits=saved['logits_nn_2class'].copy()
    n=index['n_trials']
    if logits.shape!=(n,2) or not np.isfinite(logits).all():raise ValueError('Invalid stored neural logits')
    if any(y.shape!=(n,) or not np.isfinite(y).all() for y in values.values()):raise ValueError('Y row alignment failed')
    error_logit=float(np.max(np.abs(values['logit_nn']-(logits[:,1]-logits[:,0]))))
    error_p=float(np.max(np.abs(values['p_nn']-expit(logits[:,1].astype(float)-logits[:,0]))))
    mismatch=int(np.sum(values['choice_nn']!=np.argmax(logits,axis=1)))
    if error_logit>1e-6*max(1.,float(np.max(np.abs(values['logit_nn'])))) or error_p>2e-6 or mismatch:
        raise ArithmeticError('Stored NN outputs are internally inconsistent')
    if not all(np.all((values[c]==0)|(values[c]==1)) for c in ('choice_nn','choice_observed')):
        raise ValueError('Choice channels are not binary')
    bank=source.parent/'step5_matrix_metrics'/'matrix_response_table_seed383'
    channels=json.loads((bank/'protocol.json').read_text())['channels']
    return dict(model=index['model'],all_subjects=index['n_subjects'],all_original_rows=n,Y_npz=str(file),
        Y_sha256=digest,Y_fields=fields,current_response_bank_channels=channels,
        channel_provenance={c:response_fields(c) for c in values},
        maximum_stored_NN_logit_difference_error=error_logit,maximum_stored_NN_probability_error=error_p,
        stored_NN_argmax_mismatches=mismatch,analytic_response_arrays_read=False,decoder_or_generator_evaluated=False,
        current_Y_sources_eligible=all(c in FINAL_RESPONSE_SOURCES for c in channels))


if __name__=='__main__':
    output=wave.ROOT/'outputs/supervised_abs_rotation/final_Y_provenance_audit';output.mkdir(parents=True,exist_ok=True)
    rows=[audit(wave.io.default_step2(wave.ROOT,m)) for m in ('CR3','C1','C2','C3')]
    wave.json_file(output/'four_model_Y_audit.json',dict(status='COMPLETE',
        exporter_source=str(wave.ROOT/'supervised_abs_rotation/export_wy.py'),
        exporter_code_sha256=hashlib.sha256((wave.ROOT/'supervised_abs_rotation/export_wy.py').read_bytes()).hexdigest(),
        exporter_definitions='NN class1-minus-class0 logit, NN softmax probability, NN argmax, and original observed choice',
        rows=rows,eligibility_is_not_a_claim_of_metric_success=True))
    lines=['# Final-validation response provenance','',
        'Allowed: observed choice; NN logit, probability or choice. Analytic logit/probability are auxiliary only.','',
        '| Model | Subjects | Original rows | Current channels | Eligible Y sources |',
        '|---|---:|---:|---|---|']
    for r in rows:lines.append(f'| {r["model"]} | {r["all_subjects"]} | {r["all_original_rows"]} | {", ".join(r["current_response_bank_channels"])} | {r["current_Y_sources_eligible"]} |')
    lines+=['','Read only existing case Y/index and response-bank metadata. No decoder, original weights or response generator was run.',
        'Stored NN logit/probability/choice are checked against the stored two-class NN scores on every original row.',
        'Source eligibility is separate from rotation quality or a claim that any metric has succeeded.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[Y provenance] all four actual case banks use NN logit/p; see',output,flush=True)
