"""Checkpoint-only validation/test predictive evaluation of the frozen NP.

No re-training, no Z extraction, no true theta. Uses S3's originally saved
fixed-context VALIDATION and TEST metrics and per-epoch history.
"""
from __future__ import annotations
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from modlearn_rot.training.continuous import load_checkpoint
from .pipeline import train_path,stage_folder,condition,sha256,dataset_path


def report(args):
    ckpt=train_path(args)
    if not ckpt.is_file():
        raise FileNotFoundError('Train first: '+str(ckpt))
    state=load_checkpoint(ckpt)
    if state.get('extra_state',{}).get('dataset_sha256')!=sha256(dataset_path(args.model)):
        raise RuntimeError('NN checkpoint dataset provenance mismatch')
    folder=stage_folder(args,'s3_evaluate')
    folder.mkdir(parents=True,exist_ok=True)
    hist=state['history']
    summary=dict(model=args.model,condition=condition(args),
                 best_epoch=int(state['best_epoch'])+1,
                 actual_epochs=len(hist['train_loss']),
                 model_config=state['model_config'],
                 training_config=state['training_config'],
                 validation=state['validation_metrics'],
                 test=state['test_metrics'],
                 checkpoint_sha256=sha256(ckpt),
                 metric_protocol='original frozen validation/test subject context masks; sampled choice reconstruction targets')
    (folder/'s3_evaluate__summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (folder/'s3_evaluate__history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch',*hist.keys()])
        writer.writeheader()
        for i in range(len(hist['train_loss'])):
            writer.writerow(dict(epoch=i+1,**{k:v[i] for k,v in hist.items()}))
    fig,axs=plt.subplots(2,3,figsize=(13,7.5))
    for ax,key in zip(axs.flat,('loss','reconstruction','accuracy','brier','kl','sparse')):
        ax.plot(hist['train_'+key],label='TRAIN')
        ax.plot(hist['validation_'+key],label='VAL')
        ax.axvline(int(state['best_epoch']),ls='--',c='k',alpha=.4)
        ax.set_title(key);ax.set_xlabel('Epoch');ax.legend()
    fig.suptitle(f'{args.model}: original binary-choice NP learning curves')
    fig.tight_layout();fig.savefig(folder/'s3_evaluate__learning_curves.png',dpi=170);plt.close(fig)
    print(args.model,'original NP validation metrics:',state['validation_metrics'])
    print(args.model,'original NP TEST metrics:',state['test_metrics'])
    print('Saved NN evaluation:',folder)