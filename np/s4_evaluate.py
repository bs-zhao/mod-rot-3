from pathlib import Path
import sys
# Support direct invocation as: python np/sN_*.py ...
NP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NP_ROOT / 'src'))
"""S4: Display and plot original NP checkpoint validation/test behavior.

This stage reads fixed-context test metrics saved by the S3 trainer; it does
not retrain, rerun inference, or use theta.
"""
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modrot3.cli import parse_args
from modrot3.pipeline import train_path, stage_folder, condition, sha256, dataset_path
from modlearn_rot.training.continuous import load_checkpoint

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

if __name__ == '__main__':
    report(parse_args('S4: evaluate saved NP checkpoint and training curves'))