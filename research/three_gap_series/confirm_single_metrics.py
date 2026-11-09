"""Confirm all-four atlas survivors using independent full SO(d) multistart runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from core import dump

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['C1','C2','C3','CR3'],required=True)
    p.add_argument('--beta',type=float,required=True);p.add_argument('--zdim',type=int,required=True)
    p.add_argument('--pca',type=int,choices=[0,1],required=True)
    p.add_argument('--atlas-label',default='s01_single_metric_atlas')
    p.add_argument('--label',default='s02_single_joint_confirm')
    p.add_argument('--starts',type=int,default=64)
    args=p.parse_args()
    atlas_root=ROOT/'results'/args.atlas_label
    candidates=json.loads((atlas_root/'joint_candidates.json').read_text(encoding='utf8'))
    if not candidates['complete']:raise ValueError('Wait for every model before selecting candidates')
    atlas=json.loads((atlas_root/args.model/'atlas.json').read_text(encoding='utf8'))
    reference=Path(atlas['reference_summary']);source=json.loads(reference.read_text(encoding='utf8'))
    if source['args']['channel']!=candidates['channel']:raise ValueError('Channel mismatch')
    for key in ['model','beta','zdim','pca']:
        if source['args'][key]!=getattr(args,key):raise ValueError('Wrong source condition')
    out=ROOT/'results'/args.label/args.model;out.mkdir(parents=True,exist_ok=True)
    completed=[]
    for metric in candidates['metrics']:
        found=[]
        for path in (ROOT/'results'/args.label).glob('**/summary.json'):
            j=json.loads(path.read_text(encoding='utf8'));a=j['args']
            if a['model']==args.model and a['metric']==metric and a['starts']==args.starts:found.append(path)
        if not found:
            command=[sys.executable,str(ROOT/'run.py')]
            # Copy the exact frozen field protocol, not the old metric/rotation.
            for key,value in source['args'].items():
                if key in ['metric','label','starts','maxiter','field_reference','combination_rationale','gate_prescreen']:continue
                if value is None:continue
                flag='--'+key.replace('_','-')
                if isinstance(value,bool):
                    if value:command.append(flag)
                else:command.extend([flag,str(value)])
            command+=['--metric',metric,'--label',args.label,'--starts',str(args.starts),'--maxiter','250',
                '--field-reference',str(reference),'--gate-prescreen']
            with (out/(metric+'.log')).open('w',encoding='utf8') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        completed.append(metric)
        dump(out/'queue.json',dict(metrics=candidates['metrics'],completed=completed,starts=args.starts))
        print(args.model,metric,'confirmed',len(completed),'of',len(candidates['metrics']),flush=True)


if __name__=='__main__':main()
