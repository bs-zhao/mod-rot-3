"""Sequential, resumable doc-05 runs. Each experiment remains independently runnable."""
from __future__ import annotations
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
import core as C

DIRS={14:'exp14_probability',15:'exp15_finite_scale',16:'exp16_local_context',17:'exp17_reliability',18:'exp18_matrix_only',19:'exp19_model_library'}

def main(a):
    logdir=C.HERE/'logs'/a.tag; logdir.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for exp in a.experiments:
        if exp==19:
            jobs.append((exp,'library',[sys.executable,'-u',str(C.HERE/DIRS[exp]/'run.py'),'--tag',a.tag])); continue
        for model,rep in itertools.product(a.models,a.repeats):
            channels=['prob'] if exp==14 else a.channels
            for channel in channels:
                sources=(['analytic','nn','choice'] if exp==18 else
                         ['decoder','analytic'] if exp in (14,16) and a.analytic else [None])
                for source in sources:
                    if exp==18 and source=='choice' and channel=='logit': continue
                    label=f'{model}_rep{rep}_{channel}'+(f'_{source}' if source else '')
                    cmd=[sys.executable,'-u',str(C.HERE/DIRS[exp]/'run.py'),'--model',model,'--repeat',str(rep),
                         '--channel',channel,'--tag',a.tag]
                    if exp in (14,17): cmd+=['--subjects','300','--trials','500']
                    if exp==16: cmd+=['--subjects',str(a.field_subjects),'--trials',str(a.field_trials),'--local-points','128']
                    if exp==15: cmd+=['--subjects','8','--trials','8','--starts','24','--steps','250','--random-n','200']
                    if source: cmd+=['--y-source' if exp==18 else '--source',source]
                    jobs.append((exp,label,cmd))
    state=[]
    for exp,label,cmd in jobs:
        key=f'exp{exp}_{label}'; done=logdir/(key+'.done.json')
        if done.exists() and not a.rerun:
            state.append(json.loads(done.read_text(encoding='utf8'))); continue
        log=logdir/(key+'.log'); print('START',key,flush=True); t=time.time()
        with log.open('w',encoding='utf8') as f:
            run=subprocess.run(cmd,cwd=C.ROOT,stdout=f,stderr=subprocess.STDOUT)
        record=dict(experiment=exp,label=label,command=cmd,log=str(log.relative_to(C.HERE)),
                    returncode=run.returncode,seconds=time.time()-t)
        state.append(record); C.dump(logdir/'progress.json',state)
        if run.returncode:
            print(log.read_text(encoding='utf8')[-8000:],flush=True)
            raise SystemExit(run.returncode)
        C.dump(done,record); print('DONE',key,f'{record["seconds"]:.1f}s',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiments',nargs='+',type=int,choices=list(DIRS),default=[14,15,17,16,18,19])
    p.add_argument('--models',nargs='+',choices=list(C.COND),default=list(C.COND))
    p.add_argument('--repeats',nargs='+',type=int,default=[1,2]); p.add_argument('--channels',nargs='+',default=['prob','logit'])
    p.add_argument('--analytic',action='store_true'); p.add_argument('--tag',default='round1')
    p.add_argument('--field-subjects',type=int,default=64,help='EXP16 field subject budget')
    p.add_argument('--field-trials',type=int,default=64,help='EXP16 field stimulus budget')
    p.add_argument('--rerun',action='store_true'); main(p.parse_args())
