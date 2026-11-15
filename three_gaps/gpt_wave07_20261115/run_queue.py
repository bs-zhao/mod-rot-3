"""Sequential execution helper, each experiment remains independently runnable.

Does not select methods from TEST metrics. Durable commands/logs/failure records.
"""
import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path
import core as c

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['baseline_cr3','20a_cr3','20b_cr3','22pilot_cr3','22full_cr3','baseline_c','20a_c','20b_c','20bconfirm_c','22pilot_c'])
    p.add_argument('--channel',choices=['prob','logit'],default='prob');p.add_argument('--run-name')
    a=p.parse_args();models=['CR3'] if a.stage.endswith('cr3') else ['C1','C2','C3']
    scripts={'baseline':'exp13_reproduce/run.py','20a':'exp20A/run.py','20b':'exp20B/run.py','20bconfirm':'exp20B/confirm.py','22pilot':'exp22/run.py','22full':'exp22/run.py'}
    phase=a.stage.rsplit('_',1)[0];jobs=[]
    for model in models:
        for rep in (1,2):
            base=['--model',model,'--beta','.00005' if model=='CR3' else '.0005','--zdim','6' if model=='C3' else '4','--pca','0','--repeat',str(rep),'--channel',a.channel,'--resume']
            extra=['--stage','pilot'] if phase=='22pilot' else ['--stage','full','--inner-nodes','64'] if phase=='22full' else []
            if phase=='baseline' and a.run_name:extra+=['--run-name',a.run_name]
            jobs.append([sys.executable,str(c.HERE/scripts[phase]),*base,*extra])
    logdir=c.OUT/'logs';logdir.mkdir(exist_ok=True)
    for idx,cmd in enumerate(jobs):
        stem=f'{a.stage}_{a.channel}_{a.run_name or "default"}_{idx}';out=logdir/f'{stem}.log'
        c.dump(logdir/f'{stem}.command.json',dict(argv=cmd,cwd=str(c.ROOT),protocol_sha256=c.sha(c.HERE/'protocol.json')))
        print('START',cmd,flush=True);t=time.time()
        # Append preserves failed attempt output and rerun history.
        with open(out,'a',encoding='utf-8') as f:
            f.write('\nRUN '+time.strftime('%Y-%m-%d %H:%M:%S')+'\n');f.flush()
            proc=subprocess.run(cmd,cwd=c.ROOT,stdout=f,stderr=subprocess.STDOUT)
        status=dict(returncode=proc.returncode,seconds=time.time()-t,log=str(out))
        c.dump(logdir/f'{stem}.status.json',status)
        if proc.returncode:
            print(out.read_text(encoding='utf-8')[-5000:],flush=True)
            raise RuntimeError(f'job failed; preserved at {out}')
        print('DONE',stem,status,flush=True)

if __name__=='__main__':main()
