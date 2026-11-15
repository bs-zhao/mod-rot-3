"""Durable sequential continuation of authorized independent experiments.

No TEST metric is used by this scheduler. Each experiment keeps its own CLI.
Run after the 1024-node CR3 verification finishes. Stop on any failure.
"""
import argparse,subprocess,sys,time
from pathlib import Path
import core as c

def common(model,rep,tag='independent_v1',channel='prob'):
    return ['--model',model,'--beta','.00005' if model=='CR3' else '.0005','--zdim','6' if model=='C3' else '4',
        '--pca','0','--repeat',str(rep),'--channel',channel,'--tag',tag,'--resume']

def jobs(segment):
    result=[]
    def add(script,model,rep,tag='independent_v1',channel='prob',extra=()):
        result.append([sys.executable,str(c.HERE/script),*common(model,rep,tag,channel),*extra])
    if segment in ['rest','full-small']:
        for r in [1,2]:add('exp22/run.py','CR3',r,extra=['--stage','full','--inner-nodes','64','--sigmas','0,.1'])
    if segment in ['rest','remaining','finite']:
        for r in [1,2]:
            add('exp21/check_limit.py','CR3',r,extra=['--device','cpu'])
            add('exp21/run.py','CR3',r,'fd_vectorized',extra=['--engine','vectorized'])
    if segment in ['rest','remaining','multiscale']:
        for r in [1,2]:
            add('exp22/fixed_scales.py','CR3',r,'verified1024',extra=['--inner-nodes','1024','--device','cpu'])
            add('exp23/run.py','CR3',r,'verified1024',extra=['--inner-nodes','1024'])
    if segment in ['rest','remaining','joint']:
        for r in [1,2]:
            for stage in ['joint_train','joint_gaussian']:
                add('exp22/run.py','CR3',r,'joint128',extra=['--stage',stage,'--outer-nodes','128','--joint-trials','64','--inner-nodes','512','--offset-batch','4'])
    if segment in ['rest','remaining','validate']:
        for m in ['C1','C2','C3']:
            for r in [1,2]:add('exp13_reproduce/run.py',m,r)
        for m in ['C1','C2','C3']:
            for r in [1,2]:add('exp20A/run.py',m,r)
        for m in ['C1','C2','C3']:
            for r in [1,2]:add('exp20B/confirm.py',m,r)
        # Fixed small/middle scales, same rules in every model; no MCC selection.
        for m in ['C1','C2','C3']:
            for r in [1,2]:add('exp22/run.py',m,r,'confirm128',extra=['--stage','pilot','--inner-cap','128','--sigmas','0,.1,.25','--offset-batch','4'])
    if segment in ['rest','remaining','logit']:
        for r in [1,2]:
            add('exp13_reproduce/run.py','CR3',r,channel='logit')
            add('exp20A/run.py','CR3',r,channel='logit')
    if segment in ['rest','remaining','evaluate']:
        for m in ['CR3','C1','C2','C3']:
            for r in [1,2]:
                for tag in (['independent_v1','refined512','verified1024','joint128','fd_vectorized'] if m=='CR3' else ['independent_v1','confirm128']):
                    add('evaluate.py',m,r,tag,extra=['--device','cpu'])
        for r in [1,2]:add('evaluate.py','CR3',r,channel='logit',extra=['--device','cpu'])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('segment',choices=['rest','remaining','full-small','finite','multiscale','joint','validate','logit','evaluate'])
    a=p.parse_args();logdir=c.OUT/'logs/continuation';logdir.mkdir(parents=True,exist_ok=True)
    for i,cmd in enumerate(jobs(a.segment)):
        name=f'{a.segment}_{i:03d}';path=logdir/f'{name}.log';t=time.time()
        c.dump(logdir/f'{name}.command.json',dict(argv=cmd,cwd=str(c.ROOT)))
        print('START',name,cmd,flush=True)
        with open(path,'a',encoding='utf-8') as f:
            f.write('\nRUN '+time.strftime('%Y-%m-%d %H:%M:%S')+'\n');f.flush()
            r=subprocess.run(cmd,cwd=c.ROOT,stdout=f,stderr=subprocess.STDOUT)
        status=dict(argv=cmd,returncode=r.returncode,seconds=time.time()-t,log=str(path),log_sha256=c.sha(path))
        c.dump(logdir/f'{name}.status.json',status)
        if r.returncode:
            c.dump(c.HERE/'continuation_failure.json',status)
            print(path.read_text(encoding='utf-8')[-5000:],flush=True)
            raise RuntimeError(f'failed {name}; all previous outputs retained')
        print('DONE',name,round(time.time()-t,2),flush=True)

if __name__=='__main__':main()
