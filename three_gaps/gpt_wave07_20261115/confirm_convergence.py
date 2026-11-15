"""T-free continuation of failed four-model smoothing quadrature caps."""
import json,subprocess,sys,time
import core as c
from continue_run import common

def run(cmd,stem):
    folder=c.OUT/'logs/convergence_final';folder.mkdir(parents=True,exist_ok=True)
    log=folder/(stem+'.log');c.dump(folder/(stem+'.command.json'),dict(argv=cmd,cwd=str(c.ROOT)))
    t=time.time();print('START',stem,flush=True)
    with open(log,'a',encoding='utf-8') as f:r=subprocess.run(cmd,cwd=c.ROOT,stdout=f,stderr=subprocess.STDOUT)
    c.dump(folder/(stem+'.status.json'),dict(argv=cmd,seconds=time.time()-t,returncode=r.returncode,log=str(log),log_sha256=c.sha(log)))
    if r.returncode:raise RuntimeError(str(log))
    print('DONE',stem,round(time.time()-t,2),flush=True)

def main():
    decisions=[]
    for model in ['C1','C2','C3']:
        for rep in [1,2]:
            # Explicit CLI setting key avoids depending on directory-name globs.
            a=c.args('convergence key').parse_args(common(model,rep,'confirm128')+['--device','cpu'])
            st=c.Setting(a);base=c.HERE/'exp22/results'/st.key
            audit=json.loads((base/'confirm128/pilot_cap_status.json').read_text())
            if not audit['unresolved']:continue
            for cap in [512,1024]:
                tag=f'confirm{cap}'
                decisions.append(dict(model=model,repeat=rep,previous_unresolved=audit['unresolved'],next_cap=cap,rule='final adjacent SX/T4<1% and landscape rho>.99; no TEST read'))
                c.dump(c.HERE/'reports/confirmation_extensions.json',decisions)
                run([sys.executable,str(c.HERE/'exp22/run.py'),*common(model,rep,tag),
                     '--stage','pilot','--inner-cap',str(cap),'--sigmas','0,.1,.25','--offset-batch','4'],f'{model}_r{rep}_{tag}')
                subprocess.run([sys.executable,str(c.HERE/'exp22/convergence_audit.py')],cwd=c.ROOT,check=True,stdout=subprocess.DEVNULL)
                audit=json.loads((base/tag/'pilot_cap_status.json').read_text())
                run([sys.executable,str(c.HERE/'evaluate.py'),*common(model,rep,tag),'--device','cpu'],f'{model}_r{rep}_{tag}_evaluate')
                if not audit['unresolved']:break
            if audit['unresolved']:c.dump(base/'confirmation_UNRESOLVED.json',audit)

if __name__=='__main__':main()
