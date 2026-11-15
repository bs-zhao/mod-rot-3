"""Run the complete predeclared tuning grid in a new output namespace."""
import argparse
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as C

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--tag',default='grid_v1')
    p.add_argument('--quick',action='store_true'); p.add_argument('--models',nargs='+',default=list(C.COND))
    a=p.parse_args(); logs=Path(__file__).parent/'logs'/a.tag; logs.mkdir(parents=True,exist_ok=True)
    for model in a.models:
        for rep in (1,2):
            for channel in ('prob','logit'):
                key=f'{model}_rep{rep}_{channel}'; out=C.result_dir('baseline','tuning_'+a.tag,model,rep,channel)
                if (out/'COMPLETE.json').exists(): continue
                cmd=[sys.executable,'-u',str(Path(__file__).parent/'tune.py'),'--model',model,'--repeat',str(rep),
                     '--channel',channel,'--tag',a.tag]
                if a.quick: cmd+=['--quick']
                print('START',key,flush=True)
                with (logs/(key+'.log')).open('w',encoding='utf8') as f:
                    result=subprocess.run(cmd,cwd=C.ROOT,stdout=f,stderr=subprocess.STDOUT)
                if result.returncode:
                    print((logs/(key+'.log')).read_text(encoding='utf8')[-6000:],flush=True)
                    raise SystemExit(result.returncode)
                print('DONE',key,flush=True)
