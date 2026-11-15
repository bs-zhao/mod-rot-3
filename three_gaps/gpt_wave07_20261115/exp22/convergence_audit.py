"""Judge the final adjacent budgets, not a lucky earlier passing prefix."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c

def main():
    for p in (c.HERE/'exp22/results').glob('*/*/*_convergence.json'):
        d=json.loads(p.read_text());last={}
        for row in d['rows']:
            if row['sigma']==0:continue
            key=f"{row['sigma']:g}_{row['order']}"
            if key not in last or row['inner']>last[key]['inner']:last[key]=row
        audit=[]
        for key,row in last.items():
            err=row['convergence_vs_previous']
            audit.append(dict(key=key,inner=row['inner'],SX_relative=None if err is None else err[0],
                T4_relative=None if err is None else err[1],spearman=None if err is None else err[2],
                cap_converged=bool(err and max(err[:2])<.01 and err[2]>.99)))
        dest=p.with_name(p.stem.replace('_convergence','_cap_status')+'.json')
        c.dump(dest,dict(source=str(p),source_sha256=c.sha(p),rows=audit,unresolved=[r['key'] for r in audit if not r['cap_converged']]))
        print(p.parent.parent.name,p.parent.name,p.stem,'unresolved',[r['key'] for r in audit if not r['cap_converged']])

if __name__=='__main__':main()
