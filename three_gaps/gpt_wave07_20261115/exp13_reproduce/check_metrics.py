"""Post-hoc reproduction check: complete rotations required before theta read."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import numpy as np
import pandas as pd
from common.rotlib import match_report

def main():
    hist=pd.read_csv(c.ROOT/'three_gaps/post_exp13_20261114/baseline/results/metrics_all.csv')
    rows=[]
    for m in ['CR3','C1','C2','C3']:
        for rep in [1,2]:
            a=c.args(__doc__).parse_args(['--model',m,'--beta','.00005' if m=='CR3' else '.0005','--zdim','6' if m=='C3' else '4','--pca','0','--repeat',str(rep),'--device','cpu'])
            st=c.Setting(a);base=c.HERE/'exp13_reproduce/results'/st.key/a.tag
            p=base/'A0_strict/rotations.npz'
            if not p.exists():p=base/'A0/rotations.npz'
            cp=p.parent/'COMPLETE.json'
            if not cp.exists():raise FileNotFoundError('no completed blind rotation: '+str(p))
            if c.sha(p)!=json.loads(cp.read_text())['rotations_sha256']:raise ValueError('rotation checksum')
            with np.load(p) as z:R=z['R'][int(z['best_index'])]
            with np.load(st.paths['latent']) as z:T=z['theta_true']
            out=match_report(st.V[st.te]@R.T,T[st.te]);h=hist[(hist.model==m)&(hist['repeat']==rep)&(hist.channel=='prob')].iloc[0]
            err=[out['mcc']-h.mcc,out['off_mean']-h.leakage,out['off_max']-h.max_leakage]
            rows.append(dict(model=m,repeat=rep,mcc=out['mcc'],leakage=out['off_mean'],max_leakage=out['off_max'],mcc_delta=err[0],leakage_delta=err[1],max_leakage_delta=err[2],passed=max(abs(v) for v in err)<1e-4))
    df=pd.DataFrame(rows);df.to_csv(c.HERE/'exp13_reproduce/results/reproduction_check.csv',index=False)
    print(df.to_string(index=False))
    if not df.passed.all():raise AssertionError('historical reproduction outside declared tolerance; do not silently change tolerance')

if __name__=='__main__':main()
