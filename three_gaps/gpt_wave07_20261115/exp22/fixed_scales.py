"""Historical Haar-SD control using this wave's own smoothed moments."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import numpy as np

def main(a):
    st=c.Setting(a)
    hp=c.HERE/'exp13_reproduce/results'/st.key/a.baseline_tag/'A0_strict/diagnostics.json'
    if not hp.exists():hp=hp.parent.parent/'A0/diagnostics.json'
    scales=json.loads(hp.read_text())['haar_scales']
    path=c.OUT/'exp22'/st.key/a.tag/a.outer/'moments.npz'
    with np.load(path) as z:
        for sigma in [0,.05,.1,.25,.5,1]:
            n=1 if sigma==0 else a.inner_nodes
            for order in ['s','e']:
                prefix=f's{sigma:g}_n{n}'
                if prefix+'_SX_'+order not in z.files:continue
                c.run_loss(st,f'{a.outer}_sigma{sigma:g}_n{n}_{order}_histSD',z[prefix+'_SX_'+order].mean(0),z[prefix+'_T4_'+order].mean(0),
                    scales=scales,experiment='exp22',extra=dict(sigma=sigma,inner=n,order=order,stage=a.outer,fixed_historical_scales=True,source=str(path),source_sha256=c.sha(path)))

if __name__=='__main__':
    p=c.args(__doc__);p.add_argument('--outer',choices=['pilot','full'],default='pilot');p.add_argument('--inner-nodes',type=int,default=128)
    main(p.parse_args())
