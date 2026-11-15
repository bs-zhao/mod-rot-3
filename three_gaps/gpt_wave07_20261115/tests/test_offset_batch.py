"""Actual frozen decoder: grouped query batching leaves moments unchanged."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'exp22'))
import core as c
from kernels import smooth
import numpy as np

def main(a):
    st=c.Setting(a);V=st.V[st.ids0[:4]];X=st.X[:8];root=c.OUT/'tests/offset_batch'/st.key/str(st.dev).replace(':','_')
    one=smooth(st,V,X,[0,.1,.25,.5],[8,16],root/'one',query_batch=1)
    four=smooth(st,V,X,[0,.1,.25,.5],[8,16],root/'four',query_batch=4)
    errs={k:float(np.linalg.norm(one[k]-four[k])/max(np.linalg.norm(one[k]),1e-12)) for k in one}
    if max(errs.values())>2e-6:raise AssertionError(errs)
    c.dump(c.HERE/'tests/offset_batch_audit.json',dict(passed=True,max_relative=max(errs.values()),errors=errs))
    print('GROUPED PHYSICAL NODE QUERY TEST PASS',max(errs.values()))

if __name__=='__main__':main(c.args(__doc__).parse_args())
