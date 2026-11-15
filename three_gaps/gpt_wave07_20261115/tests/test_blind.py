"""Actual loader and optimizer, with forbidden truth reads made fatal."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import numpy as np
import torch

def main(a):
    getitem=np.lib.npyio.NpzFile.__getitem__;load=np.load;seen=[]
    def guarded(z,key):
        seen.append(str(key))
        if key in ['theta_true','parameters','predictive_true','Tst','T','oracle']:raise AssertionError('truth field opened by optimizer')
        return getitem(z,key)
    def guarded_load(path,*args,**kwargs):
        if 'oracle' in str(path).lower():raise AssertionError('Oracle file opened by optimizer')
        return load(path,*args,**kwargs)
    np.lib.npyio.NpzFile.__getitem__=guarded;np.load=guarded_load
    try:
        st=c.Setting(a);b=c.fields(st,st.V[st.tr[:3]],st.X[:8]);sx,t4=c.moments(b['Htt'],b['Htx'])
        R=c.haar_set(st.d,1,17)[0];sd=np.diag(R@st.C@R.T)**.5
        V=st.V[st.tr[:3]];U=(V@R.T)/sd;recovered=(U*sd)@R
        np.testing.assert_allclose(recovered,V,atol=1e-12)
        with torch.no_grad():
            for v,vr in zip(V,recovered):
                p=st.h(torch.tensor(v,device=st.dev),torch.tensor(st.X[0],device=st.dev))
                pr=st.h(torch.tensor(vr,device=st.dev),torch.tensor(st.X[0],device=st.dev))
                torch.testing.assert_close(p,pr,rtol=1e-6,atol=1e-7)
        obj=c.Loss(sx.mean(0),t4.mean(0),st.C);R,L,_,_=c.search(obj,starts=2,steps=3)
        assert np.isfinite(L).all()
    finally:
        np.lib.npyio.NpzFile.__getitem__=getitem;np.load=load
    c.dump(c.HERE/'tests/blind_access_audit.json',dict(passed=True,accessed_NPZ_keys=sorted(set(seen)),oracle_opened=False,truth_opened=False))
    print('ACTUAL LOADER + DERIVATIVES + OPTIMIZER BLIND ACCESS TEST PASS')

if __name__=='__main__':main(c.args(__doc__).parse_args())
