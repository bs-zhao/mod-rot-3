"""T-free observed validation prediction audit; function smoothing changes f."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import nodes
import numpy as np
import torch

def audit(st,sigmas,n=128):
    with np.load(st.paths['dataset'],allow_pickle=False) as z:
        si=z['subject_index'];mask=np.isin(si,st.va[:64]);si=si[mask]
        X=z['x'][mask];Y=z['y'].reshape(-1)[mask]
    Z=st.z[si];out=[]
    delta=nodes.antithetic(n,st.d,20261115)@np.linalg.cholesky(st.C).T
    delta=delta@st.Minv.T
    def query(shift):
        result=[]
        with torch.no_grad():
            for i in range(0,len(Z),8192):
                x=torch.tensor(X[i:i+8192],dtype=torch.float32,device=st.dev)
                zz=torch.tensor(Z[i:i+8192]+shift,dtype=torch.float32,device=st.dev)
                lg=st.net.decoder(torch.cat([x,zz],1));result.append(torch.sigmoid(lg[:,1]-lg[:,0]).double().cpu().numpy())
        return np.concatenate(result)
    p0=query(np.zeros(st.d))
    for sigma in sigmas:
        p=p0 if sigma==0 else np.mean([query(sigma*shift) for shift in delta],0)
        pc=np.clip(p,1e-12,1-1e-12)
        out.append(dict(sigma=sigma,inner=n,validation_subjects=st.va[:64].tolist(),trials=len(Y),
            mean_abs_dp=float(abs(p-p0).mean()),rms_dp=float(np.sqrt(((p-p0)**2).mean())),
            CE=float(-(Y*np.log(pc)+(1-Y)*np.log1p(-pc)).mean()),Brier=float(((p-Y)**2).mean()),
            saturation=float(((p<.01)|(p>.99)).mean())))
    return out
