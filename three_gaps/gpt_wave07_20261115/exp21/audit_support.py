"""Post-search geometric support audit including off-diagonal UU corners.

No decoder queries or truth access. Mahalanobis and nearest-neighbour coverage
are geometric proxies, not proof that counterfactual trials are identifiable.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import numpy as np
from scipy.spatial import cKDTree

def main(a):
    st=c.Setting(a);inv=np.linalg.inv(st.C)
    train_radius=np.einsum('na,ab,nb->n',st.V[st.tr],inv,st.V[st.tr]);cap=np.quantile(train_radius,.99)
    sd=st.X.std(0);mean=st.X.mean(0);tree=cKDTree((st.X-mean)/sd)
    nn95=np.quantile(tree.query((st.X-mean)/sd,k=2)[0][:,1],.95)
    for cp in (c.HERE/'exp21/results'/st.key/a.tag).glob('*/COMPLETE.json'):
        folder=cp.parent;cfg=json.loads((folder/'config.json').read_text())
        if 'mode' not in cfg:continue
        rp=folder/'rotations.npz';complete=json.loads(cp.read_text())
        if c.sha(rp)!=complete['rotations_sha256']:raise ValueError('rotation hash')
        with np.load(rp) as z:R=z['R'][int(z['best_index'])].astype(float)
        D=np.sqrt(np.diag(R@st.C@R.T));A=D[:,None]*np.linalg.inv(R).T
        step=cfg['h_U_halfwidth']*A;V=st.V[cfg['subjects']];X=st.X[cfg['X_indices']]
        shifts=[np.zeros(st.d)]+[sign*step[i] for i in range(st.d) for sign in [-1,1]]
        shifts += [s*step[i]+t*step[j] for i in range(st.d) for j in range(i+1,st.d) for s in [-1,1] for t in [-1,1]]
        q=(V[:,None,:]+np.asarray(shifts)[None,:,:]).reshape(-1,st.d)
        radius=np.einsum('na,ab,nb->n',q,inv,q)
        xshifts=[np.zeros(st.p)]
        if cfg['h_X_halfwidth'] is not None:
            xshifts += [sign*np.eye(st.p)[j]*cfg['h_X_halfwidth'][j] for j in range(st.p) for sign in [-1,1]]
        xq=(X[:,None,:]+np.asarray(xshifts)[None,:,:]).reshape(-1,st.p)
        distances=tree.query((xq-mean)/sd)[0]
        c.dump(folder/'support_all_stencil.json',dict(truth_loaded=False,decoder_queried=False,
            rotation_sha256=c.sha(rp),geometry='equal weight per unique U stencil location and centre, not query-frequency weighted',
            U_locations=len(q),U_in_train_mahalanobis99=float((radius<=cap).mean()),
            U_mahalanobis_quantiles=np.quantile(radius,[.5,.9,.99]),X_locations=len(xq),
            X_in_marginal_box=float(((xq>=st.X.min(0))&(xq<=st.X.max(0))).all(1).mean()),
            X_within_bank_nn95=float((distances<=nn95).mean()),X_nearest_distance_quantiles=np.quantile(distances,[.5,.9,.99]),
            warning='These proxies do not establish joint counterfactual identifiability; cross-coordinate UU corners are included.'))
        print(folder.name,float((radius<=cap).mean()),float((distances<=nn95).mean()))

if __name__=='__main__':main(c.args(__doc__).parse_args())
