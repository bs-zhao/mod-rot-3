"""WY-only predictive estimators (never imports simulation formulas)."""
from __future__ import annotations
import argparse
import json
import pickle
import csv
import time
from pathlib import Path
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss


def metrics(y, pred, binary):
    y=np.asarray(y).ravel(); pred=np.asarray(pred).ravel()
    if binary:
        pred=np.clip(pred,1e-6,1-1e-6)
        return {'log_loss':float(log_loss(y,pred,labels=[0,1])),
                'brier':float(np.mean((y-pred)**2))}
    return {'rmse':float(np.sqrt(mean_squared_error(y,pred))),
            'r2':float(1-np.sum((pred-y)**2)/max(np.sum((y-y.mean())**2),1e-12))}


class SplineANOVA:
    """Concrete spline main/pairwise approximation, NOT SKIM-FA or OAK."""
    def __init__(self, order=2, knots=5, alpha=10.):
        self.order=order; self.knots=knots; self.alpha=alpha
    def _features(self,x):
        b=[t.transform(x[:,j:j+1])-self.means[j] for j,t in enumerate(self.transforms)]
        features=b.copy()
        if self.order>=2:
            for i in range(len(b)):
                for j in range(i+1,len(b)):
                    features.append(np.einsum('na,nb->nab',b[i],b[j]).reshape(len(x),-1))
        return np.concatenate(features,axis=1)
    def fit(self,x,y,binary=False):
        self.binary=binary
        self.transforms=[]; self.means=[]
        for j in range(x.shape[1]):
            t=SplineTransformer(n_knots=self.knots,degree=2,include_bias=False,
                                extrapolation='constant').fit(x[:,j:j+1])
            self.transforms.append(t)
            self.means.append(t.transform(x[:,j:j+1]).mean(axis=0))
        f=self._features(x)
        self.reg=(LogisticRegression(C=1/max(self.alpha,1e-8),max_iter=500) if binary else Ridge(alpha=self.alpha)).fit(f,y)
        return self
    def predict(self,x):
        f=self._features(np.asarray(x))
        return self.reg.predict_proba(f)[:,1] if self.binary else self.reg.predict(f)


# Canonicalize pickled class identity when invoked as: python models.py.
if __name__ == "__main__":
    sys.modules.setdefault("models",sys.modules[__name__])
    SplineANOVA.__module__ = "models"


class SmoothMLP:
    def __init__(self,d, binary, hidden=96,depth=3, device='cpu'):
        import torch
        from torch import nn
        self.device=device; self.binary=binary
        parts=[]; inp=d
        for _ in range(depth):
            parts.extend([nn.Linear(inp,hidden),nn.SiLU()]); inp=hidden
        parts.append(nn.Linear(inp,1))
        self.net=nn.Sequential(*parts).to(device)
        self.hidden=hidden; self.depth=depth
    def fit(self,x,y,xval,yval,seed=31,epochs=200,patience=25,batch=1024,lr=8e-4):
        import torch
        torch.manual_seed(seed)
        self.mean=x.mean(axis=0); self.std=np.maximum(x.std(axis=0),1e-6)
        self.ymean=float(y.mean()) if not self.binary else 0.
        self.ystd=float(max(y.std(),1e-6)) if not self.binary else 1.
        def pack(xx,yy):
            tx=torch.as_tensor((xx-self.mean)/self.std,device=self.device,dtype=torch.float32)
            ty=torch.as_tensor(((yy-self.ymean)/self.ystd),device=self.device,dtype=torch.float32).reshape(-1,1)
            return tx,ty
        tx,ty=pack(x,y); vx,vy=pack(xval,yval)
        opt=torch.optim.AdamW(self.net.parameters(),lr=lr,weight_decay=1e-5)
        fn=torch.nn.functional.binary_cross_entropy_with_logits if self.binary else torch.nn.functional.mse_loss
        best=float('inf'); stale=0; best_state=None
        self.training_history=[]; self.optimizer_steps=0; self.best_epoch=None
        training_started=time.perf_counter()
        rng=np.random.default_rng(seed)
        for epoch in range(epochs):
            self.net.train()
            for idx in np.array_split(rng.permutation(len(tx)),max(1,int(np.ceil(len(tx)/batch)))):
                opt.zero_grad(set_to_none=True)
                loss=fn(self.net(tx[idx]),ty[idx]);loss.backward();opt.step()
                self.optimizer_steps+=1
            self.net.eval()
            with torch.no_grad():
                value=float(fn(self.net(vx),vy).item())
                train_loss=float(fn(self.net(tx),ty).item())
            self.training_history.append({'epoch':epoch+1,'train_loss':train_loss,'val_loss':value})
            if epoch==0 or (epoch+1)%20==0 or epoch+1==epochs:
                print(f'[MLP] epoch={epoch+1}/{epochs} train_loss={train_loss:.8f} val_loss={value:.8f}',flush=True)
            if value<best-1e-5:
                self.best_epoch=epoch+1
                best=value;stale=0
                best_state={k:v.detach().cpu().clone() for k,v in self.net.state_dict().items()}
            else:
                stale+=1
                if stale>=patience:break
        self.net.load_state_dict(best_state)
        self.net.eval()
        self.epochs_used=epoch+1
        self.train_seconds=float(time.perf_counter()-training_started)
        self.early_stopped=bool(self.epochs_used<epochs)
        print(f'[MLP] FINISHED epochs={self.epochs_used} best_epoch={self.best_epoch} '
              f'optimizer_steps={self.optimizer_steps} elapsed={self.train_seconds:.2f}s',flush=True)
        return self
    def tensor_forward(self, w):
        import torch
        mu=torch.as_tensor(self.mean,device=w.device,dtype=w.dtype)
        sd=torch.as_tensor(self.std,device=w.device,dtype=w.dtype)
        y=self.net((w-mu)/sd).squeeze(-1)
        return torch.sigmoid(y) if self.binary else (y*self.ystd+self.ymean)
    def predict(self,x):
        import torch
        with torch.no_grad():
            chunks=[]
            for chunk in np.array_split(x,max(1,int(np.ceil(len(x)/4096)))):
                t=torch.as_tensor(chunk,device=self.device,dtype=torch.float32)
                chunks.append(self.tensor_forward(t).cpu().numpy())
        return np.concatenate(chunks)


def load_fit(folder, device='cpu'):
    folder=Path(folder)
    meta=json.loads((folder/'fit.json').read_text())
    if meta['method']=='mlp':
        import torch
        ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        m=SmoothMLP(ck['dim'],ck['binary'],ck['hidden'],ck['depth'],device=device)
        m.net.load_state_dict(ck['net']);m.net.eval()
        m.mean=np.asarray(ck['mean']);m.std=np.asarray(ck['std'])
        m.ymean=ck['ymean'];m.ystd=ck['ystd']
    else:
        with (folder/'model.pkl').open('rb') as fp:m=pickle.load(fp)
    return m,meta


def fit(data_dir,method,output,seed=31,max_train=12000,epochs=200,hidden=96):
    data=Path(data_dir); out=Path(output);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((data/'meta.json').read_text());binary=meta['label']=='bernoulli'
    with np.load(data/'train.npz') as d:xt=d['W'];yt=d['y']
    with np.load(data/'val.npz') as d:xv=d['W'];yv=d['y']
    with np.load(data/'test.npz') as d:xx=d['W'];yy=d['y']
    if max_train>0 and len(xt)>max_train:
        idx=np.random.default_rng(seed).choice(len(xt),max_train,replace=False)
        xt,yt=xt[idx],yt[idx]
    if method=='mlp':
        import torch
        torch.manual_seed(seed)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        m=SmoothMLP(xt.shape[1],binary,hidden=hidden,device=device)
        m.fit(xt,yt,xv,yv,seed=seed,epochs=epochs)
        ck={'dim':xt.shape[1],'binary':binary,'hidden':hidden,'depth':3,
            'net':{k:v.cpu() for k,v in m.net.state_dict().items()},
            'mean':m.mean.tolist(),'std':m.std.tolist(),
            'ymean':m.ymean,'ystd':m.ystd}
        torch.save(ck,out/'model.pt')
        with (out/'training_history.csv').open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=['epoch','train_loss','val_loss'])
            writer.writeheader();writer.writerows(m.training_history)
        training_info={'epochs_requested':int(epochs),'epochs_used':int(m.epochs_used),
                       'best_epoch':int(m.best_epoch),'optimizer_steps':int(m.optimizer_steps),
                       'training_seconds':float(m.train_seconds),'early_stopped':bool(m.early_stopped),
                       'device':device,'parameter_count':int(sum(q.numel() for q in m.net.parameters())),
                       'checkpoint_bytes':int((out/'model.pt').stat().st_size)}
    elif method in {'spline_add','spline_pair'}:
        m=SplineANOVA(order=1 if method=='spline_add' else 2).fit(xt,yt,binary=binary)
        with (out/'model.pkl').open('wb') as fp:pickle.dump(m,fp)
    elif method=='ebm':
        try:
            from interpret.glassbox import ExplainableBoostingRegressor,ExplainableBoostingClassifier
        except ImportError as e:
            raise ImportError('EBM is optional: pip install interpret') from e
        cls=ExplainableBoostingClassifier if binary else ExplainableBoostingRegressor
        m=cls(interactions=min(25,xt.shape[1]*(xt.shape[1]-1)//2),random_state=seed)
        m.fit(xt,yt)
        with (out/'model.pkl').open('wb') as fp:pickle.dump(m,fp)
    else:
        raise ValueError('Implemented: mlp,spline_add,spline_pair,ebm; SKIM-FA, OAK, GAMI-Net, Diamond are NOT impersonated.')
    def predict(v):
        return m.predict_proba(v)[:,1] if binary and method=='ebm' else m.predict(v)
    result={'method':method,'seed':seed,'n_train':len(xt),'max_train':max_train,
            'validation':metrics(yv,predict(xv),binary),'test':metrics(yy,predict(xx),binary),
            'data_dir':str(data.resolve()), 'label':meta['label'], 'model':meta['model']}
    if method=='mlp':result['training']=training_info
    (out/'fit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2)); return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--method',choices=['mlp','spline_add','spline_pair','ebm'],required=True)
    p.add_argument('--seed',type=int,default=31);p.add_argument('--max-train',type=int,default=12000)
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--hidden',type=int,default=96)
    a=p.parse_args();fit(a.data,a.method,a.output,a.seed,a.max_train,a.epochs,a.hidden)
