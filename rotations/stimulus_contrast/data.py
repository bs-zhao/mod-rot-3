"""Truth-free NP input loader, adapted from the historical suite's I/O contract."""
from pathlib import Path
import hashlib
import numpy as np
from sklearn.decomposition import PCA
from modlearn_rot.generators.dataset import load_observed_dataset

ROOT=Path(__file__).resolve().parents[2]


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def corr(a,b):
    a=np.array(a,dtype=float,copy=True);b=np.array(b,dtype=float,copy=True)
    a-=a.mean(0);b-=b.mean(0)
    scale=np.sqrt(np.sum(a*a,axis=0)[:,None]*np.sum(b*b,axis=0)[None,:])
    return np.divide(a.T@b,scale,out=np.zeros((a.shape[1],b.shape[1])),where=scale>1e-12)


def load_frozen(args):
    folder=ROOT/'outputs'/'choice'/args.model
    latent=folder/'s4_evaluate'/args.condition/f'repeat{args.repeat:02d}'/'s4_evaluate__latent.npz'
    checkpoint=folder/'s3_train'/args.condition/f'repeat{args.repeat:02d}'/'s3_train__checkpoint.pt'
    observed=folder/'s1_generate'/'s1_generate__dataset.npz'
    for path in (latent,checkpoint,observed):
        if not path.is_file():raise FileNotFoundError(path)
    dataset_hash=sha256(observed);checkpoint_hash=sha256(checkpoint)
    with np.load(latent,allow_pickle=False) as f:
        if str(f['context_mode'])!='all_observed_trials':raise ValueError('Expected S5 all-trial latents')
        if str(f['dataset_sha256'])!=dataset_hash or str(f['checkpoint_sha256'])!=checkpoint_hash:
            raise ValueError('Dataset/checkpoint/latent provenance mismatch')
        z=f['z'].astype(float);train=f['train_ids'];val=f['validation_ids'];test=f['test_ids']
    if not np.isfinite(z).all() or not np.array_equal(np.sort(np.r_[train,val,test]),np.arange(len(z))):
        raise ValueError('Invalid latent or splits')
    k={'C1':4,'C2':4,'C3':6,'CR3':4}[args.model]
    pca=PCA(n_components=k,svd_solver='full').fit(z[train])
    w=pca.components_/np.sqrt(np.maximum(pca.explained_variance_,args.whiten_floor))[:,None]
    chosen=np.sort(np.random.default_rng(args.seed).choice(train,min(len(train),args.fit_max_subjects),replace=False))
    # Unlike the historical adapter, only observation members are read from NPZ.
    data=load_observed_dataset(observed)
    counts=np.asarray(data.trial_counts,int);n=len(counts);t=int(counts.max());d=data.x.shape[1]
    if n!=len(z) or not np.array_equal(data.subject_index,np.repeat(np.arange(n),counts)):
        raise ValueError('Dataset and latent subject order mismatch')
    x=np.zeros((n,t,d),np.float32);y=np.zeros((n,t),np.int64);mask=np.zeros((n,t),bool)
    offsets=np.r_[0,np.cumsum(counts)]
    for i,count in enumerate(counts):
        sl=slice(offsets[i],offsets[i+1])
        x[i,:count]=data.x[sl];y[i,:count]=np.asarray(data.y[sl]).reshape(-1);mask[i,:count]=True
    if not np.all(np.isin(y[mask],[0,1])):raise ValueError('Nonbinary observations')
    return dict(z=z,z_fit=z[chosen],q_fit=((z-pca.mean_)@w.T)[chosen],x_fit=x[chosen],y_fit=y[chosen],mask_fit=mask[chosen],
                checkpoint_path=checkpoint,whiten_floor=args.whiten_floor,W=w,pca=pca,chosen=chosen,tr=train,val=val,te=test,
                latent_sha=sha256(latent),dataset_sha=dataset_hash,checkpoint_sha=checkpoint_hash,model=args.model,
                condition=args.condition,native_k=z.shape[1],rotation_dim=k,output_root=folder)
