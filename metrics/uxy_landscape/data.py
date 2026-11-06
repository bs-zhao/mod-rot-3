"""Authorized supervised diagnostic preparation; no metric definition lives here."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from scipy.special import expit, logit
from modlearn_rot.generators.registry import get_model
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig, DiscreteNeuralProcess
from modlearn_rot.training.continuous import load_checkpoint

ROOT=Path(__file__).resolve().parents[2]
Y_NAMES=['observed_choice','decoder_choice','decoder_logit','true_logit','true_logit_noise']


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def corr(a,b):
    a=np.asarray(a,float)-np.mean(a,axis=0);b=np.asarray(b,float)-np.mean(b,axis=0)
    return (a.T@b)/np.maximum(np.linalg.norm(a,axis=0)[:,None]*np.linalg.norm(b,axis=0)[None,:],1e-30)


def cr3_logit(x,t):
    """Exact 4P generator, including amount /100 and probability clipping."""
    a,lam,gamma,tau=t.T
    def value(v):
        gain,loss,p=v.T
        wg=expit(gamma*logit(np.clip(p,1e-12,1-1e-12)))
        wl=expit(gamma*logit(np.clip(1-p,1e-12,1-1e-12)))
        return wg*(gain/100)**a-lam*wl*(loss/100)**a
    return tau*(value(x[:,:3])-value(x[:,3:]))


def true_logit(model,x,t):
    if model=='CR3':return cr3_logit(x,t)
    generator=get_model(model)
    # Paired row evaluation avoids broadcasting all subjects against all trials.
    return np.einsum('ij,ij->i',generator.design_matrix(x),t)


def prepare(model,cfg):
    folder=ROOT/'outputs/choice'/model
    latent=folder/'s4_evaluate'/cfg['condition']/f'repeat{cfg["repeat"]:02d}'/'s4_evaluate__latent.npz'
    checkpoint=folder/'s3_train'/cfg['condition']/f'repeat{cfg["repeat"]:02d}'/'s3_train__checkpoint.pt'
    dataset=folder/'s1_generate/s1_generate__dataset.npz'
    hashes=dict(latent=sha256(latent),checkpoint=sha256(checkpoint),dataset=sha256(dataset))
    with np.load(latent,allow_pickle=False) as f:
        assert str(f['dataset_sha256'])==hashes['dataset'] and str(f['checkpoint_sha256'])==hashes['checkpoint']
        assert str(f['context_mode'])=='all_observed_trials'
        z=f['z'].astype(float);tr=f['train_ids'];te=f['test_ids'];val=f['validation_ids']
    with np.load(dataset,allow_pickle=False) as f:
        theta=f['parameters'].astype(float);counts=f['trial_counts'].astype(int)
        offsets=np.r_[0,np.cumsum(counts)]
        rng=np.random.default_rng(cfg['seed'])
        chosen=np.sort(rng.choice(te,min(cfg['analysis_subjects'],len(te)),replace=False))
        row_ids=[];subject=[];within=[]
        for i in chosen:
            ids=np.sort(rng.choice(counts[i],min(counts[i],cfg['trials_per_subject']),replace=False))
            row_ids.extend(offsets[i]+ids);subject.extend([i]*len(ids));within.extend(ids)
        row_ids=np.asarray(row_ids);subject=np.asarray(subject)
        all_x=f['x'];x=all_x[row_ids].astype(float);observed=f['y'][row_ids].reshape(-1).astype(float)
        # Fixed TRAIN observations for X scaling and oracle noise calibration.
        train_rows=np.concatenate([offsets[i]+rng.choice(counts[i],min(counts[i],4),replace=False) for i in tr[:512]])
        train_x=all_x[train_rows].astype(float)
        train_subject=np.searchsorted(offsets[1:],train_rows,side='right')
    mu=z[tr].mean(0);tm=theta[tr].mean(0);ts=theta[tr].std(0)
    target=(theta-tm)/ts
    b=np.linalg.lstsq(z[tr]-mu,target[tr],rcond=None)[0]
    u=(z-mu)@b
    # A single isotropic scale is fixed before rotations; never restandardize
    # individual rotated axes, which would change the rotation experiment.
    scale=np.sqrt(np.mean(np.var(u[tr],axis=0)))
    u/=scale
    ck=load_checkpoint(checkpoint)
    net=DiscreteNeuralProcess(DiscreteNPConfig(**ck['model_config']))
    net.load_state_dict(ck['model_state_dict']);net=net.to(cfg['device']).double().eval();net.requires_grad_(False)
    eta=[]
    with torch.no_grad():
        for start in range(0,len(x),2048):
            sl=slice(start,start+2048)
            logits=net.decode(torch.tensor(x[sl,None],dtype=torch.float64,device=cfg['device']),
                              torch.tensor(z[subject[sl]],dtype=torch.float64,device=cfg['device']))[:,0]
            eta.append((logits[:,1]-logits[:,0]).cpu().numpy())
    eta=np.concatenate(eta)
    truth_eta=true_logit(model,x,theta[subject])
    calibration=true_logit(model,train_x,theta[train_subject])
    noise_sd=cfg['noise_sd_fraction']*calibration.std()
    response_rng=np.random.default_rng(cfg['seed']+2701)
    choice=(response_rng.random(len(x))<expit(eta)).astype(float)
    noise=response_rng.normal(0,noise_sd,len(x))
    y=np.column_stack([observed,choice,eta,truth_eta,truth_eta+noise])
    assoc_subjects=np.sort(rng.choice(chosen,min(len(chosen),cfg['association_subjects']),replace=False))
    # Exactly one actual trial per subject for iid-style dependence estimators.
    assoc=np.array([np.flatnonzero(subject==i)[0] for i in assoc_subjects])
    folds=np.array_split(rng.permutation(chosen),cfg['cv_folds'])
    split=[(np.flatnonzero(~np.isin(subject,fold)),np.flatnonzero(np.isin(subject,fold))) for fold in folds]
    x_mean=train_x.mean(0);x_scale=np.maximum(train_x.std(0),1e-8)
    reference=dict(A_star=b.T/scale,z_mean=mu,theta_mean=tm,theta_scale=ts,isotropic_scale=scale,
        theta_parameter_names=['alpha','lambda_loss','gamma','tau'] if model=='CR3' else [f'b{i+1}' for i in range(theta.shape[1])],
        train_corr=corr(theta[tr],u[tr]),test_corr=corr(theta[te],u[te]),
        train_diagonal_r2=np.diag(corr(theta[tr],u[tr]))**2,test_diagonal_r2=np.diag(corr(theta[te],u[te]))**2,
        note='TRAIN OLS on original full 8D Z; supervised diagnostic, generally rectangular/oblique, not an unsupervised solution')
    return dict(model=model,folder=folder,z=z,theta=theta,target=target,tr=tr,te=te,val=val,reference=reference,u=u,
                x=(x-x_mean)/x_scale,x_raw=x,x_mean=x_mean,x_scale=x_scale,y=y,subject=subject,
                row_ids=row_ids,within_trial=np.asarray(within),assoc=assoc,split=split,noise_sd=noise_sd,
                decoder_probability=expit(eta),hashes=hashes,chosen=chosen)


def history(source,cfg):
    """Load saved coordinates only; never import or execute rotation suites."""
    entries=[];excluded=[];root=source['folder']/'rotations';n=len(source['z']);k=source['u'].shape[1]
    expected=source['hashes']
    def valid(meta):
        keys=[('latent',['source_z_sha256','source_z_sha','latent_sha']),
              ('checkpoint',['source_checkpoint_sha256','source_checkpoint_sha','checkpoint_sha']),
              ('dataset',['source_dataset_sha256','source_dataset_sha','source_dataset_sha'])]
        for kind,alternatives in keys:
            values=[meta[key] for key in alternatives if key in meta]
            if values and values[0]!=expected[kind]:return False
        return any(key in meta for key in ['source_z_sha256','source_z_sha','latent_sha'])
    def add(name,u,path):
        if u.shape!=(n,k) or not np.isfinite(u).all():
            excluded.append(dict(path=str(path),reason='shape or finiteness'));return
        centered=u-u[source['tr']].mean(0)
        scale=np.sqrt(np.mean(np.var(centered[source['tr']],axis=0)))
        if scale<1e-12:return
        entries.append(dict(name=name,U=centered/scale,path=str(path)))
    manifest=ROOT/'rotations/historical_transfer/results/run_manifest.json'
    if manifest.exists():
        for raw in json.loads(manifest.read_text())['runs']:
            folder=Path(raw);meta=json.loads((folder/'protocol.json').read_text())
            if meta['model']!=source['model']:continue
            if meta['latent_sha']!=expected['latent'] or meta['checkpoint_sha']!=expected['checkpoint']:continue
            names=json.loads((folder/'fit_complete.json').read_text())['methods']
            for name in names:
                p=folder/name/'mapping.npz'
                with np.load(p) as f:add('historical_transfer/'+meta['representation']['kind']+'/'+name,f['U'],p)
    for suite in root.iterdir():
        if not suite.is_dir() or suite.name=='historical_transfer':continue
        parent=suite/cfg['condition']/f'repeat{cfg["repeat"]:02d}'
        for folder in sorted(parent.glob('run_*')):
            fit=next((folder/name for name in ['fit.json','rotation__fit.json'] if (folder/name).exists()),None)
            if fit is None:continue
            meta=json.loads(fit.read_text())
            if not valid(meta):excluded.append(dict(path=str(folder),reason='missing/mismatched provenance'));continue
            if (folder/'rotations.npz').exists() and (folder/'mapping.npz').exists():
                with np.load(folder/'mapping.npz') as f:
                    q=(source['z']-f['mu'])@f['W'].T
                with np.load(folder/'rotations.npz') as f:
                    for name in f.files:
                        a=f[name]
                        if a.shape==(k,k):add(suite.name+'/'+folder.name+'/'+name,q@a.T,folder/'rotations.npz')
            else:
                candidates=[folder/'rotation__U.npz']+sorted(folder.glob('candidate__*/rotation__U.npz'))+sorted(folder.glob('ablation__*/rotation__U.npz'))
                for p in candidates:
                    if not p.exists():continue
                    with np.load(p) as f:
                        if 'test_ids' in f and not np.array_equal(f['test_ids'],source['te']):continue
                        add(suite.name+'/'+folder.name+'/'+('primary' if p.parent==folder else p.parent.name),f['U'],p)
    return entries,excluded
