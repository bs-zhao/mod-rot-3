"""Leakage-safe evaluation Z from an existing, frozen NP checkpoint.
Each subject is encoded using a random fraction of trials; only complementary
trials are eligible as target responses in data.py. Does NOT retrain NP.
"""
from __future__ import annotations
import argparse,sys,hashlib,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'np'/'src'))
from modrot3.pipeline import _discrete_data,sha256
from modlearn_rot.generators.dataset import load_dataset
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig,DiscreteNeuralProcess
from modlearn_rot.training.continuous import load_checkpoint


def extract(model,checkpoint,dataset,output,seed=31,context_fraction=.5,batch=64,device='auto'):
    if not (0<context_fraction<1):raise ValueError('context_fraction must be between 0 and 1')
    # _discrete_data requires only args.model, and reconstructs the original trial order.
    args=argparse.Namespace(model=model)
    source=load_dataset(dataset);data=_discrete_data(args,source)
    ck=load_checkpoint(Path(checkpoint))
    if ck.get('extra_state',{}).get('dataset_sha256')!=sha256(Path(dataset)):
        raise ValueError('Checkpoint/dataset SHA mismatch: do not mix datasets')
    device=torch.device('cuda' if device=='auto' and torch.cuda.is_available() else
                        'cpu' if device=='auto' else device)
    net=DiscreteNeuralProcess(DiscreteNPConfig(**ck['model_config']))
    net.load_state_dict(ck['model_state_dict']);net.to(device).eval()
    mean=ck['preprocessing']['encoder_mean'].to(device).reshape(-1)
    std=ck['preprocessing']['encoder_std'].to(device).reshape(-1)
    rng=np.random.default_rng(seed);mask=np.zeros(data.trial_mask.shape,dtype=bool)
    counts=np.asarray(source.trial_counts,dtype=int)
    for i,n in enumerate(counts):
        nctx=max(1,min(n-1,int(round(n*context_fraction))))
        mask[i,rng.choice(n,nctx,replace=False)]=True
    target_mask=(~mask)&data.trial_mask.cpu().numpy()
    z=np.zeros((data.n_subjects,net.config.latent_dim),dtype=np.float32)
    with torch.no_grad():
        for k in range(0,data.n_subjects,batch):
            sl=slice(k,min(k+batch,data.n_subjects))
            x=data.x[sl].to(device)
            y=data.choice[sl].float().to(device).unsqueeze(-1)
            observed=(torch.cat((x,y),dim=-1)-mean)/std
            m=torch.as_tensor(mask[sl],device=device)
            mu,_=net.encode_context(observed[...,:data.x_dim],observed[...,data.x_dim:],m)
            z[sl]=mu.cpu().numpy()
    # Interleaved data are not supported: _discrete_data verifies or reconstructs,
    # but row-order mapping here explicitly assumes subject-concatenated samples.
    rawids=np.asarray(source.subject_index)
    if not np.array_equal(rawids,np.repeat(np.arange(len(counts)),counts)):
        raise ValueError('Expected contiguous ordered original rows for target mask')
    flatmask=np.concatenate([target_mask[i,:n] for i,n in enumerate(counts)])
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Archive existing extraction; never overwrite a latent run')
    np.savez_compressed(out,z=z,target_trial_mask=flatmask,
                        context_mode=np.asarray('context_only_heldout_targets'),
                        dataset_sha256=np.asarray(sha256(Path(dataset))),
                        checkpoint_sha256=np.asarray(sha256(Path(checkpoint))),
                        seed=seed,context_fraction=context_fraction)
    print(json.dumps({'output':str(out),'z_shape':z.shape,
                      'target_trials':int(flatmask.sum()),'context_fraction':context_fraction},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['C1','C2','C3','CR3'],required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=31)
    p.add_argument('--context-fraction',type=float,default=.5)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--device',default='auto')
    a=p.parse_args();extract(a.model,a.checkpoint,a.dataset,a.output,a.seed,a.context_fraction,a.batch,a.device)
