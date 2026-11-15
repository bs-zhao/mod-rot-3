"""Archive first, then reproduce the historical float32 24x250 optimizer."""
from __future__ import annotations
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as C
sys.path.insert(0,str(C.ROOT/'three_gaps/exp13_pure_simplicity'))
sys.path.insert(0,str(C.ROOT/'three_gaps/exp11_cr3_search'))
import losses2 as L2
import harness as H
from common.rotlib import effective_A,haar

def archive():
    out=C.ROOT/'outputs/three_gaps/post_exp13_20261114/frozen_exp13'
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text(encoding='utf8'))
        for item in previous['files']:
            if not Path(item['archive']).exists() or C.io.sha256(item['archive'])!=item['sha256']:
                raise RuntimeError(f"archive checksum failed: {item['archive']}")
        print('Existing frozen archive verified',flush=True)
        return
    files=list((C.ROOT/'outputs/three_gaps/exp13').glob('*'))
    files=[p for p in files if p.is_file()]
    for model in C.COND:
        for rep in (1,2):
            pp=C.io.paths(model,5e-5 if model=='CR3' else .0005,6 if model=='C3' else 4,rep)
            files += [Path(pp[k]) for k in ('checkpoint','latent','dataset')]
            files += list((C.ROOT/'outputs/three_gaps/exp00'/f'{model}_{C.condition(model,rep)}_pca0').glob('*.npy'))
    records=[]
    for path in sorted(set(files)):
        dest=out/path.relative_to(C.ROOT); dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): shutil.copy2(path,dest)
        sha=C.io.sha256(path)
        if C.io.sha256(dest)!=sha: raise RuntimeError('copy checksum mismatch')
        records.append(dict(source=str(path),archive=str(dest),bytes=path.stat().st_size,sha256=sha))
    zipfile=out/'frozen_source.zip'
    subprocess.run(['git','archive','--format=zip',f'--output={zipfile}','HEAD','three_gaps','np/src'],cwd=C.ROOT,check=True)
    meta=dict(environment=C.environment(),reference_commit='a515a45a1399d859e586fd8f8cf195f83aa6b51a',
              source_archive_sha256=C.io.sha256(zipfile),files=records,
              note='Exact source snapshot plus checkpoint, extracted Z, datasets, derivative caches, existing rotations/figures/logs. Reproduced scales and signed correlations saved separately in baseline/results.')
    C.dump(manifest,meta)
    C.dump(Path(__file__).parent/'results/archive_manifest.json',meta)
    print(f'Archived {len(records)} files, {sum(x["bytes"] for x in records)/1e9:.2f} GB',flush=True)

def reproduce(model,rep,channel):
    out=Path(__file__).parent/'results'/f'{model}_rep{rep}_{channel}'; out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists(): return
    c=C.load(model,rep,channel); sx,t4=L2.moments(c)
    v=torch.tensor(c['V'][c['ids']],dtype=torch.float32)
    rng=np.random.default_rng(1); rr=[torch.tensor(haar(v.shape[1],rng),dtype=torch.float32) for _ in range(200)]
    def terms(A): return torch.stack([L2.LOSSES[n](*L2.energies(A,sx,t4)) for n in ('UX_neff','UUd_share')])
    with torch.no_grad(): scales=np.std(np.stack([terms(effective_A(r,v)).numpy() for r in rr]),axis=0)+1e-9
    loss=lambda b,A,M,cc: (terms(A)/torch.tensor(scales)).sum()
    # harness requires g and X to construct an unused projected base; no role detector is called.
    opt_c={k:c[k] for k in ('V','ids','g','X')}; opt_c['cat']=None
    t=time.perf_counter(); fin=H.optimize(opt_c,loss,starts=24,steps=250,seed=7)
    rs=np.stack([x[1] for x in fin]); ls=np.array([x[0] for x in fin])
    np.savez_compressed(out/'rotations.npz',R=rs,loss=ls,scales=scales,
                        axis_sd=np.stack([np.std(c['V'][c['ids']]@r.T,axis=0) for r in rs]),
                        ids=c['ids'],V_mean=c['Vtrain'].mean(0).numpy())
    met=C.evaluate(c,rs[0]); C.dump(out/'evaluation.json',met)
    row=dict(model=model,repeat=rep,channel=channel,**{k:met[k] for k in ('mcc','leakage','max_leakage')},
             loss=float(ls[0]),seconds=time.perf_counter()-t)
    pd.DataFrame([row]).to_csv(out/'metrics.csv',index=False)
    C.dump(out/'config.json',dict(environment=C.environment(),haar_n=200,haar_seed=1,starts=24,steps=250,
                                 opt_seed=7,lr=.05,subjects=len(c['ids']),trials=len(c['X']),
                                 X_standardized=False,channel=channel,cache_sha256=C.io.sha256(c['cache_path'])))
    C.heatmap(c,{'Identity':np.eye(len(rs[0])),'Frozen EXP13':rs[0],'Oracle':C.oracle(c)},out/'UT_heatmaps.png')
    C.dump(out/'COMPLETE.json',row); print(json.dumps(row),flush=True)

if __name__=='__main__':
    p=C.parser(__doc__); p.add_argument('--archive-only',action='store_true'); p.add_argument('--all-configs',action='store_true')
    a=p.parse_args(); archive()
    if not a.archive_only:
        for m,r,ch in ([(m,r,ch) for m in C.COND for r in (1,2) for ch in ('prob','logit')] if a.all_configs else [(a.model,a.repeat,a.channel)]):
            reproduce(m,r,ch)
