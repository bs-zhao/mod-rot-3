"""Post-lock diagnostic only. Never supplies an oracle rotation to optimization."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from core import load,dump,sha
from single_metric_atlas import BatchLoss
from three_gap_batch import read_bundle,reference_path

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default=str(ROOT.parents[1]));p.add_argument('--model',choices=['C1','C2','C3','CR3'],required=True)
    p.add_argument('--beta',type=float,required=True);p.add_argument('--zdim',type=int,required=True);p.add_argument('--pca',type=int,required=True)
    args=p.parse_args();context=load(args);cases=[]
    for folder in [f'b{i:02d}_three_gap_batch' for i in range(1,8)]:
        for path in (ROOT/'results'/folder/args.model/'gaussian256'/'starts8').glob('hs_*.json'):
            if path.stem.endswith('_rotation'):continue
            j=json.loads(path.read_text(encoding='utf8'))
            if 'gaps' not in j:continue
            lock=Path(j['gaps']['G3']['rotation_lock_file'])
            locked=json.loads(lock.read_text(encoding='utf8'))
            if locked['protocol_sha256']!=j['protocol_sha256']:raise ValueError('Wrong rotation lock')
            np.testing.assert_array_equal(locked['r'],j['gaps']['G3']['best']['r'])
            cases.append((path,j))
    if not cases:raise ValueError('No pre-existing locked results to audit')
    fields,cov,provenance=read_bundle(reference_path('h09_g3_frozen_decoder',args.model),context,args)
    names=sorted(set(j['metric'] for _,j in cases));batch=BatchLoss(fields,cov,args.zdim,names)
    oracle_path=list((ROOT/'results'/'oracles').glob(f'{args.model}_*pca{args.pca}.json'))[0]
    oracle=json.loads(oracle_path.read_text(encoding='utf8'))
    if oracle['hashes']!=context['hashes']:raise ValueError('Wrong frozen oracle')
    oracle_values=batch(np.asarray(oracle['rotation']));rows=[]
    for path,j in cases:
        metric=j['metric'];g=j['gaps']['G3'];value=oracle_values[metric];best=g['best']['loss']
        rows.append(dict(metric=metric,reference_loss=value,discovered_loss=best,reference_minus_discovered=value-best,
            reference_is_lower=value<best-1e-6*(1+abs(best)),test_mean=g['evaluation']['matched_mean'],
            test_mean_gap=g['mean_gap'],case=str(path),locked_rotation_sha256=sha(g['rotation_lock_file'])))
    out=ROOT/'results'/'posthoc_oracle_audit'/args.model;out.mkdir(parents=True,exist_ok=True)
    with (out/'objectives.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(out/'audit.json',dict(args=vars(args),rows=rows,fields=provenance,oracle_sha256=sha(oracle_path),
        rule='Diagnostic after all audited rotations were locked; values never returned to the optimizer'))
    print(json.dumps(dict(model=args.model,cases=len(rows),oracle_lower=sum(r['reference_is_lower'] for r in rows))))


if __name__=='__main__':main()
