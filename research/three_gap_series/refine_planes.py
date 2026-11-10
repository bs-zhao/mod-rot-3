"""Continuously refine promising sampled plane minima without changing a metric."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from core import load,dump,sha
from strict_hessian import HessianLoss
from three_gap_batch import read_bundle

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default=str(ROOT.parents[1]));p.add_argument('--model',choices=['C1','C2','C3','CR3'],required=True)
    p.add_argument('--beta',type=float,required=True);p.add_argument('--zdim',type=int,required=True);p.add_argument('--pca',type=int,required=True)
    p.add_argument('--label',required=True);p.add_argument('--metric',required=True);p.add_argument('--inference',default='gaussian256')
    p.add_argument('--starts',type=int,default=64);args=p.parse_args()
    folder=ROOT/'results'/args.label/args.model/args.inference/f'starts{args.starts}'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    path=folder/(args.metric+'.json');case=json.loads(path.read_text(encoding='utf8'));context=load(args);results={}
    for stage in ['G1','G2']:
        fields,cov,provenance=read_bundle(Path(manifest['fields'][stage]['summary']),context,args)
        loss=HessianLoss(fields,cov,args.zdim,args.metric);minima=[]
        for i,j in itertools.combinations(range(args.zdim),2):
            plane=sorted([q for q in case['gaps'][stage]['curve'] if (q['i'],q['j'])==(i,j)],key=lambda q:q['angle'])
            def objective(angle):
                a=np.deg2rad(angle);r=np.eye(args.zdim);r[i,i]=r[j,j]=np.cos(a);r[i,j]=-np.sin(a);r[j,i]=np.sin(a)
                return loss(r)
            candidates=[dict(angle=q['angle'],loss=q['loss']) for q in plane]
            for k in range(1,len(plane)-1):
                if plane[k]['loss']<=plane[k-1]['loss'] and plane[k]['loss']<=plane[k+1]['loss']:
                    fitted=minimize_scalar(objective,bounds=(plane[k-1]['angle'],plane[k+1]['angle']),method='bounded',options=dict(xatol=1e-6))
                    candidates.append(dict(angle=float(fitted.x),loss=float(fitted.fun)))
            best=min(candidates,key=lambda q:q['loss'])
            minima.append(dict(i=i,j=j,angle=best['angle'],axis_error=min(best['angle'],90-best['angle']),loss=best['loss']))
        best=case['gaps'][stage].get('best');global_error=None if best is None else best['axis_error']
        strict_pass=bool(case['gaps'][stage]['pass_gate'] and max(q['axis_error'] for q in minima)<=6.+1e-7)
        results[stage]=dict(sampled_gate=case['gaps'][stage]['pass_gate'],refined_plane_gate=strict_pass,
            best_full_rotation_axis_error=global_error,minima=minima,max_refined_plane_error=max(q['axis_error'] for q in minima),
            note='The user tolerance concerns recovered full axes; plane conditions are additional conservative diagnostics, not proofs about every near-axis full rotation')
    out=ROOT/'results'/'plane_refinements'/args.label/args.model;out.mkdir(parents=True,exist_ok=True)
    dump(out/(args.metric+'_'+args.inference+f'_starts{args.starts}.json'),dict(args=vars(args),case_sha256=sha(path),results=results))
    print(json.dumps(dict(model=args.model,metric=args.metric,inference=args.inference,
        results={s:{k:v for k,v in r.items() if k not in ['minima','note']} for s,r in results.items()})))


if __name__=='__main__':main()
