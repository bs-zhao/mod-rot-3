"""Re-audit previous math/07 rotation metrics across three sequential gaps.

No refitting. Original math/07 criterion: within 10 degrees of 0/90,
relative range at least 2%, and Pearson >= 0.9 for formula/decoder.
Also reports strict 0/90 minima separately.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

PLANES=('01_alpha_lambda','02_alpha_gamma','03_alpha_tau',
        '12_lambda_gamma','13_lambda_tau','23_gamma_tau')
TARGETS=('logit','probability')


def read_result(source,name):
    p=Path(source)
    if p.is_dir():
        return (p/name).read_bytes()
    if p.suffix.lower()!='.zip':
        raise ValueError('Source must be math/07 output dir or zip')
    with zipfile.ZipFile(p) as z:
        found=[s for s in z.namelist() if s==name or s.endswith('/'+name)]
        if len(found)!=1:raise ValueError('Expected one '+name)
        return z.read(found[0])


def near(angles,degree):
    v=np.asarray(angles,float)
    return np.isfinite(v)&((np.abs(v)<=degree+1e-9)|(np.abs(v-90)<=degree+1e-9))


def audit(screen,tolerance=10.,min_range=.02,min_corr=.9):
    d=screen.loc[screen.target.isin(TARGETS)].copy()
    if set(d.plane)!=set(PLANES):raise ValueError('Missing a CR3 plane')
    if d.duplicated(['target','plane','metric']).any():
        raise ValueError('Duplicate metric/plane')
    for prefix,field in [('analytic','analytic_min_angle'),
                         ('formula','formula_min_angle'),
                         ('nn','nn_min_angle')]:
        variation=pd.to_numeric(d[prefix+'_relative_variation'],errors='coerce')
        d[prefix+'_near']=near(d[field],tolerance)&(variation>=min_range)
        d[prefix+'_strict']=near(d[field],0)&(variation>=min_range)
    cf=pd.to_numeric(d.analytic_formula_curve_pearson,errors='coerce')>=min_corr
    cn=pd.to_numeric(d.analytic_nn_curve_pearson,errors='coerce')>=min_corr
    for mode in ('near','strict'):
        d['gap1_'+mode]=d['analytic_'+mode]
        d['gap2_'+mode]=d['gap1_'+mode]&d['formula_'+mode]&cf
        d['gap3_'+mode]=d['gap2_'+mode]&d['nn_'+mode']&cn
    keys=['target','metric']
    fields=['gap'+str(i)+'_'+mode for mode in ('near','strict') for i in (1,2,3)]
    rank=d.groupby(keys,sort=False)[fields].sum().reset_index()
    n=d.groupby(keys).size().rename('n').reset_index()
    rank=rank.merge(n,on=keys)
    rank=rank.loc[rank.n==6].copy()
    idx=pd.MultiIndex.from_frame(rank[keys])
    for dest,col in [('formula_mean_r','analytic_formula_curve_pearson'),
                     ('nn_mean_r','analytic_nn_curve_pearson')]:
        rank[dest]=d.groupby(keys)[col].mean().reindex(idx).to_numpy()
    order=['gap3_near','gap2_near','gap1_near',
           'gap3_strict','gap2_strict','gap1_strict','nn_mean_r']
    rank=rank.sort_values(order,ascending=[False]*len(order))
    return d,rank


def run(args):
    out=Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError('Nonempty output directory; choose another --output or --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    saved=read_result(args.source,'strict_screen_all.csv')
    original=pd.read_csv(io.BytesIO(saved))
    detail,rank=audit(original,args.endpoint_tolerance,args.min_range,args.corr)
    detail.to_csv(out/'all_plane_gap_checks.csv',index=False)
    rank.to_csv(out/'all_metrics_ranked.csv',index=False)
    for mode in ('near','strict'):
        for i in (1,2,3):
            rank.loc[rank['gap'+str(i)+'_'+mode]==6].to_csv(
                out/('gap'+str(i)+'_'+mode+'_six_of_six.csv'),index=False)
    counts={}
    for target in TARGETS:
        r=rank.loc[rank.target==target]
        counts[target]={k:int((r[k]==6).sum()) for k in
                        ['gap1_near','gap2_near','gap3_near',
                         'gap1_strict','gap2_strict','gap3_strict']}
    old=json.loads(read_result(args.source,'summary.json'))
    manifest=dict(origin=str(args.source),counts=counts,candidates=len(rank),
                  endpoint_tolerance=args.endpoint_tolerance,
                  min_range=args.min_range,correlation_gate=args.corr,
                  A_star=old.get('A_star'),intercept_b=old.get('b'),
                  note='Existing 07 six-plane A*-anchored curves; not SO(4) proof. No WY estimator rerun.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    lines=['# CR3 Three-Gap Audit (existing math/07 curves)','',
           'Y source: analytic, numeric formula-Y and frozen decoder-Y, identical math/07 estimator.','',
           '| Target | G1 near 6/6 | G2 near 6/6 | G3 near 6/6 | G1 strict 6/6 |',
           '|---|---:|---:|---:|---:|']
    for name in TARGETS:
        a=counts[name]
        lines.append('| '+name+' | '+' | '.join(str(a[k]) for k in
             ('gap1_near','gap2_near','gap3_near','gap1_strict'))+' |')
    lines.extend(['','Near means min angle within '+str(args.endpoint_tolerance)+
       ' degrees of an endpoint; nonflat min relative range '+str(args.min_range)+
       '; later gaps require correlation >= '+str(args.corr)+'.',
       'A six-plane check is not global SO(4) recovery.', '',
       '## Ranked previously successful candidates', '', chr(96)*3,
       rank.loc[rank.gap1_near==6,[
           'target','metric','gap1_near','gap2_near','gap3_near','gap1_strict'
       ]].head(30).to_string(index=False),chr(96)*3])
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    if args.top_plots:
        from plots import plot_top
        plot_top(args.source,out,rank,args.top_plots,read_result)
    print('\n'.join(lines[:9]))
    print('Saved to',out.resolve())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--output',default='outputs/math/09_cr3_three_gap_reaudit')
    p.add_argument('--endpoint-tolerance',type=float,default=10.)
    p.add_argument('--min-range',type=float,default=.02)
    p.add_argument('--corr',type=float,default=.9)
    p.add_argument('--top-plots',type=int,default=6)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args(argv)
    if args.endpoint_tolerance<0 or args.min_range<0 or not -1<=args.corr<=1:
        p.error('Invalid thresholds')
    run(args)


if __name__=='__main__':
    main()
