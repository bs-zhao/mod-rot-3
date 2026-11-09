"""Cross-model, single-only atlas report. A screen survivor is never a G1 pass."""
import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
from core import dump

ROOT=Path(__file__).resolve().parent
MODELS=['C1','C2','C3','CR3']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label',default='s01_single_metric_atlas');args=p.parse_args()
    folder=ROOT/'results'/args.label;atlases={}
    for model in MODELS:
        path=folder/model/'atlas.json'
        if path.exists():atlases[model]=json.loads(path.read_text(encoding='utf8'))
    if not atlases:raise ValueError('No completed atlas')
    first=next(iter(atlases.values()));names=first['catalogue']
    for atlas in atlases.values():
        if atlas['catalogue']!=names:raise ValueError('Catalogue differs across models')
        if atlas['channel']!=first['channel'] or atlas['seed']!=first['seed']:raise ValueError('Different channel/seed')
    lookup={m:{r['metric']:r for r in a['results']} for m,a in atlases.items()}
    rows=[];survivors=[];families=defaultdict(Counter)
    for name in names:
        sample=next(lookup[m][name] for m in atlases);spec=sample['specification']
        row=dict(metric=name,family=spec['family'],specification=json.dumps(spec,sort_keys=True))
        alive=True
        for model in MODELS:
            r=lookup.get(model,{}).get(name)
            row[model]=r['state'] if r else 'PENDING'
            row[model+'_bad_planes']=sum(p['counterexample'] for p in r['planes']) if r else ''
            row[model+'_flat_planes']=sum(p['flat'] for p in r['planes']) if r else ''
            row[model+'_max_plane_error']=max(p['axis_error'] for p in r['planes']) if r else ''
            if r:families[spec['family']][model+':'+r['state']]+=1
            alive&=bool(r and r['state']=='NEEDS_JOINT_OPTIMIZATION')
        if alive and not spec.get('control',False):survivors.append(name)
        rows.append(row)
    with (folder/'single_metrics.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(folder/'joint_candidates.json',dict(channel=first['channel'],seed=first['seed'],
        complete=all(m in atlases for m in MODELS),metrics=survivors,
        rule='Same single objective survives all four analytic plane screens; controls excluded. No G1 success claimed.'))
    lines=['# Single Hessian metric atlas','',
        f'Channel: {first["channel"]}; seed: {first["seed"]}; {len(names)} canonical single definitions; '
        f'{len(first["aliases"])} exact aliases omitted. Completed models: {", ".join(atlases)}.','',
        '**Current policy: single metrics only. Combinations and weight tuning are paused.**','',
        'These are falsification screens, not completed Gap 1 tests. Plane survivors require full rotation optimization. '
        'Flat means at least one unresolved plane at the fixed numerical sensitivity. A random rotation beating identity '
        'is recorded separately and does not by itself reject the allowed neighborhood.','',
        '| Model | Plane counterexample | Flat unresolved plane | Needs joint optimization |',
        '|---|---:|---:|---:|']
    for model in MODELS:
        if model not in atlases:continue
        counts=Counter(r['state'] for r in atlases[model]['results'])
        lines.append(f'| {model} | {counts["PLANE_COUNTEREXAMPLE"]} | {counts["FLAT_PLANE"]} | {counts["NEEDS_JOINT_OPTIMIZATION"]} |')
    lines+=['','## All-four survivors requiring joint optimization','']
    lines+=['- `'+name+'`' for name in survivors] if survivors else ['No all-four survivor in the completed atlas.']
    lines+=['','## Family breakdown','',
        'Each cell is counterexample / flat / needs optimization; counts include predeclared statistic and threshold variants.',
        '', '| Family | C1 | C2 | C3 | CR3 |','|---|---:|---:|---:|---:|']
    for family,counts in sorted(families.items()):
        cells=[' / '.join(str(counts[m+':'+s]) for s in ['PLANE_COUNTEREXAMPLE','FLAT_PLANE','NEEDS_JOINT_OPTIMIZATION']) for m in MODELS]
        lines.append('| '+family+' | '+' | '.join(cells)+' |')
    lines+=['','## Interpretation limits','',
        '- For C1/C2/C3 logit, pure UU and UUU fields are analytically zero. Flatness here is structural, not a WY estimator failure.',
        '- Signed means can cancel; local absolute values, RMS, SD and quantiles answer different questions. Compare their plane curves, not raw loss scales.',
        '- Top-k row and column tails measure different interaction graphs. Every dimension uses the same k.',
        '- The full family applies one penalty to [H_UU|H_UX] at common differentiable W points. It has no separately tuned component weights.',
        '- Pure total-energy controls are retained to expose invariance or coordinate standardization effects; they are not automatically promoted.',
        '- No decoder TEST result is used for this single-metric selection.','',
        'Full metric-by-model cells: [single_metrics.csv](single_metrics.csv). Per-model plane curves and random counter-directions accompany each atlas.']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines[:19]));print('joint candidates:',len(survivors))


if __name__=='__main__':main()
