"""Summarize every batch stage, retaining failed upstream gates explicitly."""
import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
from core import dump

ROOT=Path(__file__).resolve().parent/'results'
MODELS=['C1','C2','C3','CR3']


def signature(spec):
    spec=dict(spec)
    if spec.get('latent_smoothing')==0:
        spec.pop('latent_smoothing');spec['field_scale']='common'
    if spec['family']=='ux':spec.setdefault('normalization','reference')
    if spec.get('score')=='soft@0.03':spec['score']='soft003'
    if spec['family'] in ['ux','full'] and (spec.get('score') in ['l1','lp05','soft003','soft01'] or spec.get('score','').startswith('soft@')):spec.pop('axis',None)
    if spec.get('score') in ['l1','lp05','soft003'] and spec['family']=='functional':spec.pop('axis',None)
    return json.dumps(spec,sort_keys=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label',default='b01_three_gap_batch');args=p.parse_args()
    folder=ROOT/args.label;records=[];groups=defaultdict(dict);canonical={};aliases={}
    for path in sorted(folder.glob('*/**/hs_*.json')):
        if path.stem.endswith('_rotation'):continue
        j=json.loads(path.read_text(encoding='utf8'))
        if 'gaps' not in j:continue
        refinement=ROOT/'plane_refinements'/args.label/j['model']/(j['metric']+'_'+j['inference']+f'_starts{j["starts"]}.json')
        verified=json.loads(refinement.read_text(encoding='utf8'))['results'] if refinement.exists() else {}
        key=signature(j['specification'])
        if key not in canonical:canonical[key]=j['metric']
        elif canonical[key]!=j['metric']:aliases[j['metric']]=canonical[key]
        row=dict(metric=j['metric'],model=j['model'],inference=j['inference'],starts=j['starts'],
            is_control=j['specification'].get('control',False),
            family=j['specification']['family'],g1=verified.get('G1',{}).get('refined_plane_gate',j['gaps']['G1']['pass_gate']),
            g2=verified.get('G2',{}).get('refined_plane_gate',j['gaps']['G2']['pass_gate']),g3=j['gaps']['G3']['pass_gate'],
            g1_sampled=j['gaps']['G1']['pass_gate'],g2_sampled=j['gaps']['G2']['pass_gate'],continuous_plane_check=bool(verified),
            g1_mode=j['gaps']['G1']['mode'],g1_angle=(j['gaps']['G1'].get('best') or {}).get('axis_error'),
            g2_mode=j['gaps']['G2']['mode'],g2_angle=(j['gaps']['G2'].get('best') or {}).get('axis_error'),
            g2_curve_error=j['gaps']['G2']['curve_relative_error'],g2_min_correlation=j['gaps']['G2']['minimum_plane_correlation'],
            test_mean=j['gaps']['G3']['evaluation']['matched_mean'],test_min=j['gaps']['G3']['evaluation']['matched_min'],
            test_offmax=j['gaps']['G3']['evaluation']['off_max'],test_mean_gap=j['gaps']['G3']['mean_gap'],
            oracle_offmax=j['gaps']['G3']['oracle']['off_max'],source=str(path.relative_to(folder)))
        records.append(row);groups[(j['metric'],j['inference'],j['starts'])][j['model']]=row
    if not records:raise ValueError('No completed metric cases yet')
    out=folder/'report';out.mkdir(exist_ok=True)
    with (out/'cases.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    candidates=[]
    for (metric,inference,starts),models in sorted(groups.items()):
        complete=all(m in models for m in MODELS)
        gates={stage:complete and all(models[m][stage] for m in MODELS) for stage in ['g1','g2','g3']}
        first='PENDING' if not complete else 'FAILED_G1' if not gates['g1'] else 'FAILED_G2' if not gates['g2'] else 'FAILED_G3' if not gates['g3'] else 'ALL_GATES_PROVISIONAL_CONFIRM64'
        if any(r['is_control'] for r in models.values()):first='ENERGY_CONTROL'
        if inference=='nn_direct_diagnostic':first='NN_DIRECT_DIAGNOSTIC'
        candidates.append(dict(metric=metric,inference=inference,starts=starts,state=first,complete=complete,
            gates=gates,models=models,max_mean_gap=max(r['test_mean_gap'] for r in models.values()),
            max_excess_offmax=max(r['test_offmax']-r['oracle_offmax'] for r in models.values())))
    dump(out/'candidates.json',dict(candidates=candidates,exact_aliases=aliases))
    lines=['# 批量单指标的三鸿沟结果','',
        f'已完成 {len(records)} 个“指标 × 模型 × 推断 × 优化设置”案例；候选 ID 中的精确别名另列，不能作为独立发现计数。','',
        '所有下游测量均保留；上游失败时，下游结果只作诊断。每次更换 inference 都以独立分支记录，禁止拼接四模型结果。',
        '初筛多起点成功需要 64 起点复核和新样本确认，才能声明最终成功。','',
        '| 推断 / 起点 | 案例数 | G1 通过 | G2 通过 | G3 数值达标 |',
        '|---|---:|---:|---:|---:|']
    counts=defaultdict(Counter)
    for r in records:
        key=(r['inference'],r['starts']);counts[key]['n']+=1
        for stage in ['g1','g2','g3']:counts[key][stage]+=r[stage]
    for (infer,starts),c in sorted(counts.items()):lines.append(f'| {infer} / {starts} | {c["n"]} | {c["g1"]} | {c["g2"]} | {c["g3"]} |')
    lines+=['','## 同一单指标在四模型的结果','',
        '| 指标 | 推断 | 起点 | 四模型 G1 | 四模型 G2 | 四模型 G3 | 最大均值差 | 状态 |',
        '|---|---|---:|---|---|---|---:|---|']
    for c in sorted(candidates,key=lambda c:(not c['complete'],not c['gates']['g1'],not c['gates']['g2'],c['max_mean_gap'])):
        if c['metric'] in aliases:continue
        cells=['是' if c['gates'][s] else '否' if c['complete'] else '未齐' for s in ['g1','g2','g3']]
        lines.append(f'| {c["metric"]} | {c["inference"]} | {c["starts"]} | '+ ' | '.join(cells)+f' | {c["max_mean_gap"]:.5f} | {c["state"]} |')
    lines+=['','## 证据与限制','',
        '- G1/G2 的 PLANE_COUNTEREXAMPLE 表示已有平面反例，没有将该角度冒充全空间最优误差。FLAT_UNRESOLVED 表示不可辨识方向。',
        '- G2 还检查相同 W 上的 loss 曲线误差与最低平面相关；预测拟合好不等于指标推断好。',
        '- 若已有连续平面复核，报告采用更严格的复核标记；原始采样标记仍保留在 cases.csv，避免边界网格造成假通过。',
        '- G3 每个旋转先写入独立锁定文件，随后才读取真实参数作 TEST 评价。',
        '- G3 的通过要求相对各自监督旋转参考的均值差≤.01、最大错配≤参考+.02，四模型一致。',
        '- TEST 已用于探索性比较，最终候选仍需冻结方案后用独立新样本确认。','',
        f'精确别名 {len(aliases)} 个，见 candidates.json。全部模型级数据见 [cases.csv](cases.csv)。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(cases=len(records),states=dict(Counter(c['state'] for c in candidates)),aliases=len(aliases)),ensure_ascii=False))


if __name__=='__main__':main()
