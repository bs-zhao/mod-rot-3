"""Read-only G1/G2/G3 ledger; no optimization or TEST-based selection."""
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE/'results'
MODELS={'C1','C2','C3','CR3'}


def main():
    records=[];sets=defaultdict(list)
    for path in ROOT.glob('h*/**/summary.json'):
        j=json.loads(path.read_text(encoding='utf8'));a=j['args']
        if not a['metric'].startswith('hs_'):continue
        label=a['label'];stage=j.get('stage','G1' if a['source']=='formula' and a['estimator']=='direct' else 'G2' if a['source']=='formula' else 'G3')
        if a['source']=='decoder' and a['estimator']=='direct':stage='DECODER_DIRECT_DIAGNOSTIC'
        old_rule=label.startswith(('h01_','h02_'))
        unmotivated=label.startswith('h03_') and j['matrix_metrics_best']['specification'].get('weight',0)!=0
        # User later explicitly accepted the 5.796-degree result and asked
        # for a looser first pass. Re-adjudicate completed full-space runs
        # at six degrees without changing the original experiment JSON.
        g1_revised=bool(j.get('worst_axis_error_degrees',90)<=6.+1e-7 and
            max((p['axis_error'] for p in j.get('plane_minima',[])),default=90)<=6. and
            min(j['plane_ranges'])>1e-5*(1+abs(j['truth_loss'])) and
            any(s['start']>=0 for s in j['solutions']))
        passed=g1_revised if stage=='G1' else j.get('gap2_audit',{}).get('pass_gate') if stage=='G2' else None
        if stage=='G3':
            oracle_path=list((ROOT/'oracles').glob(f'{a["model"]}_*pca{a["pca"]}.json'))
            if len(oracle_path)==1:
                oracle=json.loads(oracle_path[0].read_text(encoding='utf8'))
                if oracle['hashes']!=j['hashes']:raise ValueError('Oracle and discovery source mismatch')
                ref=oracle['rotation_test'];ev=j['evaluation']
                passed=abs(ref['matched_mean']-ev['matched_mean'])<=.01 and ev['off_max']<=ref['off_max']+.02
        status=('SUPERSEDED_EXACT_AXIS_RULE' if old_rule else 'EXPLORATORY_UNMOTIVATED_COMBINATION' if unmotivated else f'PASSED_{stage}' if passed else f'FAILED_{stage}' if passed is False else 'NOT_ASSESSED')
        record=dict(round=label,model=a['model'],channel=a['channel'],structure_link=a.get('structure_link','native'),
            metric=a['metric'],stage=stage,estimator=a['estimator'],status=status,
            axis_error_degrees=j.get('worst_axis_error_degrees') if a['source']=='formula' else None,g1_exact=j.get('gap1_strict_pass'),
            g1_angle=g1_revised if stage=='G1' else j.get('gap1_angle_pass'),g1_original_angle_flag=j.get('gap1_angle_pass'),
            g2_curve_error=j.get('gap2_audit',{}).get('curve_relative_error'),
            g2_min_curve_correlation=j.get('gap2_audit',{}).get('minimum_plane_correlation'),
            hessian_error=(j.get('estimator_info') or {}).get('relative_errors',{}).get('h'),
            test_mean=(j.get('evaluation') or {}).get('matched_mean'),test_offmax=(j.get('evaluation') or {}).get('off_max'),
            seed=a['seed'],samples=a['samples'],grid_subjects=a.get('hessian_grid',0),grid_stimuli=a.get('grid_stimuli',0),
            uu_full_domain=a.get('uu_full_domain',False),combination_rationale=a.get('combination_rationale'),
            inference_signature=json.dumps({k:a.get(k) for k in ['estimator','degree','ridge','train_rows','train_subjects','neighbors','bandwidth','rbf_centers','rbf_length','utility_ridge']},sort_keys=True),
            result=path.relative_to(HERE).as_posix())
        records.append(record)
        if not old_rule and not unmotivated and stage in ['G1','G2','G3']:
            key=(a['metric'],a['channel'],a.get('structure_link','native'),a.get('uu_full_domain',False),a.get('hessian_grid',0),a.get('grid_stimuli',0),a['samples'],a['seed'])
            sets[key].append(record)
    summary=[]
    for key,group in sets.items():
        analytic=[r for r in group if r['stage']=='G1']
        signatures={r['inference_signature'] for r in group if r['stage']!='G1'} or {'not_run'}
        for signature in signatures:
            relevant=analytic+[r for r in group if r['stage']!='G1' and r['inference_signature']==signature]
            stages={s:[r for r in relevant if r['stage']==s] for s in ['G1','G2','G3']}
            state='IN_PROGRESS_G1'
            for stage in ['G1','G2','G3']:
                rows=stages[stage]
                if any(r['status']==f'FAILED_{stage}' for r in rows):state=f'STOPPED_{stage}';break
                if {r['model'] for r in rows if r['status']==f'PASSED_{stage}'}!=MODELS:
                    state=f'AWAITING_{stage}';break
                state='ALL_THREE_GATES_PASSED' if stage=='G3' else f'PASSED_{stage}'
            summary.append(dict(metric=key[0],channel=key[1],seed=key[-1],state=state,inference=signature,
                models_g1_passed=sorted({r['model'] for r in stages['G1'] if r['status']=='PASSED_G1'}),
                models_g2_passed=sorted({r['model'] for r in stages['G2'] if r['status']=='PASSED_G2'}),
                models_g3_passed=sorted({r['model'] for r in stages['G3'] if r['status']=='PASSED_G3'}),
                failures=[dict(model=r['model'],stage=r['stage'],result=r['result']) for r in relevant if r['status'].startswith('FAILED_')]))
    out=ROOT/'hessian_search';out.mkdir(exist_ok=True)
    if records:
        with (out/'runs.csv').open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (out/'gates.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    counts=Counter(r['status'] for r in records)
    lines=['# 统一 Hessian 指标：逐关研究台账','',
        '当前研究未完成。旧 transport 结果不在本任务范围。用户随后接受约5.8°，当前 G1 初筛按最差轴误差≤6°；5°原始诊断保留。组合必须先有单项互补证据。',
        '完整条目见 [runs.csv](runs.csv) 和 [gates.json](gates.json)。旧 exact-zero 预筛与无依据组合单列，不计入当前通过证据。','',
        'G2 固定门槛：所有平面 loss 曲线相对 L2 误差≤0.10、每平面相关≥0.90、最差轴误差≤5°；字段误差单独报告。',
        'G3 预定比较：平均匹配相关与监督参考差≤0.01，最大错配不超过对应参考+0.02；原 TEST 为探索性，最终必须另做冻结后的新被试确认。','',
        '## 状态计数','']
    lines.extend(f'- {k}: {v}' for k,v in sorted(counts.items()))
    lines+=['','## 当前候选','', '| 指标 | 通道 | seed | 状态 | G1 已通过模型 | G2 已通过模型 | G3 已通过模型 |','|---|---|---:|---|---|---|---|']
    for s in sorted(summary,key=lambda x:(x['metric'],x['channel'],x['seed'])):
        lines.append(f'| {s["metric"]} | {s["channel"]} | {s["seed"]} | {s["state"]} | {", ".join(s["models_g1_passed"])} | {", ".join(s["models_g2_passed"])} | {", ".join(s["models_g3_passed"])} |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(status_counts=counts,candidates=summary),ensure_ascii=False))


if __name__=='__main__':main()
