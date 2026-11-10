"""Read-only checkpoint across successive three-gap batches."""
import csv
import json
from pathlib import Path
import subprocess
import sys
from batch_report import signature
from core import dump

HERE=Path(__file__).resolve().parent;ROOT=HERE/'results'


def main():
    folders=sorted(ROOT.glob('b[0-9][0-9]*batch'))
    summaries=[];all_rows=[];definitions=set()
    for folder in folders:
        cases=list(folder.glob('*/**/hs_*.json'))
        if not any(not p.stem.endswith('_rotation') for p in cases):continue
        completed=subprocess.run([sys.executable,str(HERE/'batch_report.py'),'--label',folder.name],capture_output=True,text=True,check=True)
        index=json.loads((folder/'report'/'candidates.json').read_text(encoding='utf8'))
        with (folder/'report'/'cases.csv').open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
        all_rows+=rows
        for path in cases:
            if path.stem.endswith('_rotation'):continue
            j=json.loads(path.read_text(encoding='utf8'))
            if 'specification' in j:definitions.add(signature(j['specification']))
        for inference in sorted(set(r['inference'] for r in rows)):
            current=[c for c in index['candidates'] if c['inference']==inference and c['starts']==8 and c['complete'] and c['metric'] not in index['exact_aliases']]
            summaries.append(dict(batch=folder.name,inference=inference,cases=sum(r['inference']==inference for r in rows),complete_common_candidates=len(current),
                all4_g1=sum(c['gates']['g1'] for c in current),all4_g1_g2=sum(c['gates']['g1'] and c['gates']['g2'] for c in current),
                all4_all3=sum(c['state']=='ALL_GATES_PROVISIONAL_CONFIRM64' for c in current)))
    batch_count=len(set(s['batch'] for s in summaries))
    out=ROOT/'batch_search';out.mkdir(exist_ok=True)
    with (out/'batches.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    dump(out/'checkpoint.json',dict(batches=summaries,cases=len(all_rows),gate_measurements=3*len(all_rows),
        unique_definitions=len(definitions),definition_rule='Exact mathematical aliases merged; statistics/threshold/normalization variants remain explicit definitions, not distinct scientific ideas'))
    lines=['# 连续批量搜索：当前检查点','',
        f'已执行 {batch_count} 批，累计完成 {len(all_rows)} 个“单指标 × 模型 × inference × 优化设置”案例，'
        f'对应 {3*len(all_rows)} 次 gap 测量。跨批去除精确别名后有 {len(definitions)} 个定义；'
        '阈值、汇总方式及尺度变体均明确列出，不把它们当作同样数量的独立思想。','',
        '研究仍未完成。每个候选在四模型用同一公式和推断规则；不同 inference 分支不能拼接。',
        '所有案例同批测 G1/G2/G3，上游失败时下游结果仍保留作诊断。','',
        '| 批次 / 推断 | 完成案例 | 四模型齐全的去重候选 | 四模型 G1 | 四模型 G1+G2 | 四模型全部通过 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        label=s['batch'];lines.append(f'| [{label} / {s["inference"]}](../{label}/report/REPORT.md) | {s["cases"]} | {s["complete_common_candidates"]} | {s["all4_g1"]} | {s["all4_g1_g2"]} | {s["all4_all3"]} |')
    lines+=['','表中四模型计数采用各行明确标注的同一推断分支、8 起点初筛；另行复核也计入完成案例。'
        'G1 平面反例和不可辨识方向均如实列出；初筛成功仍需更多起点及新样本确认。','',
        '## 已确认的推进与失败','',
        '- 无向 UU 边与 UX 边的单一 RMS 软计数（阈值 .02）在四模型的 64 起点复核中通过 G1、G2，'
        '但 G3 失败，已单独保存。不能把这个结论说成“没有通用指标”，也不能称作完整解决。',
        '- 纯 UU 非线性单项在 CR3 decoder 可达到约 .824 的平均相关，但 C1/C2/C3 的解析 logit UU 严格为零，'
        '故它单独不是通用解；还存在明显错配。',
        '- 已比较 Gaussian、ANOVA、五阶 latent 多项式、latent Gaussian 等通用 WY 推断分支。'
        '当前 CR3 decoder 的 UX 导数相对误差仍接近 1，需要与目标函数偏差分开。',
        '- 事后参考方向 loss 诊断只读已经锁定的旋转，绝不把监督 R 用作起点或供给无监督目标。',
        '- 精确网络 Hessian 另记为诊断对照，不能冒充仅凭 WY 推断成功。','',
        '![一个统一单指标的三个 Gap](confirmed_single_gaps.png)','',
        '## 可复现性','',
        '全部代码和结果均在 `research/three_gap_series`。先推送代码，再拉取到本机 py11 执行。'
        '保留模型/数据哈希、固定数值场来源、每个候选的 R 锁定文件、完整平面曲线与多起点结果。',
        '原始 NP、数据、Z 与其他实验目录未修改。TEST 已用于探索，最终候选必须冻结后再用独立新样本验证。','',
        '[指标定义与文献依据](../../BATCH_SEARCH.md) · [批次计数](batches.csv)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(batches=batch_count,cases=len(all_rows),definitions=len(definitions))))


if __name__=='__main__':main()
