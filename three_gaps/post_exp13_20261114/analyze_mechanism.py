"""Post-hoc mechanism tables and scientific figures; no parameter selection."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import core as C
from report import mdtable

def main():
    base=C.HERE/'exp14_probability/results/mechanism_v1'; rows=[]; components=[]
    for directory in sorted(base.iterdir()):
        if not (directory/'COMPLETE.json').exists(): continue
        df=pd.read_csv(directory/'metrics.csv'); df['source']=directory.name.split('_')[-1]
        rows.append(df)
        data=json.loads((directory/'decomposition.json').read_text(encoding='utf8'))
        for block,energy in data.items():
            sums={k:float(np.asarray(v).sum()) for k,v in energy.items()}
            components.append(dict(config=directory.name,block=block,**sums,
                                   B_fraction=sums['B']/max(sums['p'],1e-30),
                                   cross_fraction=sums['2EAB']/max(sums['p'],1e-30)))
    df=pd.concat(rows,ignore_index=True); energy=pd.DataFrame(components)
    out=C.HERE/'exp14_probability/results/mechanism_summary'; out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'all_candidates.csv',index=False); energy.to_csv(out/'energy_components.csv',index=False)
    terms=['eta__combo','A__combo','B__combo','p__combo','gg__combo','softplus','AUX_BUU','pUX_BUU']
    report=['# EXP14 机制分析（全量，探索性）','',
            '固定 34 候选；以下表格是预定分解/消融的事后描述，不据 TEST 选择最终算法。解析场为真实 T 坐标的 Gap1，decoder 场为原 NP V 坐标。',
            '', '## decoder 各组件 MCC','',
            mdtable(df[(df.source=='decoder')&df.method.isin(terms)].pivot(index='method',columns=['model','repeat'],values='mcc').reset_index().rename(columns=lambda x:str(x))),
            '', '## CR3：完整候选在两次重复的结果','',
            mdtable(df[(df.source=='decoder')&(df.model=='CR3')][['repeat','method','mcc','leakage','max_leakage']]),
            '', '## 能量分解','',
            'energy_components.csv 保存各坐标/块的绝对能量与 B/交叉项相对份额。2E[AB] 可为负，B_fraction 可超过 1，不能把交叉项当非负概率质量。仅能量占比不能确定最小化后的参数恢复。',
            '', '## 机制边界','',
            'CR3 两次重复中 B 单独均未超过完整 p 组合。η 组合/softplus 等少数方法的 CR3 平均可能较高，但必须同时检查 C1–C3、泄漏和独立数据；不能据单模型事后选择称为跨模型成功。']
    (out/'ANALYSIS.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    for rep in (1,2):
        c=C.load('CR3',rep,'prob'); directory=base/f'CR3_rep{rep}_decoder'
        saved=np.load(directory/'rotations.npz')
        methods={'Identity':np.eye(4),'EXP13 p':C.baseline_rotation(c)}
        for name in ('eta__combo','A__combo','B__combo','softplus'): methods[name]=saved[name+'__R'][0]
        methods['Oracle']=C.oracle(c)
        C.heatmap(c,methods,out/f'CR3_rep{rep}_components_UT.png')
        curves=pd.read_csv(directory/'plane_curves.csv'); pairs=list(itertools.combinations(range(4),2))
        fig,axes=plt.subplots(2,3,figsize=(11,6),squeeze=False)
        diag=json.loads((directory/'diagnostics.json').read_text(encoding='utf8'))
        for ax,(i,j) in zip(axes.flat,pairs):
            for name in ('eta__combo','A__combo','B__combo','p__combo'):
                sub=curves[(curves.i==i)&(curves.j==j)&(curves.method==name)]
                # Null SD normalization makes curve shapes comparable within method; show delta at Oracle.
                null=np.load(directory/'random_landscapes.npz')[name]
                y=(sub.loss.to_numpy()-diag[name]['oracle_loss'])/(null.std()+1e-12)
                ax.plot(sub.angle,y,label=name.replace('__combo',''))
            ax.set_title(f'axes {i+1},{j+1}'); ax.set_xlabel('angle (degrees)'); ax.set_ylabel('loss delta / Haar SD')
            ax.axhline(0,color='gray',linewidth=.5)
        axes[0,0].legend(fontsize=8); fig.suptitle(f'CR3 decoder: all planes through supervised reference; repeat {rep}')
        fig.tight_layout(); fig.savefig(out/f'CR3_rep{rep}_all_planes.png',dpi=150); plt.close(fig)
    print('Saved mechanism analysis',out)

if __name__=='__main__': main()
