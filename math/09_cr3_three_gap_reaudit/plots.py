"""Plots for 09's read-only re-audit of existing math/07 metrics."""
from __future__ import annotations
import io
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLANES=('01_alpha_lambda','02_alpha_gamma','03_alpha_tau',
        '12_lambda_gamma','13_lambda_tau','23_gamma_tau')
SOURCES=('analytic','formula_y','nn_y')


def plot_top(source,out,ranked,limit,read_result):
    if limit<=0:return
    data=pd.read_csv(io.BytesIO(read_result(source,'all_local_metrics.csv')),
                     low_memory=False)
    folder=Path(out)/'plots'
    folder.mkdir(parents=True,exist_ok=True)
    for _,winner in ranked.loc[ranked.gap1_near==6].head(limit).iterrows():
        metric,target=winner.metric,winner.target
        fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
        for ax,plane in zip(axes.flat,PLANES):
            rows=data[(data.target==target)&(data.plane==plane)]
            ds={k:rows.loc[rows.source==k].sort_values('angle_degrees')
                for k in SOURCES}
            ref=ds['analytic'][metric].to_numpy(float)
            scale=max(float(np.ptp(ref)),1e-10)
            for k,style in zip(SOURCES,('-','--',':')):
                z=ds[k]
                ax.plot(z.angle_degrees.to_numpy(float),
                        (z[metric].to_numpy(float)-np.nanmin(ref))/scale,
                        style,label=k,linewidth=1.8)
            ax.set(xlabel='rotation angle (degrees)',
                   ylabel='loss / oracle range',title=plane)
            ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=9)
        fig.suptitle(target+' | '+metric)
        fig.savefig(folder/(target+'__'+metric+'.png'),dpi=140)
        plt.close(fig)
