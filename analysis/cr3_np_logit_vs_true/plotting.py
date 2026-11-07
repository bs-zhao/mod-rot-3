"""Visualization for full-trial CR3 NP decoder versus simulator logits and p."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sample(a, b, cap, seed):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if len(a) <= cap:
        return a, b
    indices = np.random.default_rng(seed).choice(len(a), cap, replace=False)
    return a[indices], b[indices]


def comparison_scatter(true, predicted, label, path, cap=150_000, seed=31):
    a, b = _sample(true, predicted, cap, seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    hb = axes[0].hexbin(a, b, gridsize=110, mincnt=1, bins="log", cmap="viridis")
    fig.colorbar(hb, ax=axes[0], label="log10(number of observations)")
    low = min(np.percentile(a, .2), np.percentile(b, .2))
    high = max(np.percentile(a, 99.8), np.percentile(b, 99.8))
    axes[0].plot([low, high], [low, high], "--", c="red", lw=1, label="identity")
    axes[0].set(xlabel=f"True {label}", ylabel=f"NP {label}",
                title=f"All subjects × trials (sampled {len(a):,})")
    axes[0].legend()
    axes[1].hexbin(a, b-a, gridsize=110, mincnt=1, bins="log", cmap="viridis")
    axes[1].axhline(0, c="red", ls="--", lw=1)
    axes[1].set(xlabel=f"True {label}", ylabel="NP minus true",
                title="Residual versus true")
    fig.savefig(path, dpi=165)
    plt.close(fig)


def per_axis_hist(per_subject, per_trial, label, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for j, (items, title) in enumerate((
        (per_subject, "Per-subject (across 500 trials)"),
        (per_trial, "Per-trial (across 5,000 subjects)"),
    )):
        correlations = np.asarray(items["pearson_r"])
        correlations = correlations[np.isfinite(correlations)]
        if len(correlations):
            axes[j, 0].hist(correlations, bins=np.linspace(-1, 1, 51))
        axes[j, 0].set(xlim=(-1, 1), xlabel="Pearson r", ylabel="Count",
                       title=f"{title}: {label} correlation")
        axes[j, 1].hist(np.asarray(items["rmse"]), bins=45)
        axes[j, 1].set(xlabel="RMSE", ylabel="Count",
                       title=f"{title}: {label} RMSE")
    fig.savefig(path, dpi=165)
    plt.close(fig)


def calibration_plots(logit_bins, probability_bins, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax = axes[0]
    ax.plot([x["true_mean"] for x in logit_bins],
            [x["pred_mean"] for x in logit_bins], "o-", label="Mean NP logit")
    lo = min(x["true_min"] for x in logit_bins)
    hi = max(x["true_max"] for x in logit_bins)
    ax.plot([lo, hi], [lo, hi], "--", color="gray", label="identity")
    ax.set(xlabel="True logit (bin mean)", ylabel="NP logit (bin mean)",
           title="Logit curve, bins by true logit")
    ax.legend()
    ax = axes[1]
    ax.plot([x["pred_mean"] for x in probability_bins],
            [x["true_mean"] for x in probability_bins], "o-",
            label="Mean true p")
    ax.plot([x["pred_mean"] for x in probability_bins],
            [x["choice_mean"] for x in probability_bins], "x-",
            alpha=.65, label="Mean observed binary choice")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="identity")
    ax.set(xlim=(0,1), ylim=(0,1), xlabel="NP predicted p",
           ylabel="Mean true p / observed choice",
           title="Calibration, bins by predicted p")
    ax.legend()
    fig.savefig(path, dpi=165)
    plt.close(fig)


def split_summary_plots(metrics, path):
    labels = ["train", "validation", "test"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for j, channel in enumerate(("logit", "probability")):
        rs = [metrics[channel][split]["within_trial_centered"]["pearson_r"]
              for split in labels]
        mse = [metrics[channel][split]["pooled"]["rmse"] for split in labels]
        for k, (val, error) in enumerate(zip(rs, mse)):
            ax[j].bar(k, 0 if val is None else val, alpha=.8)
            ax[j].text(k, 0 if val is None else val, f"r={val:.3f}" if val is not None
                       else "r=undefined", ha="center", va="bottom", fontsize=9)
        ax[j].set(xticks=np.arange(3), xticklabels=labels,
                  title=f"{channel}: across-subject within-trial r",
                  ylabel="Pearson r", ylim=(-.1, 1.15))
    fig.savefig(path, dpi=165)
    plt.close(fig)
