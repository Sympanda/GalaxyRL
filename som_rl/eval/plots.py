# eval/plot.py
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .rollout import RunArtifacts


def bins_to_image(bins_1d: np.ndarray, H: int, W: int) -> np.ndarray:
    return np.asarray(bins_1d, dtype=np.int32).reshape(H, W)


def summary_dict(art: RunArtifacts) -> dict:
    df = art.log_df
    steps = int(len(df))
    noop_frac = float(df["is_noop"].mean()) if steps > 0 else 0.0
    pos_frac = float((df["delta_reward"] > 0).mean()) if steps > 0 else 0.0
    neg_frac = float((df["delta_reward"] < 0).mean()) if steps > 0 else 0.0
    return {
        "steps": steps,
        "trace_start": art.trace_start,
        "trace_end": art.trace_end,
        "trace_best": art.trace_best,
        "best_step": art.best_step,
        "noop_frac": noop_frac,
        "pos_delta_frac": pos_frac,
        "neg_delta_frac": neg_frac,
        "wall_time_s": art.wall_time_s,
        "steps_per_sec": steps / max(1e-12, art.wall_time_s),
    }


def plot_trace(art: RunArtifacts, *, show_best: bool = True, ax=None):
    df = art.log_df
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(df["t"], df["reward_trace"], label="trace")
    ax.axhline(art.trace_start, linestyle="--", linewidth=1, label="start")
    if show_best:
        ax.axhline(art.trace_best, linestyle="--", linewidth=1, label="best")
        ax.axvline(art.best_step, linestyle=":", linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("reward_trace")
    ax.set_title("Trace vs step")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_delta_hist(art: RunArtifacts, bins: int = 50, ax=None):
    df = art.log_df
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.hist(df["delta_reward"].to_numpy(), bins=bins)
    ax.set_xlabel("delta_reward")
    ax.set_ylabel("count")
    ax.set_title("Delta reward histogram")
    ax.grid(True, alpha=0.3)
    return ax


def plot_noop_rate_ma(art: RunArtifacts, window: int = 100, ax=None):
    df = art.log_df
    arr = df["is_noop"].astype(np.float64).to_numpy()
    if arr.size == 0:
        return ax
    window = min(window, arr.size)
    kernel = np.ones(window) / window
    ma = np.convolve(arr, kernel, mode="valid")
    xs = np.arange(window, window + ma.size)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(xs, ma)
    ax.set_xlabel("step")
    ax.set_ylabel(f"noop rate (MA{window})")
    ax.set_title("No-op rate over time")
    ax.grid(True, alpha=0.3)
    return ax


def plot_heatmaps(art: RunArtifacts, *, axarr=None):
    if axarr is None:
        fig, axarr = plt.subplots(1, 3, figsize=(15, 4))
    axarr[0].imshow(art.visit_count)
    axarr[0].set_title("Visit count")
    axarr[1].imshow(art.noop_count)
    axarr[1].set_title("No-op count")
    axarr[2].imshow(art.pos_delta_sum)
    axarr[2].set_title("Sum positive deltas")
    for ax in axarr:
        ax.set_xticks([]); ax.set_yticks([])
    return axarr


def plot_bin_snapshots(
    art: RunArtifacts,
    H: int,
    W: int,
    *,
    vmin: int = 0,
    vmax: int = 10,
    axarr=None
):
    if axarr is None:
        fig, axarr = plt.subplots(1, 3, figsize=(15, 4))
    axarr[0].imshow(bins_to_image(art.bins_start, H, W), vmin=vmin, vmax=vmax)
    axarr[0].set_title("Start")
    axarr[1].imshow(bins_to_image(art.bins_end, H, W), vmin=vmin, vmax=vmax)
    axarr[1].set_title("End")
    axarr[2].imshow(bins_to_image(art.bins_best, H, W), vmin=vmin, vmax=vmax)
    axarr[2].set_title(f"Best (step {art.best_step})")
    for ax in axarr:
        ax.set_xticks([]); ax.set_yticks([])
    return axarr



def save_all_plots(outdir: str, env, art: RunArtifacts) -> None:
    os.makedirs(outdir, exist_ok=True)

    # trace
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    plot_trace(art, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "trace.png"), dpi=150)
    plt.close(fig)

    # delta hist
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    plot_delta_hist(art, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "delta_hist.png"), dpi=150)
    plt.close(fig)

    # noop rate
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    plot_noop_rate_ma(art, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "noop_rate.png"), dpi=150)
    plt.close(fig)

    # heatmaps
    fig, axarr = plt.subplots(1, 3, figsize=(15, 4))
    plot_heatmaps(art, axarr=axarr)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "heatmaps.png"), dpi=150)
    plt.close(fig)

    # snapshots
    H, W = env.H, env.W
    fig, axarr = plt.subplots(1, 3, figsize=(15, 4))
    plot_bin_snapshots(art, H, W, axarr=axarr)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "bin_snapshots.png"), dpi=150)
    plt.close(fig)
