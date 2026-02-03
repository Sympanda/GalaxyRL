# eval/rollout.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


@dataclass
class RunArtifacts:
    log_df: pd.DataFrame

    bins_start: np.ndarray  # (N,)
    bins_end: np.ndarray    # (N,)
    bins_best: np.ndarray   # (N,)

    trace_start: float
    trace_end: float
    trace_best: float
    best_step: int

    visit_count: np.ndarray    # (H,W)
    noop_count: np.ndarray     # (H,W)
    pos_delta_sum: np.ndarray  # (H,W)

    seed: Optional[int]
    max_steps: int
    wall_time_s: float


def _get_trace(som) -> float:
    # Prefer explicit method if you add one; otherwise fallback to reward_trace
    if hasattr(som, "get_reward"):
        return float(som.get_reward())
    if hasattr(som, "reward_trace"):
        return float(som.reward_trace())
    raise AttributeError("Backend must expose get_reward() or reward_trace().")


def run_episode(
    env,
    policy: Callable[[Dict[str, Any], np.random.Generator], Any],
    max_steps: int,
    seed: Optional[int] = None,
) -> RunArtifacts:
    """
    Runs a single rollout and returns artifacts for plotting + saving.

    Assumptions about env/info:
      - env.reset(seed=...) -> (obs, info)
      - env.step(action) -> (obs, reward, terminated, truncated, info)
      - env has H, W
      - env.som has current_bins (N,), index_to_xy(node)->(x,y)
      - info contains:
          - node (int) when acted
          - reward (float) = trace after step (recommended)
          - delta_reward (float) (recommended)
          - is_noop (bool) (recommended; chosen_bin == old_bin)
          - old_bin, chosen_bin (optional)
    """
    t0 = time.time()
    obs, info0 = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    H, W = int(env.H), int(env.W)

    bins_start = env.som.current_bins.copy()
    trace_start = _get_trace(env.som)

    best_trace = trace_start
    best_bins = bins_start.copy()
    best_step = 0

    visit = np.zeros((H, W), dtype=np.int32)
    noop = np.zeros((H, W), dtype=np.int32)
    posd = np.zeros((H, W), dtype=np.float64)

    rows: List[Dict[str, Any]] = []

    for t in range(1, max_steps + 1):
        action = policy(obs, rng)
        obs, reward_env, terminated, truncated, info = env.step(action)

        node = int(info.get("node", -1))
        x = y = None
        if node >= 0:
            x, y = env.som.index_to_xy(node)
            visit[y, x] += 1

            if bool(info.get("is_noop", False)):
                noop[y, x] += 1

            d = float(info.get("delta_reward", 0.0))
            if d > 0:
                posd[y, x] += d

        trace = float(info.get("reward", _get_trace(env.som)))

        if trace > best_trace:
            best_trace = trace
            best_bins = env.som.current_bins.copy()
            best_step = t

        rows.append({
            "t": t,
            "action": action,
            "reward_env": float(reward_env),
            "reward_trace": float(trace),
            "delta_reward": float(info.get("delta_reward", 0.0)),
            "node": node,
            "x": x,
            "y": y,
            "old_bin": info.get("old_bin", None),
            "chosen_bin": info.get("chosen_bin", None),
            "is_noop": bool(info.get("is_noop", False)),
            "acted": bool(info.get("acted", True)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })

        if terminated or truncated:
            break

    bins_end = env.som.current_bins.copy()
    trace_end = _get_trace(env.som)

    wall = time.time() - t0

    return RunArtifacts(
        log_df=pd.DataFrame(rows),
        bins_start=bins_start,
        bins_end=bins_end,
        bins_best=best_bins,
        trace_start=float(trace_start),
        trace_end=float(trace_end),
        trace_best=float(best_trace),
        best_step=int(best_step),
        visit_count=visit,
        noop_count=noop,
        pos_delta_sum=posd,
        seed=seed,
        max_steps=int(max_steps),
        wall_time_s=float(wall),
    )


def save_run(outdir: str, env, art: RunArtifacts) -> None:
    os.makedirs(outdir, exist_ok=True)

    art.log_df.to_csv(os.path.join(outdir, "run.csv"), index=False)

    np.save(os.path.join(outdir, "bins_start.npy"), art.bins_start)
    np.save(os.path.join(outdir, "bins_end.npy"), art.bins_end)
    np.save(os.path.join(outdir, "bins_best.npy"), art.bins_best)

    np.save(os.path.join(outdir, "visit_count.npy"), art.visit_count)
    np.save(os.path.join(outdir, "noop_count.npy"), art.noop_count)
    np.save(os.path.join(outdir, "pos_delta_sum.npy"), art.pos_delta_sum)

    meta = {
        "seed": art.seed,
        "max_steps": art.max_steps,
        "wall_time_s": art.wall_time_s,
        "steps_taken": int(len(art.log_df)),
        "trace_start": art.trace_start,
        "trace_end": art.trace_end,
        "trace_best": art.trace_best,
        "best_step": art.best_step,
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
