from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SOMStateConfig:
    grid_shape: Tuple[int, int]
    num_bins: int
    eps: float = 1e-12


class SOMState:
    """
    SOM backend state + cached bin-aggregates for fast reward.

    State:
      - current_bins: (N_nodes,) int in [0..K-1]

    Static (shared/read-only):
      - dd, dr, rd, rr: (N_nodes, N_nodes) float correlation-ish matrices
      - node_gal_counts: (N_nodes,) float or int counts used in n_phot
      - valid_bins_mask: optional (N_nodes, K) bool for action masking
      - umatrix: optional (H,W) float for observation

    Cached aggregates (per state):
      - DD_sum, DR_sum, RD_sum, RR_sum: (K,K)
      - n_phot: (K,) derived from node_gal_counts and current_bins

    Reward:
      - total_reward_fast(): trace(diag(xi_bins)) where xi_bins computed from sums+counts
        (matches your current objective shape)
    """

    def __init__(
        self,
        config: SOMStateConfig,
        dd: np.ndarray,
        dr: np.ndarray,
        rd: np.ndarray,
        rr: np.ndarray,
        node_gal_counts: np.ndarray,
        initial_bins: np.ndarray,
        valid_bins_mask: Optional[np.ndarray] = None,
        umatrix: Optional[np.ndarray] = None,
    ):
        self.cfg = config
        self.dd = dd
        self.dr = dr
        self.rd = rd
        self.rr = rr

        self.node_gal_counts = node_gal_counts.astype(np.float64, copy=False)

        self.H, self.W = config.grid_shape
        self.N = self.H * self.W
        self.K = int(config.num_bins)

        initial_bins = np.asarray(initial_bins, dtype=np.int32)
        if initial_bins.shape != (self.N,):
            raise ValueError(f"initial_bins must be shape {(self.N,)}, got {initial_bins.shape}")

        self.current_bins = initial_bins.copy()
        self.initial_bins = initial_bins.copy()

        if valid_bins_mask is not None:
            valid_bins_mask = np.asarray(valid_bins_mask, dtype=bool)
            if valid_bins_mask.shape != (self.N, self.K):
                raise ValueError(f"valid_bins_mask must be {(self.N,self.K)}, got {valid_bins_mask.shape}")
        self.valid_bins_mask = valid_bins_mask

        if umatrix is not None:
            umatrix = np.asarray(umatrix, dtype=np.float32)
            if umatrix.shape != (self.H, self.W):
                raise ValueError(f"umatrix must be {(self.H,self.W)}, got {umatrix.shape}")
        self.umatrix = umatrix

        # caches
        self._reset_caches()
        self._last_reward: Optional[float] = None

    # ----------------- indexing helpers -----------------

    def xy_to_index(self, x: int, y: int) -> int:
        return int(x) * self.W + int(y)

    def index_to_xy(self, idx: int) -> Tuple[int, int]:
        x = int(idx) // self.W
        y = int(idx) % self.W
        return x, y

    # ----------------- caches + reward -----------------

    def _reset_caches(self) -> None:
        # bin counts
        self.n_phot = np.bincount(
            self.current_bins,
            weights=self.node_gal_counts,
            minlength=self.K,
        ).astype(np.float64)

        # KxK sums over node-node matrices grouped by bins.
        self.DD_sum = np.zeros((self.K, self.K), dtype=np.float64)
        self.DR_sum = np.zeros((self.K, self.K), dtype=np.float64)
        self.RD_sum = np.zeros((self.K, self.K), dtype=np.float64)
        self.RR_sum = np.zeros((self.K, self.K), dtype=np.float64)

        # Build sums from scratch (O(N^2)): MVP correctness first.
        # You can later replace with your existing pre-aggregation if you already have it.
        bins = self.current_bins
        for i in range(self.N):
            bi = bins[i]
            # vectorised over j
            bj = bins
            self.DD_sum[bi, :] += np.bincount(bj, weights=self.dd[i], minlength=self.K)
            self.DR_sum[bi, :] += np.bincount(bj, weights=self.dr[i], minlength=self.K)
            self.RD_sum[bi, :] += np.bincount(bj, weights=self.rd[i], minlength=self.K)
            self.RR_sum[bi, :] += np.bincount(bj, weights=self.rr[i], minlength=self.K)

    def total_reward_fast(self) -> float:
        """
        Trace objective based on xi_bins diagonal.

        NOTE: This follows the common SharpZ-style form:
          xi = (DD - DR - RD + RR) / (n_i * n_j)
        and returns trace(xi).

        If your exact definition differs, change it here (single source of truth).
        """
        n = self.n_phot.astype(np.float64)
        denom = (n[:, None] * n[None, :]) + self.cfg.eps

        xi = (self.DD_sum - self.DR_sum - self.RD_sum + self.RR_sum) / denom
        val = float(np.trace(xi))
        self._last_reward = val
        return val

    # ----------------- mutation -----------------

    def valid_bin_for_node(self, node_idx: int, b: int) -> bool:
        if not (0 <= node_idx < self.N and 0 <= b < self.K):
            return False
        if self.valid_bins_mask is None:
            return True
        return bool(self.valid_bins_mask[node_idx, b])

    def apply_move(self, node_idx: int, new_bin: int) -> None:
        """
        MVP: apply a node move and rebuild caches incrementally later.
        For now, simplest correct path: mutate bins + rebuild caches.
        Replace with your rank-1 cache update once wired.
        """
        node_idx = int(node_idx)
        new_bin = int(new_bin)
        if not (0 <= node_idx < self.N and 0 <= new_bin < self.K):
            raise ValueError("apply_move out of bounds")
        if not self.valid_bin_for_node(node_idx, new_bin):
            raise ValueError("apply_move invalid bin for node")

        old_bin = int(self.current_bins[node_idx])
        if old_bin == new_bin:
            return

        self.current_bins[node_idx] = new_bin
        # MVP correctness: rebuild (slow but correct).
        # Swap in your rank-1 cache update here once you wire it.
        self._reset_caches()
        self._last_reward = None

    # ----------------- snapshots -----------------

    def get_state(self) -> np.ndarray:
        return self.current_bins.copy()

    def set_state(self, bins: np.ndarray) -> None:
        bins = np.asarray(bins, dtype=np.int32)
        if bins.shape != (self.N,):
            raise ValueError(f"bins must be {(self.N,)}, got {bins.shape}")
        self.current_bins[:] = bins
        self._reset_caches()
        self._last_reward = None

    def reset_to_initial(self) -> None:
        self.set_state(self.initial_bins)

    def randomize_state(self, rng: np.random.Generator) -> None:
        bins = rng.integers(0, self.K, size=self.N, dtype=np.int32)
        if self.valid_bins_mask is not None:
            # enforce validity by resampling invalid entries
            for i in range(self.N):
                if not self.valid_bins_mask[i, bins[i]]:
                    valid = np.flatnonzero(self.valid_bins_mask[i])
                    bins[i] = int(rng.choice(valid)) if len(valid) else 0
        self.set_state(bins)
