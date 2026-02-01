from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SOMStateConfig:
    grid_shape: Tuple[int, int]   # (H, W)
    num_bins: int                 # K
    eps: float = 1e-12

    # Normalization knobs (match whatever you used when producing dd/dr/rd/rr)
    # In your earlier fast proxy you used factors of 10 for DR/RD/RR. Keep configurable.
    dr_scale: float = 10.0
    rd_scale: float = 10.0
    rr_scale: float = 100.0       # typically dr_scale * rd_scale


class SOMState:
    """
    Fast SOM binning backend using compact correlation columns.

    Required input shapes:
      - dd, dr, rd, rr: (K, N)  float
        where column n is the contribution of node n to each reference bin (row).
      - node_indices_photo: (N_gal_photo,) int mapping each photometric galaxy -> node idx [0..N-1]
      - initial_node_bins: (N,) int current photometric bin label per node [0..K-1]

    Cached:
      - galaxy_bin_counts: (K,) weighted by node_gal_counts
      - DD_sum, DR_sum, RD_sum, RR_sum: (K, K) where col j aggregates nodes assigned to phot-bin j
    """

    def __init__(
        self,
        cfg: SOMStateConfig,
        dd: np.ndarray,
        dr: np.ndarray,
        rd: np.ndarray,
        rr: np.ndarray,
        node_indices_photo: np.ndarray,
        initial_node_bins: np.ndarray,
        valid_bins_mask: Optional[np.ndarray] = None,   # (N, K) bool
        umatrix: Optional[np.ndarray] = None,           # (H, W) float
    ):
        self.cfg = cfg
        self.H, self.W = cfg.grid_shape
        self.N = int(self.H * self.W)
        self.K = int(cfg.num_bins)

        # Validate compact arrays
        self.dd = self._as_bins_nodes(dd, "dd")
        self.dr = self._as_bins_nodes(dr, "dr")
        self.rd = self._as_bins_nodes(rd, "rd")
        self.rr = self._as_bins_nodes(rr, "rr")

        node_indices_photo = np.asarray(node_indices_photo, dtype=np.int64)
        if node_indices_photo.ndim != 1:
            raise ValueError("node_indices_photo must be 1D")
        if node_indices_photo.min(initial=0) < 0 or node_indices_photo.max(initial=0) >= self.N:
            raise ValueError("node_indices_photo contains out-of-range node ids")
        self.node_indices_photo = node_indices_photo

        # Node galaxy counts (photometric population weight per node)
        self.node_gal_counts = np.bincount(node_indices_photo, minlength=self.N).astype(np.float64)

        initial_node_bins = np.asarray(initial_node_bins, dtype=np.int32)
        if initial_node_bins.shape != (self.N,):
            raise ValueError(f"initial_node_bins must be {(self.N,)}, got {initial_node_bins.shape}")
        if initial_node_bins.min(initial=0) < 0 or initial_node_bins.max(initial=0) >= self.K:
            raise ValueError("initial_node_bins out of [0..K-1]")
        self.initial_bins = initial_node_bins.copy()
        self.current_bins = initial_node_bins.copy()

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

        self.reset()

    # ----------------- shape helpers -----------------

    def _as_bins_nodes(self, A: np.ndarray, name: str) -> np.ndarray:
        A = np.asarray(A)
        if A.ndim != 2:
            raise ValueError(f"{name} must be 2D (K,N), got {A.shape}")
        # Accept either (K,N) or (N,K) and fix
        if A.shape == (self.N, self.K):
            A = A.T
        if A.shape != (self.K, self.N):
            raise ValueError(f"{name} must be shape (K,N)=({self.K},{self.N}), got {A.shape}")
        return A.astype(np.float64, copy=False)

    # ----------------- indexing helpers -----------------
    # Standard row-major: idx = y*W + x

    def xy_to_index(self, x: int, y: int) -> int:
        x = int(x); y = int(y)
        return y * self.W + x

    def index_to_xy(self, idx: int) -> Tuple[int, int]:
        idx = int(idx)
        y = idx // self.W
        x = idx % self.W
        return x, y

    # ----------------- caches -----------------

    def reset(self) -> None:
        # weighted photometric bin counts
        self.galaxy_bin_counts = np.bincount(
            self.current_bins,
            weights=self.node_gal_counts,
            minlength=self.K
        ).astype(np.float64)

        # cached sums per photometric bin (columns)
        self.DD_sum = np.zeros((self.K, self.K), dtype=np.float64)
        self.DR_sum = np.zeros((self.K, self.K), dtype=np.float64)
        self.RD_sum = np.zeros((self.K, self.K), dtype=np.float64)
        self.RR_sum = np.zeros((self.K, self.K), dtype=np.float64)

        # Build from scratch: O(N*K), fine (N=40k, K=11)
        for n in range(self.N):
            j = int(self.current_bins[n])
            self.DD_sum[:, j] += self.dd[:, n]
            self.DR_sum[:, j] += self.dr[:, n]
            self.RD_sum[:, j] += self.rd[:, n]
            self.RR_sum[:, j] += self.rr[:, n]

        # Proxy reference counts for normalization (same idea as your earlier code)
        self.n_ref_proxy = self.dd.sum(axis=1) + self.cfg.eps

        self._last_reward: Optional[float] = None

    # ----------------- reward -----------------

    def total_reward_fast(self) -> float:
        """
        Trace objective. Matches your earlier fast-proxy shape:
          DD/(nref*nphot) - DR/(dr_scale*nref*nphot) - RD/(rd_scale*nref*nphot) + RR/(rr_scale*nref*nphot)
        then xi = (...) / RRn  and return trace(xi)

        If you want a different exact form, change it here (single source of truth).
        """
        n_phot = self.galaxy_bin_counts + self.cfg.eps         # (K,)
        n_ref  = self.n_ref_proxy + self.cfg.eps               # (K,)

        denom = (n_ref[:, None] * n_phot[None, :])

        DDn = self.DD_sum / denom
        DRn = self.DR_sum / (self.cfg.dr_scale * denom)
        RDn = self.RD_sum / (self.cfg.rd_scale * denom)
        RRn = self.RR_sum / (self.cfg.rr_scale * denom)

        with np.errstate(divide="ignore", invalid="ignore"):
            xi = np.divide(DDn - DRn - RDn + RRn, RRn,
                           out=np.zeros_like(RRn), where=RRn > 0)

        val = float(np.trace(xi))
        self._last_reward = val
        return val

    # ----------------- validity -----------------

    def valid_bin_for_node(self, node_idx: int, b: int) -> bool:
        node_idx = int(node_idx); b = int(b)
        if not (0 <= node_idx < self.N and 0 <= b < self.K):
            return False
        if self.valid_bins_mask is None:
            return True
        return bool(self.valid_bins_mask[node_idx, b])

    # ----------------- fast mutation (rank-1 updates) -----------------

    def set_node_bin_idx(self, node_idx: int, new_bin: int) -> None:
        node = int(node_idx)
        newb = int(new_bin)
        if not self.valid_bin_for_node(node, newb):
            raise ValueError("Invalid node/bin")

        oldb = int(self.current_bins[node])
        if newb == oldb:
            return

        # Update cached sums (O(K))
        self.DD_sum[:, oldb] -= self.dd[:, node]; self.DD_sum[:, newb] += self.dd[:, node]
        self.DR_sum[:, oldb] -= self.dr[:, node]; self.DR_sum[:, newb] += self.dr[:, node]
        self.RD_sum[:, oldb] -= self.rd[:, node]; self.RD_sum[:, newb] += self.rd[:, node]
        self.RR_sum[:, oldb] -= self.rr[:, node]; self.RR_sum[:, newb] += self.rr[:, node]

        # Update phot counts (weighted by node occupancy)
        w = float(self.node_gal_counts[node])
        self.galaxy_bin_counts[oldb] -= w
        self.galaxy_bin_counts[newb] += w

        self.current_bins[node] = newb
        self._last_reward = None

    def estimate_delta_trace(self, node_idx: int, candidate_bin: int) -> float:
        """
        Cheap delta estimator by computing reward before/after a hypothetical move.
        K is small, so we just compute two traces with temporary updates.
        """
        node = int(node_idx)
        newb = int(candidate_bin)
        if not self.valid_bin_for_node(node, newb):
            return -np.inf

        oldb = int(self.current_bins[node])
        if newb == oldb:
            return 0.0

        r0 = self.total_reward_fast()

        # Apply temporary rank-1 update
        dd_col = self.dd[:, node]; dr_col = self.dr[:, node]; rd_col = self.rd[:, node]; rr_col = self.rr[:, node]
        w = float(self.node_gal_counts[node])

        self.DD_sum[:, oldb] -= dd_col; self.DD_sum[:, newb] += dd_col
        self.DR_sum[:, oldb] -= dr_col; self.DR_sum[:, newb] += dr_col
        self.RD_sum[:, oldb] -= rd_col; self.RD_sum[:, newb] += rd_col
        self.RR_sum[:, oldb] -= rr_col; self.RR_sum[:, newb] += rr_col
        self.galaxy_bin_counts[oldb] -= w
        self.galaxy_bin_counts[newb] += w
        self.current_bins[node] = newb

        r1 = self.total_reward_fast()

        # Revert
        self.DD_sum[:, newb] -= dd_col; self.DD_sum[:, oldb] += dd_col
        self.DR_sum[:, newb] -= dr_col; self.DR_sum[:, oldb] += dr_col
        self.RD_sum[:, newb] -= rd_col; self.RD_sum[:, oldb] += rd_col
        self.RR_sum[:, newb] -= rr_col; self.RR_sum[:, oldb] += rr_col
        self.galaxy_bin_counts[newb] -= w
        self.galaxy_bin_counts[oldb] += w
        self.current_bins[node] = oldb
        self._last_reward = None  # traces invalid now

        return float(r1 - r0)

    # ----------------- deterministic "TreeCorr picks best bin" -----------------

    def best_bin_for_node(self, node_idx: int) -> Tuple[int, float]:
        """
        Returns (best_bin, best_delta). Searches all bins (or valid bins).
        """
        node = int(node_idx)
        cur = int(self.current_bins[node])

        best_bin = cur
        best_delta = -np.inf

        # Evaluate all bins (K is tiny)
        for b in range(self.K):
            if b == cur:
                continue
            d = self.estimate_delta_trace(node, b)
            if d > best_delta:
                best_delta = d
                best_bin = b

        if best_delta == -np.inf:
            return cur, 0.0
        return int(best_bin), float(best_delta)

    def apply_best_move_for_node(self, node_idx: int) -> Tuple[int, float]:
        """
        Deterministic policy: choose best bin (TreeCorr proxy) for this node and apply it.
        Returns (chosen_bin, delta_reward).
        """
        b, d = self.best_bin_for_node(node_idx)
        self.set_node_bin_idx(node_idx, b)
        return int(b), float(d)

    # ----------------- snapshots -----------------

    def get_state(self) -> np.ndarray:
        return self.current_bins.copy()

    def set_state(self, bins: np.ndarray) -> None:
        bins = np.asarray(bins, dtype=np.int32)
        if bins.shape != (self.N,):
            raise ValueError(f"bins must be {(self.N,)}, got {bins.shape}")
        self.current_bins[:] = bins
        self.reset()

    def reset_to_initial(self) -> None:
        self.set_state(self.initial_bins)
