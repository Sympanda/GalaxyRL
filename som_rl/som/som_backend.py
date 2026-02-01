from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np


@dataclass(frozen=True)
class SOMBackendConfig:
    grid_shape: Tuple[int, int]  # (H, W)
    num_bins: int = 11

    # Match your notebook normalization factors
    eps: float = 1e-12
    dr_scale: float = 10.0
    rd_scale: float = 10.0
    rr_scale: float = 100.0  # dr_scale * rd_scale


class SOMControl:
    """
    Backend for SOM node->bin assignment with fast cached TreeCorr-style reward.

    Env action:
      - node_idx (int in [0..N-1])  (env should generally sample from eligible_nodes)

    Backend policy:
      - greedily selects the best bin for that node (evaluates all K candidates)

    Notes on indexing:
      - We use image-style convention: x=col, y=row.
      - index = row * W + col
      - index_to_xy returns (x=col, y=row)
    """

    def __init__(
        self,
        cfg: SOMBackendConfig,
        dd: np.ndarray,
        dr: np.ndarray,
        rd: np.ndarray,
        rr: np.ndarray,
        node_indices_photo: np.ndarray,
        initial_node_bins: np.ndarray,
        reference_labels: np.ndarray,
        valid_bins_mask: Optional[np.ndarray] = None,  # (N,K) bool
        umatrix: Optional[np.ndarray] = None,          # (H,W) float32 (optional, for obs later)
    ):
        self.cfg = cfg
        self.H, self.W = cfg.grid_shape
        self.N = self.H * self.W
        self.K = int(cfg.num_bins)
        if self.K != 11:
            raise ValueError("This project locks K=11. If you want otherwise, change config + checks coherently.")

        # --- load arrays as float64 views (no copies if already float64/float32) ---
        self.dd = self._as_kn(dd, "dd")
        self.dr = self._as_kn(dr, "dr")
        self.rd = self._as_kn(rd, "rd")
        self.rr = self._as_kn(rr, "rr")

        # --- indices / labels ---
        self.node_indices_photo = np.asarray(node_indices_photo, dtype=np.int64)
        if self.node_indices_photo.ndim != 1:
            raise ValueError("node_indices_photo must be 1D")
        if self.node_indices_photo.size == 0:
            raise ValueError("node_indices_photo is empty")
        if self.node_indices_photo.min(initial=0) < 0 or self.node_indices_photo.max(initial=0) >= self.N:
            raise ValueError("node_indices_photo contains out-of-range node ids")

        self.reference_labels = np.asarray(reference_labels, dtype=np.int32)
        if self.reference_labels.ndim != 1:
            raise ValueError("reference_labels must be 1D")
        if self.reference_labels.size == 0:
            raise ValueError("reference_labels is empty")
        if self.reference_labels.min(initial=0) < 0 or self.reference_labels.max(initial=0) >= self.K:
            raise ValueError("reference_labels out of [0..K-1]")

        self.initial_bins = np.asarray(initial_node_bins, dtype=np.int32)
        if self.initial_bins.shape != (self.N,):
            raise ValueError(f"initial_node_bins must be {(self.N,)}, got {self.initial_bins.shape}")
        if self.initial_bins.min(initial=0) < 0 or self.initial_bins.max(initial=0) >= self.K:
            raise ValueError("initial_node_bins out of [0..K-1]")

        self.valid_bins_mask = None
        if valid_bins_mask is not None:
            vb = np.asarray(valid_bins_mask, dtype=bool)
            if vb.shape != (self.N, self.K):
                raise ValueError(f"valid_bins_mask must be {(self.N,self.K)}, got {vb.shape}")
            self.valid_bins_mask = vb

        self.umatrix = None
        if umatrix is not None:
            u = np.asarray(umatrix, dtype=np.float32)
            if u.shape != (self.H, self.W):
                raise ValueError(f"umatrix must be {(self.H,self.W)}, got {u.shape}")
            self.umatrix = u

        # --- fixed counts ---
        self.node_gal_counts = np.bincount(self.node_indices_photo, minlength=self.N).astype(np.float64)
        self.n_ref = np.bincount(self.reference_labels, minlength=self.K).astype(np.float64)

        # nodes that actually contain photometric galaxies (use for action masking/sampling)
        self.eligible_nodes = np.flatnonzero(self.node_gal_counts > 0).astype(np.int32)

        # --- mutable state + caches ---
        self.current_bins = self.initial_bins.copy()
        self._build_caches()

    def _as_kn(self, A: np.ndarray, name: str) -> np.ndarray:
        """
        Accepts arrays shaped (K,N) or (N,K) and returns (K,N) float64 view.
        """
        A = np.asarray(A)
        if A.ndim != 2:
            raise ValueError(f"{name} must be 2D, got {A.shape}")
        # accept (N,K) and transpose to (K,N)
        if A.shape == (self.N, self.K):
            A = A.T
        if A.shape != (self.K, self.N):
            raise ValueError(f"{name} must be (K,N)=({self.K},{self.N}) (or (N,K)), got {A.shape}")
        if A.dtype == object:
            raise TypeError(f"{name} dtype=object not allowed; must be numeric compact arrays")
        return A.astype(np.float64, copy=False)

    # --- indexing helpers (x=col, y=row) ---
    def xy_to_index(self, x: int, y: int) -> int:
        x = int(x); y = int(y)
        if not (0 <= x < self.W and 0 <= y < self.H):
            raise ValueError("xy_to_index out of bounds")
        return y * self.W + x

    def index_to_xy(self, idx: int) -> Tuple[int, int]:
        idx = int(idx)
        if not (0 <= idx < self.N):
            raise ValueError("index_to_xy out of bounds")
        x = idx % self.W
        y = idx // self.W
        return x, y

    def valid_bin_for_node(self, node_idx: int, b: int) -> bool:
        node_idx = int(node_idx); b = int(b)
        if not (0 <= node_idx < self.N and 0 <= b < self.K):
            return False
        if self.valid_bins_mask is None:
            return True
        return bool(self.valid_bins_mask[node_idx, b])

    def _build_caches(self) -> None:
        """
        Rebuild caches from scratch (O(N*K)). Called on init and reset.
        """
        # photometric counts (weighted by node occupancy)
        self.n_phot = np.bincount(
            self.current_bins,
            weights=self.node_gal_counts,
            minlength=self.K
        ).astype(np.float64)

        # sums: (ref_bin j, phot_bin b)
        self.DD_sum = np.zeros((self.K, self.K), dtype=np.float64)
        self.DR_sum = np.zeros((self.K, self.K), dtype=np.float64)
        self.RD_sum = np.zeros((self.K, self.K), dtype=np.float64)
        self.RR_sum = np.zeros((self.K, self.K), dtype=np.float64)

        for node in range(self.N):
            b = int(self.current_bins[node])
            self.DD_sum[:, b] += self.dd[:, node]
            self.DR_sum[:, b] += self.dr[:, node]
            self.RD_sum[:, b] += self.rd[:, node]
            self.RR_sum[:, b] += self.rr[:, node]

    def reward_trace(self) -> float:
        """
        Trace of TreeCorr-style xi matrix (diagonal objective).
        """
        eps = self.cfg.eps
        n_phot = self.n_phot + eps                 # (phot_bin,)
        n_ref = self.n_ref + eps                   # (ref_bin,)
        denom = (n_ref[:, None] * n_phot[None, :]) # (ref_bin, phot_bin)

        DDn = self.DD_sum / denom
        DRn = self.DR_sum / (self.cfg.dr_scale * denom)
        RDn = self.RD_sum / (self.cfg.rd_scale * denom)
        RRn = self.RR_sum / (self.cfg.rr_scale * denom)

        with np.errstate(divide="ignore", invalid="ignore"):
            xi = np.divide(
                DDn - DRn - RDn + RRn,
                RRn,
                out=np.zeros_like(RRn),
                where=RRn > 0
            )

        return float(np.trace(xi))

    def _rank1_move(self, node: int, oldb: int, newb: int) -> None:
        """
        Apply/revert a single node reassignment with rank-1 cache updates.
        (Used both for real moves and temporary evaluation moves.)
        """
        # sums
        self.DD_sum[:, oldb] -= self.dd[:, node]; self.DD_sum[:, newb] += self.dd[:, node]
        self.DR_sum[:, oldb] -= self.dr[:, node]; self.DR_sum[:, newb] += self.dr[:, node]
        self.RD_sum[:, oldb] -= self.rd[:, node]; self.RD_sum[:, newb] += self.rd[:, node]
        self.RR_sum[:, oldb] -= self.rr[:, node]; self.RR_sum[:, newb] += self.rr[:, node]
        # counts
        w = self.node_gal_counts[node]
        self.n_phot[oldb] -= w
        self.n_phot[newb] += w
        # state
        self.current_bins[node] = newb

    def best_bin_for_node(self, node_idx: int, r0: Optional[float] = None) -> Tuple[int, float, bool]:
        """
        Returns (best_bin, best_delta, is_noop) for a given node under greedy evaluation.

        is_noop is True if:
          - node has no galaxies, or
          - no move yields positive improvement (best_delta <= 0), or
          - best_bin == current bin
        """
        node = int(node_idx)
        if not (0 <= node < self.N):
            raise ValueError("node_idx out of range")

        oldb = int(self.current_bins[node])

        # If node has no galaxies, moving it should do nothing; treat as noop.
        if self.node_gal_counts[node] <= 0:
            return oldb, 0.0, True

        if r0 is None:
            r0 = self.reward_trace()

        bestb = oldb
        bestd = 0.0

        for newb in range(self.K):
            if newb == oldb:
                continue
            if not self.valid_bin_for_node(node, newb):
                continue

            # temp apply
            self._rank1_move(node, oldb, newb)
            r1 = self.reward_trace()
            d = r1 - r0
            # revert
            self._rank1_move(node, newb, oldb)

            if d > bestd:
                bestd = float(d)
                bestb = int(newb)

        is_noop = (bestb == oldb) or (bestd <= 0.0)
        return bestb, bestd, is_noop

    def greedy_delta_for_node(self, node_idx: int) -> float:
        """
        Convenience: best possible greedy improvement for this node (does not change state).
        """
        r0 = self.reward_trace()
        _, bestd, _ = self.best_bin_for_node(node_idx, r0=r0)
        return float(bestd)

    def step_node(self, node_idx: int) -> Dict[str, Any]:
        """
        Env action: node_idx. Backend applies greedy best bin.

        Returns info dict:
          - is_noop: True if the node was already locally optimal under greedy rule
          - applied: True if a change was applied
        """
        node = int(node_idx)
        if not (0 <= node < self.N):
            raise ValueError("node_idx out of range")

        oldb = int(self.current_bins[node])

        # baseline
        r0 = self.reward_trace()

        bestb, bestd, is_noop = self.best_bin_for_node(node, r0=r0)

        applied = False
        if not is_noop and bestb != oldb:
            self._rank1_move(node, oldb, bestb)
            applied = True

        # avoid recomputing reward_trace(); bestd was computed against r0
        r1 = float(r0 + (bestd if applied else 0.0))

        return {
            "node": node,
            "old_bin": oldb,
            "best_bin": int(bestb),
            "is_noop": bool(is_noop),
            "applied": bool(applied),
            "delta_reward": float(bestd if applied else 0.0),
            "reward_before": float(r0),
            "reward_after": float(r1),
        }

    def reset_to_initial(self) -> None:
        self.current_bins[:] = self.initial_bins
        self._build_caches()

    def get_state(self) -> np.ndarray:
        return self.current_bins.copy()
