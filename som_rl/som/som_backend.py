from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np


@dataclass(frozen=True)
class SOMBackendConfig:
    grid_shape: Tuple[int, int]  # (H, W)
    num_bins: int = 11

    eps: float = 1e-12
    dr_scale: float = 10.0
    rd_scale: float = 10.0
    rr_scale: float = 100.0  # dr_scale * rd_scale
    noop_eps: float = 0.0    # treat best_delta <= noop_eps as noop


class SOMControl:
    """
    Backend for SOM node->bin assignment with fast cached TreeCorr-style reward.
    Env action: choose node. Backend greedily chooses best bin among K=11.
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
        umatrix: Optional[np.ndarray] = None,          # (H,W) float32
    ):
        self.cfg = cfg
        self.H, self.W = cfg.grid_shape
        self.N = self.H * self.W
        self.K = int(cfg.num_bins)
        if self.K != 11:
            raise ValueError("This project locks K=11. If you want otherwise, change config + checks coherently.")

        self.dd = self._as_kn(dd, "dd")
        self.dr = self._as_kn(dr, "dr")
        self.rd = self._as_kn(rd, "rd")
        self.rr = self._as_kn(rr, "rr")

        self.node_indices_photo = np.asarray(node_indices_photo, dtype=np.int64)
        if self.node_indices_photo.ndim != 1:
            raise ValueError("node_indices_photo must be 1D")
        if self.node_indices_photo.min(initial=0) < 0 or self.node_indices_photo.max(initial=0) >= self.N:
            raise ValueError("node_indices_photo contains out-of-range node ids")

        self.reference_labels = np.asarray(reference_labels, dtype=np.int32)
        if self.reference_labels.ndim != 1:
            raise ValueError("reference_labels must be 1D")
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

        # fixed counts
        self.node_gal_counts = np.bincount(self.node_indices_photo, minlength=self.N).astype(np.float64)
        self.n_ref = np.bincount(self.reference_labels, minlength=self.K).astype(np.float64)

        # mutable state + caches
        self.current_bins = self.initial_bins.copy()
        self._build_caches()

    def _as_kn(self, A: np.ndarray, name: str) -> np.ndarray:
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

    # row-major mapping
    def xy_to_index(self, x: int, y: int) -> int:
        return int(y) * self.W + int(x)

    def index_to_xy(self, idx: int) -> Tuple[int, int]:
        idx = int(idx)
        return idx % self.W, idx // self.W

    def valid_bin_for_node(self, node_idx: int, b: int) -> bool:
        node_idx = int(node_idx); b = int(b)
        if not (0 <= node_idx < self.N and 0 <= b < self.K):
            return False
        if self.valid_bins_mask is None:
            return True
        return bool(self.valid_bins_mask[node_idx, b])

    def _build_caches(self) -> None:
        self.n_phot = np.bincount(
            self.current_bins,
            weights=self.node_gal_counts,
            minlength=self.K
        ).astype(np.float64)

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
        eps = self.cfg.eps
        n_phot = self.n_phot + eps
        n_ref = self.n_ref + eps
        denom = (n_ref[:, None] * n_phot[None, :])

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
        self.DD_sum[:, oldb] -= self.dd[:, node]; self.DD_sum[:, newb] += self.dd[:, node]
        self.DR_sum[:, oldb] -= self.dr[:, node]; self.DR_sum[:, newb] += self.dr[:, node]
        self.RD_sum[:, oldb] -= self.rd[:, node]; self.RD_sum[:, newb] += self.rd[:, node]
        self.RR_sum[:, oldb] -= self.rr[:, node]; self.RR_sum[:, newb] += self.rr[:, node]

        w = self.node_gal_counts[node]
        self.n_phot[oldb] -= w
        self.n_phot[newb] += w

        self.current_bins[node] = newb

    def best_bin_for_node(self, node_idx: int) -> Tuple[int, float]:
        node = int(node_idx)
        if not (0 <= node < self.N):
            raise ValueError("node_idx out of range")

        oldb = int(self.current_bins[node])
        r0 = self.reward_trace()

        bestb = oldb
        bestd = 0.0

        for newb in range(self.K):
            if newb == oldb:
                continue
            if not self.valid_bin_for_node(node, newb):
                continue

            self._rank1_move(node, oldb, newb)
            r1 = self.reward_trace()
            d = float(r1 - r0)
            self._rank1_move(node, newb, oldb)

            if d > bestd:
                bestd = d
                bestb = int(newb)

        return bestb, float(bestd)

    def step_node(self, node_idx: int) -> Dict[str, Any]:
        node = int(node_idx)
        if not (0 <= node < self.N):
            raise ValueError("node_idx out of range")

        oldb = int(self.current_bins[node])
        r0 = self.reward_trace()

        bestb, bestd = self.best_bin_for_node(node)
        if bestb != oldb:
            self._rank1_move(node, oldb, bestb)

        r1 = self.reward_trace()
        bestd = float(r1 - r0)  # recompute to be consistent with applied state

        is_noop = bool(bestd <= float(self.cfg.noop_eps))

        return {
            "node": node,
            "old_bin": oldb,
            "chosen_bin": int(bestb),
            "best_delta": float(bestd),
            "is_noop": is_noop,
            "delta_reward": float(bestd),
            "reward": float(r1),
        }

    def reset_to_initial(self) -> None:
        self.current_bins[:] = self.initial_bins
        self._build_caches()

    def get_state(self) -> np.ndarray:
        return self.current_bins.copy()
