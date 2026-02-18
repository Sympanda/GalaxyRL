from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Literal
import numpy as np


# ---------------------------
# Config
# ---------------------------

ProposalBinMode = Literal[
    "uniform_other",     # random bin != current
    "neighbor_pm1",      # +/- 1 (with edge handling)
]

ObjectiveMode = Literal[
    "energy",            # SharpZ-style energy from xi
    "trace",             # trace(xi)
]


@dataclass(frozen=True)
class SharpZBackendConfig:
    grid_shape: Tuple[int, int] = (200, 200)
    num_bins: int = 11

    # numerical
    eps: float = 1e-12
    dr_scale: float = 10.0
    rd_scale: float = 10.0
    rr_scale: float = 100.0

    # objective
    objective_mode: ObjectiveMode = "energy"

    # proposal
    proposal_bin_mode: ProposalBinMode = "uniform_other"

    # constraints / masks
    valid_bins_mask: Optional[np.ndarray] = None      # (N,K) bool
    frozen_nodes_mask: Optional[np.ndarray] = None    # (N,) bool


# ---------------------------
# Backend
# ---------------------------

class SharpZSOMBackend:
    """
    Backend responsibilities ONLY:
      - store state: current_bins over N nodes
      - fast cluster membership lookup (CSR-like) from weighted_clusters
      - apply/revert proposals with rank-1 cache updates
      - compute xi / energy / trace objectives

    Explicitly NOT responsible for:
      - temperature schedules
      - level schedules
      - accept/reject decisions
      - deltaE floors
      - episode bookkeeping

    Proposal workflow (env-driven):
      proposal = backend.make_proposal_random_bin(...) or make_proposal_set_bin(...)
      eval = backend.evaluate_proposal_deltaE(proposal)  # applies proposal temporarily
      env decides accept/reject; if reject: backend.revert(eval["undo"])
    """

    def __init__(
        self,
        cfg: SharpZBackendConfig,
        dd: np.ndarray,
        dr: np.ndarray,
        rd: np.ndarray,
        rr: np.ndarray,
        node_indices_photo: np.ndarray,
        initial_node_bins: np.ndarray,
        reference_labels: np.ndarray,
        umatrix: Optional[np.ndarray] = None,
        weighted_clusters: Optional[np.ndarray] = None,  # (L,N) int labels
    ):
        self.cfg = cfg
        self.H, self.W = cfg.grid_shape
        self.N = int(self.H * self.W)
        self.K = int(cfg.num_bins)
        if self.K != 11:
            raise ValueError("Locked: K=11 expected.")

        # --- compact correlation columns (K,N) ---
        self.dd = self._as_kn(dd, "dd")
        self.dr = self._as_kn(dr, "dr")
        self.rd = self._as_kn(rd, "rd")
        self.rr = self._as_kn(rr, "rr")

        # --- mapping arrays ---
        self.node_indices_photo = np.asarray(node_indices_photo, dtype=np.int64)
        if self.node_indices_photo.ndim != 1:
            raise ValueError("node_indices_photo must be 1D")
        if self.node_indices_photo.min(initial=0) < 0 or self.node_indices_photo.max(initial=0) >= self.N:
            raise ValueError("node_indices_photo contains out-of-range ids")

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

        # masks
        self.valid_bins_mask: Optional[np.ndarray] = None
        if cfg.valid_bins_mask is not None:
            vb = np.asarray(cfg.valid_bins_mask, dtype=bool)
            if vb.shape != (self.N, self.K):
                raise ValueError(f"valid_bins_mask must be {(self.N,self.K)}, got {vb.shape}")
            self.valid_bins_mask = vb

        self.frozen_nodes_mask: Optional[np.ndarray] = None
        if cfg.frozen_nodes_mask is not None:
            fm = np.asarray(cfg.frozen_nodes_mask, dtype=bool)
            if fm.shape != (self.N,):
                raise ValueError(f"frozen_nodes_mask must be {(self.N,)}, got {fm.shape}")
            self.frozen_nodes_mask = fm

        # optional umatrix
        self.umatrix: Optional[np.ndarray] = None
        if umatrix is not None:
            u = np.asarray(umatrix, dtype=np.float32)
            if u.shape != (self.H, self.W):
                raise ValueError(f"umatrix must be {(self.H,self.W)}, got {u.shape}")
            self.umatrix = u

        # fixed counts
        self.node_gal_counts = np.bincount(self.node_indices_photo, minlength=self.N).astype(np.float64)
        self.n_ref = np.bincount(self.reference_labels, minlength=self.K).astype(np.float64)

        # state + caches
        self.current_bins = self.initial_bins.copy()
        self._build_caches()

        # multi-resolution clusters (CSR style)
        self.weighted_clusters: Optional[np.ndarray] = None  # (L,N)
        self._cluster_order: Optional[List[np.ndarray]] = None      # list[level] -> (N,) int32 order
        self._cluster_offsets: Optional[List[np.ndarray]] = None    # list[level] -> (cmax+2,) int64 offsets
        self._valid_cluster_ids: Optional[List[np.ndarray]] = None  # list[level] -> (n_valid,) int32 cluster ids

        if weighted_clusters is not None:
            self.set_weighted_clusters(weighted_clusters)

        self._rng = np.random.default_rng()

    # ---------- shape helpers ----------
    def _as_kn(self, A: np.ndarray, name: str) -> np.ndarray:
        A = np.asarray(A)
        A = A.squeeze(-1)
        if A.ndim != 2:
            raise ValueError(f"{name} must be 2D, got {A.shape}")
        if A.shape == (self.N, self.K):
            A = A.T
        if A.shape != (self.K, self.N):
            raise ValueError(f"{name} must be (K,N)=({self.K},{self.N}) (or (N,K)), got {A.shape}")
        if A.dtype == object:
            raise TypeError(f"{name} dtype=object not allowed")
        return A.astype(np.float64, copy=False)

    def xy_to_index(self, x: int, y: int) -> int:
        return int(y) * self.W + int(x)

    def index_to_xy(self, idx: int) -> Tuple[int, int]:
        idx = int(idx)
        return idx % self.W, idx // self.W

    def node_from_xy(self, x: int, y: int) -> int:
        return self.xy_to_index(int(x), int(y))

    def valid_bin_for_node(self, node: int, b: int) -> bool:
        if not (0 <= node < self.N and 0 <= b < self.K):
            return False
        if self.valid_bins_mask is None:
            return True
        return bool(self.valid_bins_mask[node, b])

    def node_is_frozen(self, node: int) -> bool:
        if self.frozen_nodes_mask is None:
            return False
        return bool(self.frozen_nodes_mask[node])

    # ---------- caches ----------
    def _build_caches(self) -> None:
        # weighted phot counts per assigned bin
        self.n_phot = np.bincount(
            self.current_bins,
            weights=self.node_gal_counts,
            minlength=self.K
        ).astype(np.float64)

        # cached sums: (ref_bin, phot_bin)
        self.DD_sum = np.zeros((self.K, self.K), dtype=np.float64)
        self.DR_sum = np.zeros((self.K, self.K), dtype=np.float64)
        self.RD_sum = np.zeros((self.K, self.K), dtype=np.float64)
        self.RR_sum = np.zeros((self.K, self.K), dtype=np.float64)

        # build by node assignment
        for node in range(self.N):
            b = int(self.current_bins[node])
            self.DD_sum[:, b] += self.dd[:, node]
            self.DR_sum[:, b] += self.dr[:, node]
            self.RD_sum[:, b] += self.rd[:, node]
            self.RR_sum[:, b] += self.rr[:, node]

    def _rank1_move(self, node: int, oldb: int, newb: int) -> None:
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

    # ---------- observables ----------
    def xi_matrix(self) -> np.ndarray:
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
        return xi

    def trace_xi(self) -> float:
        return float(np.trace(self.xi_matrix()))

    def energy_E(self) -> float:
        rho = self.xi_matrix()
        K = self.K
        diag = np.diag(rho)
        row_sums = rho.sum(axis=1)
        off = (row_sums - diag) / max(K - 1, 1)
        E = (diag - off).mean()
        return float(E)

    def objective(self) -> float:
        if self.cfg.objective_mode == "trace":
            return self.trace_xi()
        return self.energy_E()

    # ---------- multi-resolution clusters (CSR indexing) ----------
    def set_weighted_clusters(self, weighted_clusters: np.ndarray) -> None:
        wc = np.asarray(weighted_clusters)
        if wc.ndim != 2 or wc.shape[1] != self.N:
            raise ValueError(f"weighted_clusters must be (L,N)=(*,{self.N}), got {wc.shape}")
        wc = wc.astype(np.int64, copy=False)

        self.weighted_clusters = wc
        L = int(wc.shape[0])

        eligible = np.flatnonzero(self.node_gal_counts > 0).astype(np.int32)
        if eligible.size == 0:
            raise RuntimeError("No eligible nodes with galaxies (node_gal_counts all zero).")

        if self.frozen_nodes_mask is not None:
            movable = eligible[~self.frozen_nodes_mask[eligible]]
        else:
            movable = eligible

        self._cluster_order = [None] * L
        self._cluster_offsets = [None] * L
        self._valid_cluster_ids = [None] * L

        for lvl in range(L):
            labels = wc[lvl]  # (N,) int64
            cmax = int(labels.max(initial=-1))
            if cmax < 0:
                raise RuntimeError(f"Level {lvl}: invalid cluster labels (max<0).")

            counts = np.bincount(labels, minlength=cmax + 1).astype(np.int64)
            offsets = np.zeros(cmax + 2, dtype=np.int64)
            offsets[1:] = np.cumsum(counts)

            order = np.argsort(labels, kind="stable").astype(np.int32, copy=False)

            movable_labels = labels[movable]
            valid_counts = np.bincount(movable_labels, minlength=cmax + 1)
            valid_cids = np.flatnonzero(valid_counts > 0).astype(np.int32)

            self._cluster_order[lvl] = order
            self._cluster_offsets[lvl] = offsets
            self._valid_cluster_ids[lvl] = valid_cids

    def levels_available(self) -> int:
        return 0 if self.weighted_clusters is None else int(self.weighted_clusters.shape[0])

    def cluster_grid(self, level: int) -> np.ndarray:
        if self.weighted_clusters is None:
            raise RuntimeError("weighted_clusters not set")
        lvl = int(level)
        if not (0 <= lvl < self.levels_available()):
            raise ValueError("level out of range")
        return self.weighted_clusters[lvl].reshape(self.H, self.W)

    def cluster_id_for_node(self, node: int, level: int) -> int:
        if self.weighted_clusters is None:
            raise RuntimeError("weighted_clusters not set")
        return int(self.weighted_clusters[int(level), int(node)])

    def nodes_for_cluster_id(self, level: int, cluster_id: int) -> np.ndarray:
        if self.weighted_clusters is None or self._cluster_order is None or self._cluster_offsets is None:
            raise RuntimeError("weighted_clusters CSR index not built (call set_weighted_clusters first).")
        lvl = int(level)
        if not (0 <= lvl < len(self._cluster_order)):
            raise ValueError("level out of range")
        cid = int(cluster_id)

        offsets = self._cluster_offsets[lvl]
        if not (0 <= cid < (len(offsets) - 1)):
            return np.empty((0,), dtype=np.int32)

        order = self._cluster_order[lvl]
        a = int(offsets[cid])
        b = int(offsets[cid + 1])
        return order[a:b].astype(np.int32, copy=False)

    def cluster_nodes(self, level: int, cluster_id: int) -> np.ndarray:
        # alias for readability
        return self.nodes_for_cluster_id(level, cluster_id)

    # ---------- cluster bin reducers ----------
    def _cluster_current_bin_weighted_mode(self, nodes: np.ndarray) -> int:
        nodes = np.asarray(nodes, dtype=np.int32)
        if nodes.size == 0:
            raise ValueError("empty nodes")
        bins = self.current_bins[nodes].astype(np.int32, copy=False)
        w = self.node_gal_counts[nodes].astype(np.float64, copy=False)
        counts = np.bincount(bins, weights=w, minlength=self.K)
        return int(np.argmax(counts))

    # ---------- proposal generation (PURE) ----------
    def _propose_new_bin(self, cur_bin: int, rng: np.random.Generator) -> int:
        K = self.K
        if self.cfg.proposal_bin_mode == "uniform_other":
            r = int(rng.integers(0, K - 1))
            return r if r < cur_bin else (r + 1)

        if self.cfg.proposal_bin_mode == "neighbor_pm1":
            if cur_bin == 0:
                return 1
            if cur_bin == K - 1:
                return K - 2
            return cur_bin + (1 if rng.random() < 0.5 else -1)

        raise ValueError(f"Unknown proposal_bin_mode: {self.cfg.proposal_bin_mode}")

    def _nodes_for_action_target(
        self,
        *,
        node: Optional[int] = None,
        xy: Optional[Tuple[int, int]] = None,
        level: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
        """
        Returns:
          nodes_eligible: np.ndarray[int32]
          level_used: int|None
          cluster_id: int|None

        If weighted_clusters is not set OR level is None -> acts on single node.
        Eligibility filter: node_gal_counts>0 and not frozen.
        """
        if (node is None) == (xy is None):
            raise ValueError("Provide exactly one of node= or xy=")

        if xy is not None:
            node = self.node_from_xy(int(xy[0]), int(xy[1]))
        node = int(node)
        if not (0 <= node < self.N):
            raise ValueError("node out of range")

        # single node mode
        if self.weighted_clusters is None or level is None:
            if self.node_gal_counts[node] <= 0:
                return np.empty((0,), dtype=np.int32), None, None
            if self.node_is_frozen(node):
                return np.empty((0,), dtype=np.int32), None, None
            return np.array([node], dtype=np.int32), None, None

        # cluster mode
        lvl = int(level)
        cid = self.cluster_id_for_node(node, lvl)
        nodes = self.nodes_for_cluster_id(lvl, cid)

        if nodes.size == 0:
            return np.empty((0,), dtype=np.int32), lvl, cid

        # eligible: has galaxies
        nodes = nodes[self.node_gal_counts[nodes] > 0]
        if nodes.size == 0:
            return np.empty((0,), dtype=np.int32), lvl, cid

        # eligible: not frozen
        if self.frozen_nodes_mask is not None:
            nodes = nodes[~self.frozen_nodes_mask[nodes]]
            if nodes.size == 0:
                return np.empty((0,), dtype=np.int32), lvl, cid

        return nodes.astype(np.int32, copy=False), lvl, cid

    def make_proposal_random_bin(
        self,
        *,
        node: Optional[int] = None,
        xy: Optional[Tuple[int, int]] = None,
        level: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """
        Pure proposal builder:
          - resolve action target (node or cluster at level)
          - pick new_bin randomly != current "cluster bin"
        Does NOT apply/revert and does NOT accept/reject.
        """
        if rng is None:
            rng = self._rng

        nodes, lvl_used, cid = self._nodes_for_action_target(node=node, xy=xy, level=level)
        if nodes.size == 0:
            return {"type": "noop", "reason": "no_eligible_nodes"}

        if lvl_used is None:
            curb = int(self.current_bins[int(nodes[0])])
            cur_cluster_bin = None
        else:
            curb = self._cluster_current_bin_weighted_mode(nodes)
            cur_cluster_bin = int(curb)

        newb = self._propose_new_bin(curb, rng)

        if self.valid_bins_mask is not None:
            if not np.all(self.valid_bins_mask[nodes, newb]):
                return {"type": "noop", "reason": "invalid_bin_for_nodes"}

        return {
            "type": "cluster" if (lvl_used is not None) else "node",
            "level": lvl_used,
            "cluster_id": cid,
            "nodes": nodes.astype(np.int32, copy=False),
            "cur_cluster_bin": cur_cluster_bin,
            "new_bin": int(newb),
        }

    def make_proposal_set_bin(
        self,
        *,
        new_bin: int,
        node: Optional[int] = None,
        xy: Optional[Tuple[int, int]] = None,
        level: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Pure proposal builder:
          - resolve action target (node or cluster at level)
          - set new_bin explicitly
        Does NOT apply/revert and does NOT accept/reject.
        """
        newb = int(new_bin)
        if not (0 <= newb < self.K):
            raise ValueError("new_bin out of range")

        nodes, lvl_used, cid = self._nodes_for_action_target(node=node, xy=xy, level=level)
        if nodes.size == 0:
            return {"type": "noop", "reason": "no_eligible_nodes"}

        if self.valid_bins_mask is not None:
            if not np.all(self.valid_bins_mask[nodes, newb]):
                return {"type": "noop", "reason": "invalid_bin_for_nodes"}

        if lvl_used is None:
            cur_cluster_bin = None
        else:
            cur_cluster_bin = int(self._cluster_current_bin_weighted_mode(nodes))

        return {
            "type": "cluster" if (lvl_used is not None) else "node",
            "level": lvl_used,
            "cluster_id": cid,
            "nodes": nodes.astype(np.int32, copy=False),
            "cur_cluster_bin": cur_cluster_bin,
            "new_bin": int(newb),
        }

    # ---------- apply / revert ----------
    def apply_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply proposal by moving all affected nodes to proposal["new_bin"].
        Returns undo payload for exact revert.
        """
        if proposal.get("type") == "noop":
            return {"type": "noop"}

        nodes = np.asarray(proposal["nodes"], dtype=np.int32)
        newb = int(proposal["new_bin"])

        old_bins = self.current_bins[nodes].copy()
        for node, oldb in zip(nodes, old_bins):
            ob = int(oldb)
            if ob == newb:
                continue
            self._rank1_move(int(node), ob, newb)

        return {"nodes": nodes, "old_bins": old_bins, "new_bin": newb}

    def revert(self, undo: Dict[str, Any]) -> None:
        if undo.get("type") == "noop":
            return
        nodes = np.asarray(undo["nodes"], dtype=np.int32)
        old_bins = np.asarray(undo["old_bins"], dtype=np.int32)
        for node, oldb in zip(nodes, old_bins):
            node_i = int(node)
            cur = int(self.current_bins[node_i])
            ob = int(oldb)
            if cur != ob:
                self._rank1_move(node_i, cur, ob)

    # ---------- evaluation helper ----------
    def evaluate_proposal_deltaE(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience for env:
          - computes E0
          - applies proposal
          - computes E1 and dE
        Leaves proposal APPLIED. Env must revert() if it rejects.
        """
        E0 = float(self.objective())
        if proposal.get("type") == "noop":
            return {
                "is_noop": True,
                "E0": E0,
                "E1": E0,
                "dE": 0.0,
                "undo": {"type": "noop"},
                "proposal": proposal,
            }

        undo = self.apply_proposal(proposal)
        E1 = float(self.objective())
        dE = float(E1 - E0)
        return {
            "is_noop": False,
            "E0": E0,
            "E1": E1,
            "dE": dE,
            "undo": undo,
            "proposal": proposal,
        }

    # ---------- grids / env-facing helpers ----------
    def bins_grid(self) -> np.ndarray:
        """(H,W) bins, including bins for empty nodes."""
        return self.current_bins.reshape(self.H, self.W)

    def bins_grid_int(self, empty_value: int = -1) -> np.ndarray:
        """
        (H,W) int16 bins where nodes with node_gal_counts==0 are filled with empty_value.
        """
        out = self.current_bins.astype(np.int16, copy=True)
        out[self.node_gal_counts <= 0] = np.int16(empty_value)
        return out.reshape(self.H, self.W)

    def bins_grid_float(self) -> np.ndarray:
        """Scale the bins to be between 0 and 1 like all other channels (There will be no empty bins)"""
        return self.current_bins.astype(np.float32) / (self.K - 1)

    def umatrix_grid(self) -> np.ndarray:
        if self.umatrix is None:
            raise RuntimeError("umatrix not set")
        return self.umatrix

    def cluster_ids_at_level(self, level: int) -> np.ndarray:
        return self.cluster_grid(level)

    def bins_grid_at_level(
        self,
        level: int,
        reducer: Literal["weighted_mode", "mode", "mean_round"] = "weighted_mode",
    ) -> np.ndarray:
        """
        Debug/plot only: returns (H,W) where each cluster at `level` is filled with a representative bin.
        """
        if self.weighted_clusters is None:
            raise ValueError("weighted_clusters not provided")

        lvl = int(level)
        clusters = self.weighted_clusters[lvl]  # (N,)
        bins = self.current_bins                # (N,)
        cmax = int(clusters.max(initial=0))

        if reducer == "weighted_mode":
            key = clusters.astype(np.int64) * self.K + bins.astype(np.int64)
            wcounts = np.bincount(
                key,
                weights=self.node_gal_counts.astype(np.float64, copy=False),
                minlength=(cmax + 1) * self.K
            ).reshape(cmax + 1, self.K)
            rep = wcounts.argmax(axis=1).astype(np.int32)

        elif reducer == "mode":
            key = clusters.astype(np.int64) * self.K + bins.astype(np.int64)
            counts = np.bincount(key, minlength=(cmax + 1) * self.K).reshape(cmax + 1, self.K)
            rep = counts.argmax(axis=1).astype(np.int32)

        else:  # mean_round
            sums = np.bincount(clusters, weights=bins.astype(np.float64), minlength=cmax + 1)
            nums = np.bincount(clusters, minlength=cmax + 1).astype(np.float64)
            means = np.divide(sums, nums, out=np.zeros_like(sums), where=nums > 0)
            rep = np.clip(np.rint(means), 0, self.K - 1).astype(np.int32)

        return rep[clusters].reshape(self.H, self.W)

    # ---------- greedy bin helper ----------
    def greedy_bin_for_node(
        self,
        node: int,
        level: Optional[int] = None,
    ) -> Tuple[int, float]:
        """
        For a given node, scan over all possible bins and return the bin that
        maximises ΔE = E1 - E0 when that node is moved to that bin.

        This is intended for hybrid policies where an external agent (e.g. PPO)
        chooses *where* to act (the node index), and the backend chooses
        greedily *how* to act (the bin assignment) for that node.

        Notes:
          - The current state is restored before returning; this does not leave
            the backend in a modified state.
          - If no valid non-noop proposals are found (e.g. node has no
            galaxies or is fully constrained), this returns the current bin and
            dE = 0.0.
        """
        node_i = int(node)
        if not (0 <= node_i < self.N):
            raise ValueError("node out of range")

        # If node has no galaxies or is frozen, there's nothing meaningful to do.
        if self.node_gal_counts[node_i] <= 0:
            return int(self.current_bins[node_i]), 0.0
        if self.node_is_frozen(node_i):
            return int(self.current_bins[node_i]), 0.0

        cur_bin = int(self.current_bins[node_i])
        best_bin = cur_bin
        best_dE = -np.inf

        for b in range(self.K):
            b_int = int(b)
            if b_int == cur_bin:
                continue

            proposal = self.make_proposal_set_bin(
                new_bin=b_int,
                node=node_i,
                level=level,
            )
            if proposal.get("type") == "noop":
                continue

            evald = self.evaluate_proposal_deltaE(proposal)
            dE = float(evald["dE"])
            self.revert(evald["undo"])

            if dE > best_dE:
                best_dE = dE
                best_bin = b_int

        if not np.isfinite(best_dE):
            # No valid moves found; fall back to current bin.
            return cur_bin, 0.0

        return int(best_bin), float(best_dE)

    # ---------- convenience ----------
    def reset_to_initial(self) -> None:
        self.current_bins[:] = self.initial_bins
        self._build_caches()

    def get_state(self) -> np.ndarray:
        return self.current_bins.copy()
