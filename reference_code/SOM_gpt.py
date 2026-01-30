# som.py
# A compact, env-ready SOM class that wraps your fast TreeCorr reward engine.
# - Keeps O(B) rank-1 cache updates
# - Adds grid layout, getters, batch ops, masks, episode control, snapshots
# - “full” reward == fast reward (by design here)

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Iterable

# ----------------- helpers -----------------
def _as_2d_bins_nodes(x, name, B=None, N=None, dtype=np.float32):
    A = np.asarray(x)
    if A.ndim > 2:
        if A.shape[-1] == 1:
            A = A[..., 0]
        else:
            A = A.sum(axis=-1)
    A = np.squeeze(A)
    if A.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {A.shape}")
    if B is not None and N is not None:
        r, c = A.shape
        if r != B and c == B:
            # user likely transposed dims; fix once
            A = A.T
            r, c = A.shape
        if r != B or c != N:
            raise ValueError(f"{name} has shape {A.shape}, expected ({B},{N})")
    return A.astype(dtype, copy=False)

def _maybe_infer_grid(num_nodes: int) -> Tuple[int, int]:
    """Infer (H, W) if possible (perfect square -> square grid; else (num_nodes, 1))."""
    r = int(round(num_nodes ** 0.5))
    if r * r == num_nodes:
        return (r, r)
    return (num_nodes, 1)

# ----------------- core SOM class (fast-TC only) -----------------
class SOM:
    """
    SOM class exposing everything the RL envs need, using your fast TreeCorr reward.

    Required inputs:
        node_indices_photo : (N_phot,) int64  -> node id for each photometric-sample galaxy
        initial_node_bins  : (N_nodes,) int32 -> starting tomographic bin per node in [0..K-1]
        dd, dr, rd, rr     : (K, N_nodes) float (or castable) compact pair-count "columns"

    Optional layout:
        grid_shape : (H, W)    -> interpret nodes in row-major order on a grid
        index_to_xy: (N_nodes,2) explicit integer mapping from node index -> (x, y)
        umatrix    : (H,W) float (optional), exposed for visualization/obs only

    Notes on speed:
        - set_node_bin / delta_node_bin are O(K) via rank-1 cache updates.
        - total_reward_fast() is O(K^2) on the cached (K×K) aggregates.
        - Snapshot/restore APIs let you vectorize episodes across many envs.
    """

    # ---------- construction ----------
    def __init__(
        self,
        node_indices_photo: np.ndarray,
        initial_node_bins: np.ndarray,
        dd: np.ndarray, dr: np.ndarray, rd: np.ndarray, rr: np.ndarray,
        num_bins: int = 11,
        grid_shape: Optional[Tuple[int, int]] = None,
        index_to_xy: Optional[np.ndarray] = None,
        umatrix: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.num_bins: int = int(num_bins)

        # Core state
        self.current_bins = np.asarray(initial_node_bins, dtype=np.int32)
        self.init_bins    = self.current_bins.copy()
        self.num_nodes    = int(self.current_bins.size)

        self.node_indices = np.asarray(node_indices_photo, dtype=np.int64)
        # per-node galaxy counts within the photometric (non-reference) sample
        self.node_gal_counts = np.bincount(self.node_indices, minlength=self.num_nodes).astype(np.int64)

        # Compact correlation "columns" (K × N_nodes)
        self.dd = _as_2d_bins_nodes(dd, "dd", self.num_bins, self.num_nodes, np.float32)
        self.dr = _as_2d_bins_nodes(dr, "dr", self.num_bins, self.num_nodes, np.float32)
        self.rd = _as_2d_bins_nodes(rd, "rd", self.num_bins, self.num_nodes, np.float32)
        self.rr = _as_2d_bins_nodes(rr, "rr", self.num_bins, self.num_nodes, np.float32)

        # Layout
        if index_to_xy is not None:
            xy = np.asarray(index_to_xy, dtype=np.int64)
            if xy.shape != (self.num_nodes, 2):
                raise ValueError(f"index_to_xy must be (N_nodes,2), got {xy.shape}")
            self._xy = xy
            H = int(xy[:, 0].max()) + 1
            W = int(xy[:, 1].max()) + 1
            self._grid_shape = (H, W)
        else:
            self._grid_shape = grid_shape or _maybe_infer_grid(self.num_nodes)
            H, W = self._grid_shape
            # row-major map: idx -> (x, y)
            xs = np.arange(self.num_nodes) // W
            ys = np.arange(self.num_nodes) %  W
            self._xy = np.stack([xs, ys], axis=1).astype(np.int64)

        # Optional U-matrix for visualization/obs channels (zeros if not provided)
        if umatrix is not None:
            U = np.asarray(umatrix, dtype=np.float32)
            if U.shape != self._grid_shape:
                raise ValueError(f"umatrix must match grid shape {self._grid_shape}, got {U.shape}")
            self._umatrix = U
        else:
            self._umatrix = np.zeros(self._grid_shape, dtype=np.float32)

        # Cached aggregates & counters
        self._reset_caches()

        # Keep a rolling value if you want delta rewards without recomputing
        self._last_reward_val: Optional[float] = None

    # ---------- internal: caches & normalization ----------
    def _reset_caches(self):
        """Rebuild all K×K cached sums and counts from current_bins."""
        B = self.num_bins
        self.galaxy_bin_counts = np.bincount(
            self.current_bins, weights=self.node_gal_counts, minlength=B
        ).astype(np.float64)

        self.DD_sum = np.zeros((B, B), dtype=np.float64)
        self.DR_sum = np.zeros((B, B), dtype=np.float64)
        self.RD_sum = np.zeros((B, B), dtype=np.float64)
        self.RR_sum = np.zeros((B, B), dtype=np.float64)

        # Accumulate columns per photometric bin j
        for n in range(self.num_nodes):
            j = int(self.current_bins[n])  # photometric bin (column)
            self.DD_sum[:, j] += self.dd[:, n]
            self.DR_sum[:, j] += self.dr[:, n]
            self.RD_sum[:, j] += self.rd[:, n]
            self.RR_sum[:, j] += self.rr[:, n]

        # reference-side scale proxy (from dd mass)
        self._n_ref_proxy = self.dd.sum(axis=1).astype(np.float64) + 1e-12

    # ---------- properties ----------
    @property
    def grid_shape(self) -> Tuple[int, int]:
        return self._grid_shape

    @property
    def H(self) -> int:
        return self._grid_shape[0]

    @property
    def W(self) -> int:
        return self._grid_shape[1]

    # ---------- getters (grid-shaped or flat) ----------
    def get_assigned_bins(self, grid: bool = True) -> np.ndarray:
        if not grid:
            return self.current_bins
        A = np.full(self._grid_shape, -1, dtype=np.int32)
        A[self._xy[:, 0], self._xy[:, 1]] = self.current_bins
        return A

    def get_node_counts_total(self, grid: bool = True) -> np.ndarray:
        if not grid:
            return self.node_gal_counts
        C = np.zeros(self._grid_shape, dtype=np.int64)
        C[self._xy[:, 0], self._xy[:, 1]] = self.node_gal_counts
        return C

    def get_node_umatrix(self) -> np.ndarray:
        return self._umatrix

    # Placeholders to keep env code unified (return zeros if not provided)
    def get_photoz_bin_mode(self) -> np.ndarray:
        # Not used by our reward; provided for compatibility with earlier env code.
        A = self.get_assigned_bins(grid=True)
        return A  # using assigned bin as a stand-in

    def get_truez_bin_mode(self) -> np.ndarray:
        # If you later compute true-z per-node modes, wire them here.
        return np.zeros(self._grid_shape, dtype=np.int32)

    def get_node_counts_spec(self) -> np.ndarray:
        # If you have per-node spec counts separately, expose them; else zeros.
        return np.zeros(self._grid_shape, dtype=np.int64)

    # ---------- validity/masks ----------
    def valid_bin_for_node(self, x: int, y: int, b: int) -> bool:
        idx = self.xy_to_index(x, y)
        return 0 <= b < self.num_bins and int(self.current_bins[idx]) != int(b)

    def valid_delta_for_node(self, x: int, y: int, d: int) -> bool:
        idx = self.xy_to_index(x, y)
        nb = int(self.current_bins[idx]) + int(d)
        return 0 <= nb < self.num_bins and d != 0

    # ---------- edit ops (O(K)) ----------
    def set_node_bin(self, x: int, y: int, new_bin: int) -> None:
        """Set a node’s bin (no wraparound)."""
        idx = self.xy_to_index(x, y)
        self._set_node_bin_by_index(idx, new_bin)

    def delta_node_bin(self, x: int, y: int, step: int) -> None:
        idx = self.xy_to_index(x, y)
        newb = int(self.current_bins[idx]) + int(step)
        if 0 <= newb < self.num_bins:
            self._set_node_bin_by_index(idx, newb)

    def batch_set_node_bins(self, updates: Dict[Tuple[int, int], int]) -> None:
        """Batch updates; still O(K*len(updates))."""
        for (x, y), b in updates.items():
            self._set_node_bin_by_index(self.xy_to_index(x, y), int(b))

    # Internal index-based setter with cache updates
    def _set_node_bin_by_index(self, node_idx: int, new_bin: int) -> None:
        node = int(node_idx)
        newb = int(new_bin)
        if not (0 <= newb < self.num_bins):
            return
        oldb = int(self.current_bins[node])
        if newb == oldb:
            return

        # rank-1 cache updates across reference bins
        self.DD_sum[:, oldb] -= self.dd[:, node]; self.DD_sum[:, newb] += self.dd[:, node]
        self.DR_sum[:, oldb] -= self.dr[:, node]; self.DR_sum[:, newb] += self.dr[:, node]
        self.RD_sum[:, oldb] -= self.rd[:, node]; self.RD_sum[:, newb] += self.rd[:, node]
        self.RR_sum[:, oldb] -= self.rr[:, node]; self.RR_sum[:, newb] += self.rr[:, node]

        # counts per photometric bin
        w = int(self.node_gal_counts[node])
        self.galaxy_bin_counts[oldb] -= w
        self.galaxy_bin_counts[newb] += w

        self.current_bins[node] = newb
        self._last_reward_val = None  # invalidate cached reward


    # ---------- reward deltas (no mutation) ----------
    def _xi_diag_from_raw(self, j: int, DDjj: float, DRjj: float, RDjj: float, RRjj: float, n_phot_j: float) -> float:
        """Compute xi[j,j] used in trace, matching total_reward_fast()."""
        jj = int(j)
        n_ref_j = float(self._n_ref_proxy[jj]) + 1e-12
        n_ph = float(n_phot_j) + 1e-12

        DDn = DDjj / (n_ref_j * n_ph)
        DRn = DRjj / (10.0 * n_ref_j * n_ph)
        RDn = RDjj / (10.0 * n_ref_j * n_ph)
        RRn = RRjj / (100.0 * n_ref_j * n_ph)

        if RRn > 0:
            return float((DDn - DRn - RDn + RRn) / RRn)
        return 0.0

    def delta_reward_for_move(self, node_idx: int, new_bin: int) -> float:
        """Exact delta in trace reward if `node_idx` were moved to `new_bin`, without mutating state."""
        node = int(node_idx)
        nb = int(new_bin)
        if node < 0 or node >= self.num_nodes:
            return 0.0
        if nb < 0 or nb >= self.num_bins:
            return 0.0

        oldb = int(self.current_bins[node])
        if nb == oldb:
            return 0.0

        # Current n_phot (photometric bin counts) for affected bins only.
        n_phot_o = float(self.galaxy_bin_counts[oldb])
        n_phot_n = float(self.galaxy_bin_counts[nb])
        w = float(self.node_gal_counts[node])

        # Current diagonal raw sums
        DD_oo = float(self.DD_sum[oldb, oldb]); DR_oo = float(self.DR_sum[oldb, oldb]); RD_oo = float(self.RD_sum[oldb, oldb]); RR_oo = float(self.RR_sum[oldb, oldb])
        DD_nn = float(self.DD_sum[nb, nb]);     DR_nn = float(self.DR_sum[nb, nb]);     RD_nn = float(self.RD_sum[nb, nb]);     RR_nn = float(self.RR_sum[nb, nb])

        # Before
        xi_oo_before = self._xi_diag_from_raw(oldb, DD_oo, DR_oo, RD_oo, RR_oo, n_phot_o)
        xi_nn_before = self._xi_diag_from_raw(nb,   DD_nn, DR_nn, RD_nn, RR_nn, n_phot_n)

        # Apply hypothetical move to diagonals and counts
        DD_oo_after = DD_oo - float(self.dd[oldb, node])
        DR_oo_after = DR_oo - float(self.dr[oldb, node])
        RD_oo_after = RD_oo - float(self.rd[oldb, node])
        RR_oo_after = RR_oo - float(self.rr[oldb, node])

        DD_nn_after = DD_nn + float(self.dd[nb, node])
        DR_nn_after = DR_nn + float(self.dr[nb, node])
        RD_nn_after = RD_nn + float(self.rd[nb, node])
        RR_nn_after = RR_nn + float(self.rr[nb, node])

        xi_oo_after = self._xi_diag_from_raw(oldb, DD_oo_after, DR_oo_after, RD_oo_after, RR_oo_after, n_phot_o - w)
        xi_nn_after = self._xi_diag_from_raw(nb,   DD_nn_after, DR_nn_after, RD_nn_after, RR_nn_after, n_phot_n + w)

        return (xi_oo_after - xi_oo_before) + (xi_nn_after - xi_nn_before)

    def apply_move(self, node_idx: int, new_bin: int) -> None:
        """Apply a node->bin change (mutates). Alias for internal rank-1 update."""
        self._set_node_bin_by_index(node_idx, new_bin)
    # ---------- reward ----------
    def total_reward_fast(self) -> float:
        """Fast TreeCorr-style trace; your “only” reward."""
        n_phot = self.galaxy_bin_counts.astype(np.float64) + 1e-12   # (phot_bin,)
        n_ref  = self._n_ref_proxy.astype(np.float64) + 1e-12        # (ref_bin,)

        DDn = self.DD_sum / (n_ref[:, None] * n_phot[None, :])
        DRn = self.DR_sum / (10.0 * n_ref[:, None] * n_phot[None, :])
        RDn = self.RD_sum / (10.0 * n_ref[:, None] * n_phot[None, :])
        RRn = self.RR_sum / (10.0 * 10.0 * n_ref[:, None] * n_phot[None, :])

        with np.errstate(divide='ignore', invalid='ignore'):
            xi = np.divide(DDn - DRn - RDn + RRn, RRn, out=np.zeros_like(RRn), where=RRn > 0)
        val = float(np.trace(xi))
        self._last_reward_val = val
        return val

    # For API compatibility if you want to call env.som.reward_full()
    def reward_full(self) -> float:
        return self.total_reward_fast()

    def reward_proxy(self) -> float:
        return self.total_reward_fast()

    # ---------- episode control ----------
    def reset_to_initial(self) -> None:
        self.current_bins = self.init_bins.copy()
        self._reset_caches()
        self._last_reward_val = None

    def randomize_state(self, severity: float = 0.0) -> None:
        """
        Randomly alter a fraction of nodes to a random *valid* bin.
        severity in [0,1]: expected fraction of nodes changed.
        """
        if severity <= 0:
            return
        m = min(self.num_nodes, int(round(severity * self.num_nodes)))
        if m <= 0:
            return
        idxs = self.rng.choice(self.num_nodes, size=m, replace=False)
        for n in idxs:
            cur = int(self.current_bins[n])
            # pick any bin != cur
            choices = np.arange(self.num_bins, dtype=np.int32)
            if self.num_bins > 1:
                choices = choices[choices != cur]
            nb = int(self.rng.choice(choices))
            # node index → (x,y) is available via _xy
            x, y = int(self._xy[n, 0]), int(self._xy[n, 1])
            self.set_node_bin(x, y, nb)

    # ---------- layout helpers ----------
    def xy_to_index(self, x: int, y: int) -> int:
        """Map (x,y) to node index under our stored layout."""
        # If index_to_xy was passed, we stored only forward map; build reverse lazily:
        if not hasattr(self, "_index_lut"):
            H, W = self._grid_shape
            lut = -np.ones((H, W), dtype=np.int64)
            lut[self._xy[:, 0], self._xy[:, 1]] = np.arange(self.num_nodes, dtype=np.int64)
            self._index_lut = lut
        return int(self._index_lut[x, y])

    # ---------- snapshots (for vectorized multi-env) ----------
    def get_state(self) -> np.ndarray:
        """Return a copy of the current per-node bin vector (flat)."""
        return self.current_bins.copy()

    def set_state(self, bins_flat: np.ndarray) -> None:
        """Restore from a per-node bin vector (flat)."""
        bins_flat = np.asarray(bins_flat, dtype=np.int32)
        if bins_flat.shape != (self.num_nodes,):
            raise ValueError(f"state must be shape ({self.num_nodes},), got {bins_flat.shape}")
        self.current_bins = bins_flat.copy()
        self._reset_caches()
        self._last_reward_val = None


# ============================== TESTS ==============================
if __name__ == "__main__":
    # Synthetic, tiny K×N test to validate cache updates & API surface.
    rng = np.random.default_rng(0)
    K = 4
    H, W = 5, 6
    N = H * W

    # Fake node mapping for photometric sample (randomly assign 50k phot galaxies to nodes)
    n_phot = 50_000
    node_indices_photo = rng.integers(0, N, size=n_phot, dtype=np.int64)

    # Initial bins
    init_bins = rng.integers(0, K, size=N, dtype=np.int32)

    # Correlation “columns” (K × N); keep positive values to avoid divide-by-zero
    dd = rng.random((K, N)).astype(np.float32) * 100 + 1e-3
    dr = rng.random((K, N)).astype(np.float32) * 10  + 1e-3
    rd = rng.random((K, N)).astype(np.float32) * 10  + 1e-3
    rr = rng.random((K, N)).astype(np.float32) * 50  + 1e-3

    # Optional U-matrix just to exercise the getter
    U = rng.random((H, W)).astype(np.float32)

    som = SOM(
        node_indices_photo=node_indices_photo,
        initial_node_bins=init_bins,
        dd=dd, dr=dr, rd=rd, rr=rr,
        num_bins=K,
        grid_shape=(H, W),
        umatrix=U,
        seed=123,
    )

    # 1) Basic reward reproducibility
    r0 = som.total_reward_fast()
    r1 = som.total_reward_fast()
    assert np.isfinite(r0) and abs(r0 - r1) < 1e-12, "Reward should be deterministic without edits."

    # 2) Single edit equals batch edit of size 1
    x, y = 2, 3
    cur = som.get_assigned_bins(grid=True)[x, y]
    newb = (cur + 1) if cur + 1 < K else (cur - 1)
    s2 = SOM(node_indices_photo, init_bins.copy(), dd, dr, rd, rr, num_bins=K, grid_shape=(H, W), umatrix=U, seed=42)
    r_before = som.total_reward_fast()
    som.set_node_bin(x, y, newb)
    r_after_single = som.total_reward_fast()

    s2.batch_set_node_bins({(x, y): newb})
    r_after_batch = s2.total_reward_fast()
    assert abs(r_after_single - r_after_batch) < 1e-9, "Single vs batch update mismatch."

    # 3) Delta edit bounds check (no wrap)
    x2, y2 = 0, 0
    cur2 = som.get_assigned_bins(grid=True)[x2, y2]
    if cur2 == 0:
        som.delta_node_bin(x2, y2, -1)  # should no-op
        assert som.get_assigned_bins(grid=True)[x2, y2] == cur2, "Lower-bound delta should be masked."
    if cur2 == K - 1:
        som.delta_node_bin(x2, y2, +1)  # should no-op
        assert som.get_assigned_bins(grid=True)[x2, y2] == cur2, "Upper-bound delta should be masked."

    # 4) Snapshot/restore consistency  ✅ independent of earlier edits
    som.reset_to_initial()
    r_base = som.total_reward_fast()   # reward of the baseline state
    state = som.get_state()

    # Do a couple of random valid edits
    for _ in range(5):
        xi, yi = int(rng.integers(0, H)), int(rng.integers(0, W))
        curb = som.get_assigned_bins(grid=True)[xi, yi]
        choices = np.arange(K)[np.arange(K) != curb]
        som.set_node_bin(xi, yi, int(rng.choice(choices)))

    r_changed = som.total_reward_fast()

    # Restore and verify we recover r_base (the baseline snapshot)
    som.set_state(state)
    r_restored = som.total_reward_fast()
    assert np.allclose(r_restored, r_base, rtol=1e-12, atol=1e-9), \
        "Restored state should reproduce previous reward."


    # 5) Randomize severity sanity
    som.reset_to_initial()
    r_base = som.total_reward_fast()
    som.randomize_state(severity=0.05)  # ~5% nodes move
    r_rand = som.total_reward_fast()
    assert np.isfinite(r_rand), "Randomized reward should be finite."

    print("[OK] SOM class tests passed.",
          f"grid={som.grid_shape}, nodes={som.num_nodes}, bins={som.num_bins}, r0={r0:.6f}, r_rand={r_rand:.6f}")
