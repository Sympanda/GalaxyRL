# som_rl.py
from __future__ import annotations
import numpy as np
from gymnasium import Env, spaces

# ----------------- SOM (fast TC) with Δ-predictor -----------------

class SOM:
    """
    Same fast-TreeCorr core + incremental caches, now with:
      - estimate_delta_trace(node_idx, new_bin): O(1)
      - estimate_delta_trace_bins_for_node(node_idx): O(K)
    Public methods unchanged from earlier drop (set_node_bin, delta_node_bin, reward, etc.).
    """

    def __init__(
        self,
        node_indices_photo: np.ndarray,         # (N_phot,)
        initial_node_bins: np.ndarray,          # (N_nodes,)
        dd: np.ndarray, dr: np.ndarray, rd: np.ndarray, rr: np.ndarray,  # (K, N_nodes)
        num_bins: int = 11,
        grid_shape: tuple[int, int] | None = None,
        index_to_xy: np.ndarray | None = None,
        umatrix: np.ndarray | None = None,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.num_bins = int(num_bins)

        # core state
        self.current_bins = np.asarray(initial_node_bins, dtype=np.int32)
        self.init_bins = self.current_bins.copy()
        self.num_nodes = int(self.current_bins.size)

        self.node_indices = np.asarray(node_indices_photo, dtype=np.int64)
        self.node_gal_counts = np.bincount(self.node_indices, minlength=self.num_nodes).astype(np.int64)

        self.dd = np.asarray(dd, dtype=np.float32)
        self.dr = np.asarray(dr, dtype=np.float32)
        self.rd = np.asarray(rd, dtype=np.float32)
        self.rr = np.asarray(rr, dtype=np.float32)

        # layout
        if index_to_xy is not None:
            xy = np.asarray(index_to_xy, dtype=np.int64)
            self._xy = xy
            H = int(xy[:, 0].max()) + 1
            W = int(xy[:, 1].max()) + 1
            self._grid_shape = (H, W)
        else:
            if grid_shape is None:
                r = int(round(self.num_nodes ** 0.5))
                self._grid_shape = (r, r) if r * r == self.num_nodes else (self.num_nodes, 1)
            else:
                self._grid_shape = tuple(grid_shape)
            H, W = self._grid_shape
            xs = np.arange(self.num_nodes) // W
            ys = np.arange(self.num_nodes) % W
            self._xy = np.stack([xs, ys], axis=1).astype(np.int64)

        if umatrix is not None:
            U = np.asarray(umatrix, dtype=np.float32)
            if tuple(U.shape) != self._grid_shape:
                raise ValueError("umatrix shape must equal grid_shape")
            self._umatrix = U
        else:
            self._umatrix = np.zeros(self._grid_shape, dtype=np.float32)

        # caches
        self._reset_caches()
        self._last_reward_val: float | None = None

    # ----- caches -----
    def _reset_caches(self):
        B = self.num_bins
        self.galaxy_bin_counts = np.bincount(
            self.current_bins, weights=self.node_gal_counts, minlength=B
        ).astype(np.float64)

        self.DD_sum = np.zeros((B, B), dtype=np.float64)
        self.DR_sum = np.zeros((B, B), dtype=np.float64)
        self.RD_sum = np.zeros((B, B), dtype=np.float64)
        self.RR_sum = np.zeros((B, B), dtype=np.float64)

        for n in range(self.num_nodes):
            j = int(self.current_bins[n])
            self.DD_sum[:, j] += self.dd[:, n]
            self.DR_sum[:, j] += self.dr[:, n]
            self.RD_sum[:, j] += self.rd[:, n]
            self.RR_sum[:, j] += self.rr[:, n]

        self._n_ref_proxy = self.dd.sum(axis=1).astype(np.float64) + 1e-12

    # ----- grid helpers -----
    @property
    def grid_shape(self) -> tuple[int, int]:
        return self._grid_shape

    def xy_to_index(self, x: int, y: int) -> int:
        if not hasattr(self, "_index_lut"):
            H, W = self._grid_shape
            lut = -np.ones((H, W), dtype=np.int64)
            lut[self._xy[:, 0], self._xy[:, 1]] = np.arange(self.num_nodes, dtype=np.int64)
            self._index_lut = lut
        return int(self._index_lut[x, y])

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

    # placeholders to match earlier env API
    def get_photoz_bin_mode(self) -> np.ndarray:
        return self.get_assigned_bins(grid=True)

    def get_truez_bin_mode(self) -> np.ndarray:
        return np.zeros(self._grid_shape, dtype=np.int32)

    def get_node_counts_spec(self) -> np.ndarray:
        return np.zeros(self._grid_shape, dtype=np.int64)

    # ----- validity -----
    def valid_bin_for_node(self, x: int, y: int, b: int) -> bool:
        idx = self.xy_to_index(x, y)
        return 0 <= b < self.num_bins and int(self.current_bins[idx]) != int(b)

    def valid_delta_for_node(self, x: int, y: int, d: int) -> bool:
        idx = self.xy_to_index(x, y)
        nb = int(self.current_bins[idx]) + int(d)
        return 0 <= nb < self.num_bins and d != 0

    # ----- edits (O(K)) -----
    def _apply_move_index(self, node: int, newb: int):
        oldb = int(self.current_bins[node])
        if newb == oldb:
            return
        # update sums (rank-1)
        self.DD_sum[:, oldb] -= self.dd[:, node]; self.DD_sum[:, newb] += self.dd[:, node]
        self.DR_sum[:, oldb] -= self.dr[:, node]; self.DR_sum[:, newb] += self.dr[:, node]
        self.RD_sum[:, oldb] -= self.rd[:, node]; self.RD_sum[:, newb] += self.rd[:, node]
        self.RR_sum[:, oldb] -= self.rr[:, node]; self.RR_sum[:, newb] += self.rr[:, node]
        # update counts
        w = int(self.node_gal_counts[node])
        self.galaxy_bin_counts[oldb] -= w
        self.galaxy_bin_counts[newb] += w
        # commit
        self.current_bins[node] = newb
        self._last_reward_val = None

    def set_node_bin(self, x: int, y: int, new_bin: int) -> None:
        idx = self.xy_to_index(x, y)
        if 0 <= new_bin < self.num_bins and new_bin != int(self.current_bins[idx]):
            self._apply_move_index(idx, int(new_bin))

    def delta_node_bin(self, x: int, y: int, step: int) -> None:
        idx = self.xy_to_index(x, y)
        nb = int(self.current_bins[idx]) + int(step)
        if 0 <= nb < self.num_bins:
            self._apply_move_index(idx, nb)

    def batch_set_node_bins(self, updates: dict[tuple[int,int], int]) -> None:
        for (x, y), b in updates.items():
            idx = self.xy_to_index(x, y)
            if 0 <= b < self.num_bins and b != int(self.current_bins[idx]):
                self._apply_move_index(idx, int(b))

    # ----- reward -----
    def _xi_diag(self, DDjj, DRjj, RDjj, RRjj, nref_j, nphot_j) -> float:
        # Normalize like in the fast reward (DR,RD divide by 10, RR by 100)
        den = (nref_j * nphot_j)
        DDn = DDjj / den
        DRn = DRjj / (10.0 * den)
        RDn = RDjj / (10.0 * den)
        RRn = RRjj / (100.0 * den)
        # xi = (DDn - DRn - RDn + RRn) / RRn
        if RRn <= 0:
            return 0.0
        return float((DDn - DRn - RDn + RRn) / RRn)

    def total_reward_fast(self) -> float:
        n_phot = self.galaxy_bin_counts + 1e-12
        n_ref = self._n_ref_proxy
        # Compute xi diag via vector ops (still cheap at K<=11)
        DDn = self.DD_sum.diagonal() / (n_ref * n_phot)
        DRn = self.DR_sum.diagonal() / (10.0 * n_ref * n_phot)
        RDn = self.RD_sum.diagonal() / (10.0 * n_ref * n_phot)
        RRn = self.RR_sum.diagonal() / (100.0 * n_ref * n_phot)
        with np.errstate(divide='ignore', invalid='ignore'):
            xi_diag = np.divide(DDn - DRn - RDn + RRn, RRn, out=np.zeros_like(RRn), where=RRn > 0)
        val = float(xi_diag.sum())
        self._last_reward_val = val
        return val

    reward_full = total_reward_fast
    reward_proxy = total_reward_fast

    # ----- Δ-predictor -----
    def estimate_delta_trace(self, node_idx: int, new_bin: int) -> float:
        """O(1) ΔTrace for moving node_idx to new_bin (no state change)."""
        node = int(node_idx)
        newb = int(new_bin)
        oldb = int(self.current_bins[node])
        if newb == oldb or not (0 <= newb < self.num_bins):
            return 0.0

        w = float(self.node_gal_counts[node])
        if w <= 0:
            return 0.0

        # current diagonal components
        DD_old = self.DD_sum[oldb, oldb]; DR_old = self.DR_sum[oldb, oldb]
        RD_old = self.RD_sum[oldb, oldb]; RR_old = self.RR_sum[oldb, oldb]
        DD_new = self.DD_sum[newb, newb]; DR_new = self.DR_sum[newb, newb]
        RD_new = self.RD_sum[newb, newb]; RR_new = self.RR_sum[newb, newb]

        # per-node row contributions (only the two needed rows)
        dd_oldrow = float(self.dd[oldb, node]); dr_oldrow = float(self.dr[oldb, node])
        rd_oldrow = float(self.rd[oldb, node]); rr_oldrow = float(self.rr[oldb, node])
        dd_newrow = float(self.dd[newb, node]); dr_newrow = float(self.dr[newb, node])
        rd_newrow = float(self.rd[newb, node]); rr_newrow = float(self.rr[newb, node])

        # n_ref (fixed) and n_phot updates
        nref_old = float(self._n_ref_proxy[oldb])
        nref_new = float(self._n_ref_proxy[newb])
        nphot_old = float(self.galaxy_bin_counts[oldb]); nphot_old_p = nphot_old - w
        nphot_new = float(self.galaxy_bin_counts[newb]); nphot_new_p = nphot_new + w
        # guard
        if nphot_old_p <= 0 or nphot_new_p <= 0:
            # moving the last galaxy out of a bin would make normalization degenerate; predictor says "very bad"
            return -1e9

        # previous xi
        xi_old_prev = self._xi_diag(DD_old, DR_old, RD_old, RR_old, nref_old, nphot_old)
        xi_new_prev = self._xi_diag(DD_new, DR_new, RD_new, RR_new, nref_new, nphot_new)
        # after-move xi (update sums & counts)
        xi_old_after = self._xi_diag(
            DD_old - dd_oldrow, DR_old - dr_oldrow, RD_old - rd_oldrow, RR_old - rr_oldrow,
            nref_old, nphot_old_p
        )
        xi_new_after = self._xi_diag(
            DD_new + dd_newrow, DR_new + dr_newrow, RD_new + rd_newrow, RR_new + rr_newrow,
            nref_new, nphot_new_p
        )
        return (xi_old_after - xi_old_prev) + (xi_new_after - xi_new_prev)

    def estimate_delta_trace_bins_for_node(self, node_idx: int) -> np.ndarray:
        """O(K) vector of ΔTrace for all bins for a given node (no state change)."""
        K = self.num_bins
        out = np.zeros(K, dtype=np.float64)
        for b in range(K):
            out[b] = self.estimate_delta_trace(node_idx, b)
        return out

    # ----- episode control & snapshots -----
    def reset_to_initial(self) -> None:
        self.current_bins = self.init_bins.copy()
        self._reset_caches()
        self._last_reward_val = None

    def randomize_state(self, severity: float = 0.0) -> None:
        if severity <= 0:
            return
        m = min(self.num_nodes, int(round(severity * self.num_nodes)))
        if m <= 0:
            return
        idxs = self.rng.choice(self.num_nodes, size=m, replace=False)
        for node in idxs:
            cur = int(self.current_bins[node])
            choices = np.arange(self.num_bins, dtype=np.int32)
            if self.num_bins > 1:
                choices = choices[choices != cur]
            nb = int(self.rng.choice(choices))
            self._apply_move_index(node, nb)

    def get_state(self) -> np.ndarray:
        return self.current_bins.copy()

    def set_state(self, bins_flat: np.ndarray) -> None:
        bins_flat = np.asarray(bins_flat, dtype=np.int32)
        if bins_flat.shape != (self.num_nodes,):
            raise ValueError("bad state shape")
        self.current_bins = bins_flat.copy()
        self._reset_caches()
        self._last_reward_val = None


# ----------------- GlobalPointerTwoStageEnv -----------------

class GlobalPointerTwoStageEnv(Env):
    """
    Phase 1: SELECT NODE  (action in [0..N-1])
    Phase 2: SELECT BIN   (action in [0..K-1])  -> applies change, returns Δreward

    - Single Discrete action space of size max(N, K); we expose a mask per phase.
    - Uses SOM.estimate_delta_trace_bins_for_node() to mask down to top-M bins.
    - Cooldown (tabu) for (node,bin) with negative Δ; neighbor unmask halves cooldown.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        som: SOM,
        top_M_bins: int = 3,
        cooldown_steps: int = 12,
        neg_eps: float = 0.0,         # treat <= this as "negative"
        epsilon_unmask: float = 0.05, # exploration
        neighbor_mode: str = "4",     # "4" or "8"
        seed: int | None = None,
        max_steps: int = 200000,
    ):
        super().__init__()
        self.som = som
        self.H, self.W = som.grid_shape
        self.N = self.H * self.W
        self.K = som.num_bins
        self.top_M = max(1, int(top_M_bins))
        self.cool_T = int(cooldown_steps)
        self.neg_eps = float(neg_eps)
        self.eps_unmask = float(epsilon_unmask)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(max(self.N, self.K))
        # observation: full grid channels (assigned, photo-mode, true-mode, delta, U, n_total, n_spec)
        self._obs_channels = 7
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(self.H, self.W, self._obs_channels), dtype=np.float32
        )

        # caches for observation
        self._norm_counts = max(1.0, float(np.max(self.som.get_node_counts_total(grid=True))))
        self._obs_grid = self._build_obs()

        # two-stage state
        self.phase = "node"
        self.selected_node = None  # flat index (0..N-1)
        self.steps = 0

        # cooldown storage as "expires at step t" (no per-step decrements)
        self._cool_until = np.zeros((self.N, self.K), dtype=np.int64)
        # neighbor graph
        self._neighbors = self._build_neighbors(mode=neighbor_mode)
        # version counter per node for neighbor-triggered unmasking
        self._node_version = np.zeros(self.N, dtype=np.int64)

    # ---- observation build/update ----
    def _build_obs(self) -> np.ndarray:
        A = self.som.get_assigned_bins(grid=True).astype(np.float32)
        P = self.som.get_photoz_bin_mode().astype(np.float32)
        T = self.som.get_truez_bin_mode().astype(np.float32)
        U = self.som.get_node_umatrix().astype(np.float32)
        Nt = self.som.get_node_counts_total(grid=True).astype(np.float32) / self._norm_counts
        Ns = self.som.get_node_counts_spec().astype(np.float32)
        # normalize
        K1 = max(1.0, self.K - 1.0)
        A /= K1; P /= K1; T /= K1
        if np.std(U) > 0:
            U = (U - np.mean(U)) / (np.std(U) + 1e-6)
        D = (A - T).astype(np.float32)
        return np.stack([A, P, T, D, U, Nt, Ns], axis=-1).astype(np.float32)

    def _update_cell_in_obs(self, node_idx: int):
        x = int(node_idx // self.W); y = int(node_idx % self.W)
        # Only channels depending on assigned bin and delta need updates
        bin_val = float(self.som.get_assigned_bins(grid=True)[x, y]) / max(1.0, self.K - 1.0)
        self._obs_grid[x, y, 0] = bin_val           # A
        self._obs_grid[x, y, 1] = bin_val           # P (stand-in)
        self._obs_grid[x, y, 3] = bin_val - self._obs_grid[x, y, 2]  # D = A - T

    # ---- neighbors (for cooldown halving) ----
    def _build_neighbors(self, mode="4"):
        nbrs = [[] for _ in range(self.N)]
        for i in range(self.N):
            x, y = divmod(i, self.W)
            for dx, dy in ([(-1,0),(1,0),(0,-1),(0,1)] if mode=="4"
                           else [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]):
                xx, yy = x+dx, y+dy
                if 0 <= xx < self.H and 0 <= yy < self.W:
                    nbrs[i].append(xx*self.W + yy)
        return [np.array(v, dtype=np.int64) for v in nbrs]

    # ---- masks ----
    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        if self.phase == "node":
            # mask valid node choices (not expired cooldown entirely; node cooldown is per-bin, so allow node)
            mask[:self.N] = True
        else:
            # Bin phase: start with impossible bins masked
            mask[:self.K] = True
            node = self.selected_node
            cur_bin = int(self.som.current_bins[node])
            mask[cur_bin] = False  # no-op forbidden
            # cooldown mask
            now = self.steps
            cool = self._cool_until[node, :self.K]
            mask[:self.K] &= (cool <= now)

            # Δ-predictor top-M gating (keep a few best + eps)
            deltas = self.som.estimate_delta_trace_bins_for_node(node)
            # Always allow any with positive predicted Δ
            positive = deltas > 0.0
            if positive.any():
                keep = positive
            else:
                # keep top-M (even if negative)
                top_idx = np.argpartition(deltas, -self.top_M)[-self.top_M:]
                keep = np.zeros(self.K, dtype=bool); keep[top_idx] = True
            # apply ε-unmask randomly among currently masked bins
            if self.eps_unmask > 0:
                masked = (~keep) & mask[:self.K]
                if masked.any() and self.rng.random() < self.eps_unmask:
                    j = int(self.rng.choice(np.flatnonzero(masked)))
                    keep[j] = True
            mask[:self.K] &= keep

        return mask

    # ---- gym api ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.som.reset_to_initial()
        if options and options.get("randomize", 0) > 0:
            self.som.randomize_state(options["randomize"])
        self._obs_grid = self._build_obs()
        self.phase = "node"; self.selected_node = None
        self.steps = 0
        return self._obs_grid, {"phase": self.phase}

    def step(self, action: int):
        self.steps += 1
        info = {"phase": self.phase}
        reward = 0.0
        terminated = False
        truncated = self.steps >= self.max_steps

        if self.phase == "node":
            node = int(action)
            if not (0 <= node < self.N):
                node = int(np.clip(node, 0, self.N-1))
            self.selected_node = node
            self.phase = "bin"
            info["selected_node"] = node
            return self._obs_grid, reward, terminated, truncated, info

        # phase == "bin"
        b = int(action)
        node = int(self.selected_node)
        cur = int(self.som.current_bins[node])
        if 0 <= b < self.K and b != cur:
            # exact Δreward for learning
            r_before = self.som.total_reward_fast()
            # apply move
            x, y = divmod(node, self.W)
            self.som.set_node_bin(x, y, b)
            r_after = self.som.total_reward_fast()
            reward = float(r_after - r_before)

            # update obs cache
            self._update_cell_in_obs(node)

            # cooldown: negative (or <=neg_eps) gets tabu
            if reward <= self.neg_eps:
                self._cool_until[node, b] = self.steps + self.cool_T

            # neighbor-triggered cooldown halving
            self._node_version[node] += 1
            for nb in self._neighbors[node]:
                # bring cooldowns closer to expiry
                future = self._cool_until[nb, :]
                active = future > self.steps
                future[active] = self.steps + ((future[active] - self.steps) // 2)
                self._cool_until[nb, :] = future

        # return to node phase
        self.phase = "node"; self.selected_node = None
        info["reward_abs"] = float(self.som.total_reward_fast())
        return self._obs_grid, reward, terminated, truncated, info

    def render(self):
        return f"phase={self.phase} steps={self.steps}"


# ----------------- ConveyorSweepEnv -----------------

class ConveyorSweepEnv(Env):
    """
    Deterministic order over nodes (row-major + random rotations/flips).
    At each step, the agent picks an ABSOLUTE bin for the current node; env advances pointer.
    Supports multi-pass episodes (coarse -> fine).
    Uses Δ-predictor to mask to top-M + epsilon.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        som: SOM,
        passes: int = 2,
        top_M_bins: int = 3,
        cooldown_steps: int = 12,
        neg_eps: float = 0.0,
        epsilon_unmask: float = 0.05,
        seed: int | None = None,
    ):
        super().__init__()
        self.som = som
        self.H, self.W = som.grid_shape
        self.N = self.H * self.W
        self.K = som.num_bins
        self.passes = int(passes)
        self.top_M = int(top_M_bins)
        self.cool_T = int(cooldown_steps)
        self.neg_eps = float(neg_eps)
        self.eps_unmask = float(epsilon_unmask)
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(self.K)
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(self.H, self.W, 7), dtype=np.float32
        )
        self._norm_counts = max(1.0, float(np.max(self.som.get_node_counts_total(grid=True))))
        self._obs_grid = self._build_obs()

        self.order: np.ndarray | None = None
        self.ptr = 0
        self.pass_no = 0
        self._cool_until = np.zeros((self.N, self.K), dtype=np.int64)

    def _build_obs(self):
        A = self.som.get_assigned_bins(grid=True).astype(np.float32)
        P = self.som.get_photoz_bin_mode().astype(np.float32)
        T = self.som.get_truez_bin_mode().astype(np.float32)
        U = self.som.get_node_umatrix().astype(np.float32)
        Nt = self.som.get_node_counts_total(grid=True).astype(np.float32) / self._norm_counts
        Ns = self.som.get_node_counts_spec().astype(np.float32)
        K1 = max(1.0, self.K - 1.0)
        A /= K1; P /= K1; T /= K1
        if np.std(U) > 0:
            U = (U - np.mean(U)) / (np.std(U) + 1e-6)
        D = (A - T).astype(np.float32)
        return np.stack([A, P, T, D, U, Nt, Ns], axis=-1).astype(np.float32)

    def _make_order(self):
        # Row-major base, with random flip/rotation for variety
        order = np.arange(self.N, dtype=np.int64).reshape(self.H, self.W)
        # random transforms
        if self.rng.random() < 0.5: order = np.flipud(order)
        if self.rng.random() < 0.5: order = np.fliplr(order)
        k = self.rng.integers(0, 4)
        order = np.rot90(order, k)
        return order.ravel()

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.K, dtype=bool)
        node = int(self.order[self.ptr])
        cur = int(self.som.current_bins[node])
        mask[:] = True
        mask[cur] = False  # no-op forbidden
        # cooldown
        now = self.ptr + self.pass_no * self.N
        cool = self._cool_until[node, :]
        mask &= (cool <= now)
        # Δ predictor top-M
        deltas = self.som.estimate_delta_trace_bins_for_node(node)
        positive = deltas > 0.0
        if positive.any():
            keep = positive
        else:
            top_idx = np.argpartition(deltas, -self.top_M)[-self.top_M:]
            keep = np.zeros(self.K, dtype=bool); keep[top_idx] = True
        # epsilon unmask
        if self.eps_unmask > 0:
            masked = (~keep) & mask
            if masked.any() and self.rng.random() < self.eps_unmask:
                j = int(self.rng.choice(np.flatnonzero(masked)))
                keep[j] = True
        mask &= keep
        return mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.som.reset_to_initial()
        if options and options.get("randomize", 0) > 0:
            self.som.randomize_state(options["randomize"])
        self._obs_grid = self._build_obs()
        self.order = self._make_order()
        self.ptr = 0
        self.pass_no = 0
        return self._obs_grid, {"node": int(self.order[0]), "pass": self.pass_no}

    def step(self, action: int):
        node = int(self.order[self.ptr])
        cur = int(self.som.current_bins[node])
        reward = 0.0
        terminated = False
        truncated = False
        info = {"node": node, "pass": self.pass_no}

        if 0 <= action < self.K and action != cur:
            r0 = self.som.total_reward_fast()
            x, y = divmod(node, self.W)
            self.som.set_node_bin(x, y, int(action))
            r1 = self.som.total_reward_fast()
            reward = float(r1 - r0)
            # cooldown on negative
            now = self.ptr + self.pass_no * self.N
            if reward <= 0.0:
                self._cool_until[node, action] = now + self.cool_T
            # update obs cell
            bin_val = float(self.som.get_assigned_bins(grid=True)[x, y]) / max(1.0, self.K - 1.0)
            self._obs_grid[x, y, 0] = bin_val
            self._obs_grid[x, y, 1] = bin_val
            self._obs_grid[x, y, 3] = bin_val - self._obs_grid[x, y, 2]

        # advance pointer
        self.ptr += 1
        if self.ptr >= self.N:
            self.ptr = 0
            self.pass_no += 1
            if self.pass_no >= self.passes:
                terminated = True

        return self._obs_grid, reward, terminated, truncated, info


# ----------------- Sanity tests for Δ-predictor -----------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K = 5; H, W = 8, 10; N = H*W
    n_phot = 60000
    node_idx_photo = rng.integers(0, N, size=n_phot)
    init_bins = rng.integers(0, K, size=N, dtype=np.int32)
    dd = rng.random((K, N)).astype(np.float32) * 100 + 1e-3
    dr = rng.random((K, N)).astype(np.float32) * 10  + 1e-3
    rd = rng.random((K, N)).astype(np.float32) * 10  + 1e-3
    rr = rng.random((K, N)).astype(np.float32) * 50  + 1e-3

    som = SOM(node_idx_photo, init_bins, dd, dr, rd, rr, num_bins=K, grid_shape=(H, W), seed=123)

    # Test Δ predictor matches exact Δ within numerical tolerance
    for _ in range(50):
        node = int(rng.integers(0, N))
        cur = int(som.current_bins[node])
        # choose a different bin
        choices = np.arange(K, dtype=int); choices = choices[choices != cur]
        newb = int(rng.choice(choices))
        x, y = divmod(node, W)

        # predicted
        d_pred = som.estimate_delta_trace(node, newb)

        # exact
        r0 = som.total_reward_fast()
        som.set_node_bin(x, y, newb)
        r1 = som.total_reward_fast()
        d_exact = r1 - r0

        assert np.allclose(d_pred, d_exact, rtol=1e-10, atol=1e-9), \
            f"Δ predictor mismatch: pred={d_pred}, exact={d_exact}"

        # undo (swap back)
        som.set_node_bin(x, y, cur)

    print("[OK] Δ-predictor sanity tests passed.")

'''
'''