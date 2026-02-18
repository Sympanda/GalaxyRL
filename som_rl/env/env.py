from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Literal, Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# schedulers live elsewhere; env only needs this minimal interface:
#   scheduler.reset()
#   scheduler.current_params() -> (level:int, T:float, floor:float)
#   scheduler.update(accepted:bool, dE:float, E:float, step_idx:int|None, is_noop:bool=False)

ObsMode = Literal["global", "patch"]
AcceptRule = Literal["sa", "positive_only"]


# ---------------------------
# Config helpers
# ---------------------------

@dataclass(frozen=True)
class EnergyFloorSchedule:
    """
    Level-dependent minimum +ΔE threshold required to accept a move.

    If accept_rule="sa":
      - Moves with dE < floor(level) are hard-rejected BEFORE SA.
      - Moves with dE >= floor(level) use standard SA accept/reject.
    If accept_rule="positive_only":
      - Accept only if dE >= floor(level) and dE > 0.

    Provide either:
      - table: dict[level] -> min_delta_E
      - or a callable f(level)->float
    """
    table: Optional[Dict[int, float]] = None
    func: Optional[Callable[[int], float]] = None
    default: float = 0.0

    def min_delta(self, level: int) -> float:
        if self.func is not None:
            return float(self.func(int(level)))
        if self.table is not None and int(level) in self.table:
            return float(self.table[int(level)])
        return float(self.default)


@dataclass(frozen=True)
class SOMEnergyEnvConfig:
    grid_shape: Tuple[int, int] = (200, 200)
    torus: bool = True

    # observation
    obs_mode: ObsMode = "patch"
    patch_size: int = 9  # odd
    include_umatrix: bool = True
    include_cluster_features: bool = True  # boundary + size, not raw ids
    include_empty_mask: bool = True

    # extra channels
    include_visit_count: bool = True
    include_recent_mask: bool = False
    recent_cooldown_steps: int = 50

    # cursor
    start_xy: Optional[Tuple[int, int]] = None

    # episode
    max_steps: int = 2000

    # accept/reject
    accept_rule: AcceptRule = "sa"
    energy_floor: EnergyFloorSchedule = EnergyFloorSchedule(default=0.0)

    # reward
    reward_type: Literal["deltaE", "rawE"] = "deltaE"
    clip_reward: Optional[float] = None
    step_penalty: float = 0.0

    # reward shaping / scaling
    # If True and reward_type == "deltaE", apply a tanh nonlinearity to dE
    # before scaling. This keeps rewards bounded and reduces sensitivity to
    # rare large |dE| jumps while still preserving sign and relative size.
    use_tanh_reward: bool = True
    # Characteristic scale for the tanh nonlinearity; roughly the |dE| value
    # where tanh(dE / reward_tanh_temperature) ~ 0.76.
    reward_tanh_temperature: float = 0.0001
    # Overall scale factor for the energy-based reward term. This replaces the
    # hard-coded "10 * dE" in the original implementation.
    reward_energy_scale: float = 1
    # Small penalties for unproductive moves. These are intended to be much
    # smaller than typical positive rewards so that exploration is not
    # discouraged, but repeated rejections/noops are still slightly costly.
    reject_penalty: float = -1e5
    noop_penalty: float = -1e5

    # Zelda-specific shaping (used only by EnvZelda in env_multi.py)
    # Small positive bonus for visiting less-visited nodes in the Zelda
    # archetype. Computed as beta * (1 - vc_norm) at the cursor/touched nodes,
    # where vc_norm is the visit count normalized to [0,1]. Set to 0.0 to
    # disable.
    zelda_explore_bonus: float = 0.0
    # Forced-interact behaviour in Zelda:
    #   - After at least zelda_force_interact_min_steps since the last
    #     interact, movement actions may be overridden with an interact at the
    #     current cursor position with a probability that increases with the
    #     number of steps since last interact.
    #   - zelda_force_interact_alpha controls how quickly this probability
    #     rises; set to 0.0 to disable.
    zelda_force_interact_min_steps: int = 0
    zelda_force_interact_alpha: float = 0.0

    # bin selection policy
    # If True, when the env needs to choose a bin for a specific node (i.e.
    # for action spaces that do NOT specify a bin explicitly), it will call
    # backend.greedy_bin_for_node(...) to pick the bin that maximises ΔE for
    # that node at the current level, instead of using a random-bin proposal.
    #
    # This is intended for hybrid policies where an external agent (e.g. PPO)
    # chooses the node/location and the backend chooses the bin greedily.
    use_greedy_bin_for_node: bool = False

    # scheduler integration
    # If scheduler is provided, it is the source of (level, temperature, floor).
    # If not, you can still set a fixed initial_level and fixed temperature in code.
    initial_level: int = 30
    initial_temperature: float = 1.0
    allowed_levels: Tuple[int, ...] = (30,)  # only for action spaces that choose level explicitly
    # If False, env.reset() will NOT call som.reset_to_initial() even if the
    # backend exposes that method. This is useful for multi-stage runs where
    # you want to reuse the same backend state across different schedulers or
    # level configurations without reinitialising the SOM.
    reset_backend_on_reset: bool = True

    # If False, the env will NOT terminate based on its own max_steps counter
    # (cfg.max_steps); only the scheduler's done() flag will control natural
    # episode termination. This is useful when you want the scheduler to be
    # the single source of truth for when an episode should end.
    use_env_max_steps: bool = True


# ---------------------------
# Utils
# ---------------------------

def _wrap(v: int, m: int) -> int:
    return int(v) % int(m)

def _clip(v: int, lo: int, hi: int) -> int:
    return int(np.clip(v, lo, hi))

def _extract_patch_torus(A: np.ndarray, x: int, y: int, ps: int) -> np.ndarray:
    H, W = A.shape
    r = ps // 2
    xs = (np.arange(x - r, x + r + 1) % W).astype(int)
    ys = (np.arange(y - r, y + r + 1) % H).astype(int)
    return A[np.ix_(ys, xs)]

def _extract_patch_clipped(A: np.ndarray, x: int, y: int, ps: int) -> np.ndarray:
    H, W = A.shape
    r = ps // 2
    x0 = _clip(x - r, 0, W - ps)
    y0 = _clip(y - r, 0, H - ps)
    return A[y0:y0 + ps, x0:x0 + ps]

def cluster_boundary_from_grid(cluster_grid: np.ndarray) -> np.ndarray:
    H, W = cluster_grid.shape
    b = np.zeros((H, W), dtype=np.uint8)
    b[:-1, :] |= (cluster_grid[:-1, :] != cluster_grid[1:, :])
    b[1:,  :] |= (cluster_grid[1:,  :] != cluster_grid[:-1, :])
    b[:, :-1] |= (cluster_grid[:, :-1] != cluster_grid[:, 1:])
    b[:, 1: ] |= (cluster_grid[:, 1: ] != cluster_grid[:, :-1])
    return b


# ---------------------------
# Base Env
# ---------------------------

class SOMEnergyBaseEnv(gym.Env):
    """
    Env responsibilities:
      - construct obs (global/patch + optional channels)
      - keep node-level stats (visit count, recent)
      - decode action -> target node (and optionally desired bin)
      - build proposal via backend (random-bin or set-bin)
      - evaluate proposal -> dE (backend applies temporarily)
      - accept/reject using accept_rule + scheduler-provided temperature/floor
      - revert if rejected
      - call scheduler.update(...) every step (if provided)

    Backend contract (methods used):
      - N, K, H, W (or infer from cfg)
      - current_bins: (N,)
      - node_gal_counts: (N,)
      - energy_E()
      - apply_proposal(proposal)->undo
      - revert(undo)
      - make_proposal_random_bin(node=..., level=..., rng=...)
      - make_proposal_set_bin(new_bin=..., node=..., level=...)
      - evaluate_proposal_deltaE(proposal)->dict with keys:
            is_noop, E0, E1, dE, undo, proposal
      - index_to_xy(idx), xy_to_index(x,y)
      - optional: umatrix_grid()
      - optional: cluster_grid(level) OR cluster_ids_at_level(level)
    """

    metadata = {"render_modes": []}

    def __init__(self, backend, cfg: SOMEnergyEnvConfig, scheduler: Optional[Any] = None):
        super().__init__()
        self.som = backend
        self.cfg = cfg
        self.scheduler = scheduler

        self.H, self.W = cfg.grid_shape
        if (self.H, self.W) != (200, 200):
            raise ValueError("Locked to 200x200 for this project.")
        if cfg.patch_size % 2 == 0:
            raise ValueError("patch_size must be odd.")
        self.ps = int(cfg.patch_size)
        self.r = self.ps // 2

        self._rng = np.random.default_rng(0)
        self._step = 0

        # schedule-driven params
        self.level = int(cfg.initial_level)
        self.temperature = float(cfg.initial_temperature)
        self.min_delta_E = float(cfg.energy_floor.min_delta(int(self.level)))

        # cursor
        self.cursor_x = 0
        self.cursor_y = 0

        # visit stats
        self.visit_count = np.zeros(self.H * self.W, dtype=np.int32)
        self.last_visit_step = np.full(self.H * self.W, -10**9, dtype=np.int32)

        # spaces
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

    # ---- spaces ----
    def _build_action_space(self) -> spaces.Space:
        raise NotImplementedError

    def _build_observation_space(self) -> spaces.Space:
        shape = (self.H, self.W) if self.cfg.obs_mode == "global" else (self.ps, self.ps)

        obs: Dict[str, spaces.Space] = {}
        # Reshape bins to be between 0 and 1 like all other channels (There will be no empty bins)
        obs["bins"] = spaces.Box(low=-1, high=1, shape=shape, dtype=np.float32)

        if self.cfg.include_empty_mask:
            obs["empty"] = spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)

        if self.cfg.include_umatrix and self._has_umatrix():
            obs["umatrix"] = spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)

        if self.cfg.include_cluster_features:
            obs["cluster_boundary"] = spaces.Box(low=0, high=1, shape=shape, dtype=np.uint8)
            obs["cluster_size"] = spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)

        if self.cfg.include_visit_count:
            obs["visit_count"] = spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)

        if self.cfg.include_recent_mask:
            obs["recent"] = spaces.Box(low=0, high=1, shape=shape, dtype=np.uint8)

        # scalars: [level_norm, T, E, steps_left_norm]
        obs["scalars"] = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        obs["cursor"] = spaces.Box(low=0, high=max(self.H, self.W), shape=(2,), dtype=np.int32)
        return spaces.Dict(obs)

    def _has_umatrix(self) -> bool:
        if hasattr(self.som, "umatrix_grid"):
            try:
                _ = self.som.umatrix_grid()
                return True
            except Exception:
                return False
        # legacy attribute
        return getattr(self.som, "umatrix", None) is not None

    def _get_umatrix(self) -> np.ndarray:
        if hasattr(self.som, "umatrix_grid"):
            return np.asarray(self.som.umatrix_grid(), dtype=np.float32)
        return np.asarray(self.som.umatrix, dtype=np.float32)

    # ---- reset ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed if seed is not None else 0)
        self._step = 0

        if self.cfg.reset_backend_on_reset and hasattr(self.som, "reset_to_initial"):
            self.som.reset_to_initial()

        self.visit_count[:] = 0
        self.last_visit_step[:] = -10**9

        if self.scheduler is not None:
            self.scheduler.reset()
            self._sync_scheduler_params()
        else:
            self.level = int(self.cfg.initial_level)
            self.temperature = float(self.cfg.initial_temperature)
            self.min_delta_E = float(self.cfg.energy_floor.min_delta(int(self.level)))

        # cursor init
        if self.cfg.start_xy is None:
            self.cursor_x = int(self._rng.integers(0, self.W))
            self.cursor_y = int(self._rng.integers(0, self.H))
        else:
            self.cursor_x, self.cursor_y = map(int, self.cfg.start_xy)
            if self.cfg.torus:
                self.cursor_x %= self.W
                self.cursor_y %= self.H
            else:
                self.cursor_x = _clip(self.cursor_x, 0, self.W - 1)
                self.cursor_y = _clip(self.cursor_y, 0, self.H - 1)

        obs = self._obs()
        info = {
            "energy": float(self.som.energy_E()),
            "level": int(self.level),
            "temperature": float(self.temperature),
        }
        return obs, info

    # ---- grids ----
    def _bins_grid(self) -> np.ndarray:
        # use backend helper if present
        if hasattr(self.som, "bins_grid_float"):
            bins = np.asarray(self.som.bins_grid_float(), dtype=np.int16)
            if bins.ndim == 1 and bins.size == self.H * self.W:
                return bins.reshape(self.H, self.W)
            return bins
        # fallback
        bins = np.asarray(self.som.current_bins, dtype=np.int16).reshape(self.H, self.W)
        empty = (np.asarray(self.som.node_gal_counts) <= 0).reshape(self.H, self.W)
        out = bins.copy()
        out[empty] = -1
        return out

    def _bins_grid_int(self) -> np.ndarray:
        if hasattr(self.som, "bins_grid_int"):
            bins = np.asarray(self.som.bins_grid_int(empty_value=-1), dtype=np.int16)
            if bins.ndim == 1 and bins.size == self.H * self.W:
                return bins.reshape(self.H, self.W)
            return bins
        # fallback
        bins = np.asarray(self.som.current_bins, dtype=np.int16).reshape(self.H, self.W)
        empty = (np.asarray(self.som.node_gal_counts) <= 0).reshape(self.H, self.W)
        out = bins.copy()
        out[empty] = -1
        return out

    def _empty_grid(self, bins_grid: np.ndarray) -> np.ndarray:
        return (bins_grid < 0).astype(np.uint8)

    def _cluster_feature_grids(self, level: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        boundary(uint8), size_norm(float32) at this level.
        If no hierarchy: boundary=0, size=1.
        """
        if getattr(self.som, "weighted_clusters", None) is None:
            boundary = np.zeros((self.H, self.W), dtype=np.uint8)
            size = np.ones((self.H, self.W), dtype=np.float32)
            return boundary, size

        if hasattr(self.som, "cluster_grid"):
            cg = np.asarray(self.som.cluster_grid(level), dtype=np.int32)
        elif hasattr(self.som, "cluster_ids_at_level"):
            cg = np.asarray(self.som.cluster_ids_at_level(level), dtype=np.int32)
        else:
            cg = np.asarray(self.som.weighted_clusters[int(level)].reshape(self.H, self.W), dtype=np.int32)

        boundary = cluster_boundary_from_grid(cg)

        flat = cg.reshape(-1)
        max_id = int(flat.max(initial=-1))
        sizes = np.bincount(flat, minlength=max_id + 1).astype(np.float32)
        denom = sizes.max(initial=1.0) + 1e-12
        sizes = sizes / denom
        size_grid = sizes[flat].reshape(self.H, self.W)
        return boundary, size_grid.astype(np.float32, copy=False)

    def _visit_count_grid_norm(self) -> np.ndarray:
        vc = self.visit_count.astype(np.float32)
        m = float(vc.max(initial=0))
        if m > 0:
            vc = vc / m
        return vc.reshape(self.H, self.W)

    def _recent_grid(self) -> np.ndarray:
        cooldown = int(self.cfg.recent_cooldown_steps)
        dt = self._step - self.last_visit_step
        recent = (dt <= cooldown).astype(np.uint8)
        return recent.reshape(self.H, self.W)

    # ---- obs ----
    def _obs(self) -> Dict[str, Any]:
        bins = self._bins_grid()

        if self.cfg.obs_mode == "global":
            out: Dict[str, Any] = {
                "bins": bins,
                "cursor": np.array([self.cursor_x, self.cursor_y], dtype=np.int32),
            }

            if self.cfg.include_empty_mask:
                out["empty"] = self._empty_grid(bins)

            if "umatrix" in self.observation_space.spaces:
                out["umatrix"] = self._get_umatrix()

            if self.cfg.include_cluster_features:
                boundary, size_grid = self._cluster_feature_grids(self.level)
                out["cluster_boundary"] = boundary
                out["cluster_size"] = size_grid

            if self.cfg.include_visit_count:
                out["visit_count"] = self._visit_count_grid_norm()

            if self.cfg.include_recent_mask:
                out["recent"] = self._recent_grid()

        else:
            if self.cfg.torus:
                bins_p = _extract_patch_torus(bins, self.cursor_x, self.cursor_y, self.ps)
            else:
                bins_p = _extract_patch_clipped(bins, self.cursor_x, self.cursor_y, self.ps)

            out = {
                "bins": bins_p.astype(np.int16, copy=False),
                "cursor": np.array([self.cursor_x, self.cursor_y], dtype=np.int32),
            }

            if self.cfg.include_empty_mask:
                out["empty"] = (bins_p < 0).astype(np.uint8)

            if "umatrix" in self.observation_space.spaces:
                umat = self._get_umatrix()
                out["umatrix"] = (_extract_patch_torus(umat, self.cursor_x, self.cursor_y, self.ps) if self.cfg.torus
                                 else _extract_patch_clipped(umat, self.cursor_x, self.cursor_y, self.ps)).astype(np.float32, copy=False)

            if self.cfg.include_cluster_features:
                boundary, size_grid = self._cluster_feature_grids(self.level)
                out["cluster_boundary"] = (_extract_patch_torus(boundary, self.cursor_x, self.cursor_y, self.ps) if self.cfg.torus
                                          else _extract_patch_clipped(boundary, self.cursor_x, self.cursor_y, self.ps))
                out["cluster_size"] = (_extract_patch_torus(size_grid, self.cursor_x, self.cursor_y, self.ps) if self.cfg.torus
                                      else _extract_patch_clipped(size_grid, self.cursor_x, self.cursor_y, self.ps)).astype(np.float32, copy=False)

            if self.cfg.include_visit_count:
                vc = self._visit_count_grid_norm()
                out["visit_count"] = (_extract_patch_torus(vc, self.cursor_x, self.cursor_y, self.ps) if self.cfg.torus
                                     else _extract_patch_clipped(vc, self.cursor_x, self.cursor_y, self.ps)).astype(np.float32, copy=False)

            if self.cfg.include_recent_mask:
                recent = self._recent_grid()
                out["recent"] = (_extract_patch_torus(recent, self.cursor_x, self.cursor_y, self.ps) if self.cfg.torus
                                else _extract_patch_clipped(recent, self.cursor_x, self.cursor_y, self.ps)).astype(np.uint8, copy=False)

        # scalars
        E = float(self.som.energy_E())
        steps_left = float(self.cfg.max_steps - self._step)
        level_norm = float(self.level) / float(max(1, self.H))
        out["scalars"] = np.array(
            [level_norm, float(self.temperature), E, steps_left / float(max(1, self.cfg.max_steps))],
            dtype=np.float32,
        )
        return out

    # ---- decode / targeting ----
    def _decode_action(self, action) -> Dict[str, Any]:
        """
        Must return dict with:
          - "node" OR ("x","y")
          - optional "bin"  (if env selects bin)
        Note: level comes from scheduler for these base envs.
        """
        raise NotImplementedError

    def _target_node_from_decoded(self, decoded: Dict[str, Any]) -> int:
        if "node" in decoded:
            node = int(decoded["node"])
            if not (0 <= node < self.H * self.W):
                raise ValueError("node out of range")
            return node

        x = int(decoded["x"]); y = int(decoded["y"])
        if self.cfg.torus:
            x, y = _wrap(x, self.W), _wrap(y, self.H)
        else:
            x, y = _clip(x, 0, self.W - 1), _clip(y, 0, self.H - 1)
        return int(self.som.xy_to_index(x, y))

    def _sync_scheduler_params(self) -> None:
        if self.scheduler is None:
            return
        lvl, T, floor = self.scheduler.current_params()
        self.level = int(lvl)
        self.temperature = float(T)
        self.min_delta_E = float(floor)

    # ---- reward ----
    def _compute_reward(self, step_out: Dict[str, Any]) -> float:
        """
        Central reward computation.

        Uses:
          - reward_type:
              * "deltaE": based on dE = E1 - E0 with optional tanh shaping
              * "rawE":   based directly on E1
          - reward_energy_scale: multiplicative scale for the energy term
          - use_tanh_reward / reward_tanh_temperature: optional nonlinearity
          - reject_penalty / noop_penalty: small penalties for unproductive moves
          - step_penalty / clip_reward: as in the original config
        """
        cfg = self.cfg

        # base energy term
        if cfg.reward_type == "deltaE":
            dE = float(step_out.get("dE", 0.0))
            if cfg.use_tanh_reward:
                # protect against zero temperature parameter
                temp = float(cfg.reward_tanh_temperature)
                if temp <= 0.0:
                    temp = 1e-12
                base = float(np.tanh(dE / temp))
            else:
                base = dE
            reward = float(cfg.reward_energy_scale) * base
        else:
            # rawE: use E1 directly (higher is better)
            E1 = float(step_out.get("E1", self.som.energy_E()))
            reward = E1

        # small penalties for clearly unproductive moves
        accepted = bool(step_out.get("accepted", False))
        is_noop = bool(step_out.get("is_noop", False))

        if is_noop and cfg.noop_penalty != 0.0:
            reward -= float(cfg.noop_penalty)
        elif (not accepted) and cfg.reject_penalty != 0.0:
            reward -= float(cfg.reject_penalty)

        # per-step shaping + clipping (unchanged behaviour, but applied after
        # the new energy/penalty terms)
        reward += float(cfg.step_penalty)
        if cfg.clip_reward is not None:
            c = float(cfg.clip_reward)
            reward = float(np.clip(reward, -c, c))

        return float(reward)

    # ---- accept/reject ----
    def _accept_sa(self, dE: float, T: float) -> bool:
        if dE > 0.0:
            return True
        P = float(np.exp(dE / max(T, 1e-30)))
        return bool(P > float(self._rng.random()))

    def _attempt_move(self, *, node: int, new_bin: Optional[int]) -> Dict[str, Any]:
        """
        Builds a backend proposal, evaluates dE (proposal is applied), then
        accepts/rejects (reverting if rejected). Returns standardized dict.
        """
        # build proposal (PURE)
        if new_bin is None:
            proposal = self.som.make_proposal_random_bin(node=int(node), level=int(self.level), rng=self._rng)
        else:
            proposal = self.som.make_proposal_set_bin(new_bin=int(new_bin), node=int(node), level=int(self.level))

        if proposal.get("type") == "noop":
            return {
                "accepted": False,
                "is_noop": True,
                "reason": proposal.get("reason", "noop"),
                "T": float(self.temperature),
                "E0": float(self.som.energy_E()),
                "E1": float(self.som.energy_E()),
                "dE": 0.0,
                "proposal": proposal,
                "level": int(self.level),
                "cluster_id": proposal.get("cluster_id", None),
                "n_nodes_touched": 0,
            }

        # evaluate (APPLIES proposal)
        evald = self.som.evaluate_proposal_deltaE(proposal)
        dE = float(evald["dE"])
        E0 = float(evald["E0"])
        E1 = float(evald["E1"])
        undo = evald["undo"]

        min_dE = float(self.min_delta_E)

        # floor gate
        if dE < min_dE:
            self.som.revert(undo)
            return {
                "accepted": False,
                "is_noop": False,
                "reason": "energy_floor",
                "T": float(self.temperature),
                "E0": float(E0),
                "E1": float(E0),
                "dE": float(dE),
                "min_delta_E": float(min_dE),
                "proposal": proposal,
                "level": int(self.level),
                "cluster_id": proposal.get("cluster_id", None),
                "n_nodes_touched": int(len(proposal.get("nodes", []))),
            }

        # accept logic
        if self.cfg.accept_rule == "positive_only":
            accepted = (dE > 0.0)
        else:
            accepted = self._accept_sa(dE=dE, T=float(self.temperature))

        if not accepted:
            self.som.revert(undo)
            return {
                "accepted": False,
                "is_noop": False,
                "T": float(self.temperature),
                "E0": float(E0),
                "E1": float(E0),
                "dE": 0.0,
                "min_delta_E": float(min_dE),
                "proposal": proposal,
                "level": int(self.level),
                "cluster_id": proposal.get("cluster_id", None),
                "n_nodes_touched": int(len(proposal.get("nodes", []))),
            }

        return {
            "accepted": True,
            "is_noop": False,
            "T": float(self.temperature),
            "E0": float(E0),
            "E1": float(E1),
            "dE": float(dE),
            "min_delta_E": float(min_dE),
            "proposal": proposal,
            "level": int(self.level),
            "cluster_id": proposal.get("cluster_id", None),
            "n_nodes_touched": int(len(proposal.get("nodes", []))),
        }

    # ---- step ----
    def step(self, action):
        step_idx = int(self._step)

        # scheduler drives (level, temperature, floor) for this step
        if self.scheduler is not None:
            self._sync_scheduler_params()

        decoded = self._decode_action(action)
        node = self._target_node_from_decoded(decoded)

        # move cursor to node
        x, y = self.som.index_to_xy(int(node))
        self.cursor_x, self.cursor_y = int(x), int(y)

        # decide bin selection strategy:
        # - if action explicitly provides a bin, honour it by default
        # - otherwise:
        #     * if use_greedy_bin_for_node: use backend.greedy_bin_for_node
        #     * else: fall back to random-bin proposal in _attempt_move
        new_bin = decoded.get("bin", None)
        if new_bin is None and self.cfg.use_greedy_bin_for_node:
            best_bin, _ = self.som.greedy_bin_for_node(node=int(node), level=int(self.level))
            new_bin = int(best_bin)

        out = self._attempt_move(node=node, new_bin=new_bin)

        # propagate "interaction" to all nodes touched by the action target (cluster or node)
        touched = out.get("proposal", {}).get("nodes", None)
        if touched is None:
            touched = np.array([node], dtype=np.int32)
        else:
            touched = np.asarray(touched, dtype=np.int32)

        self.visit_count[touched] += 1
        self.last_visit_step[touched] = int(step_idx)


        # scheduler update (post decision)
        if self.scheduler is not None:
            self.scheduler.update(
                accepted=bool(out["accepted"]),
                dE=float(out.get("dE", 0.0)),
                E=float(out.get("E1", self.som.energy_E())),
                step_idx=step_idx,
                is_noop=bool(out.get("is_noop", False)),
            )

        # env step increments after update
        self._step = step_idx + 1

        # reward (centralised)
        reward = self._compute_reward(out)

        truncated = (self._step >= int(self.cfg.max_steps)) if self.cfg.use_env_max_steps else False
        terminated = False
        if self.scheduler is not None and hasattr(self.scheduler, "done"):
            try:
                if bool(self.scheduler.done()):
                    terminated = True
            except Exception:
                # Be robust to schedulers that don't fully implement done()
                pass

        info = {"step": int(step_idx), "level": int(self.level), **out}
        obs = self._obs()
        return obs, float(reward), terminated, truncated, info


# -------------------------
# Sub-envs (action spaces)
# -------------------------

class EnvTeleportNodeRandomBin(SOMEnergyBaseEnv):
    """Action: node_idx. Effect: random-bin proposal at scheduler-provided level."""
    def _build_action_space(self):
        return spaces.Discrete(self.H * self.W)

    def _decode_action(self, action):
        node = int(action)
        return {"node": node}


class EnvTeleportXYRandomBin(SOMEnergyBaseEnv):
    """Action: (x,y). Effect: random-bin proposal at scheduler-provided level."""
    def _build_action_space(self):
        return spaces.MultiDiscrete([self.W, self.H])

    def _decode_action(self, action):
        x, y = int(action[0]), int(action[1])
        return {"x": x, "y": y}


class EnvCursorLocalRandomBin(SOMEnergyBaseEnv):
    """
    Action: local index in ps*ps patch around current cursor.
    Effect: random-bin proposal at scheduler-provided level.
    """
    def _build_action_space(self):
        return spaces.Discrete(self.ps * self.ps)

    def _decode_action(self, action):
        a = int(action)
        if not (0 <= a < self.ps * self.ps):
            raise ValueError("local action out of range")
        dy = a // self.ps
        dx = a % self.ps
        x = self.cursor_x + (dx - self.r)
        y = self.cursor_y + (dy - self.r)
        return {"x": x, "y": y}


class EnvNodeAndBin(SOMEnergyBaseEnv):
    """
    Action: (node_idx, bin).
    Effect: explicit set-bin proposal at scheduler-provided level.
    """
    def _build_action_space(self):
        return spaces.MultiDiscrete([self.H * self.W, int(self.som.K)])

    def _decode_action(self, action):
        node = int(action[0])
        b = int(action[1])
        return {"node": node, "bin": b}


class EnvXYAndBin(SOMEnergyBaseEnv):
    """
    Action: (x,y,bin).
    Effect: explicit set-bin proposal at scheduler-provided level.
    """
    def _build_action_space(self):
        return spaces.MultiDiscrete([self.W, self.H, int(self.som.K)])

    def _decode_action(self, action):
        x = int(action[0])
        y = int(action[1])
        b = int(action[2])
        return {"x": x, "y": y, "bin": b}


# Optional: if you still want action to control level explicitly (overrides scheduler)
class EnvNodeLevelBin(SOMEnergyBaseEnv):
    """
    Action: (node_idx, level_choice, bin).
    Effect: explicit set-bin proposal at chosen level.
    If a scheduler is attached, this env overrides self.level for that step but still uses
    scheduler temperature unless you decide otherwise.
    """
    def _build_action_space(self):
        return spaces.MultiDiscrete([self.H * self.W, len(self.cfg.allowed_levels), int(self.som.K)])

    def step(self, action):
        # Let scheduler set temperature/floor, but override level from action
        step_idx = int(self._step)
        if self.scheduler is not None:
            self._sync_scheduler_params()

        node = int(action[0])
        li = int(action[1])
        b = int(action[2])
        if not (0 <= li < len(self.cfg.allowed_levels)):
            raise ValueError("level index out of range")
        self.level = int(self.cfg.allowed_levels[li])
        if self.scheduler is None:
            self.min_delta_E = float(self.cfg.energy_floor.min_delta(int(self.level)))

        decoded = {"node": node, "bin": b}
        node = self._target_node_from_decoded(decoded)

        x, y = self.som.index_to_xy(int(node))
        self.cursor_x, self.cursor_y = int(x), int(y)

        self.visit_count[node] += 1
        self.last_visit_step[node] = int(step_idx)

        # In this env variant the action provides a bin explicitly, but if
        # use_greedy_bin_for_node is enabled we allow the backend to override
        # that bin choice by scanning all bins for this node.
        if self.cfg.use_greedy_bin_for_node:
            best_bin, _ = self.som.greedy_bin_for_node(node=int(node), level=int(self.level))
            chosen_bin = int(best_bin)
        else:
            chosen_bin = int(b)

        out = self._attempt_move(node=node, new_bin=chosen_bin)

        if self.scheduler is not None:
            self.scheduler.update(
                accepted=bool(out["accepted"]),
                dE=float(out.get("dE", 0.0)),
                E=float(out.get("E1", self.som.energy_E())),
                step_idx=step_idx,
                is_noop=bool(out.get("is_noop", False)),
            )

        self._step = step_idx + 1

        # reward (reuse central helper)
        reward = self._compute_reward(out)

        truncated = (self._step >= int(self.cfg.max_steps))
        terminated = False
        info = {"step": int(step_idx), "level": int(self.level), **out}
        obs = self._obs()
        return obs, float(reward), terminated, truncated, info
