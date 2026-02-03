from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Literal

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------
# Env configuration
# ---------------------------

ActionMode = Literal[
    "teleport_node",     # action = node_idx
    "teleport_xy",       # action = (x, y)  -> node_idx
    "factor_xy",         # action = (x_action, y_action) -> node_idx
    "cursor_move",       # action = move (up/down/left/right) + optional act
    "cursor_move_act",   # action = move or "act" (apply to current cursor)
    "cursor_local_k",    # action = local index within ps x ps patch around cursor (apply there)
]

ObsMode = Literal[
    "global_bins",       # (H,W) current bins (int)
    "local_patch",       # patch of bin-map + umatrix (+ optional extra channels)
]

CooldownPolicy = Literal[
    "none",          # ignore cooldown; still track channels if enabled
    "no_act",        # if target node on cooldown, do not act this step (reward=0)
    "invalid",       # if target node on cooldown, treat as invalid action (penalty)
]


@dataclass(frozen=True)
class SOMEnvConfig:
    grid_shape: Tuple[int, int] = (200, 200)
    torus: bool = True

    action_mode: ActionMode = "teleport_node"
    obs_mode: ObsMode = "local_patch"

    # cursor-based modes
    start_xy: Optional[Tuple[int, int]] = None
    patch_size: int = 9
    allow_act_every_step: bool = True

    # episode control
    max_steps: int = 2000
    terminate_on_no_improve: bool = False
    patience: int = 200

    # reward shaping
    reward_type: Literal["delta", "raw"] = "delta"
    clip_reward: Optional[float] = None
    no_op_penalty: float = 0.0
    invalid_action_penalty: float = -1.0

    # observation toggles
    include_umatrix: bool = True

    # extra observation channels (all derived; no leakage)
    include_visit_count: bool = True          # how many times node has been acted on
    include_cooldown: bool = True             # remaining cooldown (steps)
    include_steps_since_act: bool = True      # steps since last act on node (capped)

    # cooldown dynamics
    cooldown_steps: int = 0                   # 0 disables cooldown (still can track)
    cooldown_policy: CooldownPolicy = "no_act"
    steps_since_cap: int = 10_000             # cap for steps_since_act channel


# ---------------------------
# Helpers
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


# ---------------------------
# SOM Environment
# ---------------------------

class SOMBinningEnv(gym.Env):
    """
    Gymnasium env wrapping SOMControl backend.

    Requires som_control expose:
      - H, W, N
      - current_bins: (N,) int in [0..10]
      - xy_to_index(x,y)->node
      - step_node(node_idx)-> dict:
            { node, old_bin, chosen_bin, delta_reward, reward, is_noop }
      - optional: umatrix (H,W) float32
    """

    metadata = {"render_modes": []}

    def __init__(self, som_control, cfg: SOMEnvConfig):
        super().__init__()
        self.som = som_control
        self.cfg = cfg

        self.H, self.W = cfg.grid_shape
        if (self.H, self.W) != (200, 200):
            raise ValueError("This project locks 200x200 unless you change everything coherently.")
        self.N = self.H * self.W

        if cfg.patch_size % 2 == 0:
            raise ValueError("patch_size must be odd")
        self.ps = cfg.patch_size
        self.r = self.ps // 2

        # per-node bookkeeping (derived only)
        self.visit_count = np.zeros(self.N, dtype=np.int32)
        self.cooldown = np.zeros(self.N, dtype=np.int32)
        self.steps_since_act = np.full(self.N, cfg.steps_since_cap, dtype=np.int32)

        self._step = 0
        self._no_improve_streak = 0

        self.cursor_x = 0
        self.cursor_y = 0

        self.action_space = self._build_action_space(cfg.action_mode)
        self.observation_space = self._build_observation_space(cfg.obs_mode)

    # ---------- spaces ----------
    def _build_action_space(self, mode: ActionMode) -> spaces.Space:
        if mode == "teleport_node":
            return spaces.Discrete(self.N)
        if mode in ("teleport_xy", "factor_xy"):
            return spaces.MultiDiscrete([self.W, self.H])
        if mode == "cursor_move":
            return spaces.Discrete(4)
        if mode == "cursor_move_act":
            return spaces.Discrete(5)
        if mode == "cursor_local_k":
            return spaces.Discrete(self.ps * self.ps)
        raise ValueError(f"Unknown action_mode: {mode}")

    def _build_observation_space(self, mode: ObsMode) -> spaces.Space:
        # Always Dict. We add optional extra channels as separate entries to keep it explicit.
        obs: Dict[str, spaces.Space] = {
            "cursor": spaces.Box(low=0, high=max(self.H, self.W), shape=(2,), dtype=np.int32),
        }

        if mode == "global_bins":
            obs["bins"] = spaces.Box(low=0, high=10, shape=(self.H, self.W), dtype=np.int32)
            if self.cfg.include_umatrix and getattr(self.som, "umatrix", None) is not None:
                obs["umatrix"] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.H, self.W), dtype=np.float32)

            if self.cfg.include_visit_count:
                obs["visit_count"] = spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(self.H, self.W), dtype=np.int32)
            if self.cfg.include_cooldown:
                obs["cooldown"] = spaces.Box(low=0, high=max(0, int(self.cfg.cooldown_steps)), shape=(self.H, self.W), dtype=np.int32)
            if self.cfg.include_steps_since_act:
                obs["steps_since_act"] = spaces.Box(low=0, high=int(self.cfg.steps_since_cap), shape=(self.H, self.W), dtype=np.int32)

            return spaces.Dict(obs)

        if mode == "local_patch":
            obs["bins"] = spaces.Box(low=0, high=10, shape=(self.ps, self.ps), dtype=np.int32)
            if self.cfg.include_umatrix and getattr(self.som, "umatrix", None) is not None:
                obs["umatrix"] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.ps, self.ps), dtype=np.float32)

            if self.cfg.include_visit_count:
                obs["visit_count"] = spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(self.ps, self.ps), dtype=np.int32)
            if self.cfg.include_cooldown:
                obs["cooldown"] = spaces.Box(low=0, high=max(0, int(self.cfg.cooldown_steps)), shape=(self.ps, self.ps), dtype=np.int32)
            if self.cfg.include_steps_since_act:
                obs["steps_since_act"] = spaces.Box(low=0, high=int(self.cfg.steps_since_cap), shape=(self.ps, self.ps), dtype=np.int32)

            return spaces.Dict(obs)

        raise ValueError(f"Unknown obs_mode: {mode}")

    # ---------- reset/obs ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._step = 0
        self._no_improve_streak = 0

        # backend
        if hasattr(self.som, "reset_to_initial"):
            self.som.reset_to_initial()

        # bookkeeping
        self.visit_count.fill(0)
        self.cooldown.fill(0)
        self.steps_since_act.fill(int(self.cfg.steps_since_cap))

        # cursor init
        if self.cfg.start_xy is None:
            rng = np.random.default_rng(seed)
            self.cursor_x = int(rng.integers(0, self.W))
            self.cursor_y = int(rng.integers(0, self.H))
        else:
            self.cursor_x, self.cursor_y = map(int, self.cfg.start_xy)
            if self.cfg.torus:
                self.cursor_x %= self.W
                self.cursor_y %= self.H
            else:
                self.cursor_x = _clip(self.cursor_x, 0, self.W - 1)
                self.cursor_y = _clip(self.cursor_y, 0, self.H - 1)

        obs = self._obs()
        info = {"reward": float(self.som.reward_trace()) if hasattr(self.som, "reward_trace") else None}
        return obs, info

    def _bins_grid(self) -> np.ndarray:
        return self.som.current_bins.reshape(self.H, self.W)

    def _grid_from_flat(self, A_flat: np.ndarray) -> np.ndarray:
        return np.asarray(A_flat).reshape(self.H, self.W)

    def _extract(self, A2d: np.ndarray) -> np.ndarray:
        if self.cfg.obs_mode == "global_bins":
            return A2d
        if self.cfg.torus:
            return _extract_patch_torus(A2d, self.cursor_x, self.cursor_y, self.ps)
        return _extract_patch_clipped(A2d, self.cursor_x, self.cursor_y, self.ps)

    def _obs(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "cursor": np.array([self.cursor_x, self.cursor_y], dtype=np.int32),
        }

        bins_grid = self._bins_grid()
        out["bins"] = self._extract(bins_grid).astype(np.int32, copy=False)

        if "umatrix" in self.observation_space.spaces and getattr(self.som, "umatrix", None) is not None:
            out["umatrix"] = self._extract(np.asarray(self.som.umatrix, dtype=np.float32)).astype(np.float32, copy=False)

        if "visit_count" in self.observation_space.spaces:
            out["visit_count"] = self._extract(self._grid_from_flat(self.visit_count)).astype(np.int32, copy=False)

        if "cooldown" in self.observation_space.spaces:
            out["cooldown"] = self._extract(self._grid_from_flat(self.cooldown)).astype(np.int32, copy=False)

        if "steps_since_act" in self.observation_space.spaces:
            out["steps_since_act"] = self._extract(self._grid_from_flat(self.steps_since_act)).astype(np.int32, copy=False)

        return out

    # ---------- step ----------
    def step(self, action):
        self._step += 1

        # global decay / increment trackers (O(N), but N=40k so fine)
        if self.cfg.cooldown_steps > 0 and np.any(self.cooldown):
            self.cooldown = np.maximum(self.cooldown - 1, 0)
        self.steps_since_act = np.minimum(self.steps_since_act + 1, int(self.cfg.steps_since_cap))

        info: Dict[str, Any] = {"step": self._step}
        reward = 0.0
        acted = False
        node_idx: Optional[int] = None

        try:
            node_idx, acted = self._resolve_action_to_node(action)
        except Exception as e:
            obs = self._obs()
            reward = float(self.cfg.invalid_action_penalty)
            terminated = False
            truncated = self._step >= self.cfg.max_steps
            info.update({"error": str(e), "acted": False})
            return obs, reward, terminated, truncated, info

        # cooldown gating
        if acted and node_idx is not None and self.cfg.cooldown_policy != "none":
            if self.cooldown[int(node_idx)] > 0:
                if self.cfg.cooldown_policy == "no_act":
                    acted = False
                    info["cooldown_blocked"] = True
                elif self.cfg.cooldown_policy == "invalid":
                    obs = self._obs()
                    reward = float(self.cfg.invalid_action_penalty)
                    terminated = False
                    truncated = self._step >= self.cfg.max_steps
                    info.update({"cooldown_blocked": True, "acted": False})
                    return obs, reward, terminated, truncated, info

        if acted and node_idx is not None:
            out = self.som.step_node(node_idx)
            raw_delta = float(out["delta_reward"])
            raw_reward = float(out["reward"])
            is_noop = bool(out.get("is_noop", False))


            # reward signal
            reward = raw_delta if self.cfg.reward_type == "delta" else raw_reward
            if is_noop and self.cfg.no_op_penalty != 0.0:
                reward += float(self.cfg.no_op_penalty)

            # update bookkeeping for the acted node
            n = int(node_idx)
            self.visit_count[n] += 1
            self.steps_since_act[n] = 0
            if self.cfg.cooldown_steps > 0:
                self.cooldown[n] = int(self.cfg.cooldown_steps)

            info.update(out)
            info["acted"] = True
        else:
            reward = 0.0
            info["acted"] = False

        if self.cfg.clip_reward is not None:
            c = float(self.cfg.clip_reward)
            reward = float(np.clip(reward, -c, c))

        truncated = self._step >= self.cfg.max_steps
        terminated = False

        if self.cfg.terminate_on_no_improve:
            improved = bool(info.get("acted", False) and reward > 0.0)
            if improved:
                self._no_improve_streak = 0
            else:
                self._no_improve_streak += 1
            if self._no_improve_streak >= self.cfg.patience:
                terminated = True
                info["terminated_reason"] = "no_improve"

        obs = self._obs()
        return obs, float(reward), terminated, truncated, info

    # ---------- action resolution ----------
    def _resolve_action_to_node(self, action) -> Tuple[Optional[int], bool]:
        mode = self.cfg.action_mode

        if mode == "teleport_node":
            node = int(action)
            if not (0 <= node < self.N):
                raise ValueError("node_idx out of range")
            return node, True

        if mode in ("teleport_xy", "factor_xy"):
            x = int(action[0]); y = int(action[1])
            if self.cfg.torus:
                x = _wrap(x, self.W); y = _wrap(y, self.H)
            else:
                if not (0 <= x < self.W and 0 <= y < self.H):
                    raise ValueError("x,y out of range (torus=False)")
            self.cursor_x, self.cursor_y = x, y
            return int(self.som.xy_to_index(x, y)), True

        if mode == "cursor_move":
            self._cursor_move(int(action))
            if self.cfg.allow_act_every_step:
                return int(self.som.xy_to_index(self.cursor_x, self.cursor_y)), True
            return None, False

        if mode == "cursor_move_act":
            a = int(action)
            if a in (0, 1, 2, 3):
                self._cursor_move(a)
                if self.cfg.allow_act_every_step:
                    return int(self.som.xy_to_index(self.cursor_x, self.cursor_y)), True
                return None, False
            if a == 4:
                return int(self.som.xy_to_index(self.cursor_x, self.cursor_y)), True
            raise ValueError("invalid cursor_move_act action")

        if mode == "cursor_local_k":
            a = int(action)
            if not (0 <= a < self.ps * self.ps):
                raise ValueError("local_k action out of range")
            dy = a // self.ps
            dx = a % self.ps
            x = self.cursor_x + (dx - self.r)
            y = self.cursor_y + (dy - self.r)
            if self.cfg.torus:
                x = _wrap(x, self.W); y = _wrap(y, self.H)
            else:
                if not (0 <= x < self.W and 0 <= y < self.H):
                    raise ValueError("selected local cell out of bounds (torus=False)")
            return int(self.som.xy_to_index(x, y)), True

        raise ValueError(f"Unknown action_mode: {mode}")

    def _cursor_move(self, move: int) -> None:
        x, y = self.cursor_x, self.cursor_y
        if move == 0:
            y -= 1
        elif move == 1:
            y += 1
        elif move == 2:
            x -= 1
        elif move == 3:
            x += 1
        else:
            raise ValueError("move must be in {0,1,2,3}")

        if self.cfg.torus:
            self.cursor_x = _wrap(x, self.W)
            self.cursor_y = _wrap(y, self.H)
        else:
            self.cursor_x = _clip(x, 0, self.W - 1)
            self.cursor_y = _clip(y, 0, self.H - 1)
