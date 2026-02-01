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
    "factor_xy",         # action = (x_action, y_action) -> node_idx (200+200)
    "cursor_move",       # action = move (up/down/left/right) + optional act
    "cursor_move_act",   # action = move or "act" (apply to current cursor)
    "cursor_local_k",    # action = local index within KxK patch around cursor (apply there)
]

ObsMode = Literal[
    "global_bins",       # (H,W) current bins (int)
    "local_patch",       # patch of bin-map + umatrix (+ optional extra channels later)
]


@dataclass(frozen=True)
class SOMEnvConfig:
    grid_shape: Tuple[int, int] = (200, 200)
    torus: bool = True

    action_mode: ActionMode = "teleport_node"
    obs_mode: ObsMode = "local_patch"

    # cursor-based modes
    start_xy: Optional[Tuple[int, int]] = None
    patch_size: int = 9               # for local observations / local actions; must be odd
    allow_act_every_step: bool = True # if True: every step can trigger backend; else only on explicit ACT action

    # episode control
    max_steps: int = 2000
    terminate_on_no_improve: bool = False
    patience: int = 200

    # reward shaping
    reward_type: Literal["delta", "raw"] = "delta"  # delta trace recommended
    clip_reward: Optional[float] = None             # e.g. 1.0 to clip
    no_op_penalty: float = 0.0                      # penalty if backend chooses same bin
    invalid_action_penalty: float = -1.0            # e.g. out-of-range x/y if torus=False, invalid local etc.

    # observation toggles
    include_umatrix: bool = True


# ---------------------------
# Helpers (torus wrapping + patch extraction)
# ---------------------------

def _wrap(v: int, m: int) -> int:
    return int(v) % int(m)

def _clip(v: int, lo: int, hi: int) -> int:
    return int(np.clip(v, lo, hi))

def _extract_patch_torus(A: np.ndarray, x: int, y: int, ps: int) -> np.ndarray:
    # A is (H,W)
    H, W = A.shape
    r = ps // 2
    xs = (np.arange(x - r, x + r + 1) % W).astype(int)  # x is column
    ys = (np.arange(y - r, y + r + 1) % H).astype(int)  # y is row
    return A[np.ix_(ys, xs)]  # (ps, ps)

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

    Design goal:
      - Action selects *where* to act (node/cursor/local cell/etc).
      - Backend greedily picks best bin for that node (TreeCorr-consistent).
      - Reward returned as delta trace (recommended) or raw trace.

    This env is intentionally flexible via action_mode; choose one at init.
    """

    metadata = {"render_modes": []}

    def __init__(self, som_control, cfg: SOMEnvConfig):
        """
        som_control must expose:
          - H, W, N
          - current_bins: (N,) int
          - index_to_xy(node)->(x,y) and xy_to_index(x,y)->node
          - step_node(node_idx)-> dict with keys: delta_reward, reward, chosen_bin, old_bin, node
          - optional: umatrix (H,W) float32
        """
        super().__init__()
        self.som = som_control
        self.cfg = cfg

        self.H, self.W = cfg.grid_shape
        assert (self.H, self.W) == (200, 200), "You said 200x200; keep this locked unless you change config."

        if cfg.patch_size % 2 == 0:
            raise ValueError("patch_size must be odd")
        self.ps = cfg.patch_size
        self.r = self.ps // 2

        self._step = 0
        self._no_improve_streak = 0
        self._last_reward = None

        # cursor state (used by cursor_* modes)
        self.cursor_x = 0
        self.cursor_y = 0

        # ---------------- action space ----------------
        self.action_space = self._build_action_space(cfg.action_mode)

        # ---------------- observation space ----------------
        self.observation_space = self._build_observation_space(cfg.obs_mode)

    # ---------- spaces ----------
    def _build_action_space(self, mode: ActionMode) -> spaces.Space:
        if mode == "teleport_node":
            return spaces.Discrete(self.H * self.W)

        if mode == "teleport_xy":
            # (x,y) in [0..199]
            return spaces.MultiDiscrete([self.W, self.H])

        if mode == "factor_xy":
            # same as teleport_xy, but intended conceptual factorization
            return spaces.MultiDiscrete([self.W, self.H])

        if mode == "cursor_move":
            # movement only (up/down/left/right)
            return spaces.Discrete(4)

        if mode == "cursor_move_act":
            # 0..3 move, 4 act
            return spaces.Discrete(5)

        if mode == "cursor_local_k":
            # select one cell in local patch around cursor (ps*ps choices)
            return spaces.Discrete(self.ps * self.ps)

        raise ValueError(f"Unknown action_mode: {mode}")

    def _build_observation_space(self, mode: ObsMode) -> spaces.Space:
        # We keep obs compact and stable across modes.
        # Use Dict with:
        #   - "bins": (ps,ps) int32 for local_patch, or (H,W) for global_bins
        #   - "umatrix": (ps,ps) float32 optional
        #   - "cursor": (2,) int32 for cursor-based modes (always included; cheap)
        if mode == "global_bins":
            obs = {
                "bins": spaces.Box(low=0, high=10, shape=(self.H, self.W), dtype=np.int32),
                "cursor": spaces.Box(low=0, high=max(self.H, self.W), shape=(2,), dtype=np.int32),
            }
            if self.cfg.include_umatrix and getattr(self.som, "umatrix", None) is not None:
                obs["umatrix"] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.H, self.W), dtype=np.float32)
            return spaces.Dict(obs)

        if mode == "local_patch":
            obs = {
                "bins": spaces.Box(low=0, high=10, shape=(self.ps, self.ps), dtype=np.int32),
                "cursor": spaces.Box(low=0, high=max(self.H, self.W), shape=(2,), dtype=np.int32),
            }
            if self.cfg.include_umatrix and getattr(self.som, "umatrix", None) is not None:
                obs["umatrix"] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.ps, self.ps), dtype=np.float32)
            return spaces.Dict(obs)

        raise ValueError(f"Unknown obs_mode: {mode}")

    # ---------- reset/obs ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._step = 0
        self._no_improve_streak = 0
        self._last_reward = None

        # Reset backend bins
        if hasattr(self.som, "reset_to_initial"):
            self.som.reset_to_initial()

        # Cursor init
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
        # current_bins is (N,) with node idx mapping to (x,y) via row-major in SOMControl
        bins = self.som.current_bins.reshape(self.H, self.W)  # assumes same mapping as SOMControl (y*W+x)
        return bins

    def _obs(self) -> Dict[str, Any]:
        bins_grid = self._bins_grid()

        if self.cfg.obs_mode == "global_bins":
            out = {
                "bins": bins_grid.astype(np.int32, copy=False),
                "cursor": np.array([self.cursor_x, self.cursor_y], dtype=np.int32),
            }
            if "umatrix" in self.observation_space.spaces and getattr(self.som, "umatrix", None) is not None:
                out["umatrix"] = np.asarray(self.som.umatrix, dtype=np.float32)
            return out

        # local patch
        if self.cfg.torus:
            bins_patch = _extract_patch_torus(bins_grid, self.cursor_x, self.cursor_y, self.ps)
            umat = getattr(self.som, "umatrix", None)
            umat_patch = _extract_patch_torus(umat, self.cursor_x, self.cursor_y, self.ps) if umat is not None else None
        else:
            bins_patch = _extract_patch_clipped(bins_grid, self.cursor_x, self.cursor_y, self.ps)
            umat = getattr(self.som, "umatrix", None)
            umat_patch = _extract_patch_clipped(umat, self.cursor_x, self.cursor_y, self.ps) if umat is not None else None

        out = {
            "bins": bins_patch.astype(np.int32, copy=False),
            "cursor": np.array([self.cursor_x, self.cursor_y], dtype=np.int32),
        }
        if "umatrix" in self.observation_space.spaces and umat_patch is not None:
            out["umatrix"] = umat_patch.astype(np.float32, copy=False)
        return out

    # ---------- step ----------
    def step(self, action):
        self._step += 1

        acted = False
        info: Dict[str, Any] = {"step": self._step}
        reward = 0.0

        node_idx: Optional[int] = None

        try:
            node_idx, acted = self._resolve_action_to_node(action)
        except Exception as e:
            # Invalid action -> penalize; no backend step.
            obs = self._obs()
            reward = float(self.cfg.invalid_action_penalty)
            terminated = False
            truncated = self._step >= self.cfg.max_steps
            info.update({"error": str(e), "acted": False})
            return obs, reward, terminated, truncated, info

        # Apply backend move if this step is an "act"
        if acted and node_idx is not None:
            out = self.som.step_node(node_idx)
            raw_delta = float(out["delta_reward"])
            raw_reward = float(out["reward"])
            chosen_bin = int(out["chosen_bin"])
            old_bin = int(out["old_bin"])

            if self.cfg.reward_type == "delta":
                reward = raw_delta
            else:
                reward = raw_reward

            if chosen_bin == old_bin and self.cfg.no_op_penalty != 0.0:
                reward += float(self.cfg.no_op_penalty)

            info.update(out)
        else:
            # movement-only steps
            reward = 0.0
            info.update({"acted": False})

        if self.cfg.clip_reward is not None:
            c = float(self.cfg.clip_reward)
            reward = float(np.clip(reward, -c, c))

        # termination / truncation
        truncated = self._step >= self.cfg.max_steps
        terminated = False

        if self.cfg.terminate_on_no_improve:
            # define "improve" as positive delta when acting
            if acted and reward > 0:
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
        """
        Returns:
          (node_idx, acted)

        acted=False means "just moved cursor" or "no-op step".
        """
        mode = self.cfg.action_mode

        if mode == "teleport_node":
            node = int(action)
            if not (0 <= node < self.H * self.W):
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
            node = self.som.xy_to_index(x, y)
            return node, True

        if mode == "cursor_move":
            # move only; acting happens every step only if allow_act_every_step
            self._cursor_move(int(action))
            if self.cfg.allow_act_every_step:
                node = self.som.xy_to_index(self.cursor_x, self.cursor_y)
                return node, True
            return None, False

        if mode == "cursor_move_act":
            a = int(action)
            if a in (0, 1, 2, 3):
                self._cursor_move(a)
                if self.cfg.allow_act_every_step:
                    node = self.som.xy_to_index(self.cursor_x, self.cursor_y)
                    return node, True
                return None, False
            if a == 4:
                node = self.som.xy_to_index(self.cursor_x, self.cursor_y)
                return node, True
            raise ValueError("invalid cursor_move_act action")

        if mode == "cursor_local_k":
            # action selects a local cell in patch around cursor; always acts
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
            node = self.som.xy_to_index(x, y)
            return node, True

        raise ValueError(f"Unknown action_mode: {mode}")

    def _cursor_move(self, move: int) -> None:
        # 0 up, 1 down, 2 left, 3 right
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
