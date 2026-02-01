from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
from gymnasium import Env, spaces

from som_rl.old.som_state import SOMState


class GlobalPointerTwoStageEnv(Env):
    """
    Two-stage:
      phase 'node': pick (x,y) or node index
      phase 'bin' : pick new bin for that node
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        som: SOMState,
        max_steps: int = 500,
        invalid_penalty: float = -1.0,
        neg_eps: float = 0.0,
        cooldown_T: int = 0,
        seed: Optional[int] = None,
        render_scale: int = 4,
    ):
        super().__init__()
        self.som = som
        self.rng = np.random.default_rng(seed)

        self.H, self.W = self.som.grid_shape
        self.K = int(self.som.num_bins)  # SOM.py uses num_bins
        self.N = int(self.som.num_nodes)

        self.max_steps = int(max_steps)
        self.invalid_penalty = float(invalid_penalty)
        self.neg_eps = float(neg_eps)
        self.cool_T = int(cooldown_T)
        self.render_scale = int(render_scale)

        # Observation: keep same channels you already used (bins + umatrix + counts etc).
        # MVP: just bins + umatrix
        self.C = 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.H, self.W, self.C), dtype=np.float32
        )

        # One action space; interpret based on phase:
        self.action_space = spaces.Discrete(max(self.N, self.K))

        self.phase = "node"
        self.selected_node: Optional[int] = None
        self.steps = 0

        self._cool_until = np.zeros((self.N, self.K), dtype=np.int32)
        self._obs = np.zeros((self.H, self.W, self.C), dtype=np.float32)
        self.current_reward: float = 0.0

    def _rebuild_obs(self) -> None:
        bins_grid = self.som.get_assigned_bins(grid=True).astype(np.float32)
        self._obs[..., 0] = bins_grid / max(1, (self.K - 1))

        # SOM.py stores umatrix internally; expose via getter if you have it; otherwise zeros.
        # If SOM.py has no getter, you can add one (recommended) or keep zeros for MVP.
        try:
            U = self.som.get_umatrix()  # add this if missing
            U = U.astype(np.float32)
            mu, sig = float(U.mean()), float(U.std() + 1e-6)
            self._obs[..., 1] = (U - mu) / sig
        except Exception:
            self._obs[..., 1] = 0.0

    def valid_action_mask(self) -> np.ndarray:
        if self.phase == "node":
            return np.ones((self.N,), dtype=bool)

        node = int(self.selected_node) if self.selected_node is not None else -1
        mask = np.zeros((self.K,), dtype=bool)
        # Use your SOM.py delta predictor if available:
        try:
            deltas = self.som.estimate_delta_trace_bins_for_node(node)
            # Valid if delta is finite and cooldown allows
            for b in range(self.K):
                if self._cool_until[node, b] > self.steps:
                    continue
                if np.isfinite(deltas[b]):
                    mask[b] = True
        except Exception:
            # fallback: allow all bins (except cooldown)
            for b in range(self.K):
                if self._cool_until[node, b] <= self.steps:
                    mask[b] = True
        return mask

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.som.reset_to_initial()
        self.phase = "node"
        self.selected_node = None
        self.steps = 0
        self._cool_until.fill(0)

        self._rebuild_obs()
        self.current_reward = float(self.som.total_reward_fast())
        return self._obs, {"phase": self.phase, "reward_abs": self.current_reward}

    def step(self, action: int):
        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps
        info: Dict[str, Any] = {"phase": self.phase}

        if self.phase == "node":
            node = int(action)
            if not (0 <= node < self.N):
                info["invalid"] = True
                info["invalid_reason"] = "node_oob"
                info["reward_abs"] = self.current_reward
                return self._obs, self.invalid_penalty, terminated, truncated, info

            self.selected_node = node
            self.phase = "bin"
            info["selected_node"] = node
            return self._obs, 0.0, terminated, truncated, info

        node = int(self.selected_node) if self.selected_node is not None else -1
        b = int(action)
        if not (0 <= b < self.K):
            info["invalid"] = True
            info["invalid_reason"] = "bin_oob"
            info["reward_abs"] = self.current_reward
            self.phase = "node"
            self.selected_node = None
            return self._obs, self.invalid_penalty, terminated, truncated, info

        # compute delta using the fast estimator if present; else fallback to recompute diff
        try:
            delta = float(self.som.estimate_delta_trace(node, b))
            # apply move
            old_bin = int(self.som.current_bins[node])
            self.som.set_node_bin_idx(node, b)  # add this helper, or use existing set_node_bin(x,y,b)
            self.current_reward += delta
            reward = delta
        except Exception:
            r_before = self.current_reward
            old_bin = int(self.som.current_bins[node])
            self.som.set_node_bin_idx(node, b)  # add this helper
            self.current_reward = float(self.som.total_reward_fast())
            reward = float(self.current_reward - r_before)

        # cooldown
        if reward <= self.neg_eps and self.cool_T > 0:
            self._cool_until[node, b] = self.steps + self.cool_T

        # update obs bin channel only (cheap)
        x, y = self.som.index_to_xy(node)
        self._obs[x, y, 0] = float(b) / max(1, (self.K - 1))

        info.update(
            selected_node=node,
            old_bin=old_bin,
            new_bin=b,
            reward_abs=self.current_reward,
        )

        self.phase = "node"
        self.selected_node = None
        return self._obs, reward, terminated, truncated, info
