from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from gymnasium import Env, spaces

from som_rl.som.som_state import SOMState


class ConveyorSweepEnv(Env):
    """
    Single-stage:
      action = bin choice for current pointer node
      pointer advances each step (wraps).
    """

    def __init__(
        self,
        som: SOMState,
        max_steps: int = 500,
        invalid_penalty: float = -1.0,
        start_node: int = 0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.som = som
        self.H, self.W = som.H, som.W
        self.N, self.K = som.N, som.K

        self.max_steps = int(max_steps)
        self.invalid_penalty = float(invalid_penalty)

        self.ptr = int(start_node)
        self.steps = 0
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.H, self.W, 2), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.K)

        self._obs = np.zeros((self.H, self.W, 2), dtype=np.float32)
        self.current_reward: float = 0.0

    def _rebuild_obs(self) -> None:
        bins = self.som.current_bins.reshape(self.H, self.W).astype(np.float32)
        self._obs[..., 0] = bins / max(1, (self.K - 1))
        self._obs[..., 1] = 0.0 if self.som.umatrix is None else self.som.umatrix.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.som.reset_to_initial()
        self.steps = 0
        self.ptr = 0

        self._rebuild_obs()
        self.current_reward = self.som.total_reward_fast()
        return self._obs, {"ptr": self.ptr, "reward_abs": self.current_reward}

    def step(self, action: int):
        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps
        info: Dict[str, Any] = {"ptr": self.ptr}

        b = int(action)
        if not (0 <= b < self.K):
            info["invalid"] = True
            info["invalid_reason"] = "bin_oob"
            info["reward_abs"] = self.current_reward
            self.ptr = (self.ptr + 1) % self.N
            return self._obs, self.invalid_penalty, terminated, truncated, info

        if not self.som.valid_bin_for_node(self.ptr, b):
            info["invalid"] = True
            info["invalid_reason"] = "bin_invalid_for_node"
            info["reward_abs"] = self.current_reward
            self.ptr = (self.ptr + 1) % self.N
            return self._obs, self.invalid_penalty, terminated, truncated, info

        r_before = self.current_reward
        old_bin = int(self.som.current_bins[self.ptr])

        self.som.apply_move(self.ptr, b)
        self.current_reward = self.som.total_reward_fast()
        reward = float(self.current_reward - r_before)

        # update obs bin channel for that cell only
        if old_bin != b:
            x, y = self.som.index_to_xy(self.ptr)
            self._obs[x, y, 0] = float(b) / max(1, (self.K - 1))

        info["old_bin"] = old_bin
        info["new_bin"] = b
        info["reward_abs"] = self.current_reward

        self.ptr = (self.ptr + 1) % self.N
        return self._obs, reward, terminated, truncated, info
