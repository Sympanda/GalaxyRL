from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .env import SOMEnergyBaseEnv, SOMEnergyEnvConfig, _wrap, _clip


class MultiBaseEnv(SOMEnergyBaseEnv):
    """
    Thin wrapper over SOMEnergyBaseEnv intended for archetype-specific envs
    (bandit, global "minesweeper", local "zelda").

    This class does not change the core backend/scheduler/reward interaction,
    but provides a clear place to override:
      - action space / decoding
      - observation construction
      - (optionally) small archetype-specific reward bonuses on top of
        _compute_reward in the parent class.
    """

    def __init__(self, backend, cfg: SOMEnergyEnvConfig, scheduler: Optional[Any] = None):
        super().__init__(backend=backend, cfg=cfg, scheduler=scheduler)


class EnvBandit(MultiBaseEnv):
    """
    Bandit-style env:
      - Action: node_idx in [0, H*W)
      - Observation: no spatial grids, only scalars and (optionally) cursor.

    This is intentionally minimal: the agent treats each node as an arm, with
    the global energy E providing the main signal via the shaped reward.
    """

    def _build_action_space(self) -> spaces.Space:
        return spaces.Discrete(self.H * self.W)

    def _build_observation_space(self) -> spaces.Space:
        # Minimal bandit-style observation:
        #   - scalars: [level_norm, T, E, steps_left_norm]
        #   - cursor: [x, y] (optional but often still useful)
        obs: Dict[str, spaces.Space] = {}
        obs["scalars"] = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        obs["cursor"] = spaces.Box(low=0, high=max(self.H, self.W), shape=(2,), dtype=np.int32)
        return spaces.Dict(obs)

    def _decode_action(self, action) -> Dict[str, Any]:
        node = int(action)
        return {"node": node}

    def _obs(self) -> Dict[str, Any]:
        # Reuse the scalar packing logic from the base class, but omit grids.
        E = float(self.som.energy_E())
        steps_left = float(self.cfg.max_steps - self._step)
        level_norm = float(self.level) / float(max(1, self.H))
        out: Dict[str, Any] = {
            "scalars": np.array(
                [level_norm, float(self.temperature), E, steps_left / float(max(1, self.cfg.max_steps))],
                dtype=np.float32,
            ),
            "cursor": np.array([self.cursor_x, self.cursor_y], dtype=np.int32),
        }
        return out


class EnvMinesweeper(MultiBaseEnv):
    """
    Global-grid archetype ("minesweeper"):
      - Observation: full SOM grids (bins, optional umatrix/cluster/visit).
      - Action: node_idx in [0, H*W) selecting a single node to modify.

    This is effectively a thin wrapper that uses obs_mode="global" and the
    same action semantics as EnvTeleportNodeRandomBin in env.py.
    """

    def _build_action_space(self) -> spaces.Space:
        return spaces.Discrete(self.H * self.W)

    def _build_observation_space(self) -> spaces.Space:
        # Delegate to the parent implementation but force obs_mode="global"
        # semantics via cfg.
        self.cfg = SOMEnergyEnvConfig(**{**self.cfg.__dict__, "obs_mode": "global"})
        return super()._build_observation_space()

    def _decode_action(self, action) -> Dict[str, Any]:
        node = int(action)
        return {"node": node}


class EnvZelda(MultiBaseEnv):
    """
    Local patch + movement + interact archetype ("zelda"):

      - Observation: patch around the cursor (cfg.obs_mode should be "patch").
      - Action space (Discrete(5)):
          0: move up
          1: move down
          2: move left
          3: move right
          4: interact (propose a bin change at the current cursor node)

      - Movement actions:
          * only move the cursor (respecting torus/clipping as in base env)
          * do NOT touch the backend state
          * treated as noop for the scheduler (accepted=False, is_noop=True)
          * reward comes purely from step_penalty / any exploration bonus you
            may later add on top of _compute_reward if desired.

      - Interact action:
          * uses the same backend/scheduler/reward pipeline as the base env:
                - resolve node from current cursor
                - choose bin (random vs greedy) according to cfg
                - call _attempt_move(...)
                - call scheduler.update(...)
                - compute reward via _compute_reward(...)
    """

    def __init__(self, backend, cfg: SOMEnergyEnvConfig, scheduler: Optional[Any] = None):
        super().__init__(backend=backend, cfg=cfg, scheduler=scheduler)
        # Track how many steps since the last interact action; used for
        # optional forced-interact behaviour controlled by cfg.
        self._since_interact: int = 0

    def _build_action_space(self) -> spaces.Space:
        # 4 directions + 1 interact
        return spaces.Discrete(5)

    def _build_observation_space(self) -> spaces.Space:
        # Ensure obs_mode is patch for this archetype.
        self.cfg = SOMEnergyEnvConfig(**{**self.cfg.__dict__, "obs_mode": "patch"})
        return super()._build_observation_space()

    def _decode_action(self, action) -> Dict[str, Any]:
        # For interact steps we don't need extra decoding; cursor holds the
        # target location. Movement is handled directly in step(...).
        # This method is unused for pure movement actions but we keep a
        # consistent signature.
        return {"x": self.cursor_x, "y": self.cursor_y}

    def step(self, action):
        a = int(action)
        if not (0 <= a < 5):
            raise ValueError("Zelda action out of range")

        step_idx = int(self._step)

        # scheduler drives (level, temperature, floor) for this step
        if self.scheduler is not None:
            self._sync_scheduler_params()

        # Optional forced-interact: if we have not interacted for a while,
        # movement actions may be overridden with an interact at the current
        # cursor location with a probability that increases with
        # self._since_interact.
        if a in (0, 1, 2, 3):
            alpha = float(self.cfg.zelda_force_interact_alpha)
            min_steps = int(self.cfg.zelda_force_interact_min_steps)
            if alpha > 0.0 and self._since_interact >= min_steps:
                # Simple increasing probability, clipped to [0,1].
                k = self._since_interact - min_steps + 1
                p_force = float(np.clip(alpha * k, 0.0, 1.0))
                if float(self._rng.random()) < p_force:
                    a = 4  # force interact

        # --- movement-only actions ---
        if a in (0, 1, 2, 3):
            dx, dy = 0, 0
            if a == 0:   # up
                dy = -1
            elif a == 1:  # down
                dy = 1
            elif a == 2:  # left
                dx = -1
            elif a == 3:  # right
                dx = 1

            x = self.cursor_x + dx
            y = self.cursor_y + dy
            if self.cfg.torus:
                x = _wrap(x, self.W)
                y = _wrap(y, self.H)
            else:
                x = _clip(x, 0, self.W - 1)
                y = _clip(y, 0, self.H - 1)
            self.cursor_x, self.cursor_y = int(x), int(y)

            # movement counts as visiting this node
            node = int(self.som.xy_to_index(self.cursor_x, self.cursor_y))
            self.visit_count[node] += 1
            self.last_visit_step[node] = int(step_idx)

            # update interact counter: this was a movement, not an interact
            self._since_interact += 1

            # scheduler update as a noop step
            if self.scheduler is not None:
                self.scheduler.update(
                    accepted=False,
                    dE=0.0,
                    E=float(self.som.energy_E()),
                    step_idx=step_idx,
                    is_noop=True,
                )

            # env step increments after update
            self._step = step_idx + 1

            # zero "move" reward by default; any step_penalty still applies via
            # the standard reward helper, but we provide a synthetic out dict.
            synthetic_out = {
                "accepted": False,
                "is_noop": True,
                "T": float(self.temperature),
                "E0": float(self.som.energy_E()),
                "E1": float(self.som.energy_E()),
                "dE": 0.0,
                "proposal": {"type": "noop", "reason": "movement"},
                "level": int(self.level),
                "cluster_id": None,
                "n_nodes_touched": 0,
            }
            reward = self._compute_reward(synthetic_out)

            # Zelda-specific exploration bonus for visiting new/rarely visited nodes
            beta = float(self.cfg.zelda_explore_bonus)
            if beta != 0.0:
                vc = float(self.visit_count[node])
                vmax = float(self.visit_count.max(initial=0))
                vc_norm = (vc / vmax) if vmax > 0.0 else 0.0
                reward += beta * (1.0 - vc_norm)

            truncated = (self._step >= int(self.cfg.max_steps))
            terminated = False
            if self.scheduler is not None and hasattr(self.scheduler, "done"):
                try:
                    if bool(self.scheduler.done()):
                        terminated = True
                except Exception:
                    pass

            info = {"step": int(step_idx), "level": int(self.level), **synthetic_out}
            obs = self._obs()
            return obs, float(reward), terminated, truncated, info

        # --- interact action (a == 4) ---
        # Use the same pipeline as the base env, targeting the current cursor.
        node = int(self.som.xy_to_index(self.cursor_x, self.cursor_y))

        # bin selection: use greedy helper if configured and no explicit bin
        new_bin = None
        if self.cfg.use_greedy_bin_for_node:
            best_bin, _ = self.som.greedy_bin_for_node(node=int(node), level=int(self.level))
            new_bin = int(best_bin)

        out = self._attempt_move(node=node, new_bin=new_bin)

        touched = out.get("proposal", {}).get("nodes", None)
        if touched is None:
            touched = np.array([node], dtype=np.int32)
        else:
            touched = np.asarray(touched, dtype=np.int32)

        self.visit_count[touched] += 1
        self.last_visit_step[touched] = int(step_idx)

        # reset interact counter
        self._since_interact = 0

        if self.scheduler is not None:
            self.scheduler.update(
                accepted=bool(out["accepted"]),
                dE=float(out.get("dE", 0.0)),
                E=float(out.get("E1", self.som.energy_E())),
                step_idx=step_idx,
                is_noop=bool(out.get("is_noop", False)),
            )

        self._step = step_idx + 1

        reward = self._compute_reward(out)

        # Zelda-specific exploration bonus based on all touched nodes
        beta = float(self.cfg.zelda_explore_bonus)
        if beta != 0.0:
            vc = float(self.visit_count[touched].mean())
            vmax = float(self.visit_count.max(initial=0))
            vc_norm = (vc / vmax) if vmax > 0.0 else 0.0
            reward += beta * (1.0 - vc_norm)

        truncated = (self._step >= int(self.cfg.max_steps))
        terminated = False
        if self.scheduler is not None and hasattr(self.scheduler, "done"):
            try:
                if bool(self.scheduler.done()):
                    terminated = True
            except Exception:
                pass

        info = {"step": int(step_idx), "level": int(self.level), **out}
        obs = self._obs()
        return obs, float(reward), terminated, truncated, info

