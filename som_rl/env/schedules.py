from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Deque, Literal
from collections import deque
import numpy as np

ScheduleMode = Literal["fixed", "hit_gated", "plateau_gated", "level_hits"]


@dataclass
class SchedulerConfig:
    # episode
    max_steps: int = 2000

    # levels
    level_min: int = 30
    level_max: int = 80
    # Optional explicit list of levels to step through instead of the
    # continuous [level_min..level_max] range. Useful for "level-based games"
    # where you want, e.g., levels=[30,50,80] rather than every integer.
    level_list: Optional[Tuple[int, ...]] = None

    # temperature
    T_max: float = 1.0
    T_min: float = 0.01
    temp_mode: Literal["exp", "linear"] = "exp"
    temp_evolve_with: Literal["steps", "level"] = "steps"
    temp_fixed: Optional[float] = None

    # deltaE floor
    # If max_abs_deltaE is None => fixed floor = min_abs_deltaE
    # If max_abs_deltaE is set   => floor ramps from min->max as temperature decreases (or as step advances)
    min_abs_deltaE: float = 1e-5
    max_abs_deltaE: Optional[float] = None

    # how to evolve floor when max_abs_deltaE is set
    floor_evolve_with: Literal["temp", "steps", "level"] = "temp"
    # Optional shaping exponent applied to the progress variable p in [0,1]:
    # p <- p ** floor_power. With floor_power=1.0, no additional shaping.
    floor_power: float = 1.0
    # Functional form for the interpolation between min_abs_deltaE and
    # max_abs_deltaE when max_abs_deltaE is not None:
    #   - "linear": affine interpolation between endpoints
    #   - "exp":    geometric / exponential interpolation (like temperature)
    floor_mode: Literal["linear", "exp"] = "linear"
    # Direction of floor evolution between min_abs_deltaE and max_abs_deltaE
    # when max_abs_deltaE is not None:
    #   - "increasing": floor rises from min_abs_deltaE (at p=0) to max_abs_deltaE (at p=1)
    #   - "decreasing": floor falls from max_abs_deltaE (at p=0) to min_abs_deltaE (at p=1)
    floor_direction: Literal["increasing", "decreasing"] = "increasing"
    floor_fixed: Optional[float] = None

    # level control
    level_fixed: Optional[int] = None
    freeze_level: bool = False

    # time control (for temp/floor evolution)
    advance_time_on: Literal["step", "accept", "accepted_non_noop"] = "step"

    # gating (hit)
    min_accepts_per_level: int = 50

    # If True, schedulers that use a discrete level_list (e.g. LevelHitsScheduler)
    # will loop back to the first level after finishing the last one, instead of
    # marking themselves done(). SchedulerConfig.max_steps still acts as a hard
    # upper bound on the total number of update steps.
    repeat_levels: bool = False
    # When repeat_levels is True, optionally limit how many full traversals
    # over level_list are performed before marking the scheduler as done().
    # If None, repeats indefinitely (subject to max_steps). If set to an
    # integer L, the scheduler will loop over level_list at most L times.
    level_loops: Optional[int] = None

    # gating (plateau)
    plateau_window: int = 200
    plateau_eps: float = 1e-6
    plateau_min_accept_rate: float = 0.001

    # optional: reset temp when level advances
    reset_temp_on_level_up: bool = False


class BaseScheduler:
    """
    Standard API:
      - current_params() -> (level, T, floor)
      - step_params() -> (level, T, floor)
      - update(...) called once per env.step after accept/reject
    """

    def __init__(self, cfg: SchedulerConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.step_idx = 0
        if self.cfg.level_fixed is not None:
            self.level = int(self.cfg.level_fixed)
        else:
            self.level = int(self.cfg.level_min)
        self._done = False

        # internal flag to handle reset_temp_on_level_up without being overwritten by update()
        self._requested_step_reset = False

    def done(self) -> bool:
        return bool(self._done)

    # --- main API ---
    def current_params(self) -> Tuple[int, float, float]:
        """Return (level, T, min_abs_deltaE_floor) for the current step."""
        return self.step_params(step_idx=self.step_idx)

    def step_params(self, step_idx: Optional[int] = None) -> Tuple[int, float, float]:
        """Return (level, T, min_abs_deltaE_floor) for *this* env step."""
        if step_idx is None:
            step_idx = self.step_idx

        lvl = int(self.level)
        T = float(self._temperature(step_idx))
        floor = float(self._floor(step_idx=step_idx, level=lvl, T=T))
        return self._apply_fixed(lvl, T, floor)

    def update(
        self,
        *,
        accepted: bool,
        dE: float,
        E: float,
        step_idx: Optional[int] = None,
        is_noop: bool = False,
    ) -> None:
        """Called once per env.step after accept/reject decision."""
        use_env_step = (self.cfg.advance_time_on == "step")
        if step_idx is None or (not use_env_step):
            step_idx = self.step_idx
        step_idx = int(step_idx)

        self._update_impl(
            accepted=bool(accepted),
            dE=float(dE),
            E=float(E),
            step_idx=step_idx,
            is_noop=bool(is_noop),
        )

        should_advance = True
        if self.cfg.advance_time_on == "accept":
            should_advance = bool(accepted)
        elif self.cfg.advance_time_on == "accepted_non_noop":
            should_advance = bool(accepted) and (not bool(is_noop))

        if should_advance:
            # advance time unless a level-up requested a temp reset
            if self._requested_step_reset:
                # reset happens "between" steps: next step starts at 0
                self.step_idx = 0
                self._requested_step_reset = False
            else:
                self.step_idx = step_idx + 1

            if self.step_idx >= int(self.cfg.max_steps):
                self._done = True

    # --- helpers ---
    def _temperature(self, step_idx: int) -> float:
        Tmax, Tmin = float(self.cfg.T_max), float(self.cfg.T_min)
        if self.cfg.temp_fixed is not None:
            return float(self.cfg.temp_fixed)

        if self.cfg.temp_evolve_with == "level":
            denom = max(int(self.cfg.level_max) - int(self.cfg.level_min), 1)
            t = (int(self.level) - int(self.cfg.level_min)) / denom
        else:
            s = int(step_idx)
            n = max(int(self.cfg.max_steps) - 1, 1)
            t = s / n

        if self.cfg.temp_mode == "linear":
            return float(Tmax + t * (Tmin - Tmax))

        # exp interpolation
        # T = Tmax * (Tmin/Tmax)^t
        return float(Tmax * (Tmin / Tmax) ** t)

    def _advance_level(self) -> None:
        if self.cfg.level_fixed is not None or self.cfg.freeze_level:
            return
        if self.level < int(self.cfg.level_max):
            self.level += 1
            if self.cfg.reset_temp_on_level_up:
                # request time reset for next step
                self._requested_step_reset = True

    def _floor(self, *, step_idx: int, level: int, T: float) -> float:
        """
        Floor evolution policy (simple, matches your config):
          - If max_abs_deltaE is None: constant floor = min_abs_deltaE
          - Else: ramp min->max according to cfg.floor_evolve_with
        """
        if self.cfg.floor_fixed is not None:
            return float(max(self.cfg.floor_fixed, 0.0))
        lo = float(self.cfg.min_abs_deltaE)
        hi = self.cfg.max_abs_deltaE
        if hi is None:
            return max(lo, 0.0)
        hi = float(hi)

        # progress p in [0,1]
        mode = self.cfg.floor_evolve_with
        if mode == "steps":
            n = max(int(self.cfg.max_steps) - 1, 1)
            p = float(step_idx) / n

        elif mode == "level":
            denom = max(int(self.cfg.level_max) - int(self.cfg.level_min), 1)
            p = (int(level) - int(self.cfg.level_min)) / denom

        else:  # "temp" (default)
            # p=0 at Tmax, p=1 at Tmin (as T decreases)
            Tmax = max(float(self.cfg.T_max), 1e-30)
            Tmin = max(float(self.cfg.T_min), 1e-30)
            # clamp T
            Tc = float(np.clip(T, Tmin, Tmax))
            # map to [0,1]
            # when Tc=Tmax => p=0; when Tc=Tmin => p=1
            p = (Tmax - Tc) / (Tmax - Tmin) if (Tmax > Tmin) else 1.0

        p = float(np.clip(p, 0.0, 1.0))
        p = p ** float(self.cfg.floor_power)

        # Interpolate according to desired direction and mode.
        mode_floor = self.cfg.floor_mode
        direction = self.cfg.floor_direction

        if mode_floor == "exp":
            # Geometric/exponential interpolation, analogous to temperature.
            # Ensure endpoints are positive to avoid issues.
            lo_pos = max(lo, 1e-30)
            hi_pos = max(hi, lo_pos)
            if direction == "decreasing":
                # hi -> lo as p:0->1
                ratio = lo_pos / hi_pos
                floor = hi_pos * (ratio ** p)
            else:  # "increasing"
                # lo -> hi as p:0->1
                ratio = hi_pos / lo_pos
                floor = lo_pos * (ratio ** p)
        else:
            # Linear interpolation.
            if direction == "decreasing":
                floor = hi + p * (lo - hi)   # hi -> lo
            else:  # "increasing" (default)
                floor = lo + p * (hi - lo)   # lo -> hi

        return float(max(floor, 0.0))

    def _update_impl(self, *, accepted: bool, dE: float, E: float, step_idx: int, is_noop: bool) -> None:
        raise NotImplementedError

    def _apply_fixed(self, lvl: int, T: float, floor: float) -> Tuple[int, float, float]:
        if self.cfg.level_fixed is not None:
            lvl = int(self.cfg.level_fixed)
            self.level = int(lvl)
        if self.cfg.temp_fixed is not None:
            T = float(self.cfg.temp_fixed)
        if self.cfg.floor_fixed is not None:
            floor = float(self.cfg.floor_fixed)
        return int(lvl), float(T), float(floor)


class FixedScheduler(BaseScheduler):
    """
    Monotonic schedule for both T and level across steps:
      level(step) = round(level_min + t*(level_max-level_min))
      T(step) from BaseScheduler._temperature()
    """

    def step_params(self, step_idx: Optional[int] = None) -> Tuple[int, float, float]:
        if step_idx is None:
            step_idx = self.step_idx

        s = int(step_idx)
        n = max(int(self.cfg.max_steps) - 1, 1)
        t = s / n

        lvl = int(round(self.cfg.level_min + t * (self.cfg.level_max - self.cfg.level_min)))
        lvl = max(int(self.cfg.level_min), min(int(self.cfg.level_max), lvl))
        self.level = lvl

        T = float(self._temperature(s))
        floor = float(self._floor(step_idx=s, level=lvl, T=T))
        return self._apply_fixed(lvl, T, floor)

    def _update_impl(self, *, accepted: bool, dE: float, E: float, step_idx: int, is_noop: bool) -> None:
        # no gating; level determined by step_params
        return


class HitGatedScheduler(BaseScheduler):
    """
    Hold level until accepted (and non-noop) moves count reaches min_accepts_per_level, then advance.
    Temperature still decays with step unless reset_temp_on_level_up=True.
    """

    def reset(self):
        super().reset()
        self._acc = 0

    def _update_impl(self, *, accepted: bool, dE: float, E: float, step_idx: int, is_noop: bool) -> None:
        if accepted and (not is_noop):
            self._acc += 1
        if self._acc >= int(self.cfg.min_accepts_per_level):
            self._acc = 0
            self._advance_level()


class PlateauGatedScheduler(BaseScheduler):
    """
    Hold level until accepted dE plateaus in a rolling window:
      - if accept_rate < plateau_min_accept_rate => advance
      - OR if mean accepted dE < plateau_eps     => advance

    Notes:
      - mean_dE computed over accepted, non-noop moves only.
      - accept_rate computed over all steps (accepted flags) in the window.
    """

    def reset(self):
        super().reset()
        W = int(self.cfg.plateau_window)
        self._win_accept: Deque[int] = deque(maxlen=W)
        self._win_dE_acc: Deque[float] = deque(maxlen=W)

    def _update_impl(self, *, accepted: bool, dE: float, E: float, step_idx: int, is_noop: bool) -> None:
        self._win_accept.append(1 if accepted else 0)
        if accepted and (not is_noop):
            self._win_dE_acc.append(float(dE))

        W = int(self.cfg.plateau_window)
        if len(self._win_accept) < W:
            return

        accept_rate = float(np.mean(self._win_accept))
        mean_dE = float(np.mean(self._win_dE_acc)) if len(self._win_dE_acc) > 0 else 0.0

        if (accept_rate < float(self.cfg.plateau_min_accept_rate)) or (mean_dE < float(self.cfg.plateau_eps)):
            self._win_accept.clear()
            self._win_dE_acc.clear()
            self._advance_level()


class LevelHitsScheduler(BaseScheduler):
    """
    Level-list + hit-based progression with an explicit "game complete" flag.

    Semantics:
      - You define a sequence of levels (cfg.level_list). For example:
            level_list = (30, 50, 80)
      - At each level, you must accumulate at least cfg.min_accepts_per_level
        accepted, non-noop moves ("hits") before advancing to the next level.
      - When the last level in level_list has also reached its required hits,
        the scheduler marks itself done() = True. The env can then treat this
        as a natural episode termination condition (in addition to its own
        max_steps cap).

    Notes:
      - Temperature and floor evolution follow BaseScheduler (temp_evolve_with,
        floor_evolve_with, etc.).
      - cfg.max_steps is still honoured via BaseScheduler.update(): if the
        scheduler's internal step_idx reaches cfg.max_steps (according to
        advance_time_on), done() will also become True.
    """

    def reset(self):
        super().reset()
        levels = self._levels()
        self._level_idx = 0
        if len(levels) > 0:
            self.level = int(levels[0])
        self._acc = 0
        # number of completed full traversals over level_list
        self._cycles = 0

    def _levels(self) -> Tuple[int, ...]:
        if self.cfg.level_list is not None and len(self.cfg.level_list) > 0:
            return tuple(int(l) for l in self.cfg.level_list)
        # Fallback: derive a list from [level_min..level_max] inclusive
        start = int(self.cfg.level_min)
        end = int(self.cfg.level_max)
        if end < start:
            start, end = end, start
        return tuple(range(start, end + 1))

    def _advance_level_list(self) -> None:
        levels = self._levels()
        if len(levels) == 0:
            return
        if self._level_idx + 1 < len(levels):
            self._level_idx += 1
            self.level = int(levels[self._level_idx])
            if self.cfg.reset_temp_on_level_up:
                self._requested_step_reset = True
        else:
            # Reached the final level and satisfied its hit quota.
            if self.cfg.repeat_levels:
                # Completed a full cycle over all levels.
                self._cycles += 1
                # If level_loops is set and we've reached the limit, stop.
                if (self.cfg.level_loops is not None) and (self._cycles >= int(self.cfg.level_loops)):
                    self._done = True
                    return
                # Otherwise, loop back to the first level; BaseScheduler.max_steps
                # still provides the overall termination cap.
                self._level_idx = 0
                self.level = int(levels[0])
                if self.cfg.reset_temp_on_level_up:
                    self._requested_step_reset = True
            else:
                # Mark scheduler as done; BaseScheduler.update() will keep it done.
                self._done = True

    def _update_impl(self, *, accepted: bool, dE: float, E: float, step_idx: int, is_noop: bool) -> None:
        if self.done():
            return
        if accepted and (not is_noop):
            self._acc += 1
        if self._acc >= int(self.cfg.min_accepts_per_level):
            self._acc = 0
            self._advance_level_list()


def make_scheduler(mode: ScheduleMode, cfg: SchedulerConfig) -> BaseScheduler:
    mode = str(mode)
    if mode == "fixed":
        return FixedScheduler(cfg)
    if mode == "hit_gated":
        return HitGatedScheduler(cfg)
    if mode == "plateau_gated":
        return PlateauGatedScheduler(cfg)
    if mode == "level_hits":
        return LevelHitsScheduler(cfg)
    raise ValueError(f"Unknown schedule mode: {mode}")
