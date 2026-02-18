"""
Save/load ΔE floor profiles (level -> floor) for use with EnergyFloorSchedule.

Profiles are stored as JSON: {"level_min": 30, "level_max": 80, "floors": [f30, f31, ...]}
or as a dict mapping level (int) -> floor (float). The loader returns a dict so you can
pass it to EnergyFloorSchedule(table=load_floor_profile(path)).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

from som_rl.env.env import EnergyFloorSchedule


def load_floor_profile(path: Union[str, Path]) -> Dict[int, float]:
    """
    Load a floor profile from disk.

    Supports:
      - JSON: object with "floors" (list, index 0 = level_min) and "level_min" (int),
        or object with string keys "30", "31", ... mapping to float floor values.
      - .npz: arrays "levels" (int) and "floors" (float), same length.

    Returns:
      dict[int, float]: level -> floor, suitable for EnergyFloorSchedule(table=...).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Floor profile not found: {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            data = json.load(f)

        if "floors" in data and "level_min" in data:
            level_min = int(data["level_min"])
            floors = [float(f) for f in data["floors"]]
            return {level_min + i: f for i, f in enumerate(floors)}

        # Fallback: keys are level strings
        return {int(k): float(v) for k, v in data.items() if k.lstrip("-").isdigit()}

    if path.suffix.lower() == ".npz":
        import numpy as np

        npz = np.load(path)
        levels = np.asarray(npz["levels"]).astype(int)
        floors = np.asarray(npz["floors"]).astype(float)
        return dict(zip(levels.tolist(), floors.tolist()))

    raise ValueError(f"Unknown format for {path}; use .json or .npz")


def save_floor_profile(
    profile: Dict[int, float],
    path: Union[str, Path],
    *,
    level_min: int,
    level_max: int,
    format: str = "json",
) -> None:
    """
    Save a floor profile to disk.

    Args:
      profile: dict level -> floor (can be sparse; missing levels use default in loader).
      path: output path.
      level_min, level_max: range of levels (inclusive). Used to emit a dense "floors" list.
      format: "json" or "npz".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    levels = list(range(level_min, level_max + 1))
    floors = [float(profile.get(l, 0.0)) for l in levels]

    if format == "json":
        data = {"level_min": level_min, "level_max": level_max, "floors": floors}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return

    if format == "npz":
        import numpy as np

        np.savez(path, levels=np.array(levels), floors=np.array(floors))
        return

    raise ValueError(f"format must be 'json' or 'npz', got {format!r}")


def floor_schedule_from_profile(
    path: Union[str, Path], default: float = 0.0
) -> EnergyFloorSchedule:
    """
    Load a floor profile and return an EnergyFloorSchedule for use in env config.

    Usage:
      from som_rl.env.floor_profile import floor_schedule_from_profile
      cfg = SOMEnergyEnvConfig(energy_floor=floor_schedule_from_profile("data/floor_profile.json"))
    """
    table = load_floor_profile(path)
    return EnergyFloorSchedule(table=table, default=default)
