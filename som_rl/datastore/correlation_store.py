from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pickle
import numpy as np


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_npy(path: Path, memmap: bool) -> np.ndarray:
    # memmap=True keeps it off-RAM (OS pages it in) and shareable across processes.
    return np.load(path, mmap_mode="r" if memmap else None)


def _load(path: Optional[Union[str, Path]], memmap: bool) -> Any:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix == ".npy":
        return _load_npy(p, memmap=memmap)
    if p.suffix in (".pkl", ".pickle"):
        # Pickle cannot be memmapped; loads fully into RAM.
        return _load_pickle(p)
    raise ValueError(f"Unsupported file type: {p}")


@dataclass(frozen=True)
class CorrelationPaths:
    # Required by SOM.py
    dd: Union[str, Path]
    dr: Union[str, Path]
    rd: Union[str, Path]
    rr: Union[str, Path]
    node_indices_photo: Union[str, Path]
    initial_node_bins: Union[str, Path]

    # Optional
    index_to_xy: Optional[Union[str, Path]] = None   # (N_nodes,2) int64
    umatrix: Optional[Union[str, Path]] = None       # (H,W) float32


class CorrelationStore:
    """
    Loads and caches the big dataset once.
    Use .npy + memmap=True for 9GB-scale arrays.
    """

    def __init__(self, paths: CorrelationPaths, memmap: bool = True):
        self.paths = paths
        self.memmap = bool(memmap)
        self._cache: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        if self._cache:
            return self._cache

        d: Dict[str, Any] = {
            "dd": _load(self.paths.dd, self.memmap),
            "dr": _load(self.paths.dr, self.memmap),
            "rd": _load(self.paths.rd, self.memmap),
            "rr": _load(self.paths.rr, self.memmap),
            "node_indices_photo": _load(self.paths.node_indices_photo, self.memmap),
            "initial_node_bins": _load(self.paths.initial_node_bins, self.memmap),
            "index_to_xy": _load(self.paths.index_to_xy, self.memmap),
            "umatrix": _load(self.paths.umatrix, self.memmap),
        }

        # Convert lists to arrays if pickles returned python lists
        for k, v in list(d.items()):
            if isinstance(v, list):
                d[k] = np.asarray(v)

        # Mark big arrays as read-only (prevents accidental copies from in-place ops)
        for k in ("dd", "dr", "rd", "rr", "node_indices_photo", "index_to_xy", "umatrix"):
            arr = d.get(k, None)
            if isinstance(arr, np.ndarray):
                try:
                    arr.flags.writeable = False
                except Exception:
                    pass

        self._cache = d
        return d
