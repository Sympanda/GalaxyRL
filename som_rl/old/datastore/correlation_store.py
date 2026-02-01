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
    return np.load(path, mmap_mode="r" if memmap else None)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    # NOTE: npz is not memmapped; arrays are loaded via zip.
    obj = np.load(path, allow_pickle=False)
    return {k: obj[k] for k in obj.files}


def _load(path: Optional[Union[str, Path]], memmap: bool) -> Any:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix == ".npy":
        return _load_npy(p, memmap=memmap)
    if p.suffix == ".npz":
        return _load_npz(p)
    if p.suffix in (".pkl", ".pickle"):
        return _load_pickle(p)
    raise ValueError(f"Unsupported file type: {p}")


@dataclass(frozen=True)
class CorrelationPaths:
    # Option 1: explicit arrays
    dd: Optional[Union[str, Path]] = None
    dr: Optional[Union[str, Path]] = None
    rd: Optional[Union[str, Path]] = None
    rr: Optional[Union[str, Path]] = None

    # Option 2: bundled npz containing dd/dr/rd/rr
    compact_npz: Optional[Union[str, Path]] = None

    # Required
    node_indices_photo: Union[str, Path] = ""
    initial_node_bins: Union[str, Path] = ""

    # Optional
    index_to_xy: Optional[Union[str, Path]] = None   # (N_nodes,2) int64
    umatrix: Optional[Union[str, Path]] = None       # (H,W) float32


class CorrelationStore:
    """
    Loads and caches the big dataset once.

    Training-time expectation:
      dd/dr/rd/rr should be numeric arrays (float32/float64), ideally memmapped .npy,
      in compact (K,N) or transposable (N,K) form.

    Pickled TreeCorr objects are not supported for training (you can still load them,
    but validation will reject them unless you disable validation explicitly).
    """

    def __init__(self, paths: CorrelationPaths, memmap: bool = True):
        self.paths = paths
        self.memmap = bool(memmap)
        self._cache: Dict[str, Any] = {}

    def load(
        self,
        *,
        expected_num_bins: Optional[int] = None,   # K
        expected_num_nodes: Optional[int] = None,  # N
        allow_pickle_corrs: bool = False,
        auto_transpose_kn: bool = True,
    ) -> Dict[str, Any]:
        if self._cache:
            return self._cache

        d: Dict[str, Any] = {}

        # Load correlations
        if self.paths.compact_npz is not None:
            bundle = _load(self.paths.compact_npz, memmap=False)
            if not isinstance(bundle, dict):
                raise TypeError("compact_npz must load to a dict-like of arrays")
            for k in ("dd", "dr", "rd", "rr"):
                if k not in bundle:
                    raise KeyError(f"compact_npz missing key '{k}'")
                d[k] = bundle[k]
        else:
            for k in ("dd", "dr", "rd", "rr"):
                p = getattr(self.paths, k)
                if p is None:
                    raise ValueError(f"Missing correlation path for '{k}' (or provide compact_npz).")
                d[k] = _load(p, self.memmap)

        # Load indices / metadata
        d["node_indices_photo"] = _load(self.paths.node_indices_photo, self.memmap)
        d["initial_node_bins"] = _load(self.paths.initial_node_bins, self.memmap)
        d["index_to_xy"] = _load(self.paths.index_to_xy, self.memmap)
        d["umatrix"] = _load(self.paths.umatrix, self.memmap)

        # Convert lists to arrays if pickles returned python lists
        for k, v in list(d.items()):
            if isinstance(v, list):
                d[k] = np.asarray(v)

        # Basic sanity checks
        self._validate_loaded(
            d,
            expected_num_bins=expected_num_bins,
            expected_num_nodes=expected_num_nodes,
            allow_pickle_corrs=allow_pickle_corrs,
            auto_transpose_kn=auto_transpose_kn,
        )

        # Mark arrays as read-only (prevents accidental in-place mutation)
        for k in ("dd", "dr", "rd", "rr", "node_indices_photo", "index_to_xy", "umatrix", "initial_node_bins"):
            arr = d.get(k, None)
            if isinstance(arr, np.ndarray):
                try:
                    arr.flags.writeable = False
                except Exception:
                    pass

        self._cache = d
        return d

    def _validate_loaded(
        self,
        d: Dict[str, Any],
        *,
        expected_num_bins: Optional[int],
        expected_num_nodes: Optional[int],
        allow_pickle_corrs: bool,
        auto_transpose_kn: bool,
    ) -> None:
        # Required vectors
        node_idx = d.get("node_indices_photo")
        init_bins = d.get("initial_node_bins")
        if not isinstance(node_idx, np.ndarray) or node_idx.ndim != 1:
            raise TypeError("node_indices_photo must be a 1D numpy array")
        if not isinstance(init_bins, np.ndarray) or init_bins.ndim != 1:
            raise TypeError("initial_node_bins must be a 1D numpy array")

        N = init_bins.shape[0] if expected_num_nodes is None else int(expected_num_nodes)

        # Validate correlations
        for k in ("dd", "dr", "rd", "rr"):
            A = d.get(k)
            if not isinstance(A, np.ndarray):
                if allow_pickle_corrs:
                    continue
                raise TypeError(f"{k} must be a numpy array for training; got {type(A)}")
            if A.dtype == object:
                raise TypeError(f"{k} has dtype=object (likely pickled TreeCorr objects). Not supported.")
            if A.ndim != 2:
                raise ValueError(f"{k} must be 2D array (K,N) or (N,K), got shape {A.shape}")

            r, c = A.shape
            # Reject N x N early
            if expected_num_nodes is not None and r == N and c == N:
                raise ValueError(f"{k} appears to be (N,N)={A.shape}. This is not feasible for 200x200.")

            # Infer K/N if not supplied
            if expected_num_bins is not None:
                K = int(expected_num_bins)
                if (r, c) == (K, N):
                    pass
                elif (r, c) == (N, K) and auto_transpose_kn:
                    d[k] = A.T
                else:
                    raise ValueError(f"{k} has shape {A.shape}; expected (K,N)=({K},{N}) or (N,K).")
            else:
                # Heuristic: if one dim matches N, treat other as K
                if r == N and auto_transpose_kn:
                    d[k] = A.T
                elif c != N:
                    # Can't validate without K, but at least require one dim equals N
                    raise ValueError(f"{k} shape {A.shape}: neither dim equals N={N}. Provide expected_num_bins/nodes.")

        # Validate index_to_xy if present
        itxy = d.get("index_to_xy")
        if itxy is not None:
            if not isinstance(itxy, np.ndarray) or itxy.shape != (N, 2):
                raise ValueError(f"index_to_xy must be shape (N,2)=({N},2), got {None if itxy is None else itxy.shape}")

        # Validate umatrix if present (can’t fully validate H,W here)
        uma = d.get("umatrix")
        if uma is not None and not isinstance(uma, np.ndarray):
            raise TypeError("umatrix must be a numpy array if provided")
