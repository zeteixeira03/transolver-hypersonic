"""
SU2 hypersonic dataset for the Phase 4 transfer-learning study.

Each training case is one converged SU2 axisymmetric sphere-cone simulation,
stored as a compressed .npz produced by
:func:`src.cfd.postprocess.extract_training_tensors`. Fields per case:

    x, r                            -- node coordinates, shape (N,)
    rho, u, v, T                    -- primitive flow fields, shape (N,)
    case_params                     -- 8 case parameters, shape (8,)

Pressure is not stored; it is reconstructed at inference time as
``p = rho * R_specific * T`` per the CLAUDE.md output convention.

Inputs to the model are per-node and stack as

    [x, r, R_n, theta_c, R_b, R_s, M, T_inf, p_inf, T_w]    (10 channels)

with the 8 case parameters broadcast from shape (8,) to (N, 8). Outputs are
``[rho, u, v, T]`` (4 channels) per node.

The dataset loads .npz files lazily on ``__getitem__``; at ~60k nodes and
six float32 fields per case the on-disk footprint is roughly 1.4 MB compressed
per case (~1 GB across the planned 780-case sweep), well under the memory
budget for a 4-core CPU box but too large to hold in RAM as float tensors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================================================
#                                       channel layout
# ============================================================================================

CASE_PARAM_ORDER = (
    "R_n", "theta_c_deg", "R_b", "R_s",
    "mach", "T_inf", "p_inf", "T_w",
)
N_CASE_PARAMS = len(CASE_PARAM_ORDER)
POS_DIM = 2                                   # (x, r)
INPUT_DIM = POS_DIM + N_CASE_PARAMS           # 10
TARGET_ORDER = ("rho", "u", "v", "T")
TARGET_DIM = len(TARGET_ORDER)                # 4

R_SPECIFIC_AIR = 287.058                      # J/(kg K), reconstruct p = rho R T


# ============================================================================================
#                                       normalization stats
# ============================================================================================

# targets standardized in log10 space by default. the W0 normalization study
# (data/samples/phase4_normalization.png) pooled 21 cases: plain z on rho has
# skew 7.0, 91% of nodes inside |z| < 0.5 and a 25-sigma shock-layer tail; T
# reaches 29450 K with a 4-sigma tail. log10 flattens both (max |z| < 4, skew
# < 0.6). u and v are well-behaved under plain standardization.
DEFAULT_LOG_TARGETS = ("rho", "T")


@dataclass
class SU2NormStats:
    """Per-channel mean and std for inputs and targets.

    Inputs are 10-channel ``[x, r, R_n, theta_c, R_b, R_s, M, T_inf, p_inf, T_w]``.
    Targets are 4-channel ``[rho, u, v, T]``. Each channel is standardized
    independently; the CLAUDE.md numerical-scale rule forbids sharing
    constants across fields with different orders of magnitude. Channels named
    in ``log_targets`` are mapped to log10 before standardization (stats are
    fit in log space); the flag persists with the stats so training and
    inference cannot disagree.
    """

    x_mean: torch.Tensor   # (10,)
    x_std: torch.Tensor    # (10,)
    y_mean: torch.Tensor   # (4,)
    y_std: torch.Tensor    # (4,)
    log_targets: tuple[str, ...] = DEFAULT_LOG_TARGETS

    @property
    def log_mask(self) -> torch.Tensor:
        return torch.tensor(
            [name in self.log_targets for name in TARGET_ORDER], dtype=torch.bool,
        )

    def to(self, device: torch.device) -> "SU2NormStats":
        return SU2NormStats(
            x_mean=self.x_mean.to(device),
            x_std=self.x_std.to(device),
            y_mean=self.y_mean.to(device),
            y_std=self.y_std.to(device),
            log_targets=self.log_targets,
        )

    def save(self, path: str | Path) -> None:
        torch.save({
            "x_mean": self.x_mean, "x_std": self.x_std,
            "y_mean": self.y_mean, "y_std": self.y_std,
            "log_targets": list(self.log_targets),
        }, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "SU2NormStats":
        d = torch.load(str(path), map_location="cpu")
        return cls(d["x_mean"], d["x_std"], d["y_mean"], d["y_std"],
                   tuple(d.get("log_targets", ())))


def transform_targets(y_raw: np.ndarray, log_targets: tuple[str, ...]) -> np.ndarray:
    """Map physical targets into the space stats are fit in (log10 where flagged)."""
    y = y_raw.copy()
    for i, name in enumerate(TARGET_ORDER):
        if name in log_targets:
            y[:, i] = np.log10(np.clip(y[:, i], 1e-30, None))
    return y


# ============================================================================================
#                                       per-case loading
# ============================================================================================

def load_case_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load one case.npz and return its arrays.

    Raises ``ValueError`` if ``case_params`` is missing -- the dataset
    requires self-describing tensors so that downstream training is not
    coupled to a separate ledger or manifest.
    """
    d = np.load(str(path))
    required = ("x", "r", "rho", "u", "v", "T", "case_params")
    missing = [k for k in required if k not in d.files]
    if missing:
        raise ValueError(
            f"case.npz at {path} is missing required keys {missing}; "
            f"re-extract with extract_training_tensors(..., case=case)"
        )
    case_params = d["case_params"]
    if case_params.shape != (N_CASE_PARAMS,):
        raise ValueError(
            f"case_params at {path} has shape {case_params.shape}, "
            f"expected ({N_CASE_PARAMS},)"
        )
    return {k: np.asarray(d[k], dtype=np.float32) for k in required}


def stack_case_features(case_data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Build per-node (inputs, targets) for one case.

    Returns
    -------
    x : np.ndarray, shape (N, 10)
        Stacked ``[x, r, ...case_params (broadcast)]``.
    y : np.ndarray, shape (N, 4)
        Stacked ``[rho, u, v, T]``.
    """
    x_node = case_data["x"]
    r_node = case_data["r"]
    params = case_data["case_params"]                          # (8,)
    n = x_node.shape[0]
    case_broadcast = np.broadcast_to(params, (n, N_CASE_PARAMS))
    feats = np.concatenate(
        [x_node[:, None], r_node[:, None], case_broadcast], axis=1,
    )                                                          # (N, 10)
    targets = np.stack(
        [case_data["rho"], case_data["u"], case_data["v"], case_data["T"]],
        axis=1,
    )                                                          # (N, 4)
    return feats.astype(np.float32, copy=False), targets.astype(np.float32, copy=False)


# ============================================================================================
#                                       streaming Welford stats
# ============================================================================================

def compute_norm_stats(
    paths: list[str | Path],
    *,
    eps: float = 1e-8,
    log_targets: tuple[str, ...] = DEFAULT_LOG_TARGETS,
) -> SU2NormStats:
    """Compute per-channel mean/std across all training points.

    Loads one case at a time (constant peak memory in the number of cases)
    and merges per-case (mean, M2, count) summaries with the parallel-Welford
    update -- same pattern as :func:`src.data.airfrans.compute_norm_stats`.

    Parameters
    ----------
    paths : list of paths
        One ``case.npz`` per training case. Must be the train split only --
        leaking val/test points into the stats inflates apparent train
        accuracy.
    eps : float
        Added to the std to avoid divide-by-zero on degenerate channels.
    """
    n_total = 0
    x_mean: np.ndarray | None = None
    x_m2: np.ndarray | None = None
    y_mean: np.ndarray | None = None
    y_m2: np.ndarray | None = None

    for path in paths:
        case = load_case_npz(path)
        feats, targets = stack_case_features(case)
        targets = transform_targets(targets, log_targets)
        n = feats.shape[0]
        if n == 0:
            continue

        mu_x = feats.mean(axis=0)
        d_x = feats - mu_x
        m2_x = (d_x * d_x).sum(axis=0)
        mu_y = targets.mean(axis=0)
        d_y = targets - mu_y
        m2_y = (d_y * d_y).sum(axis=0)

        if x_mean is None:
            x_mean, x_m2, y_mean, y_m2, n_total = mu_x, m2_x, mu_y, m2_y, n
            continue

        n_new = n_total + n
        delta_x = mu_x - x_mean
        x_mean = x_mean + delta_x * (n / n_new)
        x_m2 = x_m2 + m2_x + (delta_x * delta_x) * (n_total * n / n_new)
        delta_y = mu_y - y_mean
        y_mean = y_mean + delta_y * (n / n_new)
        y_m2 = y_m2 + m2_y + (delta_y * delta_y) * (n_total * n / n_new)
        n_total = n_new

    if x_mean is None:
        raise ValueError("compute_norm_stats received no cases")

    denom = max(n_total - 1, 1)
    x_std = np.sqrt(x_m2 / denom) + eps
    y_std = np.sqrt(y_m2 / denom) + eps
    return SU2NormStats(
        x_mean=torch.tensor(x_mean, dtype=torch.float32),
        x_std=torch.tensor(x_std, dtype=torch.float32),
        y_mean=torch.tensor(y_mean, dtype=torch.float32),
        y_std=torch.tensor(y_std, dtype=torch.float32),
        log_targets=log_targets,
    )


# ============================================================================================
#                                       torch dataset
# ============================================================================================

class SU2Dataset(Dataset):
    """Disk-backed dataset over per-case ``case.npz`` files.

    Loads one case per ``__getitem__`` call (lazy), optionally subsampling
    to a fixed number of nodes. Returned tensors are normalized on the fly
    using ``stats``.

    Parameters
    ----------
    paths : list of paths
        One ``case.npz`` per case.
    stats : SU2NormStats
        Normalization stats fit on the train split.
    subsample : int, optional
        If given, draw this many nodes per item (uniform without
        replacement). Resampled per call so each epoch sees a fresh subset.
    """

    def __init__(
        self,
        paths: list[str | Path],
        stats: SU2NormStats,
        subsample: int | None = None,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.stats = stats
        self.subsample = subsample

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[idx]
        case = load_case_npz(path)
        feats, targets = stack_case_features(case)
        n = feats.shape[0]
        if self.subsample is not None and n > self.subsample:
            indices = random.sample(range(n), self.subsample)
            feats = feats[indices]
            targets = targets[indices]

        x_raw = torch.from_numpy(feats)
        y_raw = torch.from_numpy(targets)
        pos = x_raw[:, :POS_DIM].clone()                           # unnormalized (x, r)

        y_t = torch.from_numpy(transform_targets(targets, self.stats.log_targets))
        x = (x_raw - self.stats.x_mean) / self.stats.x_std
        y = (y_t - self.stats.y_mean) / self.stats.y_std

        return {
            "x": x,
            "y": y,
            "y_raw": y_raw,
            "pos": pos,
            "case_params": torch.from_numpy(case["case_params"]),
            "name": path.parent.name if path.name == "case.npz" else path.stem,
        }


# ============================================================================================
#                                       inference helpers
# ============================================================================================

def denormalize_targets(y_norm: torch.Tensor, stats: SU2NormStats) -> torch.Tensor:
    """Map a normalized prediction or target back to physical units (SI).

    Inverts the standardization, then the log10 transform on channels named
    in ``stats.log_targets``.
    """
    y = y_norm * stats.y_std.to(y_norm.device) + stats.y_mean.to(y_norm.device)
    mask = stats.log_mask.to(y_norm.device)
    if mask.any():
        y = y.clone()
        y[..., mask] = torch.pow(10.0, y[..., mask])
    return y


def reconstruct_pressure(rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Pressure from the ideal-gas equation of state, ``p = rho R_specific T``.

    The model predicts ``(rho, u, v, T)`` and ``p`` is reconstructed at
    output time. This is the hard architectural constraint from CLAUDE.md;
    the EoS holds exactly at inference, the network has no degree of
    freedom to violate it.
    """
    return rho * R_SPECIFIC_AIR * T


# ============================================================================================
#                                       case-listing helpers
# ============================================================================================

def list_case_npzs(workdir: str | Path) -> list[Path]:
    """Return the sorted list of case tensors under ``workdir``.

    Accepts both layouts: ``case_XXXX/case.npz`` (sweep workdir) and flat
    ``case_XXXX.npz`` (Kaggle dataset upload). Used both by training (to
    build the path list for ``SU2Dataset``) and by the supervisor for
    completion checks. Sorting keeps the order stable across machines.
    """
    workdir = Path(workdir)
    nested = sorted(workdir.glob("case_*/case.npz"))
    if nested:
        return nested
    return sorted(workdir.glob("case_*.npz"))


def case_id_from_npz_path(path: Path) -> int:
    """``case_0042/case.npz`` or ``case_0042.npz`` -> 42."""
    stem = path.parent.name if path.name == "case.npz" else path.stem
    return int(stem.split("_")[1])
