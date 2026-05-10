"""
AirfRANS dataset wrapper for Phase 1 stack validation.

The official `airfrans` pip package returns one numpy array per simulation of
shape (N, 12) with the following column layout (verified against the package
source at github.com/Extrality/airfrans_lib):

    cols  0..1   position (x, y) in meters
    cols  2..3   inlet velocity (vx, vy), replicated per point
    col   4      signed distance to the airfoil
    cols  5..6   surface normals (zero off the airfoil)
    cols  7..8   target velocity (u, v)
    col   9      target pressure / specific_mass
    col  10      target turbulent kinematic viscosity
    col  11      surface flag (1 if point lies on the airfoil)

Inputs to the model are columns 0..6 (7 channels). Targets are columns 7..10
(4 channels). The surface flag drives the surface-weighted loss term and the
Cp acceptance figure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================================================
#                                       column indices
# ============================================================================================
INPUT_COLS = slice(0, 7)
TARGET_COLS = slice(7, 11)
SURFACE_COL = 11
INLET_COLS = slice(2, 4)


# ============================================================================================
#                                       normalization stats
# ============================================================================================
@dataclass
class NormStats:
    """Per-channel mean and std for inputs and targets."""

    x_mean: torch.Tensor  # (7,)
    x_std: torch.Tensor   # (7,)
    y_mean: torch.Tensor  # (4,)
    y_std: torch.Tensor   # (4,)

    def to(self, device: torch.device) -> "NormStats":
        return NormStats(
            x_mean=self.x_mean.to(device),
            x_std=self.x_std.to(device),
            y_mean=self.y_mean.to(device),
            y_std=self.y_std.to(device),
        )


def compute_norm_stats(arrays: list[np.ndarray]) -> NormStats:
    """Compute per-channel mean/std across all training points using parallel
    Welford accumulation. Constant peak memory in the number of cases.

    The previous implementation concatenated every array into one tensor
    before calling .std(), which peaked at roughly 4x the dataset size and
    OOM'd Colab's 12 GB on AirfRANS scarce. The parallel-Welford merge
    formula combines per-case (mean, M2, count) summaries without ever
    holding the full point set in memory.
    """
    n_total = 0
    mean_acc: np.ndarray | None = None
    m2_acc: np.ndarray | None = None
    for arr in arrays:
        n = arr.shape[0]
        if n == 0:
            continue
        mu = arr.mean(axis=0)
        d = arr - mu
        m2 = (d * d).sum(axis=0)
        if mean_acc is None:
            mean_acc, m2_acc, n_total = mu, m2, n
            continue
        n_new = n_total + n
        delta = mu - mean_acc
        mean_acc = mean_acc + delta * (n / n_new)
        m2_acc = m2_acc + m2 + (delta * delta) * (n_total * n / n_new)
        n_total = n_new

    if mean_acc is None:
        raise ValueError("compute_norm_stats received no points")

    var = m2_acc / max(n_total - 1, 1)
    std = np.sqrt(var)
    eps = 1e-8
    return NormStats(
        x_mean=torch.tensor(mean_acc[INPUT_COLS], dtype=torch.float32),
        x_std=torch.tensor(std[INPUT_COLS] + eps, dtype=torch.float32),
        y_mean=torch.tensor(mean_acc[TARGET_COLS], dtype=torch.float32),
        y_std=torch.tensor(std[TARGET_COLS] + eps, dtype=torch.float32),
    )


# ============================================================================================
#                                       torch dataset
# ============================================================================================
class AirfRANSDataset(Dataset):
    """
    Each item is a single simulation, optionally subsampled to a fixed number
    of points. Returned tensors are normalized on the fly using `stats`.

    Parameters
    ----------
    arrays : list of np.ndarray
        One (N_i, 12) array per simulation, as returned by airfrans.dataset.load.
    names : list of str
        Simulation names, kept for evaluation/plotting.
    stats : NormStats
        Normalization statistics computed on the training split.
    subsample : int, optional
        If given, randomly draw this many points per __getitem__ call.
        Subsampling reseeds per call so each epoch sees a fresh subset.
    """

    def __init__(
        self,
        arrays: list[np.ndarray],
        names: list[str],
        stats: NormStats,
        subsample: int | None = None,
    ) -> None:
        if len(arrays) != len(names):
            raise ValueError("arrays and names must be the same length")
        self.arrays = arrays
        self.names = names
        self.stats = stats
        self.subsample = subsample

    def __len__(self) -> int:
        return len(self.arrays)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        arr = self.arrays[idx]
        if self.subsample is not None and arr.shape[0] > self.subsample:
            indices = random.sample(range(arr.shape[0]), self.subsample)
            arr = arr[indices]

        x_raw = torch.tensor(arr[:, INPUT_COLS], dtype=torch.float32)
        y_raw = torch.tensor(arr[:, TARGET_COLS], dtype=torch.float32)
        surface = torch.tensor(arr[:, SURFACE_COL], dtype=torch.bool)
        pos = torch.tensor(arr[:, 0:2], dtype=torch.float32)

        x = (x_raw - self.stats.x_mean) / self.stats.x_std
        y = (y_raw - self.stats.y_mean) / self.stats.y_std

        return {
            "x": x,
            "y": y,
            "y_raw": y_raw,
            "pos": pos,
            "surface": surface,
            "name": self.names[idx],
        }


# ============================================================================================
#                                       loading helpers
# ============================================================================================
def load_split(
    root: Path | str,
    task: str = "scarce",
    train: bool = True,
) -> tuple[list[np.ndarray], list[str]]:
    """Thin wrapper around airfrans.dataset.load that returns plain lists.

    Cast to float32 on load. The package returns float64, which doubles the
    peak system-RAM footprint on Colab for no model-side benefit (training
    runs in float32 or AMP).
    """
    import airfrans as af

    arrays, names = af.dataset.load(root=str(root), task=task, train=train)
    arrays = [a.astype(np.float32, copy=False) for a in arrays]
    return arrays, names


def denormalize_targets(y_norm: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """Map a normalized prediction or target back to physical units."""
    return y_norm * stats.y_std.to(y_norm.device) + stats.y_mean.to(y_norm.device)
