"""
AirfRANS dataset wrapper for stack validation.

The official `airfrans` package returns one array per simulation, of shape
(N, 12). The column layout below is the one in the package source at
github.com/Extrality/airfrans_lib:

    columns  0..1   position (x, y) in metres
    columns  2..3   inlet velocity (vx, vy), repeated at every point
    column   4      signed distance to the airfoil
    columns  5..6   surface normals, zero away from the airfoil
    columns  7..8   target velocity (u, v)
    column   9      target pressure divided by specific mass
    column  10      target turbulent kinematic viscosity
    column  11      surface flag, 1 where the point lies on the airfoil

The model reads columns 0 to 6, which is 7 channels, and predicts columns 7 to
10, which is 4. The surface flag drives both the surface term of the loss and
the pressure-coefficient acceptance figure.
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
    """Per-channel mean and standard deviation for the inputs and targets."""

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
    """Per-channel mean and standard deviation over every training point.

    The parallel form of the Welford update merges the per-case
    ``(mean, M2, count)`` summaries, so peak memory does not grow with the
    number of cases. Memory is the reason for it. One concatenated tensor over
    the whole point set peaks near four times the size of the dataset, which
    exceeds the 12 GB a free Colab session offers on the scarce task.
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
    x_mean = mean_acc[INPUT_COLS].copy()
    x_std = std[INPUT_COLS].copy() + eps
    # the surface normals sit at input indices 5 and 6. they are zero away from
    # the airfoil and unit on it, so their fitted standard deviation is tiny and
    # standardization would push the on-airfoil values to 12 or 15 sigma. the
    # activations that follow then reach infinity within a few epochs. keep
    # these two channels at their native [-1, 1] scale.
    x_mean[5:7] = 0.0
    x_std[5:7] = 1.0
    return NormStats(
        x_mean=torch.tensor(x_mean, dtype=torch.float32),
        x_std=torch.tensor(x_std, dtype=torch.float32),
        y_mean=torch.tensor(mean_acc[TARGET_COLS], dtype=torch.float32),
        y_std=torch.tensor(std[TARGET_COLS] + eps, dtype=torch.float32),
    )


# ============================================================================================
#                                       torch dataset
# ============================================================================================
class AirfRANSDataset(Dataset):
    """
    One item is one simulation, drawn down to a fixed number of points when
    asked. ``stats`` normalizes the returned tensors.

    Parameters
    ----------
    arrays : list of np.ndarray
        One array of shape (N_i, 12) per simulation, as airfrans.dataset.load
        returns them.
    names : list of str
        Simulation names, kept for evaluation and for the figures.
    stats : NormStats
        Normalization statistics computed on the training split.
    subsample : int, optional
        Draw this many points at random per call to ``__getitem__``. Each call
        draws again, so every epoch sees a different subset.
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
    """Load one split through airfrans.dataset.load and return plain lists.

    The package hands back float64 arrays, and the cast to float32 here halves
    the peak memory a free Colab session must hold. The model gains nothing
    from the extra precision, because training runs in float32 or in mixed
    precision.
    """
    import airfrans as af

    arrays, names = af.dataset.load(root=str(root), task=task, train=train)
    arrays = [a.astype(np.float32, copy=False) for a in arrays]
    return arrays, names


def denormalize_targets(y_norm: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """Map a normalized prediction or target back to physical units."""
    return y_norm * stats.y_std.to(y_norm.device) + stats.y_mean.to(y_norm.device)
