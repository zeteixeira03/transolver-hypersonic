"""
SU2 hypersonic sphere-cone dataset for surrogate training.

Each training case is one converged SU2 axisymmetric sphere-cone simulation,
stored as a compressed .npz produced by
:func:`src.cfd.postprocess.extract_training_tensors`. Fields per case:

    x, r                            -- node coordinates, shape (N,)
    rho, u, v, T                    -- primitive flow fields, shape (N,)
    case_params                     -- 8 case parameters, shape (8,)

Pressure is not stored; it is reconstructed at inference time as
``p = rho * R_specific * T``, so the equation of state holds exactly.

Inputs to the model are per-node and stack as

    [x, r, R_n, theta_c, R_b, R_s, M, T_inf, p_inf, T_w]    (10 channels)

with the 8 case parameters broadcast from shape (8,) to (N, 8). Outputs are
``[rho, u, v, T]`` (4 channels) per node.

One alternative layout exists. With ``with_p`` set, p becomes a fifth free
output channel and its target is ``rho R T`` from the stored fields. That
layout measures what the hard equation-of-state constraint is worth. It is not
the project default.

The dataset loads one .npz file per call to ``__getitem__`` rather than holding
every case in memory. A case carries about 60000 nodes across six float32
fields, so it costs about 1.4 MB on disk and about 1 GB across the 753-case
sweep. That fits a four-core machine on disk. As float tensors in memory it
does not.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.analytical import R_AIR

# ============================================================================================
#                                       channel layout
# ============================================================================================

CASE_PARAM_ORDER = (
    "R_n", "theta_c_deg", "R_b", "R_s",
    "mach", "T_inf", "p_inf", "T_w",
)
N_CASE_PARAMS = len(CASE_PARAM_ORDER)
POS_DIM = 2                                   # (x, r)
TARGET_ORDER = ("rho", "u", "v", "T")

# the free-pressure layout drops the equation-of-state constraint and lets the
# network emit pressure as a fifth channel. everywhere else the project
# reconstructs it from density and temperature. this layout exists to measure
# what that constraint is worth.
TARGET_ORDER_WITH_P = TARGET_ORDER + ("p",)



# ============================================================================================
#                                       normalization stats
# ============================================================================================

# density and temperature are standardized in log10 space by default. a study
# over 21 cases (data/samples/normalization_study.png) measured why. under
# plain standardization density has a skew of 7.0, holds 91% of nodes inside
# |z| < 0.5, and trails a 25-sigma tail through the shock layer; temperature
# reaches 29450 K with a 4-sigma tail. log10 flattens both to a maximum |z|
# under 4 and a skew under 0.6. the two velocity components need no such
# treatment.
DEFAULT_LOG_TARGETS = ("rho", "T")
# pressure runs from about 10 Pa in the freestream to about 1e6 Pa at
# stagnation. the default layout reconstructs it from log-space density and
# temperature, so a log-space pressure channel keeps the two layouts
# comparable.
DEFAULT_LOG_TARGETS_WITH_P = ("rho", "T", "p")


@dataclass
class SU2NormStats:
    """Per-channel mean and std for inputs and targets.

    Inputs hold 10 channels, ``[x, r, R_n, theta_c, R_b, R_s, M, T_inf, p_inf,
    T_w]``. Targets hold ``[rho, u, v, T]``, and ``p`` as well under the
    free-pressure layout. Each channel is standardized on its own.
    """

    x_mean: torch.Tensor   # (10,)
    x_std: torch.Tensor    # (10,)
    y_mean: torch.Tensor   # (len(targets),)
    y_std: torch.Tensor    # (len(targets),)
    log_targets: tuple[str, ...] = DEFAULT_LOG_TARGETS
    targets: tuple[str, ...] = TARGET_ORDER

    @property
    def log_mask(self) -> torch.Tensor:
        return torch.tensor(
            [name in self.log_targets for name in self.targets], dtype=torch.bool,
        )

    def to(self, device: torch.device) -> SU2NormStats:
        return SU2NormStats(
            x_mean=self.x_mean.to(device),
            x_std=self.x_std.to(device),
            y_mean=self.y_mean.to(device),
            y_std=self.y_std.to(device),
            log_targets=self.log_targets,
            targets=self.targets,
        )

    def save(self, path: str | Path) -> None:
        torch.save({
            "x_mean": self.x_mean, "x_std": self.x_std,
            "y_mean": self.y_mean, "y_std": self.y_std,
            "log_targets": list(self.log_targets),
            "targets": list(self.targets),
        }, str(path))

    @classmethod
    def load(cls, path: str | Path) -> SU2NormStats:
        d = torch.load(str(path), map_location="cpu")
        # a saved file may omit "targets" and "log_targets"; both fall back to
        # the four-channel layout
        return cls(d["x_mean"], d["x_std"], d["y_mean"], d["y_std"],
                   tuple(d.get("log_targets", ())),
                   tuple(d.get("targets", TARGET_ORDER)))


def transform_targets(
    y_raw: np.ndarray,
    log_targets: tuple[str, ...],
    targets: tuple[str, ...] = TARGET_ORDER,
) -> np.ndarray:
    """Map physical targets into the space the statistics are fit in.

    Channels named in ``log_targets`` pass through log10. The clip at 1e-30
    keeps a zero or a negative value from producing a non-finite result.
    """
    y = y_raw.copy()
    for i, name in enumerate(targets):
        if name in log_targets:
            y[:, i] = np.log10(np.clip(y[:, i], 1e-30, None))
    return y


# ============================================================================================
#                                       per-case loading
# ============================================================================================

def load_case_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load one case.npz and return its arrays.

    A missing ``case_params`` array raises ``ValueError``. Every case tensor
    must describe itself, so that training reads no separate ledger and no
    separate manifest.
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


def stack_case_features(
    case_data: dict[str, np.ndarray],
    with_p: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-node (inputs, targets) for one case.

    Parameters
    ----------
    case_data : dict
        Arrays from :func:`load_case_npz`.
    with_p : bool
        Append pressure as a fifth target channel. The files store only
        (rho, u, v, T). The ground truth is a calorically perfect ideal gas,
        so ``rho R T`` reproduces the pressure SU2 itself computed.

    Returns
    -------
    x : np.ndarray, shape (N, 10)
        Stacked ``[x, r, ...case_params (broadcast)]``.
    y : np.ndarray, shape (N, 4) or (N, 5)
        Stacked ``[rho, u, v, T]``, plus ``p`` when requested.
    """
    x_node = case_data["x"]
    r_node = case_data["r"]
    params = case_data["case_params"]                          # (8,)
    n = x_node.shape[0]
    case_broadcast = np.broadcast_to(params, (n, N_CASE_PARAMS))
    feats = np.concatenate(
        [x_node[:, None], r_node[:, None], case_broadcast], axis=1,
    )                                                          # (N, 10)
    channels = [case_data["rho"], case_data["u"], case_data["v"], case_data["T"]]
    if with_p:
        channels.append(case_data["rho"] * R_AIR * case_data["T"])
    targets = np.stack(channels, axis=1)                       # (N, 4) or (N, 5)
    return feats.astype(np.float32, copy=False), targets.astype(np.float32, copy=False)


# ============================================================================================
#                                       streaming Welford stats
# ============================================================================================

def compute_norm_stats(
    paths: list[str | Path],
    *,
    eps: float = 1e-8,
    log_targets: tuple[str, ...] | None = None,
    with_p: bool = False,
) -> SU2NormStats:
    """Compute the per-channel mean and standard deviation over training points.

    The function reads one case at a time, so peak memory does not grow with
    the number of cases. It merges the per-case ``(mean, M2, count)`` summaries
    with the parallel form of the Welford update.

    Parameters
    ----------
    paths : list of paths
        One ``case.npz`` per training case. Pass the training split alone. A
        validation or test point folded into these statistics leaks held-out
        information and inflates the reported accuracy.
    eps : float
        Added to the standard deviation, so that a constant channel does not
        divide by zero.
    log_targets : tuple of str, optional
        Channels to standardize in log10 space. None takes the default for the
        active layout. An empty tuple applies plain standardization throughout.
    with_p : bool
        Fit statistics for the five-channel layout that carries pressure.
    """
    targets_order = TARGET_ORDER_WITH_P if with_p else TARGET_ORDER
    if log_targets is None:
        log_targets = DEFAULT_LOG_TARGETS_WITH_P if with_p else DEFAULT_LOG_TARGETS
    n_total = 0
    x_mean: np.ndarray | None = None
    x_m2: np.ndarray | None = None
    y_mean: np.ndarray | None = None
    y_m2: np.ndarray | None = None

    for path in paths:
        case = load_case_npz(path)
        feats, targets = stack_case_features(case, with_p=with_p)
        targets = transform_targets(targets, log_targets, targets_order)
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
        targets=targets_order,
    )


# ============================================================================================
#                                       torch dataset
# ============================================================================================

class SU2Dataset(Dataset):
    """Disk-backed dataset over per-case ``case.npz`` files.

    Each call to ``__getitem__`` reads one case from disk and may draw a fixed
    number of nodes from it. The returned tensors are normalized with
    ``stats``.

    Parameters
    ----------
    paths : list of paths
        One ``case.npz`` per case.
    stats : SU2NormStats
        Normalization statistics fit on the training split.
    subsample : int, optional
        Draw this many nodes per item, uniform and without replacement. Each
        call draws again, so every epoch sees a different subset.
    qw_by_name : dict, optional
        ``{case_name: q_w}`` in W/m^2, taken from the ledger. Items then carry
        a ``qw`` scalar for the heat-flux head. A case absent from the map
        gets NaN, which the heat-flux loss masks out.

    ``stats.targets`` alone decides the target layout. One source of truth
    leaves no second flag that can disagree with it.
    """

    def __init__(
        self,
        paths: list[str | Path],
        stats: SU2NormStats,
        subsample: int | None = None,
        qw_by_name: dict[str, float] | None = None,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.stats = stats
        self.subsample = subsample
        self.qw_by_name = qw_by_name
        self.with_p = "p" in stats.targets

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[idx]
        case = load_case_npz(path)
        feats, targets = stack_case_features(case, with_p=self.with_p)
        n = feats.shape[0]
        if self.subsample is not None and n > self.subsample:
            indices = random.sample(range(n), self.subsample)
            feats = feats[indices]
            targets = targets[indices]

        x_raw = torch.from_numpy(feats)
        y_raw = torch.from_numpy(targets)
        pos = x_raw[:, :POS_DIM].clone()                           # (x, r) in metres

        y_t = torch.from_numpy(
            transform_targets(targets, self.stats.log_targets, self.stats.targets)
        )
        x = (x_raw - self.stats.x_mean) / self.stats.x_std
        y = (y_t - self.stats.y_mean) / self.stats.y_std

        name = path.parent.name if path.name == "case.npz" else path.stem
        qw = float("nan") if self.qw_by_name is None else self.qw_by_name.get(
            name, float("nan"))

        return {
            "x": x,
            "y": y,
            "y_raw": y_raw,
            "pos": pos,
            "case_params": torch.from_numpy(case["case_params"]),
            "qw": torch.tensor(qw, dtype=torch.float32),
            "name": name,
        }


# ============================================================================================
#                                       inference helpers
# ============================================================================================

def denormalize_targets(y_norm: torch.Tensor, stats: SU2NormStats) -> torch.Tensor:
    """Map a normalized prediction or target back to physical units.

    The function inverts the standardization first, then the log10 transform on
    the channels named in ``stats.log_targets``.
    """
    y = y_norm * stats.y_std.to(y_norm.device) + stats.y_mean.to(y_norm.device)
    mask = stats.log_mask.to(y_norm.device)
    if mask.any():
        y = y.clone()
        y[..., mask] = torch.pow(10.0, y[..., mask])
    return y


def reconstruct_pressure(rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Pressure from the ideal-gas equation of state, ``p = rho R_specific T``.

    The model predicts ``(rho, u, v, T)`` and this function supplies ``p`` at
    output time. The constraint is architectural rather than a penalty term.
    The equation of state holds exactly at inference, because the network has
    no degree of freedom with which to break it.
    """
    return rho * R_AIR * T


# ============================================================================================
#                                       case-listing helpers
# ============================================================================================

def list_case_npzs(workdir: str | Path) -> list[Path]:
    """Return the sorted list of case tensors under ``workdir``.

    Three layouts appear in practice: ``case_XXXX/case.npz`` for a per-case run
    directory, a flat ``case_XXXX.npz``, and ``cases/case_XXXX.npz`` inside an
    unpacked archive. One directory can hold several at once, so the function
    collects all three and removes duplicates by case id. The result is sorted
    by case id, which keeps the order the same on every machine.
    """
    workdir = Path(workdir)
    by_id: dict[int, Path] = {}
    for pattern in ("case_*/case.npz", "case_*.npz", "cases/case_*.npz"):
        for p in workdir.glob(pattern):
            by_id.setdefault(case_id_from_npz_path(p), p)
    return [by_id[cid] for cid in sorted(by_id)]


def case_id_from_npz_path(path: Path) -> int:
    """``case_0042/case.npz`` or ``case_0042.npz`` -> 42."""
    stem = path.parent.name if path.name == "case.npz" else path.stem
    return int(stem.split("_")[1])


def case_name(path: Path) -> str:
    """``case_0042/case.npz`` or ``case_0042.npz`` -> ``case_0042``."""
    return path.parent.name if path.name == "case.npz" else path.stem


# ============================================================================================
#                                       splits
# ============================================================================================

OOD_GROUPS = ("cone_high", "mach_high", "nose_large", "nose_small")


def split_paths(
    paths: list[Path],
    groups: dict[int, str],
    geom_ids: dict[int, int],
    val_frac: float,
    test_frac: float,
    seed: int,
    interp_frac: float = 0.0,
) -> dict[str, list[Path]]:
    """Split the cases into training, validation and the evaluation tiers.

    Inside the ``core`` group a seeded shuffle assigns whole geometries to one
    split each. Cases that share a mesh sit close together in freestream space,
    so splitting a geometry would leak between splits. Each extrapolation group
    becomes a test tier of its own.

    Above zero, ``interp_frac`` holds back that fraction of the cases on the
    training geometries as ``test_interp``: seen shapes, unseen freestream. The
    geometry shuffle runs first, so this never moves a geometry between the
    validation and test splits.

    Active-learning cases (``group_name == 'loop'``) always go to training and
    stay out of the shuffle, so adding them leaves every held-out tier unchanged.
    """
    by_group: dict[str, list[Path]] = {"core": [], "loop": [], **{g: [] for g in OOD_GROUPS}}
    for p in paths:
        cid = case_id_from_npz_path(p)
        if cid in groups:
            by_group.setdefault(groups[cid], []).append(p)

    core_by_geom: dict[int, list[Path]] = {}
    for p in by_group["core"]:
        core_by_geom.setdefault(geom_ids[case_id_from_npz_path(p)], []).append(p)

    geoms = sorted(core_by_geom)
    rng = np.random.default_rng(seed)
    rng.shuffle(geoms)
    n_val = max(1, int(round(len(geoms) * val_frac)))
    n_test = max(1, int(round(len(geoms) * test_frac)))
    val_geoms = sorted(geoms[:n_val])
    test_geoms = sorted(geoms[n_val:n_val + n_test])
    train_geoms = sorted(geoms[n_val + n_test:])

    train = [p for g in train_geoms for p in core_by_geom[g]]
    interp: list[Path] = []
    if interp_frac > 0:
        held = set(rng.permutation(len(train))[:int(round(len(train) * interp_frac))].tolist())
        interp = [p for i, p in enumerate(train) if i in held]
        train = [p for i, p in enumerate(train) if i not in held]
    train += sorted(by_group["loop"], key=case_id_from_npz_path)

    splits = {
        "train": train,
        "val": [p for g in val_geoms for p in core_by_geom[g]],
        "test": [p for g in test_geoms for p in core_by_geom[g]],
        "test_interp": interp,
    }
    for g in OOD_GROUPS:
        splits[f"ood_{g}"] = by_group[g]
    return splits


def pin_splits(splits: dict[str, list[Path]], paths: list[Path],
               pinned: dict[str, list[str]]) -> dict[str, list[Path]]:
    """Replace the listed splits with fixed case lists, by case name.

    A pinned list holds the exact membership of an earlier run, so a checkpoint
    can be scored again on the same cases after the dataset grows. Names absent
    from ``paths`` are dropped with a message.
    """
    by_name = {case_name(p): p for p in paths}
    out = dict(splits)
    for split, names in pinned.items():
        missing = [n for n in names if n not in by_name]
        if missing:
            print(f"[pinned] {split}: {len(missing)} case(s) not in the dataset, dropped")
        out[split] = [by_name[n] for n in names if n in by_name]
    return out
