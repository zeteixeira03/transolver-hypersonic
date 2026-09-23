"""
Deep-ensemble uncertainty and the rule that decides whether to trust a prediction.

Several Transolvers that differ only in their starting weights and data order
make an ensemble. Their mean is the prediction, and their spread is the
uncertainty. A prediction is refused when the input lies outside the training
envelope, when the flow is too rarefied for the continuum model, or when the
members disagree strongly. Moderate disagreement earns a warning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.sampler import KN_MAX
from src.data.su2 import (
    CASE_PARAM_ORDER,
    N_CASE_PARAMS,
    POS_DIM,
    TARGET_ORDER,
    SU2NormStats,
    denormalize_targets,
)
from src.models.transolver import Transolver

# refuse above this distance outside the training box. it marks where held-out
# error starts to climb, so it moves as the extrapolation slabs gain cases.
GUARD_DIST = 0.07


@dataclass
class Ensemble:
    models: list[Transolver]
    stats: SU2NormStats
    envelope: dict[str, list[float]]


def load_member(run_dir: Path, device: str) -> Transolver:
    """Build one member from the arguments its run recorded and load ``best.pt``."""
    args = json.loads((run_dir / "final_eval.json").read_text())["args"]
    model = Transolver(
        space_dim=POS_DIM, fun_dim=N_CASE_PARAMS, out_dim=len(TARGET_ORDER),
        n_hidden=args["n_hidden"], n_layers=args["n_layers"],
        n_head=args["n_head"], slice_num=args["slice_num"],
    )
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    return model.to(device).eval()


def load_ensemble(run_dirs: list[Path], device: str = "cpu") -> Ensemble:
    """Load every member. The first one supplies the normalization and the envelope."""
    run_dirs = sorted(run_dirs)
    if not run_dirs:
        raise ValueError("no member run directories given")
    final = json.loads((run_dirs[0] / "final_eval.json").read_text())
    return Ensemble(
        models=[load_member(d, device) for d in run_dirs],
        stats=SU2NormStats.load(run_dirs[0] / "norm_stats.pt").to(device),
        envelope=final["train_envelope"],
    )


@torch.no_grad()
def member_predictions(ens: Ensemble, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Run every member on normalized inputs of shape (1, N, C). Returns (K, N, 4) in physical units."""
    return torch.stack([denormalize_targets(m(x, pos=pos).squeeze(0), ens.stats)
                        for m in ens.models])


def predict_on_nodes(ens: Ensemble, coords: np.ndarray, params: dict[str, float],
                     device: str = "cpu") -> torch.Tensor:
    """Member predictions on a raw node cloud for one set of case parameters."""
    param_vec = np.array([params[k] for k in CASE_PARAM_ORDER])
    feats = np.concatenate([coords, np.broadcast_to(param_vec, (len(coords), N_CASE_PARAMS))],
                           axis=1).astype(np.float32)
    x_raw = torch.from_numpy(feats).to(device)
    pos = x_raw[:, :POS_DIM].clone().unsqueeze(0)
    x = ((x_raw - ens.stats.x_mean) / ens.stats.x_std).unsqueeze(0)
    return member_predictions(ens, x, pos)


def spread(members: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-channel spread: the norm of the member standard deviation over the reference norm.

    In evaluation the reference is the true field. At inference no true field
    exists, so the ensemble mean stands in for it. The two agree closely where
    the ensemble is accurate, which keeps the calibrated thresholds usable.
    """
    return members.std(dim=0).pow(2).sum(dim=0).sqrt() / reference.pow(2).sum(dim=0).sqrt()


def envelope_distance(params: dict[str, float], envelope: dict[str, list[float]]) -> float:
    """How far a case lies outside the training box, in units of each parameter's width.

    Returns the largest excess over all parameters, or 0 inside the box. A
    parameter held constant, such as the wall temperature, is skipped.
    """
    d = 0.0
    for k in CASE_PARAM_ORDER:
        lo, hi = envelope[k]
        if hi > lo:
            d = max(d, (params[k] - hi) / (hi - lo), (lo - params[k]) / (hi - lo))
    return d


def calibrate_thresholds(val_spreads: list[float]) -> dict[str, float]:
    """Warn above the 90th percentile of the validation spread, refuse above four times that."""
    warn = float(np.quantile(val_spreads, 0.9))
    return {"warn": warn, "refuse": 4.0 * warn}


def decide(distance: float, kn: float, mean_spread: float, thresholds: dict[str, float]) -> str:
    """Return ``trust``, ``warn`` or ``refuse`` for one prediction."""
    if distance > GUARD_DIST or kn > KN_MAX or mean_spread >= thresholds["refuse"]:
        return "refuse"
    if mean_spread >= thresholds["warn"]:
        return "warn"
    return "trust"
