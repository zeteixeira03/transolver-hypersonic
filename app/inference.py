"""
Turn the four dashboard controls into an ensemble prediction of the flow.

:func:`predict` meshes the geometry with the recipe that produced the training
data, runs every ensemble member on the mesh nodes, and returns the mean fields,
the spread, the shock standoff with its band across members, and the decision to
trust, warn or refuse. Meshing takes most of the time, so the geometry is
meshed once per cone half-angle and scaled to each nose radius.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.analytical import R_AIR, billig_standoff, fay_riddell_qw, knudsen_number
from src.data.sampler import T_WALL, case_params
from src.data.su2 import TARGET_ORDER, reconstruct_pressure
from src.eval.uncertainty import (
    Ensemble,
    decide,
    envelope_distance,
    load_ensemble,
    predict_on_nodes,
    spread,
)
from src.geometry.sphere_cone import mesh_nodes

ENSEMBLE_DIR = Path(__file__).resolve().parents[1] / "data/processed/ensemble_v3"

# controls without a slider sit near the middle of the training box
R_B_RATIO = 3.0     # R_b / R_n
R_S_RATIO = 0.10    # R_s / R_b

# every length of the mesh recipe scales with R_n once the 5e-7 m floor on the
# first cell stops binding, at R_n = 15 mm. from there up, a mesh is a scaled
# copy of the one at 15 mm: meshes at 20 and 40 mm agree to 0.05% in node count
# and 0.03% in node spacing, which is the spread between two meshings of one case
SCALE_FROM = 0.015  # m


def load_default_ensemble(device: str = "cpu") -> tuple[Ensemble, dict[str, float]]:
    """Load the published ensemble and the thresholds calibrated for it."""
    ens = load_ensemble(list(ENSEMBLE_DIR.glob("run_*")), device)
    thresholds = json.loads((ENSEMBLE_DIR / "thresholds.json").read_text())
    return ens, thresholds


def shock_standoff(coords: np.ndarray, rho: np.ndarray, R_n: float, rho_inf: float,
                   r_band: float = 0.10, n_bins: int = 200, thresh_ratio: float = 3.0) -> float:
    """Distance from the nose to the bow shock along the stagnation line, in metres.

    The mesh holds too few nodes on the axis itself, so the function bins the
    strip ``r < r_band R_n`` along x and takes the mean density per bin. The
    shock is the most upstream bin whose density exceeds ``thresh_ratio`` times
    the freestream value, with the crossing interpolated inside the bin. Returns
    NaN when the strip holds too few nodes.
    """
    mask = coords[:, 1] < r_band * R_n
    xs, rs = coords[mask, 0], rho[mask]
    if xs.size < 16:
        return float("nan")
    x_nose = xs.max()
    edges = np.linspace(x_nose - 2.0 * R_n, x_nose, n_bins + 1)
    idx = np.digitize(xs, edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prof = np.array([rs[idx == b].mean() if np.any(idx == b) else np.nan
                     for b in range(1, n_bins + 1)])
    ok = np.isfinite(prof)
    centers, prof = centers[ok], prof[ok]
    thresh = thresh_ratio * rho_inf
    above = np.where(prof > thresh)[0]
    if above.size == 0:
        return float("nan")
    i = above[0]
    if i == 0:
        return float(x_nose - centers[0])
    p0, p1 = prof[i - 1], prof[i]
    frac = (thresh - p0) / (p1 - p0) if p1 != p0 else 1.0
    return float(x_nose - (centers[i - 1] + frac * (centers[i] - centers[i - 1])))


@dataclass
class Prediction:
    coords: np.ndarray                 # (N, 2) node coordinates (x, r)
    fields: dict[str, np.ndarray]      # ensemble mean of rho, u, v, T and p per node
    spread: float                      # mean spread over the four channels
    distance: float                    # distance outside the training box
    kn: float                          # Knudsen number on the nose radius
    decision: str                      # trust, warn or refuse
    qoi: dict[str, dict[str, float]]   # standoff and heat flux, with references
    params: dict[str, float]


@functools.cache
def template_nodes(theta_c_deg: float) -> np.ndarray:
    """Mesh nodes at R_n = ``SCALE_FROM`` for one cone half-angle, meshed once per process."""
    R_b = R_B_RATIO * SCALE_FROM
    return mesh_nodes(SCALE_FROM, theta_c_deg, R_b, R_S_RATIO * R_b)


def predict(ens: Ensemble, thresholds: dict[str, float], R_n: float, theta_c_deg: float,
            mach: float, altitude_km: float, device: str = "cpu") -> Prediction:
    """Mesh the geometry, run the ensemble, and decide whether to trust the result."""
    params = case_params(R_n, theta_c_deg, R_B_RATIO, R_S_RATIO, mach, altitude_km)
    if R_n < SCALE_FROM:
        coords = mesh_nodes(R_n, theta_c_deg, params["R_b"], params["R_s"])
    else:
        coords = template_nodes(theta_c_deg) * (R_n / SCALE_FROM)
    members = predict_on_nodes(ens, coords, params, device)      # (K, N, 4)
    mean = members.mean(dim=0)
    fields = {c: mean[:, j].cpu().numpy() for j, c in enumerate(TARGET_ORDER)}
    fields["p"] = reconstruct_pressure(mean[:, 0], mean[:, 3]).cpu().numpy()

    # each member gets its own standoff, which gives a real band
    rho_inf = params["p_inf"] / (R_AIR * params["T_inf"])
    standoffs = [shock_standoff(coords, m[:, 0].cpu().numpy(), R_n, rho_inf) for m in members]

    mean_spread = float(spread(members, mean).mean())
    distance = envelope_distance(params, ens.envelope)
    kn = knudsen_number(mach, params["T_inf"], params["p_inf"], R_n)
    qoi = {
        "standoff": {"mean": float(np.nanmean(standoffs)), "std": float(np.nanstd(standoffs)),
                     "reference": billig_standoff(mach, R_n)},
        "q_w": {"reference": fay_riddell_qw(mach, params["T_inf"], params["p_inf"], R_n, T_WALL)},
    }
    return Prediction(coords=coords, fields=fields, spread=mean_spread, distance=distance,
                      kn=kn, decision=decide(distance, kn, mean_spread, thresholds),
                      qoi=qoi, params=params)
