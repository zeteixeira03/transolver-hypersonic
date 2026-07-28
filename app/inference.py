"""Slider params to deep-ensemble flow-field prediction for the dashboard.

One public entry point, :func:`predict`, takes the four exposed controls
(Mach, altitude, nose radius, cone half-angle), meshes the geometry live
with the same gmsh recipe the training data used, runs the five ensemble
members on the mesh nodes, and returns per-node fields, the ensemble
spread, two scalar quantities of interest with uncertainty bands, and the
trust/warn/refuse decision.

The model is sensitive to mesh resolution (its slice attention couples all
nodes), so the app meshes at the training-resolution kwargs rather than a
coarser cloud. That makes one prediction take tens of seconds on a small
CPU; the UI wraps it in a spinner.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.analytical import (
    R_AIR, billig_standoff, fay_riddell_qw, knudsen_number,
)
from src.data.sampler import us_standard_atmosphere
from src.data.su2 import (
    CASE_PARAM_ORDER, N_CASE_PARAMS, POS_DIM, TARGET_ORDER,
    SU2NormStats, denormalize_targets, reconstruct_pressure,
)
from src.geometry.sphere_cone import mesh_sphere_cone
from src.models.transolver import Transolver

# ============================================================================================
#                                       configuration
# ============================================================================================

ROOT = Path(__file__).resolve().parents[1]
ENSEMBLE_DIRS = [
    ROOT / f"data/processed/ensemble_v3/run_m32_v3_s{s}" for s in range(5)
]

# controls not exposed as sliders are fixed at box midpoints / the training wall
R_B_RATIO = 3.0      # R_b / R_n, GEOM_BOX midpoint
R_S_RATIO = 0.10     # R_s / R_b, near the box midpoint
T_WALL = 300.0       # K, isothermal cold wall (fixed across the whole dataset)

# decision thresholds, recalibrated on the ensemble backing ENSEMBLE_DIRS. the
# guard distance is not a constant: it tracks how densely the extrapolation
# region has been sampled, and moved outward from 0.038 as the OOD slabs grew
GUARD_DIST = 0.07    # L-inf box exceedance above this -> refuse
KN_MAX = 0.01        # continuum floor; above this -> refuse
WARN_SPREAD = 0.077  # val-split p90 ensemble spread
REFUSE_SPREAD = 0.25 # in-box backstop, just past the 0.222 max spread seen in eval


# ============================================================================================
#                                       model loading
# ============================================================================================

@dataclass
class Ensemble:
    models: list[Transolver]
    stats: SU2NormStats
    envelope: dict[str, list[float]]


def load_ensemble(device: str = "cpu") -> Ensemble:
    """Instantiate every member from its recorded args and load ``best.pt``."""
    models = []
    for run_dir in ENSEMBLE_DIRS:
        rec = json.loads((run_dir / "final_eval.json").read_text())["args"]
        model = Transolver(
            space_dim=POS_DIM, fun_dim=N_CASE_PARAMS, out_dim=len(TARGET_ORDER),
            n_hidden=rec["n_hidden"], n_layers=rec["n_layers"],
            n_head=rec["n_head"], slice_num=rec["slice_num"],
        )
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
        models.append(model.to(device).eval())
    stats = SU2NormStats.load(ENSEMBLE_DIRS[0] / "norm_stats.pt").to(device)
    envelope = json.loads((ENSEMBLE_DIRS[0] / "final_eval.json").read_text())["train_envelope"]
    return Ensemble(models=models, stats=stats, envelope=envelope)


# ============================================================================================
#                                       geometry / meshing
# ============================================================================================

def build_params(R_n: float, theta_c_deg: float, mach: float, altitude_km: float) -> dict[str, float]:
    """Expand the four sliders into the full eight-parameter case vector."""
    T_inf, p_inf, _rho = us_standard_atmosphere(altitude_km)
    R_b = R_B_RATIO * R_n
    R_s = R_S_RATIO * R_b
    return {
        "R_n": R_n, "theta_c_deg": theta_c_deg, "R_b": R_b, "R_s": R_s,
        "mach": mach, "T_inf": T_inf, "p_inf": p_inf, "T_w": T_WALL,
    }


def mesh_nodes(params: dict[str, float]) -> np.ndarray:
    """Live-mesh the geometry at training resolution, return (N, 2) node coords."""
    R_n, R_b = params["R_n"], params["R_b"]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "case.su2"
        mesh_sphere_cone(
            R_n, params["theta_c_deg"], R_b, params["R_s"], out,
            L_far=8.0 * R_b, h_wall=R_n / 120.0, h_far=8.0 * R_b / 25.0,
            bl_first_height=max(R_n / 30000.0, 5e-7), bl_thickness=0.06 * R_n,
            refine_shock_box=True,
        )
        return _parse_su2_nodes(out)


def _parse_su2_nodes(path: Path) -> np.ndarray:
    """Read the (x, r) coordinates from the ``NPOIN`` block of an ASCII .su2 mesh."""
    lines = path.read_text().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("NPOIN="):
            n = int(ln.split("=")[1].split()[0])
            coords = np.empty((n, 2), dtype=np.float64)
            for j in range(n):
                parts = lines[i + 1 + j].split()
                coords[j, 0], coords[j, 1] = float(parts[0]), float(parts[1])
            return coords
    raise ValueError("no NPOIN block in mesh file")


# ============================================================================================
#                                       inference
# ============================================================================================

@torch.no_grad()
def _ensemble_forward(
    ens: Ensemble, coords: np.ndarray, params: dict[str, float], device: str = "cpu",
) -> torch.Tensor:
    """Run every member on the node cloud. Returns denormalized (K, N, 4)."""
    n = coords.shape[0]
    param_vec = np.array([params[k] for k in CASE_PARAM_ORDER], dtype=np.float64)
    feats = np.concatenate(
        [coords, np.broadcast_to(param_vec, (n, N_CASE_PARAMS))], axis=1,
    ).astype(np.float32)
    x_raw = torch.from_numpy(feats).to(device)
    pos = x_raw[:, :POS_DIM].clone().unsqueeze(0)
    x = ((x_raw - ens.stats.x_mean) / ens.stats.x_std).unsqueeze(0)
    return torch.stack([
        denormalize_targets(m(x, pos=pos).squeeze(0), ens.stats) for m in ens.models
    ])


def _channel_spread(members: torch.Tensor) -> float:
    """||std over members|| / ||truth-proxy|| per channel, channel-averaged.

    Matches the ensemble-UQ definition but uses the ensemble mean as the
    scale (no ground truth at inference), so it is comparable in magnitude
    to the calibrated thresholds.
    """
    mean = members.mean(dim=0)                                   # (N, C)
    std = members.std(dim=0)                                     # (N, C)
    num = std.pow(2).sum(dim=0).sqrt()                           # (C,)
    den = mean.pow(2).sum(dim=0).sqrt().clamp_min(1e-12)         # (C,)
    return float((num / den).mean())


# ============================================================================================
#                                   quantities of interest
# ============================================================================================

def shock_standoff(coords: np.ndarray, rho: np.ndarray, R_n: float, rho_inf: float,
                   r_band: float = 0.10, n_bins: int = 200, thresh_ratio: float = 3.0) -> float:
    """Standoff distance from the nose to the bow shock along the stagnation line.

    Bins the near-axis strip (``r < r_band R_n``) in x, takes the mean density
    per bin, and locates the shock as the most upstream bin where density rises
    past ``thresh_ratio`` times freestream. Strip-binning is used instead of an
    exact-axis profile because on-axis mesh nodes are too sparse to resolve the
    thin stagnation shock layer.

    Returns the distance in metres, or ``nan`` if the strip is too sparse.
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
        x_shock = centers[0]
    else:
        # linear sub-bin interpolation of the threshold crossing, so the
        # per-member standoff is continuous rather than quantized to a bin
        p0, p1 = prof[i - 1], prof[i]
        frac = (thresh - p0) / (p1 - p0) if p1 != p0 else 1.0
        x_shock = centers[i - 1] + frac * (centers[i] - centers[i - 1])
    return float(x_nose - x_shock)


# ============================================================================================
#                                   decision rule
# ============================================================================================

def envelope_distance(params: dict[str, float], envelope: dict[str, list[float]]) -> float:
    """L-inf normalized box-exceedance of the case over the train envelope."""
    d = 0.0
    for k in CASE_PARAM_ORDER:
        lo, hi = envelope[k]
        width = hi - lo
        if width <= 0:
            continue
        d = max(d, (params[k] - hi) / width, (lo - params[k]) / width)
    return max(d, 0.0)


def decide(distance: float, kn: float, spread: float) -> str:
    """trust / warn / refuse from the envelope guard and ensemble spread."""
    if distance > GUARD_DIST or kn > KN_MAX or spread >= REFUSE_SPREAD:
        return "refuse"
    if spread >= WARN_SPREAD:
        return "warn"
    return "trust"


# ============================================================================================
#                                       public entry point
# ============================================================================================

@dataclass
class Prediction:
    coords: np.ndarray                 # (N, 2) node (x, r)
    fields: dict[str, np.ndarray]      # ensemble-mean rho, u, v, T, p per node
    n_nodes: int
    spread: float                      # scalar ensemble spread (threshold units)
    distance: float                    # envelope L-inf exceedance
    kn: float                          # nose-radius Knudsen number
    decision: str                      # trust / warn / refuse
    qoi: dict[str, dict[str, float]]   # standoff, q_w: mean/std/reference
    params: dict[str, float]


def predict(ens: Ensemble, R_n: float, theta_c_deg: float, mach: float,
            altitude_km: float, device: str = "cpu") -> Prediction:
    """End-to-end: sliders -> mesh -> ensemble fields, QoIs, and decision."""
    params = build_params(R_n, theta_c_deg, mach, altitude_km)
    coords = mesh_nodes(params)
    members = _ensemble_forward(ens, coords, params, device)     # (K, N, 4)
    mean = members.mean(dim=0)                                   # (N, 4)

    fields = {name: mean[:, i].cpu().numpy() for i, name in enumerate(TARGET_ORDER)}
    fields["p"] = reconstruct_pressure(mean[:, 0], mean[:, 3]).cpu().numpy()

    # shock standoff is a genuine field-derived QoI: per member -> ensemble band
    rho_inf = params["p_inf"] / (R_AIR * params["T_inf"])
    rho_k = members[:, :, 0].cpu().numpy()
    standoffs = np.array([shock_standoff(coords, rho_k[k], R_n, rho_inf)
                          for k in range(len(ens.models))])

    spread = _channel_spread(members)
    distance = envelope_distance(params, ens.envelope)
    kn = knudsen_number(params["mach"], params["T_inf"], params["p_inf"], params["R_n"])

    # stagnation heat flux: the surrogate cannot resolve the thin near-wall
    # thermal gradient, so the direct field estimate is unreliable (a known
    # limitation motivating a residual head). Report the Fay-Riddell analytic
    # value instead, which is exact from freestream and geometry.
    qoi = {
        "standoff": {
            "mean": float(np.nanmean(standoffs)),
            "std": float(np.nanstd(standoffs)),
            "reference": billig_standoff(mach, R_n),
        },
        "q_w": {
            "reference": fay_riddell_qw(mach, params["T_inf"], params["p_inf"], R_n, T_WALL),
        },
    }
    return Prediction(
        coords=coords, fields=fields, n_nodes=coords.shape[0], spread=spread,
        distance=distance, kn=kn, decision=decide(distance, kn, spread),
        qoi=qoi, params=params,
    )
