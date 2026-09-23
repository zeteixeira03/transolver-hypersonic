"""Turn the dashboard controls into a deep-ensemble prediction of the flow.

One entry point, :func:`predict`, takes the four exposed controls, which are
Mach number, altitude, nose radius and cone half-angle. It meshes the geometry
through the same gmsh recipe that produced the training data, runs the five
ensemble members over the mesh nodes, and returns the per-node fields, the
ensemble spread, two scalar quantities of interest with their uncertainty
bands, and the decision to trust, warn or refuse.

The mesh resolution has to match the one the model saw in training, because
slice attention couples every node to every other one. A coarser cloud shifts
the prediction. Meshing at full resolution is what makes one prediction take
tens of seconds on a small processor, so the interface runs it behind a
spinner.
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

# every control without a slider holds at the midpoint of the training box, or
# at the wall temperature the whole dataset shares
R_B_RATIO = 3.0      # R_b / R_n, GEOM_BOX midpoint
R_S_RATIO = 0.10     # R_s / R_b, near the box midpoint
T_WALL = 300.0       # K, isothermal cold wall (fixed across the whole dataset)

# decision thresholds, calibrated against the ensemble in ENSEMBLE_DIRS. the
# guard distance is a measurement and not a constant. it marks where held-out
# error starts to climb, so it moves outward as the extrapolation slabs gain
# cases, and it needs recalibration whenever they do.
GUARD_DIST = 0.07    # refuse above this L-inf exceedance of the box
KN_MAX = 0.01        # refuse above this Knudsen number, the continuum floor
WARN_SPREAD = 0.077  # the 90th percentile of ensemble spread on the validation split
REFUSE_SPREAD = 0.25 # an in-box backstop above the largest spread measured, 0.222


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
#                                    geometry and meshing
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
    """Mesh the geometry at training resolution and return (N, 2) node coordinates."""
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
    """Run every member over the node cloud, and return (K, N, 4) in physical units."""
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
    """The norm of the spread over members, divided by the norm of the mean.

    The ratio is taken per channel and then averaged over channels. The
    uncertainty study defines it the same way, except that it divides by the
    true field. No true field exists at inference, so the ensemble mean stands
    in for it, which keeps the value comparable to the calibrated thresholds.
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

    The function bins the strip near the axis, ``r < r_band R_n``, along x and
    takes the mean density in each bin. The shock is the most upstream bin whose
    density passes ``thresh_ratio`` times the freestream value. A strip rather
    than the axis itself, because the mesh carries too few nodes on the axis to
    resolve the thin shock layer at the stagnation point.

    Returns the distance in metres, or ``nan`` when the strip holds too few
    nodes.
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
        # interpolate the crossing inside the bin, so that the standoff of one
        # member varies continuously rather than in steps of one bin width
        p0, p1 = prof[i - 1], prof[i]
        frac = (thresh - p0) / (p1 - p0) if p1 != p0 else 1.0
        x_shock = centers[i - 1] + frac * (centers[i] - centers[i - 1])
    return float(x_nose - x_shock)


# ============================================================================================
#                                   decision rule
# ============================================================================================

def envelope_distance(params: dict[str, float], envelope: dict[str, list[float]]) -> float:
    """How far the case falls outside the training envelope.

    Each parameter is measured in units of its own box width, and the result is
    the largest of those. A case inside the box returns 0.
    """
    d = 0.0
    for k in CASE_PARAM_ORDER:
        lo, hi = envelope[k]
        width = hi - lo
        if width <= 0:
            continue
        d = max(d, (params[k] - hi) / width, (lo - params[k]) / width)
    return max(d, 0.0)


def decide(distance: float, kn: float, spread: float) -> str:
    """Decide to trust, warn or refuse, from the envelope guard and the spread."""
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
    """Run the whole chain: controls, mesh, ensemble fields, quantities, decision."""
    params = build_params(R_n, theta_c_deg, mach, altitude_km)
    coords = mesh_nodes(params)
    members = _ensemble_forward(ens, coords, params, device)     # (K, N, 4)
    mean = members.mean(dim=0)                                   # (N, 4)

    fields = {name: mean[:, i].cpu().numpy() for i, name in enumerate(TARGET_ORDER)}
    fields["p"] = reconstruct_pressure(mean[:, 0], mean[:, 3]).cpu().numpy()

    # the standoff comes from the density field itself, so running it per member
    # gives a real uncertainty band rather than a nominal value
    rho_inf = params["p_inf"] / (R_AIR * params["T_inf"])
    rho_k = members[:, :, 0].cpu().numpy()
    standoffs = np.array([shock_standoff(coords, rho_k[k], R_n, rho_inf)
                          for k in range(len(ens.models))])

    spread = _channel_spread(members)
    distance = envelope_distance(params, ens.envelope)
    kn = knudsen_number(params["mach"], params["T_inf"], params["p_inf"], params["R_n"])

    # stagnation heat flux. the surrogate does not resolve the thin thermal
    # gradient at the wall, so an estimate read off its temperature field is not
    # a physical heat flux. report the Fay-Riddell value, which follows from the
    # freestream and the nose radius alone.
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
