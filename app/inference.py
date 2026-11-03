"""
Turn the four dashboard controls into an ensemble prediction of the flow.

:func:`predict` meshes the geometry with the recipe that produced the training
data, runs every ensemble member on the mesh nodes, and returns the mean fields,
the spread, the shock standoff with its band across members, and the decision to
trust, warn or refuse. Meshing takes most of the time, so the geometry is
meshed once per cone half-angle and scaled to each nose radius.

It also times the prediction against the SU2 wall time of the nearest solved
case, and records which learned slice each node belongs to most strongly.
"""

from __future__ import annotations

import functools
import json
import time
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
from src.geometry.sphere_cone import body_outline, mesh_nodes

ENSEMBLE_DIR = Path(__file__).resolve().parents[1] / "data/processed/ensemble_v3"
SU2_TIMES = Path(__file__).with_name("su2_wall_time.csv")

# the head whose strongest slices agree best with the flow regions, out of 4 layers
# times 8 heads of member 1: normalized mutual information 0.59 at Mach 15, 52 km,
# R_n 25 mm, theta_c 50 deg
SLICE_LAYER = 1
SLICE_HEAD = 0
REGIONS = ("freestream", "shock layer", "near wall")

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

    The mesh holds too few nodes on the axis itself, so the strip
    ``r < r_band R_n`` is binned along x. The shock is the most upstream bin
    whose mean density exceeds ``thresh_ratio`` times the freestream value,
    interpolated inside the bin. NaN when the strip holds too few nodes.
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


def wall_distance(coords: np.ndarray, outline: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Distance from each node to the nearest segment of the closed body outline."""
    a, b = outline, np.roll(outline, -1, axis=0)
    ab = b - a
    ab2 = (ab ** 2).sum(1)
    out = np.empty(len(coords))
    for i in range(0, len(coords), chunk):
        ap = coords[i:i + chunk, None, :] - a[None]
        t = np.clip((ap * ab).sum(-1) / ab2, 0.0, 1.0)
        out[i:i + chunk] = np.sqrt(((ap - t[..., None] * ab) ** 2).sum(-1)).min(1)
    return out


def flow_regions(coords: np.ndarray, T: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Index into ``REGIONS`` per node.

    Freestream has not been heated by the shock (below 1.5 T_inf). Near wall lies
    within 0.1 R_n of the body. The shock layer is the rest.
    """
    outline = body_outline(params["R_n"], params["theta_c_deg"], params["R_b"], params["R_s"])
    near = wall_distance(coords, outline) < 0.1 * params["R_n"]
    return np.where(T < 1.5 * params["T_inf"], 0, np.where(near, 2, 1))


def normalized_mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """Mutual information of two labelings over the mean of their entropies: 1 for a perfect match."""
    joint = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(joint, (a, b), 1.0 / len(a))
    pa, pb = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = (joint[nz] * np.log(joint[nz] / np.outer(pa, pb)[nz])).sum()
    entropy = [-(p[p > 0] * np.log(p[p > 0])).sum() for p in (pa, pb)]
    return float(mi / np.mean(entropy))


def slice_table(slices: np.ndarray, regions: np.ndarray, n_rows: int = 10) -> list[dict]:
    """The largest slices holding at least 1% of the nodes, each with the region most of its nodes fall in."""
    rows = []
    ids, counts = np.unique(slices, return_counts=True)
    for m, n in zip(ids, counts):
        if n < 0.01 * len(slices):
            continue
        in_region = np.bincount(regions[slices == m], minlength=len(REGIONS))
        k = int(in_region.argmax())
        rows.append({"slice": int(m), "share": n / len(slices), "region": REGIONS[k],
                     "match": in_region[k] / n})
    return sorted(rows, key=lambda row: -row["share"])[:n_rows]


@functools.cache
def su2_wall_times() -> np.ndarray:
    """(R_n, theta_c, Mach, p_inf, hours) of every finished sweep case."""
    return np.loadtxt(SU2_TIMES, delimiter=",")


def nearest_su2_case(params: dict[str, float]) -> dict[str, float]:
    """Wall time of the solved case nearest in nose radius, cone angle, Mach and log pressure."""
    rows = su2_wall_times()
    feat = np.c_[rows[:, :3], np.log(rows[:, 3])]
    lo, hi = feat.min(0), feat.max(0)
    q = np.array([params["R_n"], params["theta_c_deg"], params["mach"], np.log(params["p_inf"])])
    i = int(np.argmin((((feat - q) / (hi - lo)) ** 2).sum(1)))
    hours = rows[:, 4]
    return {"hours": float(hours[i]), "R_n": float(rows[i, 0]), "theta_c": float(rows[i, 1]),
            "mach": float(rows[i, 2]), "n_cases": len(rows), "median_hours": float(np.median(hours)),
            "p5_hours": float(np.percentile(hours, 5)), "p95_hours": float(np.percentile(hours, 95))}


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
    mesh_s: float                      # meshing time, zero when the template mesh was cached
    infer_s: float                     # time to run the five members and reduce their output
    slices: np.ndarray                 # (N,) strongest slice of SLICE_HEAD in SLICE_LAYER, member 1
    regions: np.ndarray                # (N,) index into REGIONS
    su2: dict[str, float]              # nearest solved case and the sweep's wall-time spread


@functools.cache
def template_nodes(theta_c_deg: float) -> np.ndarray:
    """Mesh nodes at R_n = ``SCALE_FROM`` for one cone half-angle, meshed once per process."""
    R_b = R_B_RATIO * SCALE_FROM
    return mesh_nodes(SCALE_FROM, theta_c_deg, R_b, R_S_RATIO * R_b)


def predict(ens: Ensemble, thresholds: dict[str, float], R_n: float, theta_c_deg: float,
            mach: float, altitude_km: float, device: str = "cpu") -> Prediction:
    """Mesh the geometry, run the ensemble, and decide whether to trust the result."""
    params = case_params(R_n, theta_c_deg, R_B_RATIO, R_S_RATIO, mach, altitude_km)
    t0 = time.perf_counter()
    if R_n < SCALE_FROM:
        coords = mesh_nodes(R_n, theta_c_deg, params["R_b"], params["R_s"])
    else:
        coords = template_nodes(theta_c_deg) * (R_n / SCALE_FROM)
    mesh_s = time.perf_counter() - t0

    captured = []

    def strongest_slice(attn, inp, _out):
        # softmax and the positive temperature keep the order, so the strongest
        # slice is the largest logit
        x = inp[0]
        x_mid = attn.in_project_x(x).reshape(*x.shape[:2], attn.heads, attn.dim_head)[:, :, SLICE_HEAD]
        captured.append(attn.in_project_slice(x_mid)[0].argmax(-1).cpu().numpy())

    t0 = time.perf_counter()
    hook = ens.models[0].blocks[SLICE_LAYER].attn.register_forward_hook(strongest_slice)
    try:
        members = predict_on_nodes(ens, coords, params, device)  # (K, N, 4)
    finally:
        hook.remove()
    mean = members.mean(dim=0)
    fields = {c: mean[:, j].cpu().numpy() for j, c in enumerate(TARGET_ORDER)}
    fields["p"] = reconstruct_pressure(mean[:, 0], mean[:, 3]).cpu().numpy()

    # one standoff per member gives the band
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
    infer_s = time.perf_counter() - t0
    return Prediction(coords=coords, fields=fields, spread=mean_spread, distance=distance,
                      kn=kn, decision=decide(distance, kn, mean_spread, thresholds),
                      qoi=qoi, params=params, mesh_s=mesh_s, infer_s=infer_s,
                      slices=captured[0], regions=flow_regions(coords, fields["T"], params),
                      su2=nearest_su2_case(params))
