"""Phase 3 parameter sampler: a nested geometry x freestream DOE plus OOD slabs.

The training set is built as a nested Latin hypercube rather than a single joint
LHS, for one practical reason: warm starting. A geometry change forces a remesh,
which means a from-scratch solve; a freestream change at fixed geometry reuses
the mesh and can restart from the previous converged solution. So we draw

    n_geom   geometries     -- LHS over (R_n, theta_c, R_b/R_n, R_s/R_b)
    n_fs     freestreams     -- LHS over (altitude, M_inf) for each geometry

giving ``n_geom * n_fs`` core cases. Cases sharing a geometry are emitted
consecutively (and ordered by a nearest-neighbour walk in freestream space), so
the first case of each geometry cluster solves cold and the rest restart from
their predecessor. Coverage of the joint parameter box is unchanged from a flat
LHS of the same size; only the layout changes.

Freestream temperature and pressure are not independent parameters; they follow
a sampled altitude through the US Standard Atmosphere 1976 so the (T_inf, p_inf)
pairs are physical. Wall temperature is fixed at 300 K per project scope.

On top of the core box, thin OOD slabs probe extrapolation: large and small nose
radii, high cone angles, and Mach above the core ceiling. Each slab is its own
small nested DOE, tagged so the Phase 4 OOD analysis can slice on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from src.analytical import knudsen_number
from src.cfd.runner import Case


# ============================================================================================
#                                   standard atmosphere
# ============================================================================================

# US Standard Atmosphere 1976, breakpoints to 86 km: (base altitude [km],
# base temperature [K], lapse rate [K/km], base pressure [Pa]).
_ATM_LAYERS = [
    (0.0,   288.15,  -6.5,   101325.0),
    (11.0,  216.65,   0.0,    22632.06),
    (20.0,  216.65,   1.0,     5474.889),
    (32.0,  228.65,   2.8,      868.0187),
    (47.0,  270.65,   0.0,      110.9063),
    (51.0,  270.65,  -2.8,       66.93887),
    (71.0,  214.65,  -2.0,        3.956420),
]
_G0 = 9.80665      # m/s^2
_R_AIR = 287.058   # J/(kg K), matches the SU2 cfg GAS_CONSTANT


def us_standard_atmosphere(h_km: float) -> tuple[float, float, float]:
    """Return (T [K], p [Pa], rho [kg/m^3]) at geometric altitude ``h_km``.

    Valid 0 to 86 km. Piecewise-linear-temperature US 1976 model with a
    hydrostatic pressure integral within each layer.

    Parameters
    ----------
    h_km : float
        Geometric altitude in kilometres.

    Returns
    -------
    tuple of float
        Static temperature, static pressure, density.
    """
    if not 0.0 <= h_km <= 86.0:
        raise ValueError("altitude must lie in [0, 86] km")
    base_h, base_T, lapse, base_p = _ATM_LAYERS[0]
    for layer in _ATM_LAYERS:
        if h_km >= layer[0]:
            base_h, base_T, lapse, base_p = layer
        else:
            break
    dh = (h_km - base_h) * 1000.0  # m
    if abs(lapse) < 1e-9:
        T = base_T
        p = base_p * np.exp(-_G0 * dh / (_R_AIR * base_T))
    else:
        lapse_si = lapse / 1000.0  # K/m
        T = base_T + lapse_si * dh
        p = base_p * (T / base_T) ** (-_G0 / (_R_AIR * lapse_si))
    return float(T), float(p), float(p / (_R_AIR * T))


# ============================================================================================
#                                   sampling boxes
# ============================================================================================

# (lo, hi) per parameter, split into geometry dims and freestream dims
GEOM_BOX = {
    "R_n":       (0.010, 0.050),   # m
    "theta_c":   (35.0,  70.0),    # deg
    "R_b_ratio": (2.0,   4.0),     # R_b / R_n
    "R_s_ratio": (0.05,  0.30),    # R_s / R_b
}
FS_BOX = {
    # altitude upper bound was 78 km in the first Phase 3 attempt. p_inf at 78 km
    # is ~1.3 Pa; the laminar-NS recipe (validated at p_inf >= ~30 Pa in stage 2)
    # collapses the bow shock onto the body at those pressures, polluting the
    # cluster via warm restart. The 60 km cap gives p_inf >= ~22 Pa, the floor
    # the discriminator analysis (analyze_gates2.py) places the broken cluster at.
    "altitude":  (45.0,  60.0),    # km
    "mach":      (8.0,   25.0),
}

# OOD slabs: each overrides a subset of GEOM_BOX or FS_BOX ranges; the rest are
# drawn from the core boxes. Kept small on purpose.
OOD_SLABS = {
    "nose_large": {"R_n":     (0.050, 0.100)},
    "nose_small": {"R_n":     (0.004, 0.010)},
    "cone_high":  {"theta_c": (70.0,  82.0)},
    "mach_high":  {"mach":    (25.0,  30.0)},
}

T_WALL = 300.0  # K, fixed isothermal cold wall

# continuum no-slip Navier-Stokes holds below this nose-radius Knudsen number;
# above it the wall slips and the shock layer merges (see analyze_gates2.py: the
# broken low-pressure cluster sits at Kn > 0.01). the altitude cap in FS_BOX is a
# coarse proxy for this; the filter in sample_cases enforces it case by case,
# since high Mach and small nose radius can push Kn past the floor even at the
# 60 km cap.
KN_MAX = 0.01

_GEOM_ORDER = ("R_n", "theta_c", "R_b_ratio", "R_s_ratio")
_FS_ORDER = ("altitude", "mach")


# ============================================================================================
#                                   case specification
# ============================================================================================

@dataclass
class CaseSpec:
    """One sampled case: its group tag, its geometry-cluster id, and the case.

    ``geom_id`` is unique per distinct meshed geometry; cases with the same
    ``geom_id`` share a mesh and can restart from one another.
    """
    group: str
    geom_id: int
    case: Case


# ============================================================================================
#                                       sampling
# ============================================================================================

def _scale(unit: np.ndarray, names: tuple[str, ...], box: dict) -> np.ndarray:
    out = np.empty_like(unit)
    for j, name in enumerate(names):
        lo, hi = box[name]
        out[:, j] = lo + unit[:, j] * (hi - lo)
    return out


def _nn_order(points: np.ndarray) -> list[int]:
    """Greedy nearest-neighbour visiting order over rows of ``points`` (min-max scaled)."""
    n = len(points)
    if n <= 2:
        return list(range(n))
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    f = (points - lo) / span
    start = int(np.argmin(((f - f.mean(axis=0)) ** 2).sum(axis=1)))
    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True
    for _ in range(n - 1):
        d2 = ((f - f[order[-1]]) ** 2).sum(axis=1)
        d2[visited] = np.inf
        nxt = int(np.argmin(d2))
        order.append(nxt)
        visited[nxt] = True
    return order


def _build_block(
    group: str,
    n_geom: int,
    n_fs: int,
    geom_box: dict,
    fs_box: dict,
    rng: np.random.Generator,
    geom_id0: int,
) -> tuple[list[CaseSpec], int]:
    """Build one nested DOE block; returns (specs, next free geom_id)."""
    geom_unit = qmc.LatinHypercube(d=len(_GEOM_ORDER), seed=rng).random(n_geom)
    geom_rows = _scale(geom_unit, _GEOM_ORDER, geom_box)
    geom_visit = _nn_order(geom_rows)

    specs: list[CaseSpec] = []
    gid = geom_id0
    for gi in geom_visit:
        R_n, theta_c, R_b_ratio, R_s_ratio = geom_rows[gi]
        R_b = float(R_n * R_b_ratio)
        R_s = float(R_b * R_s_ratio)

        fs_unit = qmc.LatinHypercube(d=len(_FS_ORDER), seed=rng).random(n_fs)
        fs_rows = _scale(fs_unit, _FS_ORDER, fs_box)
        for fi in _nn_order(fs_rows):
            altitude, mach = fs_rows[fi]
            T_inf, p_inf, _rho = us_standard_atmosphere(float(altitude))
            specs.append(CaseSpec(
                group=group,
                geom_id=gid,
                case=Case(
                    R_n=float(R_n), theta_c_deg=float(theta_c),
                    R_b=R_b, R_s=R_s, mach=float(mach),
                    T_inf=T_inf, p_inf=p_inf, T_w=T_WALL,
                ),
            ))
        gid += 1
    return specs, gid


def sample_cases(
    n_geom: int = 70,
    n_fs: int = 10,
    n_geom_ood: int = 10,
    n_fs_ood: int = 2,
    seed: int = 0,
) -> list[CaseSpec]:
    """Build the Phase 3 case list: a nested core DOE plus one per OOD slab.

    Parameters
    ----------
    n_geom : int
        Core geometries.
    n_fs : int
        Freestream points per core geometry. Core size is ``n_geom * n_fs``.
    n_geom_ood : int
        Geometries per OOD slab.
    n_fs_ood : int
        Freestream points per OOD geometry. Each slab is ``n_geom_ood * n_fs_ood``
        cases; with the four default slabs the OOD total is
        ``4 * n_geom_ood * n_fs_ood``.
    seed : int
        Seed for the LHS engines (deterministic).

    Returns
    -------
    list of CaseSpec
        Core block first, then OOD slabs in :data:`OOD_SLABS` order. Within each
        block, cases are grouped by geometry (geometries nearest-neighbour
        ordered) and freestream-ordered within a geometry, so consecutive cases
        in the same geometry cluster can restart from one another.
    """
    rng = np.random.default_rng(seed)
    specs, gid = _build_block("core", n_geom, n_fs, GEOM_BOX, FS_BOX, rng, 0)
    for name, overrides in OOD_SLABS.items():
        gb = {**GEOM_BOX, **{k: v for k, v in overrides.items() if k in GEOM_BOX}}
        fb = {**FS_BOX, **{k: v for k, v in overrides.items() if k in FS_BOX}}
        block, gid = _build_block(name, n_geom_ood, n_fs_ood, gb, fb, rng, gid)
        specs.extend(block)
    # drop slip-regime draws: continuum no-slip NS and the analytical gates are
    # invalid above KN_MAX. dropping leaves gaps in geometry clusters, which
    # init_ledger's restart_from logic chains over (it matches on geom_id).
    specs = [
        s for s in specs
        if knudsen_number(s.case.mach, s.case.T_inf, s.case.p_inf, s.case.R_n) <= KN_MAX
    ]
    return specs


def partition(n_items: int, n_blocks: int) -> list[list[int]]:
    """Split ``range(n_items)`` into ``n_blocks`` contiguous near-equal blocks.

    Applied to the case order, each block is a contiguous run of geometry
    clusters, so a worker assigned one block warm-starts within it. Static
    partitioning needs no shared ledger across machines; a final sweep picks up
    anything a machine failed to finish.

    Parameters
    ----------
    n_items : int
        Number of items to split.
    n_blocks : int
        Number of blocks (>= 1).

    Returns
    -------
    list of list of int
    """
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    edges = np.linspace(0, n_items, n_blocks + 1, dtype=int)
    return [list(range(edges[i], edges[i + 1])) for i in range(n_blocks)]
