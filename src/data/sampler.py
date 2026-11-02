"""Sweep parameter sampler over geometry, freestream, and extrapolation slabs.

Two nested Latin hypercubes: one over geometry (R_n, theta_c, R_b/R_n,
R_s/R_b), and per geometry one over freestream (altitude, M_inf). A new
geometry needs a new mesh and a cold solve, while a new freestream on the same
mesh can warm-start from its neighbour, so cases sharing a geometry come out
consecutive in nearest-neighbour order. Altitude sets (T_inf, p_inf) through
the US Standard Atmosphere 1976.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from src.analytical import R_AIR, knudsen_number
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


def us_standard_atmosphere(h_km: float) -> tuple[float, float, float]:
    """Return (T [K], p [Pa], rho [kg/m^3]) at geometric altitude ``h_km`` in km, 0 to 86."""
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
        p = base_p * np.exp(-_G0 * dh / (R_AIR * base_T))
    else:
        lapse_si = lapse / 1000.0  # K/m
        T = base_T + lapse_si * dh
        p = base_p * (T / base_T) ** (-_G0 / (R_AIR * lapse_si))
    return float(T), float(p), float(p / (R_AIR * T))


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
    # the 60 km ceiling keeps p_inf above about 22 Pa. below that the recipe
    # collapses the bow shock onto the body, and a collapsed case spreads
    # through its geometry cluster by warm restart.
    "altitude":  (45.0,  60.0),    # km
    "mach":      (8.0,   25.0),
}

# each slab overrides part of the core boxes. they stay small because they
# measure extrapolation and must not become training coverage.
OOD_SLABS = {
    "nose_large": {"R_n":     (0.050, 0.100)},
    "nose_small": {"R_n":     (0.004, 0.010)},
    "cone_high":  {"theta_c": (70.0,  82.0)},
    "mach_high":  {"mach":    (25.0,  30.0)},
}

T_WALL = 300.0  # K, fixed isothermal cold wall

# every case measured above this nose-radius Knudsen number came out broken.
# high Mach with a small nose can cross it below the 60 km ceiling.
KN_MAX = 0.01

_GEOM_ORDER = ("R_n", "theta_c", "R_b_ratio", "R_s_ratio")
_FS_ORDER = ("altitude", "mach")


# ============================================================================================
#                                   case specification
# ============================================================================================

@dataclass
class CaseSpec:
    """One sampled case with its group tag and geometry id, which names its mesh."""
    group: str
    geom_id: int
    case: Case


# ============================================================================================
#                                       sampling
# ============================================================================================

def case_params(R_n: float, theta_c: float, R_b_ratio: float, R_s_ratio: float,
                mach: float, altitude: float) -> dict[str, float]:
    """Turn the six design axes into the eight parameters of a case."""
    T_inf, p_inf, _rho = us_standard_atmosphere(altitude)
    R_b = R_b_ratio * R_n
    return {"R_n": R_n, "theta_c_deg": theta_c, "R_b": R_b, "R_s": R_s_ratio * R_b,
            "mach": mach, "T_inf": T_inf, "p_inf": p_inf, "T_w": T_WALL}


def _scale(unit: np.ndarray, names: tuple[str, ...], box: dict) -> np.ndarray:
    out = np.empty_like(unit)
    for j, name in enumerate(names):
        lo, hi = box[name]
        out[:, j] = lo + unit[:, j] * (hi - lo)
    return out


def _nn_order(points: np.ndarray) -> list[int]:
    """Greedy nearest-neighbour order over the rows of ``points``, columns scaled to [0, 1]."""
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
    """Build one nested block of cases and return it with the next free geom_id."""
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
    """Build the sweep case list: the core block, then each slab of :data:`OOD_SLABS`.

    The core holds ``n_geom * n_fs`` cases and each slab ``n_geom_ood * n_fs_ood``,
    less any case above :data:`KN_MAX`.
    """
    rng = np.random.default_rng(seed)
    specs, gid = _build_block("core", n_geom, n_fs, GEOM_BOX, FS_BOX, rng, 0)
    for name, overrides in OOD_SLABS.items():
        gb = {**GEOM_BOX, **{k: v for k, v in overrides.items() if k in GEOM_BOX}}
        fb = {**FS_BOX, **{k: v for k, v in overrides.items() if k in FS_BOX}}
        block, gid = _build_block(name, n_geom_ood, n_fs_ood, gb, fb, rng, gid)
        specs.extend(block)
    # init_ledger chains restarts on geom_id, so the gaps this leaves are harmless
    specs = [
        s for s in specs
        if knudsen_number(s.case.mach, s.case.T_inf, s.case.p_inf, s.case.R_n) <= KN_MAX
    ]
    return specs


def partition(n_items: int, n_blocks: int) -> list[list[int]]:
    """Split ``range(n_items)`` into ``n_blocks`` contiguous near-equal blocks."""
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    edges = np.linspace(0, n_items, n_blocks + 1, dtype=int)
    return [list(range(edges[i], edges[i + 1])) for i in range(n_blocks)]
