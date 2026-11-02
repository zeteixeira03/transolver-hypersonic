"""Compare one SU2 case against the analytical correlations, and estimate wall heat flux."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.analytical import (
    CP_AIR,
    PR_AIR,
    billig_standoff,
    fay_riddell_qw,
    rayleigh_pitot_p02,
    sutherland_mu,
)

# ============================================================================================
#                                       tolerances
# ============================================================================================

TOL_QW   = 0.15  # Fay-Riddell
TOL_P02  = 0.05  # Rayleigh-Pitot
TOL_DELTA = 0.20  # Billig
TOL_DELTA_HIGH_MACH = 0.50  # Billig reads low at high Mach number
M_HIGH = 22.0               # above this Mach number the wider tolerance applies
# Billig has no cone-angle term. The ratio of solver to Billig standoff tracks
# the cone angle (correlation 0.78): near 1 below 60 degrees, about 2.3 above.
# Past this angle the standoff is reported and not gated.
THETA_C_BILLIG_MAX = 60.0  # deg


# ============================================================================================
#                                       comparison
# ============================================================================================

@dataclass
class Check:
    """One quantity against its correlation. A check that is not applicable always passes."""
    name: str
    su2: float
    analytical: float
    tolerance: float
    applicable: bool = True

    @property
    def rel_err(self) -> float:
        return (self.su2 - self.analytical) / self.analytical

    @property
    def passed(self) -> bool:
        if not self.applicable:
            return True
        return abs(self.rel_err) <= self.tolerance


def compare_to_analytical(
    *,
    M_inf: float,
    T_inf: float,
    p_inf: float,
    R_n: float,
    T_w: float,
    su2_qw: float,
    su2_p02: float,
    su2_standoff: float,
    theta_c_deg: float | None = None,
) -> dict:
    """Compare heat flux (W/m^2), stagnation pressure (Pa) and standoff (m) to the correlations.

    The heat flux is compared as a magnitude. ``theta_c_deg=None`` gates the
    standoff at every cone angle.
    """
    ref_qw = fay_riddell_qw(M_inf, T_inf, p_inf, R_n, T_w)
    ref_p02 = rayleigh_pitot_p02(M_inf, p_inf)
    ref_standoff = billig_standoff(M_inf, R_n)

    tol_delta = TOL_DELTA_HIGH_MACH if M_inf > M_HIGH else TOL_DELTA
    standoff_applicable = theta_c_deg is None or theta_c_deg <= THETA_C_BILLIG_MAX

    checks = [
        Check("stagnation_heat_flux", abs(su2_qw), ref_qw,       TOL_QW),
        Check("stagnation_pressure",  su2_p02,     ref_p02,      TOL_P02),
        Check("shock_standoff",       su2_standoff, ref_standoff, tol_delta,
              applicable=standoff_applicable),
    ]
    return {
        "checks": checks,
        "all_passed": all(c.passed for c in checks),
        "ref_qw": ref_qw,
        "ref_p02": ref_p02,
        "ref_standoff": ref_standoff,
    }


# ============================================================================================
#                                  q_w from predicted T field
# ============================================================================================

def thermal_conductivity_air(T: float) -> float:
    """Thermal conductivity of air in W/(m K), ``mu(T) cp / Pr``, the solver's own model."""
    return sutherland_mu(T) * CP_AIR / PR_AIR


def identify_wall_nodes(
    T: np.ndarray, T_w: float, tol: float = 5.0,
) -> np.ndarray:
    """Indices of nodes with ``|T - T_w| < tol``, the isothermal wall.

    On the canonical case the largest wall departure from T_w is about 1.2 K.
    """
    return np.where(np.abs(np.asarray(T) - T_w) < tol)[0]


def _wall_normal(
    x: np.ndarray,
    r: np.ndarray,
    T: np.ndarray,
    T_w: float,
    wall_indices: np.ndarray,
    node_idx: int,
    k_tan: int = 4,
) -> np.ndarray:
    """Unit normal at one wall node, pointing into the hot fluid.

    The tangent is the principal direction of the ``k_tan`` nearest wall nodes.
    """
    others = wall_indices[wall_indices != node_idx]
    dx = x[others] - x[node_idx]
    dr = r[others] - r[node_idx]
    d = np.sqrt(dx * dx + dr * dr)
    nb = np.argsort(d)[:k_tan]
    pts = np.stack([dx[nb], dr[nb]], axis=1)
    pts = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts, full_matrices=False)
    tangent = vh[0]
    normal = np.array([-tangent[1], tangent[0]])
    # orient along the temperature-weighted displacement of nearby fluid nodes
    wall_set = np.zeros(x.shape[0], dtype=bool)
    wall_set[wall_indices] = True
    cand_mask = ~wall_set
    dx_all = x[cand_mask] - x[node_idx]
    dr_all = r[cand_mask] - r[node_idx]
    d_all = np.sqrt(dx_all * dx_all + dr_all * dr_all)
    nn = np.argsort(d_all)[:12]
    T_n = T[cand_mask][nn]
    weighting = T_n - T_w
    grad_dir = np.array([(dx_all[nn] * weighting).sum(),
                         (dr_all[nn] * weighting).sum()])
    if np.dot(normal, grad_dir) < 0:
        normal = -normal
    return normal


def compute_q_w_from_T(
    *,
    x: np.ndarray,
    r: np.ndarray,
    T: np.ndarray,
    T_w: float,
    wall_indices: np.ndarray | None = None,
    y_axis_skip: float = 0.0,
    n_average: int = 3,
) -> dict:
    """Stagnation-point wall heat flux magnitude in W/m^2 from a per-node temperature field.

    Averages the ``n_average`` wall nodes nearest the axis above ``y_axis_skip``,
    each from the nearest fluid node along the normal. On the fine canonical
    mesh it agrees with SU2 to about 5%. On the coarser sweep meshes it reads
    low by a large factor, so use it only to compare a predicted field with a
    true one, never as a physical heat flux.

    For a predicted field, pass ``wall_indices`` found from the true field: the
    surrogate does not hold T = T_w at the wall. Returns ``q_w``, ``per_node``,
    ``wall_node_indices`` and ``slopes`` (dT/dn in K/m).
    """
    x = np.asarray(x, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    if wall_indices is None:
        wall_indices = identify_wall_nodes(T, T_w)
    if wall_indices.size == 0:
        raise ValueError("no wall nodes found; check T_w and tol")

    r_wall = r[wall_indices]
    mask = r_wall >= y_axis_skip
    if not mask.any():
        raise ValueError(
            f"no wall nodes with r >= {y_axis_skip}; "
            f"smallest wall r = {r_wall.min():.4g}"
        )
    survivors = wall_indices[mask]
    r_surv = r[survivors]
    order = np.argsort(r_surv)[: max(1, n_average)]
    stag_nodes = survivors[order]

    k_at_wall = thermal_conductivity_air(T_w)
    slopes = np.empty(len(stag_nodes))
    q_per = np.empty(len(stag_nodes))

    # a neighbouring wall node has no temperature difference and would flatten the gradient
    fluid_mask = np.ones_like(x, dtype=bool)
    for idx in wall_indices:
        fluid_mask[idx] = False
    x_f, r_f = x[fluid_mask], r[fluid_mask]
    T_f = T[fluid_mask]

    for i, idx in enumerate(stag_nodes):
        normal = _wall_normal(x, r, T, T_w, wall_indices, int(idx))
        dx = x_f - x[idx]
        dr = r_f - r[idx]
        d = np.sqrt(dx * dx + dr * dr)
        proj = dx * normal[0] + dr * normal[1]
        cos = np.where(d > 0, proj / np.maximum(d, 1e-300), 0.0)
        keep = (proj > 0) & (cos > 0.9)
        if not keep.any():
            raise ValueError(
                f"wall node {idx}: no fluid neighbor within 26 deg of the "
                f"inward normal; mesh too coarse or normal estimate broken"
            )
        proj_k = proj[keep]
        T_k = T_f[keep]
        # the boundary-layer profile is far from linear, so a least-squares fit
        # reads the slope low. the first cell sits about R_n/30000 off the wall,
        # close enough for a one-sided difference anchored at T(0) = T_w.
        j = int(np.argmin(proj_k))
        slope = (T_k[j] - T_w) / proj_k[j]
        slopes[i] = slope
        q_per[i] = k_at_wall * abs(slope)

    return {
        "q_w": float(q_per.mean()),
        "per_node": q_per,
        "wall_node_indices": stag_nodes,
        "slopes": slopes,
    }


# ============================================================================================
#                                       formatting
# ============================================================================================

def format_summary(summary: dict) -> str:
    """Format the comparison result as a console-friendly table."""
    lines = []
    lines.append(f"{'quantity':<24} {'SU2':>14} {'analytical':>14} "
                 f"{'rel. error':>11} {'limit':>7} {'pass':>5}")
    lines.append("-" * 78)
    for c in summary["checks"]:
        verdict = "n/a" if not c.applicable else ("YES" if c.passed else "NO")
        lines.append(
            f"{c.name:<24} {c.su2:>14.4g} {c.analytical:>14.4g} "
            f"{c.rel_err * 100:>+10.2f}% {c.tolerance * 100:>5.0f}% "
            f"{verdict:>5}"
        )
    lines.append("-" * 78)
    lines.append(f"all passed: {summary['all_passed']}")
    return "\n".join(lines)
