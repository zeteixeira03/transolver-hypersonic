"""Compare SU2 single-case output against analytical hypersonic correlations.

Three acceptance gates, each with its own tolerance:

- stagnation heat flux against Fay-Riddell, within about 15%
- stagnation pressure against Rayleigh-Pitot, within about 5%
- bow-shock standoff against Billig, within about 20%

The standoff gate holds over part of the design box and not all of it. Billig
fits a spherical nose and carries no term for the cone angle. A blunt
sphere-cone presents a much wider frontal shape than a sphere of radius R_n, so
its shock stands off farther than the correlation predicts, and the difference
is physical rather than an error. Above :data:`THETA_C_BILLIG_MAX` the standoff
check reports both values and takes no part in the decision to pass or fail.

:func:`compare_to_analytical` returns a summary that the validation script
prints to the console and also writes to JSON beside the other run files.
"""

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
# Billig fits the standoff of a spherical nose and carries no cone-angle term.
# a sphere-cone blunter than this half-angle presents a far wider frontal shape
# than a sphere of radius R_n, and its shock stands off about 2.3 times farther.
# the ratio of the solver value to the Billig value tracks the cone angle with a
# correlation of 0.78: close to one below 60 degrees, about 2.3 above it. past
# this angle the standoff is reported and not gated.
THETA_C_BILLIG_MAX = 60.0  # deg


# ============================================================================================
#                                       comparison
# ============================================================================================

@dataclass
class Check:
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
        # a check outside its validity domain never fails the case. the
        # relative error is still computed and reported.
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
    """Compare three stagnation and shock quantities against the correlations.

    Parameters
    ----------
    M_inf, T_inf, p_inf, R_n, T_w : float
        Freestream conditions, the nose radius, and the wall temperature.
    su2_qw : float
        Stagnation wall heat flux from SU2, W/m^2, positive into the wall. The
        Fay-Riddell value it meets is a magnitude.
    su2_p02 : float
        Stagnation pressure from SU2, which is the peak wall pressure, Pa.
    su2_standoff : float
        Standoff distance from the nose to the bow shock along r = 0, m.
    theta_c_deg : float, optional
        Cone half-angle in degrees. It decides whether the standoff check
        gates. Above :data:`THETA_C_BILLIG_MAX` the spherical Billig reference
        does not apply, so the check reports and does not gate. None gates the
        standoff check at every angle.

    Returns
    -------
    dict
        ``checks`` as a list of Check, ``all_passed`` as a boolean, and the
        three reference values under ``ref_qw``, ``ref_p02`` and
        ``ref_standoff``.
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
    """Thermal conductivity of air from Sutherland viscosity at fixed Prandtl.

    ``k = mu(T) * cp / Pr`` at the laminar Prandtl number of 0.71. The solver
    runs the same model, so a heat flux computed here from a temperature field
    rests on the same constitutive assumptions as the Fay-Riddell reference.
    """
    return sutherland_mu(T) * CP_AIR / PR_AIR


def identify_wall_nodes(
    T: np.ndarray, T_w: float, tol: float = 5.0,
) -> np.ndarray:
    """Indices of the nodes that lie on the isothermal wall.

    A node counts as a wall node when ``|T - T_w|`` falls below ``tol``. The
    default of 5 K covers the numerical drift a converged field carries. The
    canonical case holds about 440 wall nodes and its largest departure from
    T_w is about 1.2 K.
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
    """Inward unit normal to the wall at one wall node.

    The tangent is the principal direction of the ``k_tan`` nearest other wall
    nodes, taken from a singular value decomposition of their displacements.
    The normal is perpendicular to that tangent. Its sign is chosen so that it
    points into the fluid, where the temperature exceeds T_w.
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
    # take the direction of rising temperature, which points away from the cold
    # wall and into the hot fluid behind the shock. that direction is the mean
    # displacement of the nearby off-wall nodes, weighted by temperature.
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
    """Stagnation-point wall heat flux from a per-node temperature field.

    The function finds the wall nodes, or takes the ones it is given, keeps the
    ``n_average`` of them closest to ``r = 0`` above the ``y_axis_skip`` band,
    and reads the wall temperature gradient from the nearest fluid node along
    the inward normal.

    The result depends on how finely the mesh resolves the near-wall
    temperature gradient. On the fine canonical mesh it agrees with the wall
    heat flux SU2 reports to within about 5%. At the coarser resolution of the
    sweep it underresolves that gradient and reads low by a large factor. Use it
    to compare a predicted field against a true field through the same
    estimator. Do not read it as a physical heat flux, and do not compare it
    against the value in the ledger.

    Parameters
    ----------
    x, r : np.ndarray
        Per-node coordinates, shape (N,).
    T : np.ndarray
        Per-node temperature, shape (N,), Kelvin.
    T_w : float
        Wall temperature in K, the isothermal boundary condition.
    wall_indices : np.ndarray, optional
        Wall-node indices found earlier. None finds them from ``T`` through
        :func:`identify_wall_nodes`. Against a *predicted* temperature field,
        pass the indices found from the *true* field. The surrogate carries no
        constraint that T equals T_w at the wall, so indices taken from its own
        output would be circular.
    y_axis_skip : float
        Drop wall nodes with ``r < y_axis_skip``. The axisymmetric source term
        inflates wall values within about 5% of R_n of the axis. This matches
        the convention in :func:`src.cfd.postprocess.stagnation_values`.
    n_average : int
        Average the magnitude over this many surviving wall nodes, taken at the
        smallest r.

    Returns
    -------
    dict
        ``q_w``, the averaged stagnation magnitude in W/m^2. ``per_node``, the
        individual magnitudes, of shape (n_average,). ``wall_node_indices``,
        the nodes used. ``slopes``, the fitted values of dT/dn in K/m.

    Notes
    -----
    ``q_w`` comes back as a positive magnitude, which is what Fay-Riddell
    returns. The fluid behind the shock is hot, near 5000 K at its peak, and
    the wall is cold at 300 K, so the heat flows into the wall whichever sign
    the normal takes.
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

    # only off-wall nodes can serve as neighbours. the wall is isothermal, so a
    # second wall node carries no temperature difference and would flatten the
    # gradient
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
        # temperature through the boundary layer is far from linear, so a
        # least-squares fit across several cells reads the wall slope low. the
        # first cell sits about R_n/30000 from the wall, close enough that a
        # one-sided difference anchored at the known T(0) = T_w beats the fit.
        # take the nearest cell along the normal and use (T - T_w) / d.
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
