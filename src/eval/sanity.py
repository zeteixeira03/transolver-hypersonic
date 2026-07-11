"""Compare SU2 single-case output against analytical hypersonic correlations.

The three Phase 2 acceptance gates:

- Stagnation heat flux vs Fay-Riddell:       expect agreement within ~15%
- Stagnation pressure vs Rayleigh-Pitot:     expect agreement within ~5%
- Bow-shock standoff vs Billig:              expect agreement within ~20%

The standoff gate carries a validity domain. Billig is a sphere-nose fit with no
cone-angle term; for blunt sphere-cones (large theta_c) the body is far blunter
than a sphere of radius R_n and the shock genuinely stands off much farther than
the correlation predicts. Above :data:`THETA_C_BILLIG_MAX` the standoff check is
marked not-applicable: the SU2 value and the (invalid) Billig value are still
reported, but the check is excluded from the pass/fail decision. See PHASE_LOG
2026-06-24.

:func:`compare_to_analytical` returns a structured summary that the validation
script renders to console and persists to JSON alongside the run artifacts.
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
TOL_DELTA_HIGH_MACH = 0.50  # Billig correlation under-predicts at high Mach
M_HIGH = 22.0              # threshold above which the wider Billig tol applies
# Billig is a sphere-nose standoff fit (no cone-angle term). For sphere-cones
# blunter than this half-angle the body presents a much wider frontal shape than
# a sphere of radius R_n, the shock stands off ~2.3x farther, and Billig is no
# longer a valid yardstick (CFD/Billig tracks theta_c with corr 0.78; clean
# below 60 deg, ~2.3x above). Standoff is reported but not gated past this.
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
        # a check outside its validity domain never fails the case; the rel
        # error is still computed and reported, just not gated
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
    """Compare three SU2 stagnation/shock quantities against analytical refs.

    Parameters
    ----------
    M_inf, T_inf, p_inf, R_n, T_w : float
        Case parameters (freestream + nose radius + wall T).
    su2_qw : float
        SU2 stagnation wall heat flux, W/m^2. Sign convention: positive into
        the wall (the analytical Fay-Riddell value is the magnitude).
    su2_p02 : float
        SU2 stagnation pressure (peak wall pressure), Pa.
    su2_standoff : float
        SU2 bow-shock standoff distance from the nose along r = 0, m.
    theta_c_deg : float, optional
        Cone half-angle in degrees. Gates the standoff check: above
        :data:`THETA_C_BILLIG_MAX` the sphere-Billig reference does not apply
        and the standoff check is marked not-applicable (reported, not gated).
        When None the standoff check is gated unconditionally (back-compat).

    Returns
    -------
    dict
        ``checks`` (list of Check), ``all_passed`` (bool), and the analytical
        reference values keyed ``ref_qw``, ``ref_p02``, ``ref_standoff``.
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
    """Thermal conductivity of air at T from Sutherland viscosity + constant Pr.

    ``k = mu(T) * cp / Pr`` with the calorically-perfect Pr_lam = 0.71. This is
    the same constitutive model SU2 runs against, so the post-hoc q_w computed
    on a predicted T field is dimensionally consistent with the Fay-Riddell
    reference.
    """
    return sutherland_mu(T) * CP_AIR / PR_AIR


def identify_wall_nodes(
    T: np.ndarray, T_w: float, tol: float = 5.0,
) -> np.ndarray:
    """Indices of nodes lying on the isothermal wall.

    A node is on the wall iff ``|T - T_w| < tol``. The default 5 K matches the
    small numerical drift seen on converged SU2 fields (the canonical case
    has ~440 wall nodes with peak deviation ~1.2 K).
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
    """Inward unit wall-normal at one wall node.

    Tangent is the principal direction of the k_tan nearest other wall
    nodes (SVD of their displacements). Normal is perpendicular to the
    tangent, sign chosen so that it points into the fluid (where T is
    higher than T_w).
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
    # pick the direction in which T increases (away from the cold wall,
    # into the hot post-shock fluid); the gradient direction is the
    # T-weighted mean displacement of off-wall nodes near this wall node
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

    Identifies wall nodes (or uses ones passed in), picks the ``n_average``
    closest to ``r = 0`` (subject to ``y_axis_skip`` to avoid the
    axisymmetric singularity), and fits ``dT/dn`` by least-squares over each
    wall node's nearest fluid neighbors. The wall-normal direction is not
    computed explicitly; we use Euclidean distance from the wall node to
    each fluid neighbor as the local along-normal coordinate. Near a smooth
    wall and over a few BL cells this is the same number to leading order.

    Parameters
    ----------
    x, r : np.ndarray
        Per-node coordinates, shape (N,).
    T : np.ndarray
        Per-node temperature, shape (N,), Kelvin.
    T_w : float
        Wall temperature in K (isothermal BC).
    wall_indices : np.ndarray, optional
        Pre-computed wall-node indices. If None, identified from ``T``
        using :func:`identify_wall_nodes`. For evaluating a *predicted* T
        field, pass the indices identified from the *ground-truth* T --
        the surrogate has no architectural constraint that T = T_w at the
        wall, so identification from predicted T would be circular.
    y_axis_skip : float
        Skip wall nodes with ``r < y_axis_skip``. The axisymmetric source
        term inflates wall values within ~5% of R_n of r = 0; the same
        convention used in :func:`src.cfd.postprocess.stagnation_values`.
    n_average : int
        Average the magnitude across the ``n_average`` smallest-r wall
        nodes that survive the skip.

    Returns
    -------
    dict
        ``q_w`` (float, W/m^2, the averaged stagnation magnitude);
        ``per_node`` (np.ndarray, shape (n_average,), individual q_w
        magnitudes); ``wall_node_indices`` (the ones used);
        ``slopes`` (the fitted dT/dn values, K/m).

    Notes
    -----
    Sign convention: q_w is returned as a magnitude (positive) and is
    directly comparable to Fay-Riddell. The post-shock fluid is hot
    (T peaks ~5000 K) and the wall is cold (T_w = 300 K), so the true
    heat flux is into the wall regardless of which normal direction we
    define.
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

    # neighbor candidates: any non-wall node (the wall is isothermal, so
    # neighboring wall nodes contribute no T gradient, just bias)
    wall_set = set(wall_indices.tolist())
    fluid_mask = np.ones_like(x, dtype=bool)
    for idx in wall_indices:
        fluid_mask[idx] = False
    x_f, r_f = x[fluid_mask], r[fluid_mask]
    T_f = T[fluid_mask]
    fluid_global = np.where(fluid_mask)[0]

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
        # BL T(n) is highly nonlinear; LSQ over multiple cells averages the
        # wall slope down. The fine BL mesh (first cell ~R_n/30000) makes a
        # one-sided FD anchored at T(0) = T_w the right move: take the
        # nearest along-normal cell and use (T - T_w) / d as the wall slope.
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
    lines.append(f"{'quantity':<24} {'SU2':>14} {'analytical':>14} {'rel err':>9} {'tol':>7} {'pass':>5}")
    lines.append("-" * 76)
    for c in summary["checks"]:
        verdict = "n/a" if not c.applicable else ("YES" if c.passed else "NO")
        lines.append(
            f"{c.name:<24} {c.su2:>14.4g} {c.analytical:>14.4g} "
            f"{c.rel_err * 100:>+8.2f}% {c.tolerance * 100:>5.0f}% "
            f"{verdict:>5}"
        )
    lines.append("-" * 76)
    lines.append(f"all passed: {summary['all_passed']}")
    return "\n".join(lines)
