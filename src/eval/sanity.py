"""Compare SU2 single-case output against analytical hypersonic correlations.

The three Phase 2 acceptance gates:

- Stagnation heat flux vs Fay-Riddell:       expect agreement within ~15%
- Stagnation pressure vs Rayleigh-Pitot:     expect agreement within ~5%
- Bow-shock standoff vs Billig:              expect agreement within ~20%

:func:`compare_to_analytical` returns a structured summary that the validation
script renders to console and persists to JSON alongside the run artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.analytical import billig_standoff, fay_riddell_qw, rayleigh_pitot_p02


# ============================================================================================
#                                       tolerances
# ============================================================================================

TOL_QW   = 0.15  # Fay-Riddell
TOL_P02  = 0.05  # Rayleigh-Pitot
TOL_DELTA = 0.20  # Billig


# ============================================================================================
#                                       comparison
# ============================================================================================

@dataclass
class Check:
    name: str
    su2: float
    analytical: float
    tolerance: float

    @property
    def rel_err(self) -> float:
        return (self.su2 - self.analytical) / self.analytical

    @property
    def passed(self) -> bool:
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

    Returns
    -------
    dict
        ``checks`` (list of Check), ``all_passed`` (bool), and the analytical
        reference values keyed ``ref_qw``, ``ref_p02``, ``ref_standoff``.
    """
    ref_qw = fay_riddell_qw(M_inf, T_inf, p_inf, R_n, T_w)
    ref_p02 = rayleigh_pitot_p02(M_inf, p_inf)
    ref_standoff = billig_standoff(M_inf, R_n)

    checks = [
        Check("stagnation_heat_flux",     abs(su2_qw),       ref_qw,       TOL_QW),
        Check("stagnation_pressure",      su2_p02,           ref_p02,      TOL_P02),
        Check("shock_standoff",           su2_standoff,      ref_standoff, TOL_DELTA),
    ]
    return {
        "checks": checks,
        "all_passed": all(c.passed for c in checks),
        "ref_qw": ref_qw,
        "ref_p02": ref_p02,
        "ref_standoff": ref_standoff,
    }


def format_summary(summary: dict) -> str:
    """Format the comparison result as a console-friendly table."""
    lines = []
    lines.append(f"{'quantity':<24} {'SU2':>14} {'analytical':>14} {'rel err':>9} {'tol':>7} {'pass':>5}")
    lines.append("-" * 76)
    for c in summary["checks"]:
        lines.append(
            f"{c.name:<24} {c.su2:>14.4g} {c.analytical:>14.4g} "
            f"{c.rel_err * 100:>+8.2f}% {c.tolerance * 100:>5.0f}% "
            f"{'YES' if c.passed else 'NO':>5}"
        )
    lines.append("-" * 76)
    lines.append(f"all passed: {summary['all_passed']}")
    return "\n".join(lines)
