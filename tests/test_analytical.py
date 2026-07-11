"""Unit tests for analytical hypersonic correlations.

Reference values:
- Rayleigh-Pitot ratios from NACA Report 1135 Table II (gamma=1.4).
- Billig standoff: closed-form, checked against Anderson Hypersonic Eq. 5.32.
- Fay-Riddell: self-consistent regression at a representative reentry point
  computed by hand against the same formula, plus monotonicity sanity checks.
"""

from __future__ import annotations

import math

import pytest

from src.analytical import (
    billig_standoff,
    fay_riddell_qw,
    rayleigh_pitot_p02,
    sutherland_mu,
)


# ============================================================================================
#                                       sutherland mu
# ============================================================================================

def test_sutherland_reference_temperature():
    # at T_REF the formula collapses to MU_REF
    mu = sutherland_mu(273.15)
    assert mu == pytest.approx(1.716e-5, rel=1e-9)


def test_sutherland_monotonic_in_temperature():
    temps = [200.0, 300.0, 500.0, 1000.0, 2000.0, 5000.0]
    mus = [sutherland_mu(T) for T in temps]
    assert all(b > a for a, b in zip(mus, mus[1:]))


# ============================================================================================
#                                  rayleigh-pitot (NACA 1135)
# ============================================================================================

@pytest.mark.parametrize(
    "M_inf, ratio_expected",
    [
        (2.0, 5.6404),    # NACA 1135 Table II
        (5.0, 32.653),
        (10.0, 129.22),
        (20.0, 515.49),
    ],
)
def test_rayleigh_pitot_naca_1135(M_inf, ratio_expected):
    p_inf = 100.0
    p_02 = rayleigh_pitot_p02(M_inf, p_inf)
    assert p_02 / p_inf == pytest.approx(ratio_expected, rel=1e-3)


def test_rayleigh_pitot_rejects_subsonic():
    with pytest.raises(ValueError):
        rayleigh_pitot_p02(0.8, 101325.0)


# ============================================================================================
#                                     billig standoff
# ============================================================================================

def test_billig_high_mach_limit():
    # at M -> infinity, delta/R_n -> 0.143 exactly
    delta = billig_standoff(1000.0, 1.0)
    assert delta == pytest.approx(0.143, rel=1e-4)


@pytest.mark.parametrize(
    "M_inf, expected_ratio",
    [
        (5.0, 0.143 * math.exp(3.24 / 25.0)),
        (10.0, 0.143 * math.exp(3.24 / 100.0)),
        (25.0, 0.143 * math.exp(3.24 / 625.0)),
    ],
)
def test_billig_closed_form(M_inf, expected_ratio):
    R_n = 0.0254  # 1 inch nose, Anderson example scale
    delta = billig_standoff(M_inf, R_n)
    assert delta == pytest.approx(R_n * expected_ratio, rel=1e-9)


def test_billig_scales_with_nose_radius():
    d1 = billig_standoff(10.0, 0.1)
    d2 = billig_standoff(10.0, 1.0)
    assert d2 == pytest.approx(10.0 * d1, rel=1e-9)


# ============================================================================================
#                                 fay-riddell heat flux
# ============================================================================================

def test_fay_riddell_reentry_regression():
    """Mach 10, p_inf=100 Pa, T_inf=220 K, R_n=0.5 m, T_w=300 K.

    Hand-computed against the same formula: ~0.25 MW/m^2. This pins the
    constants and unit conventions. If you change any constant (R_AIR, CP_AIR,
    PR_AIR, MU_REF, S_SUTH), update this expected value.
    """
    q_w = fay_riddell_qw(M_inf=10.0, T_inf=220.0, p_inf=100.0, R_n=0.5, T_w=300.0)
    assert q_w == pytest.approx(2.5e5, rel=0.10)  # +/- 10% on the hand calc


def test_fay_riddell_magnitude_band():
    # CLAUDE.md says hypersonic wall heat flux is O(1-100) MW/m^2 at peak.
    # Spot-check a more aggressive case sits inside that band.
    q_w = fay_riddell_qw(M_inf=20.0, T_inf=250.0, p_inf=50.0, R_n=0.3, T_w=300.0)
    assert 1e5 < q_w < 1e8


def test_fay_riddell_decreases_with_nose_radius():
    # q_w ~ 1 / sqrt(R_n) through the velocity gradient
    q_small = fay_riddell_qw(10.0, 220.0, 100.0, 0.1, 300.0)
    q_large = fay_riddell_qw(10.0, 220.0, 100.0, 1.0, 300.0)
    assert q_small > q_large
    # ratio should be roughly sqrt(10) = 3.16
    assert q_small / q_large == pytest.approx(math.sqrt(10.0), rel=0.05)


def test_fay_riddell_increases_with_mach():
    q_low = fay_riddell_qw(5.0, 220.0, 100.0, 0.5, 300.0)
    q_high = fay_riddell_qw(15.0, 220.0, 100.0, 0.5, 300.0)
    assert q_high > q_low


def test_fay_riddell_increases_with_density():
    q_thin = fay_riddell_qw(10.0, 220.0, 10.0, 0.5, 300.0)
    q_dense = fay_riddell_qw(10.0, 220.0, 1000.0, 0.5, 300.0)
    assert q_dense > q_thin


def test_fay_riddell_rejects_subsonic():
    with pytest.raises(ValueError):
        fay_riddell_qw(0.5, 220.0, 100.0, 0.5, 300.0)


# ============================================================================================
#                          standoff gate validity domain (theta_c)
# ============================================================================================

from src.eval.sanity import THETA_C_BILLIG_MAX, compare_to_analytical


def _summary_with_bad_standoff(theta_c_deg):
    """A case whose qw and p02 match the references exactly but whose standoff
    is 2.3x Billig (the blunt-cone overshoot). Isolates the standoff gate."""
    M, T, p, R_n, T_w = 12.0, 250.0, 40.0, 0.015, 300.0
    ref_qw = fay_riddell_qw(M, T, p, R_n, T_w)
    ref_p02 = rayleigh_pitot_p02(M, p)
    ref_sd = billig_standoff(M, R_n)
    return compare_to_analytical(
        M_inf=M, T_inf=T, p_inf=p, R_n=R_n, T_w=T_w,
        su2_qw=ref_qw, su2_p02=ref_p02, su2_standoff=2.3 * ref_sd,
        theta_c_deg=theta_c_deg,
    )


def _standoff_check(summary):
    return next(c for c in summary["checks"] if c.name == "shock_standoff")


def test_standoff_gated_for_slender_cone():
    # below the cutoff Billig applies: a 2.3x standoff is a real miss
    s = _summary_with_bad_standoff(45.0)
    assert _standoff_check(s).applicable is True
    assert s["all_passed"] is False


def test_standoff_not_applicable_for_blunt_cone():
    # above the cutoff Billig is the wrong yardstick: standoff is reported but
    # not gated, so the same 2.3x value does not fail the case
    s = _summary_with_bad_standoff(67.0)
    sd = _standoff_check(s)
    assert sd.applicable is False
    assert sd.passed is True
    assert sd.rel_err == pytest.approx(1.3, rel=1e-6)  # still computed and reported
    assert s["all_passed"] is True


def test_standoff_cutoff_is_inclusive():
    # exactly at the cutoff Billig still applies
    assert _standoff_check(_summary_with_bad_standoff(THETA_C_BILLIG_MAX)).applicable is True


def test_standoff_gated_when_theta_c_unknown():
    # back-compat: callers that pass no theta_c gate the standoff unconditionally
    s = _summary_with_bad_standoff(None)
    assert _standoff_check(s).applicable is True
    assert s["all_passed"] is False
