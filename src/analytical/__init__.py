"""Analytical correlations for hypersonic blunt-body validation.

Three sanity checks against SU2 axisymmetric laminar Navier-Stokes for the
stagnation region of a sphere-cone:

- :func:`fay_riddell_qw`     stagnation-point heat flux (laminar, equilibrium BL)
- :func:`rayleigh_pitot_p02` post-normal-shock stagnation pressure (calorically perfect)
- :func:`billig_standoff`    bow-shock standoff distance correlation

References
----------
Anderson, J. D., "Hypersonic and High-Temperature Gas Dynamics", 2nd ed., AIAA, 2006.
Fay, J. A. and Riddell, F. R., "Theory of Stagnation Point Heat Transfer in
    Dissociated Air", J. Aero. Sci., Vol. 25, No. 2, 1958, pp. 73-85.
Billig, F. S., "Shock-Wave Shapes Around Spherical- and Cylindrical-Nosed
    Bodies", J. Spacecraft, Vol. 4, No. 6, 1967, pp. 822-823.
NACA Report 1135, "Equations, Tables, and Charts for Compressible Flow", 1953.
"""

from __future__ import annotations

import math


# ============================================================================================
#                                       air constants
# ============================================================================================

R_AIR = 287.05          # J/(kg K), specific gas constant for dry air
GAMMA_AIR = 1.4         # ratio of specific heats, calorically perfect
CP_AIR = 1004.5         # J/(kg K), specific heat at constant pressure
PR_AIR = 0.71           # Prandtl number, cold-wall laminar boundary layer

# Sutherland's law for air viscosity (SI)
MU_REF = 1.716e-5       # Pa s, reference viscosity at T_REF
T_REF = 273.15          # K
S_SUTH = 110.4          # K, Sutherland constant


def sutherland_mu(T: float) -> float:
    """Dynamic viscosity of air via Sutherland's law.

    Parameters
    ----------
    T : float
        Static temperature in K.

    Returns
    -------
    float
        Dynamic viscosity in Pa s.
    """
    return MU_REF * (T / T_REF) ** 1.5 * (T_REF + S_SUTH) / (T + S_SUTH)


def knudsen_number(M_inf: float, T_inf: float, p_inf: float, R_n: float) -> float:
    """Freestream Knudsen number on the nose radius, Kn = 1.26 sqrt(gamma) M / Re_n.

    Re_n is the nose-radius Reynolds number rho_inf V_inf R_n / mu(T_inf). The
    continuum no-slip Navier-Stokes model the dataset assumes is valid only for
    Kn below ~0.01; above that the wall develops velocity slip and a temperature
    jump the solver does not represent, and the bow shock and boundary layer
    merge into a thick viscous layer the inviscid Fay-Riddell and Billig
    correlations no longer describe.

    Parameters
    ----------
    M_inf : float
        Freestream Mach number.
    T_inf, p_inf : float
        Freestream static temperature (K) and pressure (Pa).
    R_n : float
        Nose radius in m.

    Returns
    -------
    float
        Knudsen number on the nose radius (dimensionless).
    """
    rho_inf = p_inf / (R_AIR * T_inf)
    V_inf = M_inf * math.sqrt(GAMMA_AIR * R_AIR * T_inf)
    Re_n = rho_inf * V_inf * R_n / sutherland_mu(T_inf)
    return 1.26 * math.sqrt(GAMMA_AIR) * M_inf / Re_n


# ============================================================================================
#                              rayleigh-pitot post-shock stagnation
# ============================================================================================

def rayleigh_pitot_p02(M_inf: float, p_inf: float, gamma: float = GAMMA_AIR) -> float:
    """Post-normal-shock stagnation pressure from supersonic freestream.

    Combines the normal-shock static-pressure jump with isentropic compression
    of the subsonic post-shock flow to rest. Calorically perfect gas; this is
    the canonical "Pitot pressure" expression in NACA 1135 (Eq. 100).

    Parameters
    ----------
    M_inf : float
        Freestream Mach number, must satisfy M_inf > 1.
    p_inf : float
        Freestream static pressure in Pa.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    float
        Total (stagnation) pressure behind the normal shock, Pa.
    """
    if M_inf <= 1.0:
        raise ValueError("rayleigh_pitot_p02 requires supersonic freestream M_inf > 1")
    g = gamma
    M2 = M_inf * M_inf
    term1 = ((g + 1.0) ** 2 * M2 / (4.0 * g * M2 - 2.0 * (g - 1.0))) ** (g / (g - 1.0))
    term2 = (1.0 - g + 2.0 * g * M2) / (g + 1.0)
    return p_inf * term1 * term2


# ============================================================================================
#                               billig bow-shock standoff
# ============================================================================================

def billig_standoff(M_inf: float, R_n: float) -> float:
    """Bow-shock standoff distance ahead of a spherical nose.

    Empirical fit by Billig (1967), reproduced as Anderson Hypersonic Eq. 5.32.
    Valid for M_inf >= 2 roughly; asymptotes correctly as M_inf grows.

    Parameters
    ----------
    M_inf : float
        Freestream Mach number.
    R_n : float
        Nose radius in m.

    Returns
    -------
    float
        Standoff distance from nose to bow shock along the stagnation line, m.
    """
    if M_inf <= 1.0:
        raise ValueError("billig_standoff requires supersonic freestream M_inf > 1")
    return R_n * 0.143 * math.exp(3.24 / (M_inf * M_inf))


# ============================================================================================
#                              fay-riddell stagnation heat flux
# ============================================================================================

def fay_riddell_qw(
    M_inf: float,
    T_inf: float,
    p_inf: float,
    R_n: float,
    T_w: float,
    gamma: float = GAMMA_AIR,
) -> float:
    """Stagnation-point heat flux to a cold wall, laminar equilibrium boundary layer.

    Fay-Riddell (1958) with Lewis number set to unity (no dissociation source
    term, consistent with the ideal-gas calorically perfect SU2 setup used in
    this project). The stagnation velocity gradient uses the modified-Newtonian
    form ``(du_e/dx)_s = (1/R_n) sqrt(2 (p_02 - p_inf) / rho_02)``.

    Parameters
    ----------
    M_inf : float
        Freestream Mach number, must satisfy M_inf > 1.
    T_inf : float
        Freestream static temperature in K.
    p_inf : float
        Freestream static pressure in Pa.
    R_n : float
        Nose radius in m.
    T_w : float
        Wall temperature in K (isothermal cold wall).
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    float
        Wall heat flux at the stagnation point, W/m^2.
    """
    if M_inf <= 1.0:
        raise ValueError("fay_riddell_qw requires supersonic freestream M_inf > 1")
    g = gamma

    # post-normal-shock stagnation conditions (boundary-layer edge state at stagnation point)
    p_02 = rayleigh_pitot_p02(M_inf, p_inf, gamma=g)
    # stagnation temperature is freestream total temperature, conserved across the shock
    T_0 = T_inf * (1.0 + 0.5 * (g - 1.0) * M_inf * M_inf)
    rho_02 = p_02 / (R_AIR * T_0)
    mu_e = sutherland_mu(T_0)

    # wall state: same pressure as boundary-layer edge at stagnation (thin BL), wall temperature
    rho_w = p_02 / (R_AIR * T_w)
    mu_w = sutherland_mu(T_w)

    # stagnation velocity gradient (modified Newtonian)
    due_dx = (1.0 / R_n) * math.sqrt(2.0 * (p_02 - p_inf) / rho_02)

    # enthalpy difference, calorically perfect
    h_0e = CP_AIR * T_0
    h_w = CP_AIR * T_w

    return (
        0.763
        * PR_AIR ** (-0.6)
        * (rho_w * mu_w) ** 0.1
        * (rho_02 * mu_e) ** 0.4
        * math.sqrt(due_dx)
        * (h_0e - h_w)
    )
