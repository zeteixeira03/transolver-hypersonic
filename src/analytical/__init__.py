"""Analytical correlations for hypersonic blunt-body validation.

Three closed-form checks on the stagnation region of a sphere-cone. Each one
covers a different SU2 output, so together they catch an error anywhere in the
axisymmetric laminar Navier-Stokes pipeline.

- :func:`fay_riddell_qw`     stagnation-point heat flux, laminar boundary layer
- :func:`rayleigh_pitot_p02` stagnation pressure behind the normal shock
- :func:`billig_standoff`    standoff distance of the bow shock

Every correlation assumes a calorically perfect gas, which matches the solver
setup this project uses.

References
----------
Anderson, J. D., "Hypersonic and High-Temperature Gas Dynamics", 2nd edition,
    AIAA, 2006.
Fay, J. A. and Riddell, F. R., "Theory of Stagnation Point Heat Transfer in
    Dissociated Air", Journal of the Aeronautical Sciences, Volume 25,
    Number 2, 1958, pages 73-85.
Billig, F. S., "Shock-Wave Shapes Around Spherical- and Cylindrical-Nosed
    Bodies", Journal of Spacecraft and Rockets, Volume 4, Number 6, 1967,
    pages 822-823.
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

# Sutherland's law for the viscosity of air
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

    Re_n is the Reynolds number on the nose radius, rho_inf V_inf R_n / mu(T_inf).
    The solver treats air as a continuum and holds the no-slip condition at the
    wall. That model needs a Knudsen number below about 0.01. Above it the wall
    takes on a velocity slip and a temperature jump that the solver cannot
    represent. The bow shock and the boundary layer also merge into one thick
    viscous layer, which neither Fay-Riddell nor Billig describes.

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

    The expression joins the static-pressure jump across a normal shock to the
    isentropic compression that brings the subsonic flow behind it to rest. The
    gas is calorically perfect. NACA Report 1135 gives it as Equation 100 and
    calls the result the Pitot pressure.

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

    An empirical fit by Billig (1967), reproduced as Equation 5.32 in Anderson.
    It holds above about Mach 2 and approaches the correct limit at high Mach
    number.

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

    Fay-Riddell (1958) with the Lewis number set to unity, which drops the
    dissociation source term. That matches the solver, which models air as a
    calorically perfect ideal gas. The velocity gradient at the stagnation point
    takes the modified Newtonian form
    ``(du_e/dx)_s = (1/R_n) sqrt(2 (p_02 - p_inf) / rho_02)``.

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

    # stagnation conditions behind the normal shock, which are also the state at
    # the edge of the boundary layer on the stagnation streamline
    p_02 = rayleigh_pitot_p02(M_inf, p_inf, gamma=g)
    # stagnation temperature is freestream total temperature, conserved across the shock
    T_0 = T_inf * (1.0 + 0.5 * (g - 1.0) * M_inf * M_inf)
    rho_02 = p_02 / (R_AIR * T_0)
    mu_e = sutherland_mu(T_0)

    # wall state. the boundary layer is thin, so the wall pressure equals the
    # edge pressure; the temperature is the isothermal wall value
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
