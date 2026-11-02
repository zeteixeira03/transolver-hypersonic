"""Analytical correlations for the stagnation region of a sphere-cone.

Every correlation assumes a calorically perfect gas.

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

R_AIR = 287.058         # J/(kg K), dry air, the GAS_CONSTANT of the SU2 template
GAMMA_AIR = 1.4         # ratio of specific heats, calorically perfect
CP_AIR = 1004.5         # J/(kg K), specific heat at constant pressure
PR_AIR = 0.71           # Prandtl number, cold-wall laminar boundary layer

# Sutherland's law for the viscosity of air
MU_REF = 1.716e-5       # Pa s, reference viscosity at T_REF
T_REF = 273.15          # K
S_SUTH = 110.4          # K, Sutherland constant


def sutherland_mu(T: float) -> float:
    """Dynamic viscosity of air in Pa s from Sutherland's law, T in K."""
    return MU_REF * (T / T_REF) ** 1.5 * (T_REF + S_SUTH) / (T + S_SUTH)


def knudsen_number(M_inf: float, T_inf: float, p_inf: float, R_n: float) -> float:
    """Freestream Knudsen number on the nose radius, Kn = 1.26 sqrt(gamma) M / Re_n.

    Re_n is rho_inf V_inf R_n / mu(T_inf). The continuum, no-slip solver needs
    Kn below about 0.01. Above it the wall sees velocity slip and a temperature
    jump, and the shock and boundary layer merge into one viscous layer that
    neither Fay-Riddell nor Billig describes. T_inf in K, p_inf in Pa, R_n in m.
    """
    rho_inf = p_inf / (R_AIR * T_inf)
    V_inf = M_inf * math.sqrt(GAMMA_AIR * R_AIR * T_inf)
    Re_n = rho_inf * V_inf * R_n / sutherland_mu(T_inf)
    return 1.26 * math.sqrt(GAMMA_AIR) * M_inf / Re_n


# ============================================================================================
#                              rayleigh-pitot post-shock stagnation
# ============================================================================================

def rayleigh_pitot_p02(M_inf: float, p_inf: float, gamma: float = GAMMA_AIR) -> float:
    """Stagnation pressure behind a normal shock in Pa, p_inf in Pa.

    NACA Report 1135, Equation 100 (the Pitot pressure).
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
    """Bow-shock standoff distance on the stagnation line, in the units of R_n.

    Billig's empirical fit (Anderson, Equation 5.32), valid above about Mach 2.
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
    """Stagnation-point heat flux to an isothermal cold wall, W/m^2.

    Fay-Riddell (1958) with unit Lewis number, so no dissociation term. The
    stagnation velocity gradient is modified Newtonian,
    ``(1/R_n) sqrt(2 (p_02 - p_inf) / rho_02)``. Temperatures in K, p_inf in Pa,
    R_n in m.
    """
    if M_inf <= 1.0:
        raise ValueError("fay_riddell_qw requires supersonic freestream M_inf > 1")
    g = gamma

    p_02 = rayleigh_pitot_p02(M_inf, p_inf, gamma=g)
    # total temperature is conserved across the shock
    T_0 = T_inf * (1.0 + 0.5 * (g - 1.0) * M_inf * M_inf)
    rho_02 = p_02 / (R_AIR * T_0)
    mu_e = sutherland_mu(T_0)

    # thin boundary layer: wall pressure equals edge pressure
    rho_w = p_02 / (R_AIR * T_w)
    mu_w = sutherland_mu(T_w)

    due_dx = (1.0 / R_n) * math.sqrt(2.0 * (p_02 - p_inf) / rho_02)

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
