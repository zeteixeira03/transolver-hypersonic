"""Read SU2 output and extract validation quantities.

- :func:`extract_surface`         per-node arrays from ``surface_flow.vtu``
- :func:`extract_axis_line`       the volume solution along the symmetry axis
- :func:`stagnation_values`       pressure, heat flux and temperature at the wall
- :func:`find_shock_standoff`     the bow-shock position along the axis line
- :func:`extract_training_tensors` per-node (x, r, rho, u, v, T) from flow.vtu

SU2 version 8 writes a thin surface .csv file that carries the conserved
variables alone. The surface .vtu file carries the full set of primitives,
including pressure, temperature, heat flux and y+. This project therefore reads
the .vtu file for every surface quantity.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.cfd.runner import Case

# ============================================================================================
#                                       surface read
# ============================================================================================

_SURFACE_FIELD_ALIASES = {
    "p":   ("Pressure",),
    "rho": ("Density",),
    "T":   ("Temperature",),
    "qw":  ("Heat_Flux", "Heatflux"),
    "Cp":  ("Pressure_Coefficient", "Cp"),
    "yp":  ("Y_Plus", "Yplus"),
    "Mach": ("Mach",),
}


def extract_surface(surface_path: str | Path) -> dict[str, np.ndarray]:
    """Parse SU2 surface output and return a dict of per-node arrays.

    The function reads either the surface ``.vtu`` file, which carries the full
    set of primitives, or the surface ``.csv`` file, which carries only the
    columns SU2 chose to write. The file extension selects the reader.
    """
    path = Path(surface_path)
    if path.suffix.lower() == ".vtu":
        return _extract_surface_vtu(path)
    return _extract_surface_csv(path)


def _extract_surface_vtu(path: Path) -> dict[str, np.ndarray]:
    import pyvista as pv  # imported here because only the readers need it

    mesh = pv.read(str(path))
    pts = np.asarray(mesh.points)
    out: dict[str, np.ndarray] = {"x": pts[:, 0], "y": pts[:, 1]}
    for key, aliases in _SURFACE_FIELD_ALIASES.items():
        for a in aliases:
            if a in mesh.point_data:
                arr = np.asarray(mesh.point_data[a])
                if arr.ndim == 1:
                    out[key] = arr
                # this layer takes scalar fields only. read a vector field
                # straight from mesh.point_data when one is needed.
                break
    return out


def _extract_surface_csv(path: Path) -> dict[str, np.ndarray]:
    aliases = {"x": ("x", "Coord_x"), "y": ("y", "Coord_y"), **_SURFACE_FIELD_ALIASES}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    header_clean = [h.strip().strip('"') for h in header]

    def find_col(names: tuple[str, ...]) -> int | None:
        for alias in names:
            for i, h in enumerate(header_clean):
                if h.lower() == alias.lower():
                    return i
        return None

    if not rows:
        raise ValueError(f"empty surface CSV: {path}")
    arr_rows = np.array([[float(v) for v in row] for row in rows])
    out: dict[str, np.ndarray] = {}
    for key, names in aliases.items():
        col = find_col(names)
        if col is not None:
            out[key] = arr_rows[:, col]
    if "x" not in out or "y" not in out:
        raise ValueError(
            f"surface CSV missing x or y coordinate column; got header {header_clean}"
        )
    return out


# ============================================================================================
#                                  stagnation extraction
# ============================================================================================

def stagnation_values(
    surface: dict[str, np.ndarray],
    y_axis_skip: float = 0.0,
    n_average: int = 1,
) -> dict[str, float]:
    """Pick stagnation-region values from the surface arrays.

    On an axisymmetric blunt body the wall node that sits exactly on the
    symmetry axis carries a 1/r singularity from the source term of the
    axisymmetric Navier-Stokes discretization. Pressure and heat flux come out
    inflated at that node, so no analytical correlation matches them. The
    accepted practice skips a small band of body radius above the axis and then
    averages the surviving nodes closest to the stagnation point.

    Parameters
    ----------
    surface : dict
        From :func:`extract_surface`.
    y_axis_skip : float
        Drop wall nodes with ``y < y_axis_skip``. The default of 0.0 drops
        nothing. A sphere-cone wants about 0.05 R_n.
    n_average : int
        Average the surviving wall values over this many nodes, taken closest
        to the geometric stagnation point. The default of 1 takes one node.

    Returns
    -------
    dict
        ``idx`` for the representative node index, then ``x``, ``y``, and every
        scalar primitive that ``surface`` carries.
    """
    x, y = surface["x"], surface["y"]
    mask = y >= y_axis_skip
    if not np.any(mask):
        raise ValueError(f"no wall nodes with y >= {y_axis_skip}")
    cand = np.where(mask)[0]
    dist = np.hypot(x[cand], y[cand])
    order = cand[np.argsort(dist)]
    pick = order[: max(1, n_average)]

    result = {"idx": int(pick[0]), "x": float(x[pick[0]]), "y": float(y[pick[0]])}
    for key in ("p", "T", "rho", "qw", "Cp", "yp", "Mach"):
        if key in surface:
            result[key] = float(np.mean(surface[key][pick]))
    return result


# ============================================================================================
#                                      axis-line sampling
# ============================================================================================

def extract_axis_line(
    volume_vtu_path: str | Path,
    x_in: float,
    x_nose: float = 0.0,
    n: int = 2000,
    r_offset: float = 0.0,
) -> dict[str, np.ndarray]:
    """Sample the volume solution along the symmetry axis r = ``r_offset``.

    Parameters
    ----------
    volume_vtu_path : path
        The SU2 volume output in Paraview format, such as ``flow.vtu``.
    x_in : float
        Upstream end of the sample line. It is negative when the nose sits at
        x = 0.
    x_nose : float
        Downstream end. The default of 0 is the stagnation point of the body.
    n : int
        Number of sample points along the line.
    r_offset : float
        Offset of the line away from the axis. A line at exactly r = 0 crosses
        the singularity of the axisymmetric source term and reads inflated
        values near the body. An offset of a few percent of the nose radius
        clears the singularity and still lies on the stagnation streamline.

    Returns
    -------
    dict
        Keys ``x``, ``p``, ``rho`` and ``T``, plus any other field the file
        carries. Every value is an array of shape (n,) in physical units.
    """
    import pyvista as pv  # imported here because only the readers need it

    mesh = pv.read(str(volume_vtu_path))
    line = pv.Line((x_in, r_offset, 0.0), (x_nose, r_offset, 0.0), resolution=n - 1)
    sampled = line.sample(mesh)
    x = np.asarray(sampled.points[:, 0])

    out: dict[str, np.ndarray] = {"x": x}
    # the Paraview field names differ a little between SU2 versions
    aliases = {
        "p":   ("Pressure", "pressure"),
        "rho": ("Density", "density"),
        "T":   ("Temperature", "temperature"),
        "u":   ("Velocity_x", "velocity_x", "Momentum_x"),
        "v":   ("Velocity_y", "velocity_y", "Momentum_y"),
    }
    for key, names in aliases.items():
        for n_ in names:
            if n_ in sampled.point_data:
                out[key] = np.asarray(sampled.point_data[n_])
                break
    return out


# ============================================================================================
#                                  shock standoff finder
# ============================================================================================

def find_shock_standoff(
    axis: dict[str, np.ndarray],
    p_inf: float,
    *,
    threshold_factor: float = 2.0,
    smooth: int = 5,
    min_cells_from_body: int = 3,
) -> dict:
    """Locate the bow shock along an axis-line sample, return standoff distance.

    The function makes three picks:

    - **midpoint**: the first crossing of p above (p_inf + p_post) / 2, where
      p_post is the median of the pressure plateau between the shock and the
      body.
    - **gradient**: the position of the largest d p / d x along the line.
    - **threshold**: the first node, walking upstream, where p passes
      ``threshold_factor * p_inf``.

    The midpoint pick leads, because a numerically smeared shock moves it least.

    A pick that lands within ``min_cells_from_body`` cells of either end of the
    sample line counts as no pick at all. Two flow states produce that: a field
    with no shock the mesh can resolve between the freestream and the body, and
    a shock that has collapsed onto the wall. Both leave the standoff at NaN, so
    the sanity checks report a NaN relative error rather than a plausible
    constant.

    Parameters
    ----------
    axis : dict
        Output of :func:`extract_axis_line`, must contain ``x`` and ``p``.
    p_inf : float
        Freestream pressure for the threshold and midpoint picks.
    threshold_factor : float
        Multiplier on p_inf for the threshold pick.
    smooth : int
        Width of the boxcar window applied before the gradient.
    min_cells_from_body : int
        Least number of axis-line cells between a valid pick and either end of
        the line. A pick inside that margin becomes NaN. Default 3.

    Returns
    -------
    dict
        ``x_shock``, negative for a shock upstream of the nose. ``standoff``,
        positive, or NaN when no pick is valid. The three picks on their own.
        ``method``, one of "midpoint", "gradient", "threshold" or "none". And
        ``valid`` as a boolean.
    """
    x = np.asarray(axis["x"])
    p = np.asarray(axis["p"])
    if x.ndim != 1 or x.size < 5:
        raise ValueError("axis-line sample too short to locate a shock")

    order = np.argsort(x)
    x, p = x[order], p[order]

    dx = (x[-1] - x[0]) / (len(x) - 1)
    x_lo = x[0] + min_cells_from_body * dx
    x_hi = x[-1] - min_cells_from_body * dx

    def _valid(xs: float) -> bool:
        return not np.isnan(xs) and x_lo <= xs <= x_hi

    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        p_smooth = np.convolve(p, kernel, mode="same")
    else:
        p_smooth = p
    dp_dx = np.gradient(p_smooth, x)
    i_grad = int(np.argmax(dp_dx))
    x_grad = float(x[i_grad])

    # threshold pick: the first node above threshold_factor times p_inf
    mask_thr = p > threshold_factor * p_inf
    x_thr = float(x[np.argmax(mask_thr)]) if np.any(mask_thr) else float("nan")

    # midpoint pick. p_post is the median of the shock-layer plateau, which is
    # everything downstream of the largest gradient
    if 1 < i_grad < len(x) - 2:
        p_post = float(np.median(p_smooth[i_grad:]))
        p_mid = 0.5 * (p_inf + p_post)
        mask_mid = p_smooth > p_mid
        x_mid = float(x[np.argmax(mask_mid)]) if np.any(mask_mid) else float("nan")
    else:
        p_post = float("nan")
        x_mid = float("nan")

    # take the midpoint pick, then the gradient pick, then the threshold pick.
    # each one must sit at least min_cells_from_body cells inside the sample
    # line. a shock collapsed onto the body picks at one end and a shock the
    # mesh never resolved picks at the other, and both return NaN rather than
    # a value that looks physical
    x_shock: float = float("nan")
    method = "none"
    for cand, name in ((x_mid, "midpoint"), (x_grad, "gradient"), (x_thr, "threshold")):
        if _valid(cand):
            x_shock, method = cand, name
            break

    valid = method != "none"
    standoff = -x_shock if valid else float("nan")
    return {
        "x_shock": x_shock,
        "standoff": standoff,
        "x_shock_midpoint": x_mid,
        "x_shock_gradient": x_grad,
        "x_shock_threshold": x_thr,
        "p_post_plateau": p_post,
        "method": method,
        "valid": valid,
    }


# ============================================================================================
#                                training-tensor extraction
# ============================================================================================

def extract_training_tensors(
    volume_vtu_path: str | Path,
    *,
    case: Case | None = None,
    save_npz: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Extract per-mesh-node training arrays from a converged SU2 case.

    Returns the four primitives the model predicts, (rho, u, v, T), with the
    node coordinates (x, r). Pressure is absent on purpose. It is reconstructed
    at inference as ``p = rho * R_specific * T``, which holds the equation of
    state exactly at the output of the model.

    Parameters
    ----------
    volume_vtu_path : path
        Converged SU2 ``flow.vtu`` for one case.
    case : Case, optional
        When given, the eight case parameters are written beside the field
        arrays as ``case_params``, in the channel order of
        :data:`src.data.su2.CASE_PARAM_ORDER`. The .npz file then describes
        itself and training needs no separate manifest.
    save_npz : path, optional
        When given, write the result as a compressed .npz file at this path.

    Returns
    -------
    dict
        Keys ``x``, ``r``, ``rho``, ``u``, ``v`` and ``T``, each a
        one-dimensional array, plus ``case_params`` of shape (8,) when ``case``
        is given.
    """
    import pyvista as pv  # imported here because only the readers need it

    mesh = pv.read(str(volume_vtu_path))
    pts = np.asarray(mesh.points)
    x = pts[:, 0]
    r = pts[:, 1]

    if "Density" not in mesh.point_data:
        raise ValueError(f"flow.vtu at {volume_vtu_path} missing Density")
    rho = np.asarray(mesh.point_data["Density"])

    # velocity comes from the Velocity vector, or from Momentum over Density
    if "Velocity" in mesh.point_data:
        vel = np.asarray(mesh.point_data["Velocity"])
        u, v = vel[:, 0], vel[:, 1]
    elif "Momentum" in mesh.point_data:
        mom = np.asarray(mesh.point_data["Momentum"])
        u, v = mom[:, 0] / rho, mom[:, 1] / rho
    else:
        raise ValueError(f"flow.vtu at {volume_vtu_path} missing Velocity and Momentum")

    if "Temperature" not in mesh.point_data:
        raise ValueError(f"flow.vtu at {volume_vtu_path} missing Temperature")
    T = np.asarray(mesh.point_data["Temperature"])

    out = {"x": x, "r": r, "rho": rho, "u": u, "v": v, "T": T}
    if case is not None:
        out["case_params"] = np.array(
            [case.R_n, case.theta_c_deg, case.R_b, case.R_s,
             case.mach, case.T_inf, case.p_inf, case.T_w],
            dtype=np.float32,
        )
    if save_npz is not None:
        np.savez_compressed(save_npz, **out)
    return out
