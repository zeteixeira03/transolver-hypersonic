"""Read SU2 output and extract validation quantities.

- :func:`extract_surface`         read ``surface_flow.vtu`` (or .csv), per-node arrays.
- :func:`extract_axis_line`       sample the volume VTU along the symmetry axis.
- :func:`stagnation_values`       stagnation-point pressure, heat flux, wall T.
- :func:`find_shock_standoff`     bow-shock location along the axis line.
- :func:`extract_training_tensors` per-node (x, r, rho, u, v, T) arrays from flow.vtu.

SU2 v8 writes minimal surface CSV (conservative variables only). The surface
VTU carries the full primitive set (pressure, temperature, heat flux, y+),
so VTU is the canonical surface output for this project.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


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

    Accepts the surface ``.vtu`` (preferred, has full primitive set) or the
    surface ``.csv`` (limited to whatever columns SU2 chose to write). The
    function dispatches on file extension.
    """
    path = Path(surface_path)
    if path.suffix.lower() == ".vtu":
        return _extract_surface_vtu(path)
    return _extract_surface_csv(path)


def _extract_surface_vtu(path: Path) -> dict[str, np.ndarray]:
    import pyvista as pv  # lazy

    mesh = pv.read(str(path))
    pts = np.asarray(mesh.points)
    out: dict[str, np.ndarray] = {"x": pts[:, 0], "y": pts[:, 1]}
    for key, aliases in _SURFACE_FIELD_ALIASES.items():
        for a in aliases:
            if a in mesh.point_data:
                arr = np.asarray(mesh.point_data[a])
                if arr.ndim == 1:
                    out[key] = arr
                # vector fields are skipped at this layer; pull explicitly if needed
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

    For axisymmetric blunt bodies, the wall node exactly on the symmetry axis
    (y == 0) carries a 1/r source-term singularity in the axisymmetric NS
    discretization. The output pressure and heat flux at that node are
    inflated and unsuitable for direct comparison with analytical
    correlations. Skipping the first ``y_axis_skip`` of body radius and
    averaging over the ``n_average`` smallest-arc wall nodes that survive is
    the standard practice.

    Parameters
    ----------
    surface : dict
        From :func:`extract_surface`.
    y_axis_skip : float
        Skip wall nodes with ``y < y_axis_skip``. Default 0.0 (no skip,
        legacy behavior). For sphere-cones a typical value is 0.05 R_n.
    n_average : int
        Average the surviving wall variables across the ``n_average`` nodes
        closest to the geometric stagnation point. Default 1 (single node).

    Returns
    -------
    dict
        ``idx`` (representative node index), ``x``, ``y``, and any scalar
        primitives present in ``surface``.
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
        SU2 volume Paraview output, e.g. ``flow.vtu``.
    x_in : float
        Upstream end of the sample line (negative if nose is at x=0).
    x_nose : float
        Downstream end. Defaults to 0 (body stagnation).
    n : int
        Number of sample points along the line.
    r_offset : float
        Small r-offset for the sampling line. Exact axis sampling (r=0) hits
        the axisymmetric source-term singularity and returns inflated values
        near the body; a small offset of a few percent of the nose radius
        avoids this while still being on the stagnation streamline.

    Returns
    -------
    dict
        Keys: ``x``, ``p``, ``rho``, ``T`` (others when present). Values are
        (n,) numpy arrays in physical units.
    """
    import pyvista as pv  # lazy: pyvista is only needed at run time

    mesh = pv.read(str(volume_vtu_path))
    line = pv.Line((x_in, r_offset, 0.0), (x_nose, r_offset, 0.0), resolution=n - 1)
    sampled = line.sample(mesh)
    x = np.asarray(sampled.points[:, 0])

    out: dict[str, np.ndarray] = {"x": x}
    # SU2 paraview field names vary slightly across versions
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

    Three picks are computed:
    - **midpoint**: first crossing of p > (p_inf + p_post) / 2, where p_post
      is the median of the high-p plateau between the shock and the body.
      This is the most robust definition for a smeared shock.
    - **gradient**: argmax of d p / d x along the line.
    - **threshold**: first upstream-walking node where p > threshold_factor * p_inf.

    The midpoint pick is reported as primary because it is least sensitive
    to numerical shock smearing.

    Degenerate detections (a pick within ``min_cells_from_body`` grid cells of
    either endpoint of the sample line) are treated as missing: the field
    either has no resolvable shock between freestream and the body, or the
    shock has collapsed onto the wall and the picks have latched onto a few
    cells from the line endpoint. The returned standoff is NaN in that case
    and downstream sanity checks see NaN rel error, not a silent constant.

    Parameters
    ----------
    axis : dict
        Output of :func:`extract_axis_line`, must contain ``x`` and ``p``.
    p_inf : float
        Freestream pressure for the threshold and midpoint picks.
    threshold_factor : float
        Multiplier on p_inf for the leading-edge threshold pick.
    smooth : int
        Boxcar window for gradient smoothing.
    min_cells_from_body : int
        Minimum number of axis-line cells between a valid pick and either
        endpoint. Picks within this margin are NaN'd. Default 3.

    Returns
    -------
    dict
        ``x_shock`` (negative for an upstream shock), ``standoff`` (positive;
        NaN if no valid pick), per-method picks, ``method`` ("midpoint",
        "gradient", "threshold", or "none"), and ``valid`` (bool).
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

    # threshold leading-edge pick
    mask_thr = p > threshold_factor * p_inf
    x_thr = float(x[np.argmax(mask_thr)]) if np.any(mask_thr) else float("nan")

    # midpoint pick: use the median of the shock-layer plateau as p_post
    # (everything downstream of the gradient maximum)
    if 1 < i_grad < len(x) - 2:
        p_post = float(np.median(p_smooth[i_grad:]))
        p_mid = 0.5 * (p_inf + p_post)
        mask_mid = p_smooth > p_mid
        x_mid = float(x[np.argmax(mask_mid)]) if np.any(mask_mid) else float("nan")
    else:
        p_post = float("nan")
        x_mid = float("nan")

    # midpoint preferred; fall back to gradient, then threshold. each pick
    # must lie at least min_cells_from_body cells inside the sample line, so
    # a collapsed shock (pick at the body end) or a missed shock (pick at
    # the freestream end) returns NaN instead of a near-endpoint artifact
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
    save_npz: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Extract per-mesh-node training arrays from a converged SU2 case.

    Returns the 4 primitives Transolver will be trained to predict
    ((rho, u, v, T)), plus node coordinates (x, r). Pressure is omitted; it
    is reconstructed at inference time as ``p = rho * R_specific * T``, by
    project convention (CLAUDE.md, Phase 1 / Phase 4 output convention).

    Parameters
    ----------
    volume_vtu_path : path
        Converged SU2 ``flow.vtu`` for one case.
    save_npz : path, optional
        If provided, save the dict as a compressed .npz alongside the VTU.

    Returns
    -------
    dict
        Keys ``x``, ``r``, ``rho``, ``u``, ``v``, ``T``; all 1-D arrays.
    """
    import pyvista as pv  # lazy

    mesh = pv.read(str(volume_vtu_path))
    pts = np.asarray(mesh.points)
    x = pts[:, 0]
    r = pts[:, 1]

    if "Density" not in mesh.point_data:
        raise ValueError(f"flow.vtu at {volume_vtu_path} missing Density")
    rho = np.asarray(mesh.point_data["Density"])

    # velocity from Velocity vector (preferred) or Momentum / Density
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
    if save_npz is not None:
        np.savez_compressed(save_npz, **out)
    return out
