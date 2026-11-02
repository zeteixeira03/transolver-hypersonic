"""Read SU2 output and extract validation quantities and training tensors.

SU2 version 8 writes only conserved variables to the surface .csv file, so
surface quantities come from the surface .vtu file.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.cfd.runner import Case
from src.eval.sanity import compare_to_analytical

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
    """Parse a surface ``.vtu`` or ``.csv`` file into per-node arrays."""
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
    """Average the ``n_average`` wall nodes nearest the stagnation point.

    Wall nodes with ``y < y_axis_skip`` are dropped first: the axisymmetric
    source term carries a 1/r singularity that inflates pressure and heat flux
    on the axis. A sphere-cone wants about 0.05 R_n.
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
    """Sample the volume solution on the line r = ``r_offset`` from ``x_in`` to ``x_nose``.

    A line at exactly r = 0 reads values inflated by the axis singularity. An
    offset of a few percent of the nose radius clears it and stays on the
    stagnation streamline. Returns arrays of shape (n,).
    """
    import pyvista as pv

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
    """Locate the bow shock on an axis-line sample and return the standoff distance.

    Three picks, tried in order:

    - midpoint: first crossing of (p_inf + p_post) / 2, with p_post the median
      of the plateau behind the shock. A smeared shock moves it least.
    - gradient: the largest dp/dx, after a boxcar of width ``smooth``.
    - threshold: the first node above ``threshold_factor * p_inf``.

    A pick within ``min_cells_from_body`` cells of either end of the line is
    rejected. That catches a shock collapsed onto the wall and a shock the mesh
    never resolved, and leaves ``standoff`` at NaN instead of a plausible
    constant. ``x_shock`` is negative upstream of the nose.
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

    mask_thr = p > threshold_factor * p_inf
    x_thr = float(x[np.argmax(mask_thr)]) if np.any(mask_thr) else float("nan")

    # the plateau is everything downstream of the largest gradient
    if 1 < i_grad < len(x) - 2:
        p_post = float(np.median(p_smooth[i_grad:]))
        p_mid = 0.5 * (p_inf + p_post)
        mask_mid = p_smooth > p_mid
        x_mid = float(x[np.argmax(mask_mid)]) if np.any(mask_mid) else float("nan")
    else:
        p_post = float("nan")
        x_mid = float("nan")

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
    """Extract per-node ``x``, ``r``, ``rho``, ``u``, ``v``, ``T`` from a converged ``flow.vtu``.

    With ``case``, also stores ``case_params`` of shape (8,) in the order of
    :data:`src.data.su2.CASE_PARAM_ORDER`, so the .npz file needs no manifest.
    """
    import pyvista as pv

    mesh = pv.read(str(volume_vtu_path))
    pts = np.asarray(mesh.points)
    x = pts[:, 0]
    r = pts[:, 1]

    if "Density" not in mesh.point_data:
        raise ValueError(f"flow.vtu at {volume_vtu_path} missing Density")
    rho = np.asarray(mesh.point_data["Density"])

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


# ============================================================================================
#                                       one solved case
# ============================================================================================

def measure_case(run_dir: str | Path, case: Case) -> dict:
    """Measure stagnation heat flux, stagnation pressure and shock standoff of a solved case.

    Values are read at 5% of the nose radius off the axis, clear of the axis
    singularity. Raises when the stagnation line has no bow shock. Returns
    ``qw``, ``p02`` and ``standoff`` in SI units, plus the analytical
    comparison as ``summary``.
    """
    run_dir = Path(run_dir)
    stag = stagnation_values(extract_surface(run_dir / "surface_flow.vtu"),
                             y_axis_skip=0.05 * case.R_n, n_average=3)
    axis = extract_axis_line(run_dir / "flow.vtu", x_in=-8.0 * case.R_b, x_nose=0.0,
                             n=2000, r_offset=0.05 * case.R_n)
    shock = find_shock_standoff(axis, p_inf=case.p_inf)
    if not shock.get("valid", not np.isnan(shock["standoff"])):
        raise RuntimeError("no bow shock on the stagnation line; the field is broken")
    qw = float(abs(stag.get("qw", np.nan)))
    p02 = shock.get("p_post_plateau", np.nan)
    p02 = float(stag.get("p", np.nan) if np.isnan(p02) else p02)
    delta = float(shock["standoff"])
    summary = compare_to_analytical(
        M_inf=case.mach, T_inf=case.T_inf, p_inf=case.p_inf, R_n=case.R_n, T_w=case.T_w,
        su2_qw=qw, su2_p02=p02, su2_standoff=delta, theta_c_deg=case.theta_c_deg,
    )
    return {"qw": qw, "p02": p02, "standoff": delta, "summary": summary}
