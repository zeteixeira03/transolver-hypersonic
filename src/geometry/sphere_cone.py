"""Parametric axisymmetric sphere-cone geometry and gmsh meshing.

The body in the (x, r) half-plane, stagnation point at the origin:

    sphere arc (P0 -> P1)  ->  cone segment (P1 -> P2)  ->  shoulder arc (P2 -> P3)

Boundaries:

- WALL:     sphere + cone + shoulder
- SYMMETRY: axis segment from upstream inlet to stagnation point
- FARFIELD: upstream vertical inlet + outboard horizontal top
- OUTLET:   vertical line at the base plane (x = P3.x) from P3 up to the top
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

# ============================================================================================
#                                   analytical key points
# ============================================================================================

def sphere_cone_points(
    R_n: float,
    theta_c_deg: float,
    R_b: float,
    R_s: float,
) -> dict:
    """Key points of a sphere-cone forebody as (x, r) tuples in the units of R_n.

    R_n is the nose radius, R_b the body radius at the shoulder top, R_s the
    shoulder radius. Keys: P0 to P3, ``sphere_center``, ``shoulder_center``.
    """
    if not 0.0 < theta_c_deg < 90.0:
        raise ValueError("theta_c_deg must lie in (0, 90)")
    if R_b <= R_n:
        raise ValueError("R_b must exceed R_n")
    if not 0.0 < R_s < R_b:
        raise ValueError("R_s must satisfy 0 < R_s < R_b")

    theta = math.radians(theta_c_deg)
    sphere_center = (R_n, 0.0)

    P1 = (R_n * (1.0 - math.sin(theta)), R_n * math.cos(theta))

    # shoulder center at r = R_b - R_s, tangent to the cone and the base plane
    P2_r = R_b - R_s * (1.0 - math.cos(theta))
    if P2_r <= P1[1]:
        raise ValueError(
            "geometry degenerate: shoulder would overlap sphere-cone tangent; "
            "increase R_b or decrease R_s/theta_c"
        )
    P2_x = P1[0] + (P2_r - P1[1]) / math.tan(theta)
    P2 = (P2_x, P2_r)

    sh_x = P2[0] + R_s * math.sin(theta)
    shoulder_center = (sh_x, R_b - R_s)
    P3 = (sh_x, R_b)

    return {
        "P0": (0.0, 0.0),
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "sphere_center": sphere_center,
        "shoulder_center": shoulder_center,
    }


def body_outline(R_n: float, theta_c_deg: float, R_b: float, R_s: float,
                 n_arc: int = 48) -> np.ndarray:
    """Closed outline of the body in the (x, r) plane, both halves, shape (4 n_arc, 2)."""
    k = sphere_cone_points(R_n, theta_c_deg, R_b, R_s)
    theta = math.radians(theta_c_deg)
    a = np.linspace(math.pi, math.pi - (math.pi / 2 - theta), n_arc)
    nose = np.c_[R_n + R_n * np.cos(a), R_n * np.sin(a)]
    b = np.linspace(math.pi / 2 + theta, math.pi / 2, n_arc)
    sx, sr = k["shoulder_center"]
    shoulder = np.c_[sx + R_s * np.cos(b), sr + R_s * np.sin(b)]
    upper = np.vstack([nose, shoulder])
    lower = upper[::-1] * np.array([1.0, -1.0])
    return np.vstack([upper, lower])


# ============================================================================================
#                                       gmsh meshing
# ============================================================================================

def mesh_sphere_cone(
    R_n: float,
    theta_c_deg: float,
    R_b: float,
    R_s: float,
    output_path: str | Path,
    L_far: float | None = None,
    h_wall: float | None = None,
    h_far: float | None = None,
    bl_first_height: float | None = None,
    bl_thickness: float | None = None,
    bl_ratio: float = 1.2,
    refine_shock_box: bool = True,
    show_gui: bool = False,
) -> dict:
    """Mesh the fluid domain around a sphere-cone, write .su2, and return its key points.

    ``L_far`` is the upstream and outboard extent. ``h_wall`` sets the
    wall-tangent size and ``bl_first_height`` the wall-normal first cell; its
    default R_n / 5000 holds y+ near 1 for a cold wall at Mach 10.
    """
    pts = sphere_cone_points(R_n, theta_c_deg, R_b, R_s)
    P0, P1, P2, P3 = pts["P0"], pts["P1"], pts["P2"], pts["P3"]
    Cs, Csh = pts["sphere_center"], pts["shoulder_center"]

    if L_far is None:
        L_far = 8.0 * R_b
    if h_wall is None:
        h_wall = R_n / 80.0
    if h_far is None:
        h_far = L_far / 25.0
    if bl_first_height is None:
        bl_first_height = R_n / 5000.0
    if bl_thickness is None:
        bl_thickness = 0.25 * R_n

    x_in = -L_far
    r_far = L_far
    x_out = P3[0]

    import gmsh  # imported lazily; not needed by sphere_cone_points

    # the default interrupt-signal handler can only be installed from the main thread
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sphere_cone")
        geo = gmsh.model.geo

        p_stag = geo.addPoint(P0[0], P0[1], 0.0, h_wall)
        p_sphc = geo.addPoint(Cs[0],  Cs[1],  0.0, h_wall)
        p_tan1 = geo.addPoint(P1[0],  P1[1],  0.0, h_wall)
        p_tan2 = geo.addPoint(P2[0],  P2[1],  0.0, h_wall)
        p_shc  = geo.addPoint(Csh[0], Csh[1], 0.0, h_wall)
        p_top  = geo.addPoint(P3[0],  P3[1],  0.0, h_wall)

        p_in_axis = geo.addPoint(x_in,  0.0,   0.0, h_far)
        p_in_top  = geo.addPoint(x_in,  r_far, 0.0, h_far)
        p_out_top = geo.addPoint(x_out, r_far, 0.0, h_far)

        c_sphere = geo.addCircleArc(p_stag, p_sphc, p_tan1)
        c_cone   = geo.addLine(p_tan1, p_tan2)
        c_shoul  = geo.addCircleArc(p_tan2, p_shc, p_top)

        c_axis = geo.addLine(p_in_axis, p_stag)

        c_inlet = geo.addLine(p_in_axis, p_in_top)
        c_top   = geo.addLine(p_in_top, p_out_top)

        c_outlet = geo.addLine(p_out_top, p_top)

        # counter-clockwise around the fluid region
        loop = geo.addCurveLoop(
            [c_axis, c_sphere, c_cone, c_shoul, -c_outlet, -c_top, -c_inlet]
        )
        surf = geo.addPlaneSurface([loop])

        geo.synchronize()

        tag = gmsh.model.addPhysicalGroup(1, [c_sphere, c_cone, c_shoul])
        gmsh.model.setPhysicalName(1, tag, "WALL")
        tag = gmsh.model.addPhysicalGroup(1, [c_axis])
        gmsh.model.setPhysicalName(1, tag, "SYMMETRY")
        tag = gmsh.model.addPhysicalGroup(1, [c_inlet, c_top])
        gmsh.model.setPhysicalName(1, tag, "FARFIELD")
        tag = gmsh.model.addPhysicalGroup(1, [c_outlet])
        gmsh.model.setPhysicalName(1, tag, "OUTLET")
        tag = gmsh.model.addPhysicalGroup(2, [surf])
        gmsh.model.setPhysicalName(2, tag, "FLUID")

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", [c_sphere, c_cone, c_shoul])
        gmsh.model.mesh.field.setNumber(1, "Sampling", 400)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", h_wall)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", h_far)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5 * R_n)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 4.0 * R_b)

        field_list = [2]

        if refine_shock_box:
            # the bow shock stands about 0.14 R_n upstream of the nose
            gmsh.model.mesh.field.add("Distance", 3)
            gmsh.model.mesh.field.setNumbers(3, "PointsList", [p_stag])
            gmsh.model.mesh.field.add("Threshold", 4)
            gmsh.model.mesh.field.setNumber(4, "InField", 3)
            gmsh.model.mesh.field.setNumber(4, "SizeMin", 2.0 * h_wall)
            gmsh.model.mesh.field.setNumber(4, "SizeMax", h_far)
            gmsh.model.mesh.field.setNumber(4, "DistMin", 1.5 * R_n)
            gmsh.model.mesh.field.setNumber(4, "DistMax", 3.0 * R_b)
            field_list.append(4)

        gmsh.model.mesh.field.add("Min", 10)
        gmsh.model.mesh.field.setNumbers(10, "FieldsList", field_list)
        gmsh.model.mesh.field.setAsBackgroundMesh(10)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        bl = gmsh.model.mesh.field.add("BoundaryLayer")
        gmsh.model.mesh.field.setNumbers(bl, "CurvesList", [c_sphere, c_cone, c_shoul])
        gmsh.model.mesh.field.setNumber(bl, "Size", bl_first_height)
        gmsh.model.mesh.field.setNumber(bl, "Ratio", bl_ratio)
        gmsh.model.mesh.field.setNumber(bl, "Thickness", bl_thickness)
        gmsh.model.mesh.field.setNumber(bl, "Quads", 1)
        gmsh.model.mesh.field.setNumbers(bl, "PointsList", [p_stag, p_top])
        gmsh.model.mesh.field.setAsBoundaryLayer(bl)

        gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay
        gmsh.option.setNumber("Mesh.RecombineAll", 0)

        gmsh.model.mesh.generate(2)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(out))

        if show_gui:
            gmsh.fltk.run()
    finally:
        gmsh.finalize()

    return pts


# ============================================================================================
#                                   dataset mesh recipe
# ============================================================================================

def mesh_settings(R_n: float, R_b: float) -> dict:
    """Mesh settings of the dataset, scaled with the geometry.

    A ratio of 1.2 with a first cell at R_n/15000 misses Fay-Riddell by 127% and
    puts the standoff 27% low. The 5e-7 m floor on the first cell binds below
    R_n = 15 mm; a finer cell reads the heat flux 22 to 26% high on a 12 mm nose.
    A model trained on these meshes must predict on them too, because slice
    attention couples every node to every other.
    """
    return dict(
        L_far=8.0 * R_b,
        h_wall=R_n / 120.0,
        h_far=8.0 * R_b / 25.0,
        bl_first_height=max(R_n / 30000.0, 5e-7),
        bl_thickness=0.06 * R_n,
        bl_ratio=1.15,
        refine_shock_box=True,
    )


def read_su2_nodes(path: str | Path) -> np.ndarray:
    """Read the (x, r) node coordinates from the ``NPOIN`` block of an ASCII .su2 mesh."""
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("NPOIN="):
            n = int(line.split("=")[1].split()[0])
            return np.array([[float(v) for v in lines[i + 1 + j].split()[:2]] for j in range(n)])
    raise ValueError(f"no NPOIN block in {path}")


def mesh_nodes(R_n: float, theta_c_deg: float, R_b: float, R_s: float) -> np.ndarray:
    """Mesh one geometry with the dataset recipe and return its (N, 2) node coordinates."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "case.su2"
        mesh_sphere_cone(R_n, theta_c_deg, R_b, R_s, out, **mesh_settings(R_n, R_b))
        return read_su2_nodes(out)
