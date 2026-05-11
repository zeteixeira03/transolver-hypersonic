"""Unit tests for parametric sphere-cone geometry and gmsh meshing.

The analytical key-point math is checked directly; the gmsh mesh step is a
smoke test that runs only if gmsh is importable. Marker presence in the
written .su2 file is verified by string search.
"""

from __future__ import annotations

import math

import pytest

from src.geometry.sphere_cone import sphere_cone_points


# ============================================================================================
#                                 analytical key points
# ============================================================================================

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_sphere_cone_points_60deg_sphere():
    R_n, theta_c, R_b, R_s = 0.0254, 60.0, 0.0762, 0.00762
    pts = sphere_cone_points(R_n, theta_c, R_b, R_s)

    # P1 lies on the nose sphere
    assert _dist(pts["P1"], pts["sphere_center"]) == pytest.approx(R_n, rel=1e-12)
    # P2 and P3 lie on the shoulder arc
    assert _dist(pts["P2"], pts["shoulder_center"]) == pytest.approx(R_s, rel=1e-12)
    assert _dist(pts["P3"], pts["shoulder_center"]) == pytest.approx(R_s, rel=1e-12)
    # P3 is at body max radius
    assert pts["P3"][1] == pytest.approx(R_b, rel=1e-12)
    # shoulder center is directly below P3 (vertical tangent at top)
    assert pts["shoulder_center"][0] == pytest.approx(pts["P3"][0], rel=1e-12)


def test_sphere_cone_p2_lies_on_cone_line():
    R_n, theta_c, R_b, R_s = 0.02, 45.0, 0.10, 0.01
    pts = sphere_cone_points(R_n, theta_c, R_b, R_s)
    th = math.radians(theta_c)
    # cone line: r - P1.r = tan(theta) * (x - P1.x)
    P1, P2 = pts["P1"], pts["P2"]
    lhs = P2[1] - P1[1]
    rhs = math.tan(th) * (P2[0] - P1[0])
    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)


def test_sphere_cone_points_70deg_apollo_like():
    # Apollo-style 70 / 33 / 0.4564 m geometry, rough
    R_n, theta_c, R_b, R_s = 0.305, 33.0, 0.4566, 0.0457
    pts = sphere_cone_points(R_n, theta_c, R_b, R_s)
    assert pts["P1"][0] > 0
    assert pts["P2"][0] > pts["P1"][0]
    assert pts["P3"][0] > pts["P2"][0]
    assert pts["P3"][1] == pytest.approx(R_b, rel=1e-12)


def test_sphere_cone_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        sphere_cone_points(R_n=0.05, theta_c_deg=0.0, R_b=0.1, R_s=0.01)
    with pytest.raises(ValueError):
        sphere_cone_points(R_n=0.05, theta_c_deg=60.0, R_b=0.04, R_s=0.01)  # R_b < R_n
    with pytest.raises(ValueError):
        sphere_cone_points(R_n=0.05, theta_c_deg=60.0, R_b=0.1, R_s=0.0)
    with pytest.raises(ValueError):
        sphere_cone_points(R_n=0.05, theta_c_deg=60.0, R_b=0.1, R_s=0.2)  # R_s >= R_b


# ============================================================================================
#                                   mesh smoke test
# ============================================================================================

def test_mesh_sphere_cone_writes_su2_with_markers(tmp_path):
    pytest.importorskip("gmsh", reason="gmsh not available on this host")
    from src.geometry.sphere_cone import mesh_sphere_cone

    out = tmp_path / "case.su2"
    pts = mesh_sphere_cone(
        R_n=0.0254,
        theta_c_deg=60.0,
        R_b=0.0762,
        R_s=0.00762,
        output_path=out,
        L_far=0.3,
        h_wall=2e-3,
        h_far=2e-2,
        bl_first_height=1e-5,
        bl_thickness=2e-3,
    )
    assert out.exists()
    assert out.stat().st_size > 1024  # non-trivial file

    text = out.read_text(errors="ignore")
    for marker in ("WALL", "SYMMETRY", "FARFIELD", "OUTLET"):
        assert marker in text, f"missing marker {marker} in .su2 output"

    # geometry dict returned with expected keys
    for k in ("P0", "P1", "P2", "P3", "sphere_center", "shoulder_center"):
        assert k in pts
