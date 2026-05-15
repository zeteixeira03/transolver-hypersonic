"""Unit tests for SU2 postprocess: surface CSV parsing, stagnation pick,
shock-standoff finder. Synthetic data only -- no SU2 run required."""

from __future__ import annotations

import numpy as np
import pytest

from src.cfd.postprocess import (
    extract_surface,
    extract_training_tensors,
    find_shock_standoff,
    stagnation_values,
)


# ============================================================================================
#                                       surface CSV
# ============================================================================================

def _write_surface_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(f"{v}" for v in r) + "\n")


def test_extract_surface_basic(tmp_path):
    p = tmp_path / "surface_flow.csv"
    _write_surface_csv(
        p,
        ["PointID", "x", "y", "z", "Pressure", "Temperature", "Heat_Flux"],
        [
            (0, 0.0,   0.0, 0.0, 12345.0, 4500.0, 2.5e5),
            (1, 0.001, 0.001, 0.0, 12000.0, 4400.0, 2.4e5),
            (2, 0.01,  0.005, 0.0, 8000.0,  3500.0, 1.5e5),
        ],
    )
    out = extract_surface(p)
    assert set(out.keys()) >= {"x", "y", "p", "T", "qw"}
    np.testing.assert_allclose(out["x"], [0.0, 0.001, 0.01])
    np.testing.assert_allclose(out["p"], [12345.0, 12000.0, 8000.0])
    np.testing.assert_allclose(out["qw"], [2.5e5, 2.4e5, 1.5e5])


def test_extract_surface_requires_xy(tmp_path):
    p = tmp_path / "surface_flow.csv"
    _write_surface_csv(p, ["Pressure", "Temperature"], [(1.0, 2.0)])
    with pytest.raises(ValueError):
        extract_surface(p)


def test_extract_surface_quoted_header(tmp_path):
    p = tmp_path / "surface_flow.csv"
    # SU2 sometimes writes quoted column names
    with open(p, "w") as f:
        f.write('"x","y","Pressure","Heat_Flux"\n')
        f.write("0.0,0.0,12345.0,2.5e5\n")
        f.write("0.01,0.005,8000.0,1.5e5\n")
    out = extract_surface(p)
    np.testing.assert_allclose(out["p"], [12345.0, 8000.0])


def test_extract_surface_vtu_roundtrip(tmp_path):
    pv = pytest.importorskip("pyvista", reason="pyvista not available")
    import vtk  # noqa: F401  (pyvista needs vtk available)

    # build an UnstructuredGrid with a couple of line cells (matches the
    # shape of SU2's surface_flow.vtu output)
    pts = np.array([
        [0.0,   0.0,   0.0],
        [0.001, 0.001, 0.0],
        [0.01,  0.005, 0.0],
    ])
    cells = np.array([2, 0, 1, 2, 1, 2])  # two line cells
    celltypes = np.array([3, 3], dtype=np.uint8)  # VTK_LINE = 3
    grid = pv.UnstructuredGrid(cells, celltypes, pts)
    grid.point_data["Pressure"] = np.array([12345.0, 12000.0, 8000.0])
    grid.point_data["Heat_Flux"] = np.array([2.5e5, 2.4e5, 1.5e5])
    grid.point_data["Temperature"] = np.array([4500.0, 4400.0, 3500.0])
    out_path = tmp_path / "surface_flow.vtu"
    grid.save(str(out_path))

    out = extract_surface(out_path)
    assert "p" in out and "qw" in out and "T" in out
    np.testing.assert_allclose(out["x"], pts[:, 0])
    np.testing.assert_allclose(out["p"], [12345.0, 12000.0, 8000.0])
    np.testing.assert_allclose(out["qw"], [2.5e5, 2.4e5, 1.5e5])


# ============================================================================================
#                                  stagnation point pick
# ============================================================================================

def test_stagnation_values_picks_closest_to_origin():
    surface = {
        "x":  np.array([0.10, 0.0,  0.05, 0.001]),
        "y":  np.array([0.05, 0.0,  0.02, 0.001]),
        "p":  np.array([5e3,  1.3e4, 1.1e4, 1.29e4]),
        "qw": np.array([5e4,  3e5,  1.5e5, 2.9e5]),
    }
    stag = stagnation_values(surface)
    assert stag["idx"] == 1
    assert stag["p"] == pytest.approx(1.3e4)
    assert stag["qw"] == pytest.approx(3e5)


def test_stagnation_values_skips_axis_band():
    # axis node has an inflated value; the y_axis_skip band should exclude it
    surface = {
        "x":  np.array([0.0,    0.0001, 0.001, 0.005]),
        "y":  np.array([0.0,    0.001,  0.002, 0.01]),
        "p":  np.array([2.0e4,  1.2e4,  1.18e4, 8.0e3]),  # axis inflated
        "qw": np.array([6.0e6,  1.8e6,  1.5e6,  5.0e5]),
    }
    stag = stagnation_values(surface, y_axis_skip=5e-4, n_average=1)
    assert stag["idx"] == 1
    assert stag["p"] == pytest.approx(1.2e4)
    assert stag["qw"] == pytest.approx(1.8e6)


def test_stagnation_values_averages_when_requested():
    surface = {
        "x":  np.array([0.0001, 0.001, 0.002, 0.005]),
        "y":  np.array([0.001,  0.002, 0.003, 0.01]),
        "p":  np.array([1.2e4,  1.18e4, 1.16e4, 8.0e3]),
        "qw": np.array([1.8e6,  1.5e6,  1.4e6,  5.0e5]),
    }
    stag = stagnation_values(surface, y_axis_skip=5e-4, n_average=3)
    # mean of first 3 nodes (sorted by distance from origin)
    assert stag["p"] == pytest.approx(np.mean([1.2e4, 1.18e4, 1.16e4]))
    assert stag["qw"] == pytest.approx(np.mean([1.8e6, 1.5e6, 1.4e6]))


# ============================================================================================
#                                  shock standoff finder
# ============================================================================================

def _synthetic_axis(x_shock=-0.005, p_inf=100.0, p_post=1.2e4, n=2000, smooth_width=0.0008):
    """Construct an axis-line p(x) profile with a smeared shock at x_shock."""
    x = np.linspace(-0.04, 0.0, n)
    # tanh transition centered at x_shock, narrow width
    p = p_inf + 0.5 * (p_post - p_inf) * (1.0 + np.tanh((x - x_shock) / smooth_width))
    return {"x": x, "p": p}


def test_shock_standoff_midpoint_matches_truth():
    truth = -0.005
    axis = _synthetic_axis(x_shock=truth)
    res = find_shock_standoff(axis, p_inf=100.0)
    # the midpoint pick lands at the center of the smear == truth
    assert res["method"] == "midpoint"
    assert res["x_shock_midpoint"] == pytest.approx(truth, abs=2e-4)
    assert res["x_shock_gradient"] == pytest.approx(truth, abs=1e-4)


def test_shock_standoff_threshold_fallback_matches_truth():
    # threshold pick (first p > 2*p_inf) should land near the leading edge
    truth = -0.005
    axis = _synthetic_axis(x_shock=truth)
    res = find_shock_standoff(axis, p_inf=100.0)
    # threshold pick lands on the leading edge of the smear, which is upstream
    # of the center; tolerance reflects the smear width
    assert res["x_shock_threshold"] == pytest.approx(truth, abs=3e-3)
    assert res["x_shock_threshold"] < truth  # leading edge upstream of center


def test_extract_training_tensors_from_synthetic_vtu(tmp_path):
    pv = pytest.importorskip("pyvista", reason="pyvista not available")
    import vtk  # noqa: F401

    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.001, 0.001, 0.0],
        [0.01, 0.005, 0.0],
        [0.02, 0.01, 0.0],
    ])
    cells = np.array([3, 0, 1, 2, 3, 1, 2, 3])
    celltypes = np.array([5, 5], dtype=np.uint8)  # VTK_TRIANGLE
    grid = pv.UnstructuredGrid(cells, celltypes, pts)
    grid.point_data["Density"] = np.array([0.15, 0.10, 0.05, 0.02])
    grid.point_data["Velocity"] = np.array(
        [[100.0, 10.0, 0.0],
         [500.0, 20.0, 0.0],
         [1500.0, 50.0, 0.0],
         [2500.0, 80.0, 0.0]]
    )
    grid.point_data["Temperature"] = np.array([4000.0, 3000.0, 1500.0, 500.0])
    vtu = tmp_path / "flow.vtu"
    grid.save(str(vtu))

    out = extract_training_tensors(vtu, save_npz=tmp_path / "case.npz")
    assert set(out) == {"x", "r", "rho", "u", "v", "T"}
    np.testing.assert_allclose(out["rho"], [0.15, 0.10, 0.05, 0.02])
    np.testing.assert_allclose(out["u"], [100.0, 500.0, 1500.0, 2500.0])
    np.testing.assert_allclose(out["v"], [10.0, 20.0, 50.0, 80.0])
    np.testing.assert_allclose(out["T"], [4000.0, 3000.0, 1500.0, 500.0])

    loaded = np.load(tmp_path / "case.npz")
    np.testing.assert_allclose(loaded["rho"], out["rho"])


def test_shock_standoff_handles_no_shock():
    # uniform freestream, no shock at all -- all picks should be rejected and
    # standoff returned as NaN rather than a near-endpoint constant
    x = np.linspace(-0.04, 0.0, 1000)
    p = np.full_like(x, 100.0)
    res = find_shock_standoff({"x": x, "p": p}, p_inf=100.0)
    assert np.isnan(res["x_shock_threshold"])
    assert res["method"] == "none"
    assert res["valid"] is False
    assert np.isnan(res["standoff"])


def test_shock_standoff_collapsed_returns_nan():
    """Shock pinned within a few cells of the body endpoint: must be NaN'd.

    This is the Phase 3 sharp-cone stall failure mode -- the second-order
    SU2 solve settles with the bow shock at x ~ -1 grid cell, behind which
    a 6000 K shock layer sits against a 300 K wall. Pre-fix, the threshold
    pick latched onto the second-to-last grid index and the standoff came
    back as a finite ~3e-4 m constant, silently passing along garbage data.
    """
    n = 2000
    x = np.linspace(-0.6, 0.0, n)
    dx = (x[-1] - x[0]) / (n - 1)
    truth = x[-2]  # one cell from the body endpoint
    p = 100.0 + 0.5 * (1.4e4 - 100.0) * (1.0 + np.tanh((x - truth) / (0.5 * dx)))
    res = find_shock_standoff({"x": x, "p": p}, p_inf=100.0, min_cells_from_body=3)
    assert res["method"] == "none"
    assert res["valid"] is False
    assert np.isnan(res["standoff"])
    assert np.isnan(res["x_shock"])


def test_shock_standoff_far_upstream_pick_rejected():
    # if the only "shock-like" feature sits within min_cells of the freestream
    # end of the line, it is also degenerate (no real bow shock found)
    n = 2000
    x = np.linspace(-0.6, 0.0, n)
    p = np.full_like(x, 100.0)
    p[:2] = 1.4e4  # spurious spike at the upstream end
    res = find_shock_standoff({"x": x, "p": p}, p_inf=100.0, min_cells_from_body=3)
    assert res["method"] == "none"
    assert np.isnan(res["standoff"])


def test_shock_standoff_valid_pick_marked_valid():
    # well-resolved shock far from both endpoints stays valid and reports
    # the standoff with the same sign convention as before the NaN guard
    axis = _synthetic_axis(x_shock=-0.005)
    res = find_shock_standoff(axis, p_inf=100.0)
    assert res["valid"] is True
    assert res["method"] == "midpoint"
    assert res["standoff"] == pytest.approx(0.005, abs=3e-4)
