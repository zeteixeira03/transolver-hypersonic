"""Tests for the SU2 dataset and post-hoc q_w computation.

The synthetic-NPZ tests cover schema, normalization, lazy loading, and the
reconstruct-pressure helper without depending on any committed sample. The
canonical-NPZ regression check verifies that
:func:`src.eval.sanity.compute_q_w_from_T` reproduces the Phase 2 SU2 stagnation
heat flux to within 5% on the committed ``phase2_canonical.npz``, which is the
load-bearing check for the eval metric used in Phase 4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.su2 import (
    CASE_PARAM_ORDER,
    INPUT_DIM,
    N_CASE_PARAMS,
    R_SPECIFIC_AIR,
    SU2Dataset,
    SU2NormStats,
    TARGET_DIM,
    compute_norm_stats,
    list_case_npzs,
    load_case_npz,
    reconstruct_pressure,
    stack_case_features,
)
from src.eval.sanity import (
    compute_q_w_from_T,
    identify_wall_nodes,
    thermal_conductivity_air,
)


# ============================================================================================
#                                       synthetic case helpers
# ============================================================================================

def _synth_case(tmp_path: Path, name: str, n: int = 500, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    case_dir = tmp_path / name
    case_dir.mkdir()
    path = case_dir / "case.npz"
    # vary case params across seeds so norm stats have a non-degenerate spread
    params = np.array(
        [0.02 + 0.005 * seed, 50.0 + 5.0 * seed, 0.07 + 0.01 * seed,
         0.006 + 0.001 * seed, 8.0 + seed, 220.0 + 10.0 * seed,
         100.0 + 5.0 * seed, 300.0],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        x=rng.standard_normal(n).astype(np.float32) * 0.1,
        r=np.abs(rng.standard_normal(n)).astype(np.float32) * 0.1,
        rho=rng.uniform(1e-3, 1e-1, n).astype(np.float32),
        u=rng.uniform(0, 3000, n).astype(np.float32),
        v=rng.uniform(-100, 1000, n).astype(np.float32),
        T=rng.uniform(220, 5000, n).astype(np.float32),
        case_params=params,
    )
    return path


# ============================================================================================
#                                       schema
# ============================================================================================

def test_input_target_dims_are_consistent():
    assert INPUT_DIM == 2 + N_CASE_PARAMS == 10
    assert TARGET_DIM == 4
    assert len(CASE_PARAM_ORDER) == N_CASE_PARAMS


def test_load_case_npz_rejects_missing_case_params(tmp_path):
    path = tmp_path / "no_params.npz"
    np.savez_compressed(
        path,
        x=np.zeros(3, dtype=np.float32),
        r=np.zeros(3, dtype=np.float32),
        rho=np.ones(3, dtype=np.float32),
        u=np.ones(3, dtype=np.float32),
        v=np.zeros(3, dtype=np.float32),
        T=np.full(3, 300.0, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="missing required keys"):
        load_case_npz(path)


def test_stack_case_features_shapes(tmp_path):
    p = _synth_case(tmp_path, "case_0000", n=128)
    case = load_case_npz(p)
    feats, targets = stack_case_features(case)
    assert feats.shape == (128, INPUT_DIM)
    assert targets.shape == (128, TARGET_DIM)
    # case params broadcast: every node carries the same 8 trailing values
    assert np.allclose(feats[:, 2:], case["case_params"])


# ============================================================================================
#                                       norm stats and dataset
# ============================================================================================

def test_compute_norm_stats_unit_variance_target(tmp_path):
    paths = [_synth_case(tmp_path, f"case_{i:04d}", n=400, seed=i) for i in range(4)]
    stats = compute_norm_stats(paths)
    assert stats.x_mean.shape == (INPUT_DIM,)
    assert stats.x_std.shape == (INPUT_DIM,)
    # build the pooled training points and confirm normalization lands at 0/1
    feats = np.concatenate(
        [stack_case_features(load_case_npz(p))[0] for p in paths], axis=0,
    )
    targets = np.concatenate(
        [stack_case_features(load_case_npz(p))[1] for p in paths], axis=0,
    )
    x_norm = (feats - stats.x_mean.numpy()) / stats.x_std.numpy()
    y_norm = (targets - stats.y_mean.numpy()) / stats.y_std.numpy()
    assert np.allclose(x_norm.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(y_norm.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(y_norm.std(axis=0), 1, atol=1e-3)


def test_su2_dataset_returns_normalized_tensors(tmp_path):
    paths = [_synth_case(tmp_path, f"case_{i:04d}", n=256, seed=i) for i in range(3)]
    stats = compute_norm_stats(paths)
    ds = SU2Dataset(paths, stats, subsample=64)
    item = ds[0]
    assert item["x"].shape == (64, INPUT_DIM)
    assert item["y"].shape == (64, TARGET_DIM)
    assert item["pos"].shape == (64, 2)
    # unnormalized targets recoverable
    assert item["y_raw"].shape == (64, TARGET_DIM)
    assert item["case_params"].shape == (N_CASE_PARAMS,)
    assert item["name"] == "case_0000"


def test_norm_stats_roundtrip(tmp_path):
    paths = [_synth_case(tmp_path, "case_0000", n=128)]
    stats = compute_norm_stats(paths)
    out = tmp_path / "stats.pt"
    stats.save(out)
    loaded = SU2NormStats.load(out)
    assert torch.allclose(stats.x_mean, loaded.x_mean)
    assert torch.allclose(stats.y_std, loaded.y_std)


def test_list_case_npzs_sorted(tmp_path):
    for i in [2, 0, 1]:
        _synth_case(tmp_path, f"case_{i:04d}")
    found = list_case_npzs(tmp_path)
    assert [p.parent.name for p in found] == ["case_0000", "case_0001", "case_0002"]


# ============================================================================================
#                                       pressure reconstruction
# ============================================================================================

def test_reconstruct_pressure_matches_ideal_gas():
    rho = torch.tensor([0.01, 0.05, 0.1])
    T = torch.tensor([300.0, 2000.0, 5000.0])
    p = reconstruct_pressure(rho, T)
    expected = rho * R_SPECIFIC_AIR * T
    assert torch.allclose(p, expected)


# ============================================================================================
#                                       compute_q_w_from_T
# ============================================================================================

def test_thermal_conductivity_at_T_w_value():
    # at T_w = 300 K with Pr=0.71 and Sutherland mu, k ~ 0.026 W/(m K) -- the
    # textbook value for air at room temperature
    k = thermal_conductivity_air(300.0)
    assert 0.024 < k < 0.029


def test_compute_q_w_linear_ramp():
    """On a synthetic linear T(n) ramp from T_w into the fluid, FD recovers
    the slope exactly and q_w = k(T_w) * slope.

    Geometry: a flat wall at ``x = 0`` extending along ``r`` (wall normal
    pointing in the ``-x`` direction). One stagnation wall node at the
    origin with the rest of the wall along ``+r``; fluid nodes at
    ``x = -d, r = 0`` with ``T(d) = T_w + slope * d``.
    """
    T_w = 300.0
    slope_true = 3.0e7                                  # K/m

    # wall nodes: origin + a column along +r (defines the tangent direction
    # for the SVD wall-normal estimate -> tangent is r-axis, normal is x-axis)
    n_wall_others = 6
    x_wall = np.concatenate([[0.0], np.zeros(n_wall_others)])
    r_wall = np.concatenate([[0.0],
                              np.linspace(1e-5, 6e-5, n_wall_others)])

    # fluid nodes along the inward normal (-x) at the stagnation streamline
    n_fluid = 8
    d_in = (1 + np.arange(n_fluid)) * 1e-6
    x_fluid = -d_in
    r_fluid = np.zeros(n_fluid)

    # far fluid nodes the algorithm should ignore (large d)
    x_far = np.array([-0.01, -0.005, -0.02])
    r_far = np.array([0.005, 0.02, 0.01])

    x = np.concatenate([x_wall, x_fluid, x_far]).astype(np.float64)
    r = np.concatenate([r_wall, r_fluid, r_far]).astype(np.float64)
    T = np.empty_like(x)
    n_walls = len(x_wall)
    T[:n_walls] = T_w
    T[n_walls : n_walls + n_fluid] = T_w + slope_true * d_in
    T[n_walls + n_fluid :] = T_w + slope_true * 1e-3

    wall_indices = np.arange(n_walls)
    out = compute_q_w_from_T(
        x=x, r=r, T=T, T_w=T_w, wall_indices=wall_indices,
        y_axis_skip=0.0, n_average=1,
    )
    expected_q = thermal_conductivity_air(T_w) * slope_true
    assert abs(out["q_w"] - expected_q) / expected_q < 0.05


def test_compute_q_w_on_canonical_sample():
    """End-to-end regression: on the committed Phase 2 canonical NPZ, the
    post-hoc q_w computed from the T field must reproduce the SU2 surface-CSV
    stagnation heat flux (1.003 MW/m^2) within 5%. This is the load-bearing
    check for the Phase 4 evaluation pipeline: if the recovery on ground
    truth drifts, every reported surrogate q_w error is suspect.
    """
    p = Path("data/samples/phase2_canonical.npz")
    if not p.exists():
        pytest.skip(f"{p} not present; regenerate via scripts/phase2_validate.py")
    d = np.load(p)
    T_w = float(d["case_params"][CASE_PARAM_ORDER.index("T_w")])
    R_n = float(d["case_params"][CASE_PARAM_ORDER.index("R_n")])
    wall = identify_wall_nodes(d["T"], T_w)
    out = compute_q_w_from_T(
        x=d["x"], r=d["r"], T=d["T"], T_w=T_w,
        wall_indices=wall, y_axis_skip=0.05 * R_n, n_average=3,
    )
    su2_qw_csv = 1.003e6
    assert abs(out["q_w"] - su2_qw_csv) / su2_qw_csv < 0.05
