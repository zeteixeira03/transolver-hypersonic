"""Unit tests for the loop before/after report's parameter geometry."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.loop_report import (
    SCAN_SPANS,
    altitude_from_pressure,
    boot_ci,
    pair_records,
    scan_coords,
)
from src.data.sampler import FS_BOX, GEOM_BOX, us_standard_atmosphere


# ============================================================================================
#                                   parameter geometry
# ============================================================================================

def test_altitude_pressure_round_trip():
    for h in (45.0, 50.0, 55.0, 60.0):
        assert altitude_from_pressure(us_standard_atmosphere(h)[1]) == pytest.approx(h, abs=0.05)


def box_center_case() -> dict[str, float]:
    """A case sitting at the middle of every core-box axis."""
    mid = {k: 0.5 * (lo + hi) for k, (lo, hi) in {**GEOM_BOX, **FS_BOX}.items()}
    R_b = mid["R_b_ratio"] * mid["R_n"]
    T_inf, p_inf, _ = us_standard_atmosphere(mid["altitude"])
    return {
        "R_n": mid["R_n"], "theta_c_deg": mid["theta_c"],
        "R_b": R_b, "R_s": mid["R_s_ratio"] * R_b,
        "mach": mid["mach"], "T_inf": T_inf, "p_inf": p_inf, "T_w": 300.0,
    }


def test_box_center_maps_to_half():
    assert scan_coords(box_center_case()) == pytest.approx(np.full(6, 0.5), abs=1e-3)


def test_ood_case_falls_outside_unit_box():
    # nose_large slab: R_n above the core box, so its first coordinate exceeds
    # 1. R_b and R_s scale with it to hold both aspect ratios fixed.
    case = box_center_case()
    scale = 0.080 / case["R_n"]
    case["R_n"], case["R_b"], case["R_s"] = 0.080, case["R_b"] * scale, case["R_s"] * scale
    coords = scan_coords(case)
    assert coords[0] > 1.0
    assert coords[1:] == pytest.approx(np.full(5, 0.5), abs=1e-3)


def test_scan_axis_order_matches_spans():
    assert tuple(SCAN_SPANS) == ("R_n", "theta_c", "R_b_ratio", "R_s_ratio",
                                 "mach", "altitude")


# ============================================================================================
#                                       pairing
# ============================================================================================

def rec(name: str, split: str, err: float) -> dict:
    return {
        "name": name, "split": split, "case_params": box_center_case(),
        "rL2_ens": {c: err for c in ("rho", "u", "v", "T")},
        "spread": {c: 0.5 * err for c in ("rho", "u", "v", "T")},
    }


def test_pair_records_keeps_shared_cases_only():
    before = [rec("case_0001", "test", 0.10), rec("case_0002", "test", 0.20)]
    after = [rec("case_0001", "test", 0.08), rec("case_0003", "test", 0.30)]
    paired = pair_records(before, after)
    assert [p["name"] for p in paired] == ["case_0001"]
    assert paired[0]["err_before"] == pytest.approx(0.10)
    assert paired[0]["err_after"] == pytest.approx(0.08)


def test_pair_records_rejects_split_disagreement():
    before = [rec("case_0001", "test", 0.10)]
    after = [rec("case_0001", "test_interp", 0.10)]
    with pytest.raises(SystemExit):
        pair_records(before, after)


def test_boot_ci_brackets_the_mean():
    x = np.random.default_rng(0).normal(-0.01, 0.02, 200)
    lo, hi = boot_ci(x)
    assert lo < x.mean() < hi
