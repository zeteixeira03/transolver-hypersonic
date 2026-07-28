"""Unit tests for the sweep sampler and SQLite ledger. No SU2 run required."""

from __future__ import annotations

import numpy as np
import pytest

from src.cfd import ledger as L
from src.cfd.runner import Case
from src.analytical import knudsen_number
from src.data.sampler import (
    KN_MAX,
    CaseSpec,
    FS_BOX,
    GEOM_BOX,
    OOD_SLABS,
    partition,
    sample_cases,
    us_standard_atmosphere,
)


# ============================================================================================
#                                   standard atmosphere
# ============================================================================================

def test_atmosphere_known_points():
    # sea level and the 11 km tropopause from the 1976 standard
    T0, p0, rho0 = us_standard_atmosphere(0.0)
    assert T0 == pytest.approx(288.15, rel=1e-4)
    assert p0 == pytest.approx(101325.0, rel=1e-4)
    assert rho0 == pytest.approx(1.225, rel=1e-3)
    T11, p11, _ = us_standard_atmosphere(11.0)
    assert T11 == pytest.approx(216.65, rel=1e-4)
    assert p11 == pytest.approx(22632.0, rel=1e-3)


def test_atmosphere_pressure_monotone_decreasing():
    h = np.linspace(0.0, 80.0, 50)
    p = np.array([us_standard_atmosphere(x)[1] for x in h])
    assert np.all(np.diff(p) < 0.0)


def test_atmosphere_rejects_out_of_range():
    with pytest.raises(ValueError):
        us_standard_atmosphere(100.0)


# ============================================================================================
#                                       sampler
# ============================================================================================

def test_sample_cases_count_and_determinism():
    a = sample_cases(n_geom=8, n_fs=3, n_geom_ood=2, n_fs_ood=1, seed=1)
    b = sample_cases(n_geom=8, n_fs=3, n_geom_ood=2, n_fs_ood=1, seed=1)
    # the Kn <= KN_MAX filter drops rarefied draws, so the count is bounded
    # above by the nominal DOE size but need not reach it
    nominal = 8 * 3 + len(OOD_SLABS) * 2 * 1
    assert 0 < len(a) <= nominal
    assert len(a) == len(b)
    assert all(isinstance(s, CaseSpec) for s in a)
    for s in a:
        c = s.case
        assert knudsen_number(c.mach, c.T_inf, c.p_inf, c.R_n) <= KN_MAX
    # determinism: same seed -> identical case parameters
    for sa, sb in zip(a, b):
        assert sa.group == sb.group and sa.geom_id == sb.geom_id
        assert sa.case == sb.case
    # different seed -> different sample
    c = sample_cases(n_geom=8, n_fs=3, n_geom_ood=2, n_fs_ood=1, seed=2)
    assert any(sa.case != sc.case for sa, sc in zip(a, c))


def test_sample_cases_respects_core_bounds():
    specs = sample_cases(n_geom=12, n_fs=4, n_geom_ood=0, n_fs_ood=0, seed=3)
    for s in specs:
        c = s.case
        assert GEOM_BOX["R_n"][0] <= c.R_n <= GEOM_BOX["R_n"][1]
        assert GEOM_BOX["theta_c"][0] <= c.theta_c_deg <= GEOM_BOX["theta_c"][1]
        assert GEOM_BOX["R_b_ratio"][0] <= c.R_b / c.R_n <= GEOM_BOX["R_b_ratio"][1]
        assert GEOM_BOX["R_s_ratio"][0] <= c.R_s / c.R_b <= GEOM_BOX["R_s_ratio"][1]
        assert FS_BOX["mach"][0] <= c.mach <= FS_BOX["mach"][1]
        assert c.T_w == 300.0
        assert c.R_b > c.R_n and 0.0 < c.R_s < c.R_b


def test_sample_cases_geometry_clusters_are_contiguous():
    specs = sample_cases(n_geom=6, n_fs=5, n_geom_ood=0, n_fs_ood=0, seed=4)
    # every geom_id appears in one unbroken run
    seen_runs = []
    last = object()
    for s in specs:
        if s.geom_id != last:
            assert s.geom_id not in [g for g, _ in seen_runs], "geom cluster split"
            seen_runs.append((s.geom_id, 1))
            last = s.geom_id
    # and within a cluster the geometry is identical
    by_geom: dict[int, list[Case]] = {}
    for s in specs:
        by_geom.setdefault(s.geom_id, []).append(s.case)
    for cases in by_geom.values():
        first = cases[0]
        for c in cases:
            assert (c.R_n, c.theta_c_deg, c.R_b, c.R_s) == \
                   (first.R_n, first.theta_c_deg, first.R_b, first.R_s)


def test_ood_slabs_land_outside_core():
    specs = sample_cases(n_geom=4, n_fs=2, n_geom_ood=5, n_fs_ood=2, seed=5)
    big = [s.case for s in specs if s.group == "nose_large"]
    assert big and all(c.R_n > GEOM_BOX["R_n"][1] for c in big)
    fast = [s.case for s in specs if s.group == "mach_high"]
    assert fast and all(c.mach >= FS_BOX["mach"][1] for c in fast)


# ============================================================================================
#                                       partition
# ============================================================================================

def test_partition_is_a_partition():
    blocks = partition(17, 4)
    assert len(blocks) == 4
    flat = [i for b in blocks for i in b]
    assert flat == list(range(17))
    sizes = [len(b) for b in blocks]
    assert max(sizes) - min(sizes) <= 1


def test_partition_rejects_zero_blocks():
    with pytest.raises(ValueError):
        partition(10, 0)


# ============================================================================================
#                                       ledger
# ============================================================================================

def _tiny_ledger(tmp_path, n_geom=3, n_fs=4, n_blocks=2):
    specs = sample_cases(n_geom=n_geom, n_fs=n_fs, n_geom_ood=0, n_fs_ood=0, seed=0)
    db = tmp_path / "ledger.db"
    L.init_ledger(db, specs, n_blocks=n_blocks)
    return db, specs


def test_init_ledger_rows_and_restart_links(tmp_path):
    db, specs = _tiny_ledger(tmp_path)
    assert L.status_counts(db) == {"pending": len(specs)}
    con = L.connect(db)
    try:
        rows = con.execute("SELECT * FROM cases ORDER BY ord").fetchall()
    finally:
        con.close()
    assert [r["case_id"] for r in rows] == list(range(len(specs)))
    for r in rows:
        rf = r["restart_from"]
        if rf is None:
            continue
        prev = rows[rf]
        # a restart link only points at the immediate predecessor, same geom, same block
        assert rf == r["ord"] - 1
        assert prev["geom_id"] == r["geom_id"]
        assert prev["block"] == r["block"]
    # the first case of each block has no restart link
    first_of_block = {}
    for r in rows:
        first_of_block.setdefault(r["block"], r)
    for r in first_of_block.values():
        assert r["restart_from"] is None


def test_init_ledger_is_idempotent(tmp_path):
    db, specs = _tiny_ledger(tmp_path)
    # claim and finish one case, then re-init: the finished row must survive
    cid, case, _ = L.claim_next(db, "worker_a")
    L.mark_done(db, cid, qw=1.0, p02=2.0, standoff=3e-3, checks_passed=True)
    L.init_ledger(db, specs, n_blocks=2)
    counts = L.status_counts(db)
    assert counts.get("done") == 1
    assert counts.get("pending") == len(specs) - 1


def test_claim_mark_and_recover(tmp_path):
    db, specs = _tiny_ledger(tmp_path)
    cid1, case1, rf1 = L.claim_next(db, "worker_a")
    assert cid1 == 0 and rf1 is None and isinstance(case1, Case)
    cid2, _, _ = L.claim_next(db, "worker_b")
    assert cid2 == 1  # next by ord
    L.mark_done(db, cid1, qw=1.0, p02=2.0, standoff=3e-3, checks_passed=True)
    L.mark_failed(db, cid2, "boom")
    counts = L.status_counts(db)
    assert counts["done"] == 1 and counts["failed"] == 1
    # recent_done returns the finished one with its quantities
    rd = L.recent_done(db, limit=10)
    assert len(rd) == 1 and rd[0]["case_id"] == cid1 and rd[0]["qw"] == 1.0


def test_reclaim_stale(tmp_path):
    db, specs = _tiny_ledger(tmp_path, n_geom=1, n_fs=1, n_blocks=1)
    assert len(specs) == 1
    cid, _, _ = L.claim_next(db, "worker_a")
    # nothing is stale with a long lease
    assert L.reclaim_stale(db, max_age_s=1e9) == 0
    # a zero lease reclaims the running row back to pending (attempts 1 < MAX_ATTEMPTS)
    assert L.reclaim_stale(db, max_age_s=0.0) == 1
    assert L.status_counts(db) == {"pending": 1}
    # burn the rest of the budget; every claim costs one attempt
    for _ in range(L.MAX_ATTEMPTS - 2):
        L.claim_next(db, "worker_a")
        assert L.reclaim_stale(db, max_age_s=0.0) == 1
        assert L.status_counts(db) == {"pending": 1}
    # the last claim exhausts it, so expiring the lease marks the row failed
    L.claim_next(db, "worker_a")
    assert L.reclaim_stale(db, max_age_s=0.0) == 1
    con = L.connect(db)
    try:
        row = con.execute("SELECT * FROM cases WHERE case_id=?", (cid,)).fetchone()
    finally:
        con.close()
    assert row["status"] == "failed"
    assert row["attempts"] == L.MAX_ATTEMPTS
