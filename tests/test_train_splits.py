"""Tests for train.py split_paths, especially loop-group routing.

The active-learning loop appends new cases with ``group_name='loop'`` to the
dataset. The core val/test/interp membership must stay byte-identical for the
same seed regardless of how many loop cases are present, so before/after
comparisons across loop iterations are legitimate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train import split_paths


def _fake_paths(case_ids: list[int]) -> list[Path]:
    return [Path(f"/tmp/case_{cid:04d}/case.npz") for cid in case_ids]


def test_loop_cases_route_to_train_only():
    core_ids = list(range(100))
    loop_ids = [900, 901, 902, 903, 904]
    paths = _fake_paths(core_ids + loop_ids)
    groups = {**{c: "core" for c in core_ids}, **{c: "loop" for c in loop_ids}}
    geoms = {c: c // 10 for c in core_ids}  # 10 core geometries
    splits = split_paths(paths, groups, geoms, val_frac=0.2, test_frac=0.2,
                         seed=0, interp_frac=0.1)
    loop_names = {f"case_{c:04d}" for c in loop_ids}

    train_names = {p.parent.name for p in splits["train"]}
    for tier in ("val", "test", "test_interp"):
        tier_names = {p.parent.name for p in splits[tier]}
        assert not (tier_names & loop_names), (
            f"loop cases leaked into {tier}: {tier_names & loop_names}"
        )
    assert loop_names <= train_names, "loop cases missing from train"


def test_core_splits_identical_with_and_without_loop_cases():
    core_ids = list(range(100))
    loop_ids = [900, 901, 902, 903, 904]
    geoms = {c: c // 10 for c in core_ids}

    # baseline: no loop cases
    paths_a = _fake_paths(core_ids)
    groups_a = {c: "core" for c in core_ids}
    splits_a = split_paths(paths_a, groups_a, geoms, 0.2, 0.2, seed=0, interp_frac=0.1)

    # with loop cases
    paths_b = _fake_paths(core_ids + loop_ids)
    groups_b = {**groups_a, **{c: "loop" for c in loop_ids}}
    splits_b = split_paths(paths_b, groups_b, geoms, 0.2, 0.2, seed=0, interp_frac=0.1)

    for tier in ("val", "test", "test_interp"):
        a = [p.parent.name for p in splits_a[tier]]
        b = [p.parent.name for p in splits_b[tier]]
        assert a == b, f"{tier} membership drifted when loop cases were added"

    core_train_a = [p.parent.name for p in splits_a["train"]]
    core_train_b_no_loop = [p.parent.name for p in splits_b["train"]
                            if int(p.parent.name.split("_")[1]) < 900]
    assert core_train_a == core_train_b_no_loop, (
        "core train ordering shifted when loop cases were added"
    )


def test_loop_cases_deterministic_order_in_train():
    core_ids = list(range(30))
    loop_ids = [905, 902, 908, 900]  # unsorted on purpose
    geoms = {c: c // 10 for c in core_ids}
    paths = _fake_paths(core_ids + loop_ids)
    groups = {**{c: "core" for c in core_ids}, **{c: "loop" for c in loop_ids}}
    splits = split_paths(paths, groups, geoms, 0.2, 0.2, seed=0, interp_frac=0.0)
    train_names = [p.parent.name for p in splits["train"]]
    loop_positions = [train_names.index(f"case_{c:04d}") for c in sorted(loop_ids)]
    assert loop_positions == sorted(loop_positions), "loop cases not in case_id order"
    # loop cases occupy the tail of the train list, contiguous
    tail = train_names[-len(loop_ids):]
    assert tail == [f"case_{c:04d}" for c in sorted(loop_ids)], (
        f"loop cases should be at the tail of train in case_id order; got tail={tail}"
    )


def test_ood_groups_untouched_by_loop():
    core_ids = list(range(30))
    loop_ids = [900, 901]
    ood_ids = [800, 801, 802]
    geoms = {c: c // 10 for c in core_ids}
    paths = _fake_paths(core_ids + ood_ids + loop_ids)
    groups = {
        **{c: "core" for c in core_ids},
        **{c: "loop" for c in loop_ids},
        **{c: "nose_large" for c in ood_ids},
    }
    splits = split_paths(paths, groups, geoms, 0.2, 0.2, seed=0, interp_frac=0.0)
    ood_names = {p.parent.name for p in splits["ood_nose_large"]}
    assert ood_names == {"case_0800", "case_0801", "case_0802"}
