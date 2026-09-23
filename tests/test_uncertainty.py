"""Tests for the scoring metric and the trust/warn/refuse rule."""

from __future__ import annotations

import pytest
import torch

from src.data.sampler import KN_MAX
from src.data.su2 import CASE_PARAM_ORDER
from src.eval.metrics import relative_l2, tier_label
from src.eval.uncertainty import (
    GUARD_DIST,
    calibrate_thresholds,
    decide,
    envelope_distance,
    spread,
)


def test_relative_l2_is_per_channel():
    true = torch.tensor([[3.0, 1.0], [4.0, 1.0]])
    pred = true.clone()
    pred[:, 0] *= 1.1
    assert relative_l2(pred, true).tolist() == pytest.approx([0.1, 0.0])


def test_tier_labels():
    assert tier_label("test") == "family holdout"
    assert tier_label("ood_mach_high") == "mach_high (extrapolation)"


def test_spread_vanishes_when_members_agree():
    field = torch.rand(50, 4) + 1.0
    members = field.expand(5, 50, 4)
    assert torch.all(spread(members, field) == 0)


def test_envelope_distance():
    envelope = {k: [0.0, 1.0] for k in CASE_PARAM_ORDER}
    inside = {k: 0.5 for k in CASE_PARAM_ORDER}
    assert envelope_distance(inside, envelope) == 0.0
    outside = dict(inside, mach=1.2)
    assert envelope_distance(outside, envelope) == pytest.approx(0.2)


def test_decision_rule_order():
    t = calibrate_thresholds([0.01 * i for i in range(1, 11)])
    assert t["refuse"] == pytest.approx(4 * t["warn"])
    assert decide(0.0, 0.0, 0.0, t) == "trust"
    assert decide(0.0, 0.0, t["warn"], t) == "warn"
    assert decide(0.0, 0.0, t["refuse"], t) == "refuse"
    assert decide(GUARD_DIST * 1.01, 0.0, 0.0, t) == "refuse"
    assert decide(0.0, KN_MAX * 1.01, 0.0, t) == "refuse"
