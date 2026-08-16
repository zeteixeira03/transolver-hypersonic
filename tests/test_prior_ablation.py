"""Tests for the physics-prior ablation: q_w encoding, the head, and the table.

The load-bearing checks here are the two that would silently corrupt a
results row rather than crash: the q_w residual encoding must invert exactly
(a broken decode looks like a plausible heat-flux number), and the table must
score each q_w estimate against its own truth, since the finite-difference
and ledger values differ by more than an order of magnitude on this dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prior_ablation import (
    COMMON_CHANNELS,
    parse_run_dir,
    qw_abs_rel_median,
    render_table,
    tier_mean_rl2,
)
from scripts.train import QwSpec, collect_env, preflight_device, train_one_epoch
from src.data.su2 import N_CASE_PARAMS, POS_DIM
from src.models.transolver import Transolver


# ============================================================================================
#                                       q_w encoding
# ============================================================================================

def test_direct_encoding_round_trips():
    spec = QwSpec("direct", mean=6.5, std=0.4)
    qw, q_fr = 4.2e6, 3.9e6
    assert spec.decode(spec.encode(qw, q_fr), q_fr) == pytest.approx(qw, rel=1e-9)


def test_residual_encoding_round_trips():
    spec = QwSpec("residual", mean=0.03, std=0.02)
    qw, q_fr = 4.2e6, 3.9e6
    assert spec.decode(spec.encode(qw, q_fr), q_fr) == pytest.approx(qw, rel=1e-9)


def test_residual_target_is_the_log_ratio():
    """The residual arm must regress the departure from Fay-Riddell, not q_w."""
    spec = QwSpec("residual", mean=0.0, std=1.0)
    assert spec.log_target(2.0e6, 1.0e6) == pytest.approx(np.log10(2.0))
    # a case sitting exactly on the correlation has a zero target
    assert spec.log_target(1.0e6, 1.0e6) == pytest.approx(0.0)


def test_residual_head_at_zero_returns_the_analytical_value():
    """An untrained residual head should fall back near Fay-Riddell, which is
    the whole point of the prior."""
    spec = QwSpec("residual", mean=0.0, std=0.02)
    assert spec.decode(0.0, 3.9e6) == pytest.approx(3.9e6)


def test_direct_and_residual_differ_in_scale():
    """Direct targets live near log10 q_w; residual targets near zero."""
    qw, q_fr = 4.2e6, 3.9e6
    assert QwSpec("direct", 0.0, 1.0).log_target(qw, q_fr) > 6.0
    assert abs(QwSpec("residual", 0.0, 1.0).log_target(qw, q_fr)) < 0.1


# ============================================================================================
#                                       model head
# ============================================================================================

def _model(qw_head: bool) -> Transolver:
    return Transolver(
        space_dim=POS_DIM, fun_dim=N_CASE_PARAMS, out_dim=4,
        n_hidden=16, n_layers=2, n_head=4, slice_num=4, qw_head=qw_head,
    )


def test_default_model_returns_only_point_predictions():
    x = torch.randn(1, 64, POS_DIM + N_CASE_PARAMS)
    out = _model(qw_head=False)(x, pos=x[..., :2])
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 64, 4)


def test_qw_head_returns_point_and_scalar():
    x = torch.randn(1, 64, POS_DIM + N_CASE_PARAMS)
    out, qw = _model(qw_head=True)(x, pos=x[..., :2])
    assert out.shape == (1, 64, 4)
    assert qw.shape == (1,)


def test_qw_head_is_insensitive_to_node_count():
    """Mean pooling must give a comparable scalar on a subsample and on the
    full mesh, since training subsamples and evaluation does not."""
    torch.manual_seed(0)
    model = _model(qw_head=True).eval()
    x = torch.randn(1, 4096, POS_DIM + N_CASE_PARAMS)
    with torch.no_grad():
        _, full = model(x, pos=x[..., :2])
        _, sub = model(x[:, :1024], pos=x[:, :1024, :2])
    assert float(sub) == pytest.approx(float(full), abs=0.2)


# ============================================================================================
#                                       run capture
# ============================================================================================

# Training happens on rented GPUs whose output is destroyed when the notebook
# is re-pushed, so anything these assertions let slip is only recoverable by
# spending quota again.

def _fake_batch(n_nodes: int = 32, qw: float = 0.5) -> dict:
    x = torch.randn(1, n_nodes, POS_DIM + N_CASE_PARAMS)
    return {"x": x, "y": torch.randn(1, n_nodes, 4), "pos": x[..., :2],
            "qw": torch.tensor([qw]), "name": "case_0000"}


@pytest.mark.parametrize("qw_weight", [0.0, 0.1])
def test_epoch_record_carries_every_cheap_scalar(qw_weight):
    model = _model(qw_head=qw_weight > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rec = train_one_epoch(model, [_fake_batch(), _fake_batch()], opt, "cpu", 1.0,
                          qw_weight=qw_weight)
    for key in ("loss", "field_loss", "qw_loss", "grad_norm_mean", "grad_norm_max",
                "n_steps", "n_qw_steps", "epoch_time_s"):
        assert key in rec, f"epoch record dropped {key}"
    assert rec["n_steps"] == 2
    assert rec["grad_norm_max"] >= rec["grad_norm_mean"] > 0


def test_grad_norm_is_measured_even_without_clipping():
    """grad_clip <= 0 must still report the norm, not silently zero it."""
    model = _model(qw_head=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rec = train_one_epoch(model, [_fake_batch()], opt, "cpu", 0.0)
    assert rec["grad_norm_mean"] > 0


def test_preflight_is_a_noop_on_cpu():
    preflight_device("cpu")


def test_preflight_rejects_an_unsupported_capability(monkeypatch):
    """A card the torch build has no kernels for must abort before staging."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (6, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "Tesla P100-PCIE-16GB")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_70", "sm_75", "sm_80"])
    with pytest.raises(SystemExit, match="sm_60"):
        preflight_device("cuda")


def test_preflight_accepts_a_supported_capability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "Tesla T4")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_70", "sm_75", "sm_80"])
    preflight_device("cuda")


def test_collect_env_records_code_version():
    env = collect_env("cpu")
    for key in ("python", "torch", "device", "cuda_available", "git_commit"):
        assert key in env
    # a run traced to a commit is reproducible; one that is not, is not
    assert env["git_commit"] is None or len(env["git_commit"]) == 40


# ============================================================================================
#                                       table aggregation
# ============================================================================================

def test_parse_run_dir_splits_cell_and_seed():
    assert parse_run_dir(Path("run_base_s0")) == ("base", 0)
    assert parse_run_dir(Path("run_qw_resid_s2")) == ("qw_resid", 2)
    assert parse_run_dir(Path("run_plain_norm_s11")) == ("plain_norm", 11)


def test_parse_run_dir_rejects_unparseable_names():
    with pytest.raises(ValueError, match="cannot parse"):
        parse_run_dir(Path("ensemble_v3"))


def test_tier_mean_rl2_averages_common_channels_only():
    recs = [{"rL2": {"rho": 0.1, "u": 0.2, "v": 0.3, "T": 0.4, "p": 9.9}}]
    # p is excluded, so a five-channel cell stays comparable to a four-channel one
    assert tier_mean_rl2(recs, COMMON_CHANNELS) == pytest.approx(0.25)


def test_tier_mean_rl2_on_empty_split_is_none():
    assert tier_mean_rl2([], COMMON_CHANNELS) is None


def test_qw_columns_use_their_own_truths():
    """Crossing the truths inflates the head error by the estimator gap."""
    recs = [{"qw_true": 1.0e5, "qw_pred": 1.1e5,
             "qw_ledger": 2.7e6, "qw_head_pred": 2.7e6}]
    assert qw_abs_rel_median(recs, "qw_pred", "qw_true") == pytest.approx(0.1)
    assert qw_abs_rel_median(recs, "qw_head_pred", "qw_ledger") == pytest.approx(0.0)
    # scored against the wrong truth the head would look 26x off
    assert qw_abs_rel_median(recs, "qw_head_pred", "qw_true") > 20


def test_qw_median_is_none_without_records():
    assert qw_abs_rel_median([], "qw_pred", "qw_true") is None
    assert qw_abs_rel_median([{"qw_true": 1.0}], "qw_head_pred", "qw_ledger") is None


def test_render_table_marks_missing_values():
    run = {"n_train": 100, "interp": 0.1, "family": 0.2, "ood": None,
           "qw_posthoc": 0.3, "qw_head": None, "eos_viol": 0.0}
    table = render_table({"base": [run]})
    assert "n_train = [100]" in table
    assert "| -- |" in table


def test_render_table_reports_spread_across_seeds():
    runs = [{"n_train": 100, "interp": v, "family": 0.2, "ood": 0.3,
             "qw_posthoc": 0.3, "qw_head": None, "eos_viol": 0.0}
            for v in (0.10, 0.20)]
    table = render_table({"base": runs})
    assert "0.1500 +/- 0.0500" in table
