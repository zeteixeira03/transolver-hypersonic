"""Train Transolver on the SU2 hypersonic dataset.

One script covers both the fine-tune-from-AirfRANS-pretrain and from-scratch
baselines (toggle with ``--pretrain PATH`` or its absence). The slice-count
ablation is a single ``--slice-num`` flag; sweep it across {4, 8, 16, 32, 64}
by re-running with different values and different ``--out`` directories.

Splits
------
Train and val are drawn from the ``core`` group only, stratified by
``geom_id`` so val cases sit on geometries the train set has not seen. The
OOD slabs (``cone_high``, ``mach_high``, ``nose_large``, ``nose_small``) are
held out entirely as test sets; per-slab metrics are reported during eval.

Three evaluation tiers, in increasing difficulty:
``test_interp`` (interpolation: random cases held out from train geometries,
so unseen freestream conditions on seen shapes), ``test`` (family holdout:
whole geometry clusters unseen in training), and the OOD slabs
(extrapolation past the core parameter box).

Deep ensembles
--------------
``--init-seed`` decouples training stochasticity (model init, data order,
node subsampling) from the split seed. Ensemble members share ``--seed``
(identical splits and norm stats) and differ only in ``--init-seed``; a
run trained without ``--init-seed`` counts as the init-seed-0 member.

Eval-only mode
--------------
``--eval-only RUNDIR`` skips training, loads ``RUNDIR/best.pt`` and
``RUNDIR/norm_stats.pt``, rebuilds the splits from the current workdir and
evaluates all tiers. Use it to re-score existing checkpoints after a
dataset version update lands new OOD cases.

Physics priors
--------------
Three priors that the project normally leaves on are switchable, so the
prior ablation can price each one against a baseline on a small train split.
``--predict-p`` relaxes the hard equation of state and emits pressure as a
free fifth channel instead of reconstructing it from rho and T.
``--log-targets`` overrides which channels are standardized in log10 space
(``none`` gives plain standardization throughout). ``--qw-head`` adds a
case-level stagnation heat-flux head, either predicting log10 q_w directly
or the log-ratio against the Fay-Riddell correlation. Defaults reproduce the
project convention, so existing commands are unaffected.

Metrics
-------
Per-channel relative L2 over the held-out set is the primary number, in
physical units after de-normalization. Stagnation heat flux is the
engineering-credibility figure, and there are two of them. The post-hoc one
runs the same finite-difference estimator
(:func:`src.eval.sanity.compute_q_w_from_T`) over the predicted and the true
T field and compares the two, so it isolates the surrogate's near-wall
gradient; it runs for every model, which keeps that column comparable across
ablation cells. A heat-flux head, when present, is trained on and scored
against the ledger's SU2-postprocessed q_w. The two estimators disagree by a
median factor of about 27 on this dataset (the finite-difference value being
the smaller, since nearest-neighbor fits do not resolve the near-wall
gradient on these meshes), so the two columns are reported separately and
must not be read as one quantity.

Pretrain loading
----------------
The AirfRANS-pretrained Transolver was trained with ``space_dim=2,
fun_dim=5, out_dim=4`` and a different output convention
``(u, v, p/rho, nu_t)``. For SU2 we instantiate with ``space_dim=2,
fun_dim=8, out_dim=4`` and ``(rho, u, v, T)``. The input projection
(``preprocess``) and the final head (``blocks[-1].head``) cannot be
reused: their input/output dimensions or their target semantics differ.
We load all other parameters from the pretrain checkpoint and re-init
``preprocess`` and the final head from scratch. This is the standard
transfer move.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytical import fay_riddell_qw
from src.data.su2 import (
    CASE_PARAM_ORDER,
    DEFAULT_LOG_TARGETS,
    DEFAULT_LOG_TARGETS_WITH_P,
    N_CASE_PARAMS,
    POS_DIM,
    R_SPECIFIC_AIR,
    SU2Dataset,
    SU2NormStats,
    TARGET_ORDER,
    TARGET_ORDER_WITH_P,
    case_id_from_npz_path,
    compute_norm_stats,
    denormalize_targets,
    list_case_npzs,
)
from src.eval.sanity import (
    compute_q_w_from_T,
    identify_wall_nodes,
)
from src.models.transolver import Transolver


OOD_GROUPS = ("cone_high", "mach_high", "nose_large", "nose_small")


# ============================================================================================
#                                       physics priors
# ============================================================================================

# The three priors below are switchable so the ablation can price each one.
# Defaults reproduce the project convention: hard EoS (p reconstructed, never
# predicted), log10 normalization on rho and T, and no heat-flux head.

def _case_name(path: Path) -> str:
    """Case name as the dataset and the pinned-split files spell it."""
    return path.parent.name if path.name == "case.npz" else path.stem


def preflight_device(device: str) -> None:
    """Fail fast on an accelerator the installed torch has no kernels for.

    Rented GPU sessions are assigned a card, not asked for one, and a
    too-old card otherwise surfaces as a CUDA error at the first forward
    pass, which is after dataset staging has already burned most of an hour
    of quota. Checking up front turns that into an instant, legible failure.
    """
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise SystemExit(f"--device {device} but torch reports no CUDA device")
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"[preflight] {name}, compute capability {major}.{minor}, "
          f"{torch.cuda.device_count()} device(s)")
    arch_list = torch.cuda.get_arch_list()
    if f"sm_{major}{minor}" not in arch_list:
        raise SystemExit(
            f"{name} is compute capability {major}.{minor} (sm_{major}{minor}), "
            f"which this torch build has no kernels for (it ships {arch_list}). "
            f"Training would die at the first CUDA call. Reassign the session "
            f"to a supported accelerator and rerun."
        )


def collect_env(device: str) -> dict:
    """Provenance for one run: code version, interpreter, and accelerator.

    Runs happen on ephemeral rented GPUs whose output is destroyed if the
    notebook is re-pushed, so anything not written into the run directory is
    unrecoverable without spending quota again. Recording the commit and the
    device is what makes a surprising number explicable months later.
    """
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        env["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(ROOT), text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (subprocess.CalledProcessError, OSError):
        env["git_commit"] = None          # not a checkout, or no git present
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        env["cuda_version"] = torch.version.cuda
        env["gpu_count"] = torch.cuda.device_count()
    return env


@dataclass
class QwSpec:
    """Encoding for the auxiliary stagnation heat-flux head.

    ``direct`` regresses standardized ``log10 q_w``. ``residual`` regresses
    the standardized log-ratio against the Fay-Riddell correlation, so the
    network only has to learn the departure from the analytical value. Both
    work in log space because q_w spans 1.2e4 to 5.8e7 W/m^2 across the
    dataset.

    Attributes
    ----------
    mode : str
        ``direct`` or ``residual``.
    mean, std : float
        Standardization constants, fit on the train split only.
    """

    mode: str
    mean: float
    std: float

    def log_target(self, qw: float, q_fr: float) -> float:
        """Unstandardized log-space target for one case."""
        if self.mode == "residual":
            return float(np.log10(qw / q_fr))
        return float(np.log10(qw))

    def encode(self, qw: float, q_fr: float) -> float:
        return (self.log_target(qw, q_fr) - self.mean) / self.std

    def decode(self, z: float, q_fr: float) -> float:
        """Standardized head output back to W/m^2."""
        log_val = z * self.std + self.mean
        if self.mode == "residual":
            return float(q_fr * 10.0 ** log_val)
        return float(10.0 ** log_val)


def load_qw_from_ledger(ledger_path: Path) -> dict[int, float]:
    """Return ``{case_id: q_w}`` in W/m^2 for cases the sweep postprocessed."""
    con = _connect_ro(ledger_path)
    rows = con.execute("select case_id, qw from cases where qw is not null").fetchall()
    con.close()
    return {int(cid): float(q) for cid, q in rows}


def fay_riddell_for_case(params: dict[str, float]) -> float:
    """Analytical stagnation q_w from a case-parameter dict."""
    return fay_riddell_qw(
        M_inf=params["mach"], T_inf=params["T_inf"], p_inf=params["p_inf"],
        R_n=params["R_n"], T_w=params["T_w"],
    )


def build_qw_maps(
    paths: list[Path],
    case_params: dict[int, dict[str, float]],
    qw_by_id: dict[int, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-case-name ``(measured q_w, Fay-Riddell q_w)``, both in W/m^2.

    Cases missing a ledger q_w are omitted from both maps; downstream they
    fall out of the auxiliary loss rather than contributing a NaN.
    """
    qw, q_fr = {}, {}
    for path in paths:
        cid = case_id_from_npz_path(path)
        if cid not in qw_by_id or cid not in case_params:
            continue
        name = _case_name(path)
        qw[name] = qw_by_id[cid]
        q_fr[name] = fay_riddell_for_case(case_params[cid])
    return qw, q_fr


# ============================================================================================
#                                       split helpers
# ============================================================================================

def _connect_ro(ledger_path: Path) -> sqlite3.Connection:
    """Read-only immutable connection; works on read-only mounts (Kaggle input)."""
    return sqlite3.connect(f"file:{Path(ledger_path).as_posix()}?mode=ro&immutable=1", uri=True)


def load_case_groups(ledger_path: Path) -> dict[int, str]:
    """Return ``{case_id: group_name}`` for every row in the ledger."""
    con = _connect_ro(ledger_path)
    rows = con.execute("select case_id, group_name from cases").fetchall()
    con.close()
    return {int(cid): g for cid, g in rows}


def load_case_geom_ids(ledger_path: Path) -> dict[int, int]:
    con = _connect_ro(ledger_path)
    rows = con.execute("select case_id, geom_id from cases").fetchall()
    con.close()
    return {int(cid): int(g) for cid, g in rows}


def load_case_params_from_ledger(ledger_path: Path) -> dict[int, dict[str, float]]:
    """Return ``{case_id: {param: value}}`` for the 8 case parameters."""
    cols = ", ".join(CASE_PARAM_ORDER)
    con = _connect_ro(ledger_path)
    rows = con.execute(f"select case_id, {cols} from cases").fetchall()
    con.close()
    return {int(r[0]): dict(zip(CASE_PARAM_ORDER, map(float, r[1:]))) for r in rows}


def case_id_from_path(path: Path) -> int:
    """``case_0042/case.npz`` or ``case_0042.npz`` -> 42."""
    return case_id_from_npz_path(path)


def split_paths(
    paths: list[Path],
    groups: dict[int, str],
    geom_ids: dict[int, int],
    val_frac: float,
    test_frac: float,
    seed: int,
    interp_frac: float = 0.0,
) -> dict[str, list[Path]]:
    """Stratified train/val/test split by geometry cluster (core only).

    OOD groups go entirely to per-slab test sets. Within ``core``, full
    geometry clusters are assigned to train/val/test by a deterministic
    seeded shuffle of unique ``geom_id``s -- this prevents leakage between
    freestream-NN neighbours that share a mesh.

    With ``interp_frac > 0`` a random per-case fraction of the train
    geometries' cases is held out as ``test_interp``: unseen freestream
    conditions on shapes the model trained on. The geometry shuffle happens
    before the interp draw, so val/test geometry assignment for a given
    seed is unchanged by ``interp_frac``.

    Cases with ``group_name == 'loop'`` (active-learning acquired cases)
    are force-routed to ``train`` and never enter val/test/interp. They
    are also excluded from the geometry-shuffle input, so adding or
    removing loop cases leaves the core val/test/interp membership
    byte-identical for the same seed.
    """
    by_group: dict[str, list[Path]] = {"core": [], "loop": [], **{g: [] for g in OOD_GROUPS}}
    for p in paths:
        cid = case_id_from_path(p)
        if cid not in groups:
            continue
        by_group.setdefault(groups[cid], []).append(p)

    core_paths = by_group.pop("core")
    loop_paths = by_group.pop("loop", [])
    core_by_geom: dict[int, list[Path]] = {}
    for p in core_paths:
        gid = geom_ids[case_id_from_path(p)]
        core_by_geom.setdefault(gid, []).append(p)

    geom_ids_sorted = sorted(core_by_geom.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(geom_ids_sorted)
    n = len(geom_ids_sorted)
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, int(round(n * test_frac)))
    val_geoms = set(geom_ids_sorted[:n_val])
    test_geoms = set(geom_ids_sorted[n_val : n_val + n_test])
    train_geoms = set(geom_ids_sorted[n_val + n_test :])

    train_paths = [p for gid in sorted(train_geoms) for p in core_by_geom[gid]]
    interp_paths: list[Path] = []
    if interp_frac > 0:
        n_interp = int(round(len(train_paths) * interp_frac))
        pick = rng.permutation(len(train_paths))[:n_interp]
        interp_set = set(pick.tolist())
        interp_paths = [p for i, p in enumerate(train_paths) if i in interp_set]
        train_paths = [p for i, p in enumerate(train_paths) if i not in interp_set]

    # loop cases append after the core train, deterministic in case_id order
    train_paths = train_paths + sorted(loop_paths, key=case_id_from_path)

    splits = {
        "train": train_paths,
        "val":   [p for gid in sorted(val_geoms)   for p in core_by_geom[gid]],
        "test":  [p for gid in sorted(test_geoms)  for p in core_by_geom[gid]],
        "test_interp": interp_paths,
    }
    for g in OOD_GROUPS:
        splits[f"ood_{g}"] = by_group.get(g, [])
    return splits


# ============================================================================================
#                                       pretrain loading
# ============================================================================================

def load_pretrain_into(model: Transolver, pretrain_path: Path) -> dict[str, int]:
    """Load AirfRANS pretrain weights, skipping the input projection and head.

    The input projection (``preprocess``) is skipped because the input
    feature count differs (AirfRANS 7 channels vs SU2 10 channels). The
    final head (``blocks[-1].head``) is skipped because the 4 output
    channels carry different physical meanings (AirfRANS ``(u, v, p/rho,
    nu_t)`` vs SU2 ``(rho, u, v, T)``); reusing the AirfRANS head would
    carry irrelevant biases. The LayerNorms ``blocks[-1].ln_3`` operate on
    the hidden dim and are kept. Returns a summary dict for logging.
    """
    state = torch.load(str(pretrain_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model_state = model.state_dict()
    n_blocks = len(model.blocks)
    skip_prefixes = ("preprocess.", f"blocks.{n_blocks - 1}.head.")
    loaded, skipped_shape, skipped_unknown, skipped_explicit = 0, [], [], []
    for k, v in state.items():
        if k not in model_state:
            skipped_unknown.append(k)
            continue
        if any(k.startswith(pfx) for pfx in skip_prefixes):
            skipped_explicit.append(k)
            continue
        if v.shape != model_state[k].shape:
            skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        model_state[k] = v
        loaded += 1
    model.load_state_dict(model_state)
    print(f"[pretrain] loaded {loaded} tensors from {pretrain_path}")
    if skipped_explicit:
        print(f"[pretrain] skipped {len(skipped_explicit)} explicit (re-init): "
              f"{skipped_explicit}")
    if skipped_shape:
        print(f"[pretrain] skipped {len(skipped_shape)} shape-mismatched:")
        for k, src, dst in skipped_shape:
            print(f"  {k}: src{src} dst{dst}")
    if skipped_unknown:
        print(f"[pretrain] skipped {len(skipped_unknown)} unknown keys (first 5): "
              f"{skipped_unknown[:5]}")
    return {"loaded": loaded, "skipped_explicit": len(skipped_explicit),
            "skipped_shape": len(skipped_shape),
            "skipped_unknown": len(skipped_unknown)}


# ============================================================================================
#                                       collate + loop
# ============================================================================================

def collate_single(batch: list[dict]) -> dict:
    """Per-sample batches; meshes have variable N so no stacking."""
    if len(batch) != 1:
        raise NotImplementedError("batch_size > 1 not supported (variable N per mesh)")
    item = batch[0]
    return {
        "x": item["x"].unsqueeze(0),
        "y": item["y"].unsqueeze(0),
        "y_raw": item["y_raw"].unsqueeze(0),
        "pos": item["pos"].unsqueeze(0),
        "case_params": item["case_params"],
        "qw": item["qw"].unsqueeze(0),
        "name": item["name"],
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float,
    qw_weight: float = 0.0,
) -> dict[str, float]:
    """One pass over the train split.

    ``qw_weight`` above zero adds the auxiliary heat-flux term; the loader's
    ``qw`` entries are expected to be already encoded by :class:`QwSpec`.
    Cases with no ledger q_w arrive as NaN and are dropped from that term.
    """
    model.train()
    t0 = time.time()
    n, sum_loss, sum_field, sum_qw = 0, 0.0, 0.0, 0.0
    sum_gnorm, max_gnorm, n_qw_steps = 0.0, 0.0, 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pos = batch["pos"].to(device)
        optimizer.zero_grad()
        out = model(x, pos=pos)
        pred, qw_pred = out if qw_weight > 0 else (out, None)
        pred = pred.float()
        field_loss = F.mse_loss(pred, y.float())
        loss = field_loss
        qw_loss = torch.zeros((), device=pred.device)
        if qw_weight > 0:
            qw_true = batch["qw"].to(device).float()
            mask = torch.isfinite(qw_true)
            if mask.any():
                qw_loss = F.mse_loss(qw_pred.float()[mask], qw_true[mask])
                loss = loss + qw_weight * qw_loss
                n_qw_steps += 1
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at step={n} sim={batch.get('name', '?')}; "
                f"x.range=[{float(x.min()):.3g}, {float(x.max()):.3g}], "
                f"pred.range=[{float(pred.min()):.3g}, {float(pred.max()):.3g}]"
            )
        loss.backward()
        # always measured, never free but cheap: an inf max_norm clips nothing
        # and still returns the pre-clip norm, which is the diagnostic that
        # explains a training curve after the fact
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip if grad_clip > 0 else float("inf")))
        optimizer.step()
        sum_loss += float(loss.detach())
        sum_field += float(field_loss.detach())
        sum_qw += float(qw_loss.detach())
        sum_gnorm += gnorm
        max_gnorm = max(max_gnorm, gnorm)
        n += 1
    denom = max(n, 1)
    return {
        "loss": sum_loss / denom,
        "field_loss": sum_field / denom,
        "qw_loss": sum_qw / denom,
        "grad_norm_mean": sum_gnorm / denom,
        "grad_norm_max": max_gnorm,
        "n_steps": n,
        "n_qw_steps": n_qw_steps,
        "epoch_time_s": time.time() - t0,
    }


# ============================================================================================
#                                       evaluation
# ============================================================================================

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: SU2Dataset,
    stats: SU2NormStats,
    device: str,
    *,
    case_groups: dict[int, str] | None = None,
    qw_spec: QwSpec | None = None,
    fr_by_name: dict[str, float] | None = None,
    qw_ledger_by_name: dict[str, float] | None = None,
) -> tuple[dict, list[dict]]:
    """Per-channel rel-L2 in physical units, plus post-hoc stagnation q_w.

    For each case in the split, computes ||pred_c - true_c||_2 / ||true_c||_2
    per channel, plus q_w from the predicted T field via FD on the inward
    wall normal. The wall-node indices are identified from the ground-truth
    T field (not the predicted one) so the q_w comparison is well-defined
    even when the surrogate's near-wall T is noisy.

    The post-hoc q_w path runs for every model, with or without a heat-flux
    head, so that column stays comparable across the prior ablation. When
    ``qw_spec`` is given, the head's own prediction is recorded alongside it
    under ``qw_head_pred`` rather than replacing it.

    The two q_w numbers are scored against different truths and are not
    interchangeable. The post-hoc error compares the finite-difference
    estimate on the predicted T field against the same estimate on the true
    field, so it isolates the surrogate's near-wall gradient. The head is
    trained on the ledger's SU2-postprocessed q_w and is therefore scored
    against that. The two truths differ by a median factor of about 27 on
    this dataset, the finite-difference value being the smaller, because the
    nearest-fluid-neighbor fit cannot resolve the near-wall gradient on these
    meshes. Do not put the two columns on one axis.

    Models carrying a free pressure channel also get ``eos_viol``, the median
    relative departure from ``p = rho R T``. It is identically zero when p is
    reconstructed instead of predicted, which is the point of the comparison.

    Returns the aggregate summary dict and a per-case record list (name,
    case params, per-channel rL2, q_w pair) for envelope-distance analysis.
    """
    model.eval()
    targets = stats.targets
    saved_subsample = dataset.subsample
    dataset.subsample = None
    per_case: list[dict] = []
    try:
        for i in range(len(dataset)):
            item = dataset[i]
            x = item["x"].unsqueeze(0).to(device)
            pos = item["pos"].unsqueeze(0).to(device)
            y_true = item["y_raw"].to(device)
            out_model = model(x, pos=pos)
            qw_head_norm = None
            if qw_spec is not None:
                out_model, qw_batch = out_model
                qw_head_norm = float(qw_batch.squeeze(0))
            pred_norm = out_model.squeeze(0)
            pred = denormalize_targets(pred_norm, stats.to(device))

            num = (pred - y_true).pow(2).sum(dim=0).sqrt()
            den = y_true.pow(2).sum(dim=0).sqrt().clamp_min(1e-12)
            rl2 = (num / den).cpu().numpy()
            rec = {
                "name": item["name"],
                "case_params": {k: float(item["case_params"][j])
                                for j, k in enumerate(CASE_PARAM_ORDER)},
                "rL2": {name: float(rl2[j]) for j, name in enumerate(targets)},
                "qw_true": None,
                "qw_pred": None,
            }

            if "p" in targets:
                p_free = pred[:, targets.index("p")]
                p_eos = (pred[:, targets.index("rho")] * R_SPECIFIC_AIR
                         * pred[:, targets.index("T")])
                p_ref = y_true[:, targets.index("p")].abs().clamp_min(1e-12)
                rec["eos_viol"] = float(
                    ((p_free - p_eos).abs() / p_ref).median()
                )

            # recorded for every run, head or not: these are the reference
            # values any later heat-flux analysis needs, and re-deriving them
            # means re-running the case
            q_fr = None if fr_by_name is None else fr_by_name.get(item["name"])
            if q_fr is not None:
                rec["qw_fay_riddell"] = q_fr
            if qw_ledger_by_name is not None:
                rec["qw_ledger"] = qw_ledger_by_name.get(item["name"])
            if qw_head_norm is not None and q_fr is not None:
                rec["qw_head_norm"] = qw_head_norm      # raw, pre-decode
                rec["qw_head_pred"] = qw_spec.decode(qw_head_norm, q_fr)

            # post-hoc q_w on predicted T (Kelvin), wall ids from ground truth
            xy = item["pos"].cpu().numpy()
            T_true = y_true[:, targets.index("T")].cpu().numpy()
            T_pred = pred[:, targets.index("T")].cpu().numpy()
            T_w = float(item["case_params"][CASE_PARAM_ORDER.index("T_w")])
            R_n = float(item["case_params"][CASE_PARAM_ORDER.index("R_n")])
            try:
                wall = identify_wall_nodes(T_true, T_w)
                qw_true = float(compute_q_w_from_T(
                    x=xy[:, 0], r=xy[:, 1], T=T_true, T_w=T_w,
                    wall_indices=wall, y_axis_skip=0.05 * R_n, n_average=3,
                )["q_w"])
                qw_pred = float(compute_q_w_from_T(
                    x=xy[:, 0], r=xy[:, 1], T=T_pred, T_w=T_w,
                    wall_indices=wall, y_axis_skip=0.05 * R_n, n_average=3,
                )["q_w"])
                # assign only when both succeed; a lone qw_true poisons the
                # (true, pred) pair arithmetic downstream
                rec["qw_true"], rec["qw_pred"] = qw_true, qw_pred
            except ValueError:
                pass                       # mesh / wall-normal degeneracy; keep the rL2 record
            per_case.append(rec)
    finally:
        dataset.subsample = saved_subsample

    rl2 = np.stack([[r["rL2"][name] for name in targets] for r in per_case]).mean(axis=0)
    out = {f"rL2_{name}": float(rl2[i]) for i, name in enumerate(targets)}
    qw_pairs = [(r["qw_true"], r["qw_pred"]) for r in per_case if r["qw_true"] is not None]
    if qw_pairs:
        qw_arr = np.array(qw_pairs)
        rel = (qw_arr[:, 1] - qw_arr[:, 0]) / qw_arr[:, 0]
        out["qw_rel_err_mean"] = float(rel.mean())
        out["qw_rel_err_median"] = float(np.median(rel))
        out["qw_abs_rel_err_median"] = float(np.median(np.abs(rel)))
        out["n_qw_cases"] = int(len(qw_pairs))

    # scored against the ledger q_w the head was trained on, not against the
    # finite-difference qw_true above; see the note in the docstring
    head_pairs = [(r["qw_ledger"], r["qw_head_pred"]) for r in per_case
                  if r.get("qw_head_pred") is not None and r.get("qw_ledger")]
    if head_pairs:
        head_arr = np.array(head_pairs)
        rel_head = (head_arr[:, 1] - head_arr[:, 0]) / head_arr[:, 0]
        out["qw_head_rel_err_median"] = float(np.median(rel_head))
        out["qw_head_abs_rel_err_median"] = float(np.median(np.abs(rel_head)))
        out["n_qw_head_cases"] = int(len(head_pairs))

    eos = [r["eos_viol"] for r in per_case if "eos_viol" in r]
    if eos:
        out["eos_viol_median"] = float(np.median(eos))
    return out, per_case


# ============================================================================================
#                                       main
# ============================================================================================

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="train Transolver on SU2 hypersonic")
    p.add_argument("--workdir", default="data/raw/sweep",
                   help="dataset workdir containing case_*/case.npz and ledger.db")
    p.add_argument("--out", required=True, help="output directory for checkpoints + logs")
    p.add_argument("--pretrain", default=None,
                   help="path to AirfRANS pretrain .pt; omit for from-scratch baseline")
    p.add_argument("--slice-num", type=int, default=32)
    p.add_argument("--n-hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--subsample", type=int, default=8192,
                   help="random subsample of nodes per training step (full mesh at eval)")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--interp-frac", type=float, default=0.10,
                   help="fraction of train-geometry cases held out as the interpolation tier")
    p.add_argument("--eval-only", default=None, metavar="RUNDIR",
                   help="skip training; evaluate RUNDIR/best.pt with RUNDIR/norm_stats.pt "
                        "on all tiers and write results to --out")
    p.add_argument("--pinned-splits", default=None, metavar="JSON",
                   help="JSON of {split_name: [case_name, ...]} that overrides the "
                        "auto-computed splits for the tiers it lists; tiers absent from "
                        "the JSON keep their auto-computed membership. Use to reproduce "
                        "the exact core val/test/interp used by a prior run when "
                        "rescoring the same checkpoint against a refreshed dataset.")
    p.add_argument("--seed", type=int, default=0,
                   help="split seed; also seeds training RNGs unless --init-seed is given")
    p.add_argument("--init-seed", type=int, default=None,
                   help="training RNG seed (init, data order, subsampling); "
                        "splits stay on --seed. For deep-ensemble members.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant",
                   help="cosine anneals to zero over --epochs, stepped per epoch")
    p.add_argument("--predict-p", action="store_true",
                   help="emit pressure as a free fifth channel instead of "
                        "reconstructing it from rho and T (relaxes the hard EoS)")
    p.add_argument("--log-targets", default=None,
                   help="comma-separated target channels to standardize in log10 "
                        "space, or 'none' for plain standardization throughout. "
                        "Defaults to rho,T (plus p when --predict-p is set).")
    p.add_argument("--qw-head", choices=("none", "direct", "residual"), default="none",
                   help="auxiliary stagnation heat-flux head: predict log10 q_w "
                        "directly, or the log-ratio against Fay-Riddell")
    p.add_argument("--qw-loss-weight", type=float, default=0.1,
                   help="weight on the auxiliary q_w term; ignored without --qw-head")
    p.add_argument("--val-every", type=int, default=10)
    return p


def main() -> None:
    args = _parser().parse_args()
    workdir = Path(args.workdir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    preflight_device(args.device)
    init_seed = args.seed if args.init_seed is None else args.init_seed
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    random.seed(init_seed)
    print(f"[main] split seed {args.seed}, init seed {init_seed}")

    ledger = workdir / "ledger.db"
    paths = list_case_npzs(workdir)
    if not paths:
        raise SystemExit(f"no case_*/case.npz under {workdir}")
    print(f"[main] found {len(paths)} case.npz files under {workdir}")

    groups = load_case_groups(ledger)
    geom_ids = load_case_geom_ids(ledger)
    splits = split_paths(paths, groups, geom_ids, args.val_frac, args.test_frac,
                         args.seed, args.interp_frac)
    if args.pinned_splits is not None:
        pinned = json.loads(Path(args.pinned_splits).read_text())
        by_name = {_case_name(p): p for p in paths}
        for split_name, names in pinned.items():
            resolved = [by_name[n] for n in names if n in by_name]
            missing = [n for n in names if n not in by_name]
            if missing:
                print(f"[pinned] {split_name}: {len(missing)} case(s) not present in "
                      f"workdir; dropping ({missing[:5]}{' ...' if len(missing) > 5 else ''})")
            splits[split_name] = resolved
        print(f"[pinned] applied {len(pinned)} pinned split(s) from {args.pinned_splits}")
    for name, ps in splits.items():
        print(f"[split] {name}: {len(ps)} cases")

    if not splits["train"]:
        raise SystemExit("empty train split; check ledger / workdir")

    # per-parameter training envelope, for envelope-distance analysis downstream
    case_params = load_case_params_from_ledger(ledger)
    train_ids = [case_id_from_path(p) for p in splits["train"]]
    train_envelope = {
        k: [min(case_params[c][k] for c in train_ids),
            max(case_params[c][k] for c in train_ids)]
        for k in CASE_PARAM_ORDER
    }

    # ---- prior configuration: target layout, log channels, heat-flux head ----
    target_order = TARGET_ORDER_WITH_P if args.predict_p else TARGET_ORDER
    if args.log_targets is None:
        log_targets = DEFAULT_LOG_TARGETS_WITH_P if args.predict_p else DEFAULT_LOG_TARGETS
    elif args.log_targets.strip().lower() == "none":
        log_targets = ()
    else:
        log_targets = tuple(s.strip() for s in args.log_targets.split(",") if s.strip())
        unknown = [s for s in log_targets if s not in target_order]
        if unknown:
            raise SystemExit(f"--log-targets names unknown channels {unknown}; "
                             f"available: {list(target_order)}")
    print(f"[priors] targets={list(target_order)} log={list(log_targets)} "
          f"qw_head={args.qw_head}")

    # built unconditionally so every run's per-case record carries the ledger
    # and analytical heat flux, not just the runs that train a head
    qw_raw, fr_by_name = build_qw_maps(paths, case_params,
                                       load_qw_from_ledger(ledger))
    qw_spec: QwSpec | None = None
    qw_encoded: dict[str, float] | None = None
    if args.qw_head != "none":
        train_names = [_case_name(p) for p in splits["train"]]
        logs = [np.log10(qw_raw[n] / (fr_by_name[n] if args.qw_head == "residual" else 1.0))
                for n in train_names if n in qw_raw]
        if not logs:
            raise SystemExit("--qw-head set but no train case has a ledger q_w")
        # std floor guards a degenerate train split; real spreads are O(0.1-1)
        qw_spec = QwSpec(args.qw_head, float(np.mean(logs)),
                         max(float(np.std(logs)), 1e-6))
        qw_encoded = {n: qw_spec.encode(qw_raw[n], fr_by_name[n]) for n in qw_raw}
        print(f"[priors] q_w head on {len(logs)}/{len(train_names)} train cases, "
              f"log-target mean={qw_spec.mean:.3f} std={qw_spec.std:.3f}")

    qw_weight = args.qw_loss_weight if qw_spec is not None else 0.0

    if args.eval_only is not None:
        rundir = Path(args.eval_only).resolve()
        stats = SU2NormStats.load(rundir / "norm_stats.pt")
        target_order = stats.targets            # checkpoint decides the layout
        spec_path = rundir / "qw_spec.json"
        if spec_path.is_file():
            qw_spec = QwSpec(**json.loads(spec_path.read_text()))
            if fr_by_name is None:
                qw_raw, fr_by_name = build_qw_maps(
                    paths, case_params, load_qw_from_ledger(ledger))
        else:
            qw_spec = None
        # model shape must match the checkpoint; prefer the recorded run args
        run_args = dict(vars(args))
        prev_eval = rundir / "final_eval.json"
        if prev_eval.is_file():
            recorded = json.loads(prev_eval.read_text()).get("args", {})
            for k in ("slice_num", "n_hidden", "n_layers", "n_head"):
                if k in recorded:
                    run_args[k] = recorded[k]
        model = Transolver(
            space_dim=POS_DIM, fun_dim=N_CASE_PARAMS, out_dim=len(target_order),
            n_hidden=run_args["n_hidden"], n_layers=run_args["n_layers"],
            n_head=run_args["n_head"], slice_num=run_args["slice_num"],
            qw_head=qw_spec is not None,
        ).to(args.device)
        model.load_state_dict(torch.load(rundir / "best.pt", map_location=args.device))
        print(f"[eval-only] loaded {rundir / 'best.pt'} "
              f"(slice_num={run_args['slice_num']})")
        # the checkpoint's own layout wins over the CLI defaults for the record
        log_targets = stats.log_targets
        n_params = sum(p.numel() for p in model.parameters())
        pretrain_info, best_val, best_epoch = None, None, None
    else:
        stats = compute_norm_stats(splits["train"], log_targets=log_targets,
                                   with_p=args.predict_p)
        stats.save(out / "norm_stats.pt")
        print(f"[main] norm stats fit on {len(splits['train'])} train cases, saved")
        if qw_spec is not None:
            (out / "qw_spec.json").write_text(json.dumps(vars(qw_spec), indent=2))

        train_ds = SU2Dataset(splits["train"], stats, subsample=args.subsample,
                              qw_by_name=qw_encoded)
        val_ds = SU2Dataset(splits["val"], stats, subsample=None)
        loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_single,
                            num_workers=0)

        model = Transolver(
            space_dim=POS_DIM, fun_dim=N_CASE_PARAMS, out_dim=len(target_order),
            n_hidden=args.n_hidden, n_layers=args.n_layers, n_head=args.n_head,
            slice_num=args.slice_num, qw_head=qw_spec is not None,
        ).to(args.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[main] model: {n_params:,} params, slice_num={args.slice_num}")

        pretrain_info = None
        if args.pretrain is not None:
            pretrain_info = load_pretrain_into(model, Path(args.pretrain))

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            if args.lr_schedule == "cosine" else None
        )

        history: list[dict] = []
        best_val, best_epoch = float("inf"), None
        t_start = time.time()
        for epoch in range(1, args.epochs + 1):
            rec = train_one_epoch(model, loader, optimizer, args.device, args.grad_clip,
                                  qw_weight=qw_weight)
            # every cheap scalar goes in on every epoch: the run directory is
            # the only durable record and re-running to recover a curve costs
            # a GPU session
            rec["epoch"] = epoch
            rec["lr"] = float(optimizer.param_groups[0]["lr"])
            rec["wallclock_s"] = time.time() - t_start
            if torch.cuda.is_available():
                rec["gpu_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
            if scheduler is not None:
                scheduler.step()
            if epoch % args.val_every == 0 or epoch == args.epochs:
                val, val_per_case = evaluate(model, val_ds, stats, args.device,
                                             qw_spec=qw_spec, fr_by_name=fr_by_name,
                                             qw_ledger_by_name=qw_raw)
                mean_rl2 = float(np.mean([val[k] for k in val if k.startswith("rL2_")]))
                rec.update({"val": val, "mean_rL2": mean_rl2})
                print(f"[ep {epoch:4d}] loss={rec['loss']:.4f} "
                      f"val_mean_rL2={mean_rl2:.4f} val={val}")
                if mean_rl2 < best_val:
                    best_val, best_epoch = mean_rl2, epoch
                    torch.save(model.state_dict(), out / "best.pt")
            else:
                print(f"[ep {epoch:4d}] loss={rec['loss']:.4f}")
            history.append(rec)
            # rewritten each epoch so a killed session keeps its curve
            (out / "history.json").write_text(json.dumps(history, indent=2))

        torch.save(model.state_dict(), out / "final.pt")
        model.load_state_dict(torch.load(out / "best.pt", map_location=args.device))
        print(f"[main] best val mean_rL2 {best_val:.4f} at epoch {best_epoch} "
              f"of {args.epochs}")

    # final eval on all tiers from the best checkpoint
    final = {}
    per_case_all: dict[str, list[dict]] = {}
    for split_name in ["val", "test", "test_interp", *(f"ood_{g}" for g in OOD_GROUPS)]:
        if not splits[split_name]:
            continue
        ds = SU2Dataset(splits[split_name], stats, subsample=None)
        final[split_name], per_case_all[split_name] = evaluate(
            model, ds, stats, args.device, qw_spec=qw_spec, fr_by_name=fr_by_name,
            qw_ledger_by_name=qw_raw)
        print(f"[final] {split_name}: {final[split_name]}")

    # in eval-only mode, args.slice_num / n_hidden / n_layers / n_head reflect the
    # CLI defaults, not the model that actually ran; overlay from run_args so the
    # provenance record matches the loaded checkpoint
    args_out = dict(vars(args))
    if args.eval_only is not None:
        for k in ("slice_num", "n_hidden", "n_layers", "n_head"):
            args_out[k] = run_args[k]
    (out / "final_eval.json").write_text(json.dumps({
        "splits": {k: len(v) for k, v in splits.items()},
        # names too, not just counts: a rerun on a refreshed dataset would
        # otherwise be impossible to line up against this one case by case
        "split_names": {k: [_case_name(p) for p in v] for k, v in splits.items()},
        "args": args_out,
        "env": collect_env(args.device),
        "n_params": n_params,
        "pretrain_info": pretrain_info,
        "best_val_mean_rL2": best_val,
        "best_epoch": best_epoch,
        "qw_spec": None if qw_spec is None else vars(qw_spec),
        "target_order": list(target_order),
        "log_targets": list(log_targets),
        "train_envelope": train_envelope,
        "final": final,
    }, indent=2))
    (out / "per_case_eval.json").write_text(json.dumps(per_case_all, indent=2))


if __name__ == "__main__":
    main()
