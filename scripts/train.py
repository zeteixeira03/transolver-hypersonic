"""Train Transolver on the SU2 hypersonic dataset.

One script covers two baselines. With ``--pretrain PATH`` it fine-tunes from
the AirfRANS checkpoint, and without it the model trains from scratch. The
slice-count ablation is the single flag ``--slice-num``. Sweep it across
{4, 8, 16, 32, 64} by running again with a new value and a new ``--out``.

Splits
------
The training and validation splits come from the ``core`` group alone,
stratified by ``geom_id``, so a validation case sits on a geometry the model
never trained on. The four extrapolation slabs, ``cone_high``, ``mach_high``,
``nose_large`` and ``nose_small``, are held out whole as test sets, and each
one reports its own metrics.

Three evaluation tiers rise in difficulty. ``test_interp`` is interpolation:
cases held back from the training geometries, so the shapes are seen and the
freestream conditions are not. ``test`` is the family holdout: whole geometry
clusters absent from training. The slabs are extrapolation past the core box.

Deep ensembles
--------------
``--init-seed`` separates the randomness of training, meaning the starting
weights, the data order and the node draw, from the randomness of the split.
Every member of an ensemble shares one ``--seed``, which gives identical splits
and identical normalization statistics, and differs only in ``--init-seed``. A
run with no ``--init-seed`` counts as the member at init-seed 0.

Evaluation without training
---------------------------
``--eval-only RUNDIR`` skips training. It loads ``RUNDIR/best.pt`` and
``RUNDIR/norm_stats.pt``, rebuilds the splits from the current dataset
directory, and scores every tier. Use it to re-score existing checkpoints once
a new dataset version brings new extrapolation cases.

Physics priors
--------------
The project leaves three priors on. Each one can be switched off, so the prior
ablation can price it against a baseline on a small training split.
``--predict-p`` drops the hard equation of state and emits pressure as a free
fifth channel instead of reconstructing it from density and temperature.
``--log-targets`` names the channels standardized in log10 space, and ``none``
applies plain standardization to every channel. ``--qw-head`` adds a case-level stagnation
heat-flux head, which predicts either log10 of q_w directly or the log of its
ratio to the Fay-Riddell value. The defaults hold the project convention, so an
existing command behaves as before.

Metrics
-------
The primary number is the relative L2 per channel over the held-out set, in
physical units after the normalization is undone.

Stagnation heat flux is what makes the result credible to an engineer, and two
different numbers carry that name here. The first runs the finite-difference
estimator :func:`src.eval.sanity.compute_q_w_from_T` over the predicted field
and over the true field and compares the two, which isolates the near-wall
gradient of the surrogate. It runs for every model, so the column stays
comparable across ablation cells. The second exists only where a heat-flux head
does, and that head trains and scores against the value SU2 itself reported in
the ledger.

The two disagree by a median factor of about 27 on this dataset, the
finite-difference value being the smaller, because a fit to the nearest fluid
neighbour cannot resolve the near-wall gradient at this mesh resolution. The two
columns are reported apart and must never be read as one quantity.

Loading the pretrained weights
------------------------------
The AirfRANS checkpoint was trained at ``space_dim=2, fun_dim=5, out_dim=4``
and predicts ``(u, v, p/rho, nu_t)``. The SU2 model runs at ``space_dim=2,
fun_dim=8, out_dim=4`` and predicts ``(rho, u, v, T)``. Two parts of the network
therefore cannot transfer. The input projection ``preprocess`` takes a different
channel count, and the final head ``blocks[-1].head`` emits four channels that
mean different physical things. Every other parameter loads from the checkpoint,
and those two start fresh.
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

# each of the three priors below can be switched off, so the ablation can price
# it. the defaults hold the project convention: pressure reconstructed and never
# predicted, log10 normalization on density and temperature, no heat-flux head.

def _case_name(path: Path) -> str:
    """Return the case name in the spelling the dataset and split files use."""
    return path.parent.name if path.name == "case.npz" else path.stem


def preflight_device(device: str) -> None:
    """Stop at once on an accelerator the installed torch has no kernels for.

    A rented session assigns a card rather than offering a choice. A card too old
    for the build otherwise fails as a CUDA error at the first forward pass,
    which arrives after dataset staging has already spent most of an hour of
    quota. This check turns that into an immediate and readable failure.
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
    """Record the provenance of one run: code version, interpreter, accelerator.

    A run happens on a rented machine that does not persist, and pushing the
    notebook again destroys its output. Anything absent from the run directory
    costs another session of quota to recover. The commit and the device are what
    make a surprising number explicable months later.
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
        env["git_commit"] = None          # no git available, or not a checkout
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        env["cuda_version"] = torch.version.cuda
        env["gpu_count"] = torch.cuda.device_count()
    return env


@dataclass
class QwSpec:
    """How the stagnation heat-flux head encodes its target.

    ``direct`` regresses the standardized ``log10 q_w``. ``residual`` regresses
    the standardized log of the ratio to the Fay-Riddell value, so the network
    learns only the departure from the correlation. Both work in log space,
    because q_w runs from 1.2e4 to 5.8e7 W/m^2 across the dataset.

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
        """Return the log-space target for one case, before standardization."""
        if self.mode == "residual":
            return float(np.log10(qw / q_fr))
        return float(np.log10(qw))

    def encode(self, qw: float, q_fr: float) -> float:
        return (self.log_target(qw, q_fr) - self.mean) / self.std

    def decode(self, z: float, q_fr: float) -> float:
        """Turn a standardized head output back into W/m^2."""
        log_val = z * self.std + self.mean
        if self.mode == "residual":
            return float(q_fr * 10.0 ** log_val)
        return float(10.0 ** log_val)


def load_qw_from_ledger(ledger_path: Path) -> dict[int, float]:
    """Return ``{case_id: q_w}`` in W/m^2 for every case the sweep measured."""
    con = _connect_ro(ledger_path)
    rows = con.execute("select case_id, qw from cases where qw is not null").fetchall()
    con.close()
    return {int(cid): float(q) for cid, q in rows}


def fay_riddell_for_case(params: dict[str, float]) -> float:
    """Return the Fay-Riddell stagnation heat flux for one case-parameter dict."""
    return fay_riddell_qw(
        M_inf=params["mach"], T_inf=params["T_inf"], p_inf=params["p_inf"],
        R_n=params["R_n"], T_w=params["T_w"],
    )


def build_qw_maps(
    paths: list[Path],
    case_params: dict[int, dict[str, float]],
    qw_by_id: dict[int, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return, per case name, the measured and the Fay-Riddell q_w in W/m^2.

    A case with no q_w in the ledger is absent from both maps, so it falls out
    of the heat-flux loss rather than feeding it a NaN.
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
    """Open the ledger read-only and immutable, which a read-only mount allows."""
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
    """Return ``{case_id: {name: value}}`` over the eight case parameters."""
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
    """Split the core group into training, validation and test by geometry.

    Each extrapolation group becomes a test set of its own. Inside ``core``, a
    seeded shuffle of the unique ``geom_id`` values assigns whole geometry
    clusters to one split each. Whole clusters, because two cases that share a
    mesh sit next to each other in freestream space and would leak between
    splits.

    Above zero, ``interp_frac`` holds back that fraction of the cases on the
    training geometries as ``test_interp``, which gives unseen freestream
    conditions on shapes the model trained on. The geometry shuffle runs before
    that draw, so ``interp_frac`` never moves a geometry between the validation
    and test splits at a given seed.

    A case with ``group_name == 'loop'`` comes from the active-learning
    acquisition and always goes to ``train``. Such a case never reaches a
    held-out tier, and it never enters the geometry shuffle either. Adding or
    removing loop cases therefore leaves the core membership identical at the
    same seed.
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

    # the loop cases follow the core training cases, ordered by case_id
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
    """Load the AirfRANS weights, leaving the input projection and head fresh.

    The input projection ``preprocess`` is left out because the channel counts
    differ, at 7 for AirfRANS and 10 for SU2. The final head
    ``blocks[-1].head`` is left out because its four output channels mean
    different things, ``(u, v, p/rho, nu_t)`` against ``(rho, u, v, T)``, and
    its biases would carry no useful information. The layer normalization at
    ``blocks[-1].ln_3`` acts on the hidden dimension and does transfer. Returns
    a summary dict for the log.
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
    """Build a batch from one sample. Two meshes differ in N and cannot stack."""
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

    Above zero, ``qw_weight`` adds the heat-flux term to the loss. The ``qw``
    entry of each item must already be encoded by :class:`QwSpec`. A case with no
    q_w in the ledger arrives as NaN and drops out of that term.
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
        # measured on every step. an infinite max_norm clips nothing and
        # still returns the norm before clipping, which is the one diagnostic
        # that explains a training curve after the fact
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
    """Score one split: relative L2 per channel, with the stagnation heat flux.

    For every case the function computes ||pred_c - true_c||_2 / ||true_c||_2
    per channel, in physical units. It also reads a heat flux off the predicted
    temperature field, by finite difference along the inward wall normal. The
    wall nodes are identified from the true temperature field rather than the
    predicted one, so the comparison stays well defined even where the surrogate
    is noisy at the wall.

    That path runs for every model, with a heat-flux head or without one, which
    keeps the column comparable across the prior ablation. Where ``qw_spec`` is
    given, the prediction of the head is recorded beside it under
    ``qw_head_pred`` rather than in place of it.

    The two heat-flux numbers are scored against different truths and are not
    interchangeable. The first compares the finite-difference estimate on the
    predicted field against the same estimate on the true field, which isolates
    the near-wall gradient of the surrogate. The head trains on the value SU2
    reported in the ledger and is scored against that. The two truths differ by
    a median factor of about 27 on this dataset, the finite-difference value
    being the smaller, because a fit to the nearest fluid neighbour cannot
    resolve the near-wall gradient at this mesh resolution. Never place the two
    columns on one axis.

    A model with a free pressure channel also gets ``eos_viol``, the median
    relative departure from ``p = rho R T``. That number is identically zero
    where pressure is reconstructed, which is what the comparison measures.

    Returns the summary dict and a list of per-case records. Each record holds
    the name, the case parameters, the relative L2 per channel and the pair of
    heat-flux values, which the envelope-distance analysis reads.
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

            # physics requires density, temperature and pressure to stay
            # positive. a relative L2 in physical units moves little on a
            # negative freestream density, because the magnitudes there are
            # small, and yet such a field is unusable: it drives p = rho R T
            # negative. a log-space channel cannot reach zero at all, and
            # this column is where that guarantee becomes visible.
            for j, ch_name in enumerate(targets):
                if ch_name in ("rho", "T", "p"):
                    rec[f"frac_nonpos_{ch_name}"] = float(
                        (pred[:, j] <= 0).float().mean()
                    )

            if "p" in targets:
                p_free = pred[:, targets.index("p")]
                p_eos = (pred[:, targets.index("rho")] * R_SPECIFIC_AIR
                         * pred[:, targets.index("T")])
                p_ref = y_true[:, targets.index("p")].abs().clamp_min(1e-12)
                rec["eos_viol"] = float(
                    ((p_free - p_eos).abs() / p_ref).median()
                )

            # recorded on every run, with a head or without one. any later
            # heat-flux analysis needs these reference values, and recovering
            # them otherwise means running the case again
            q_fr = None if fr_by_name is None else fr_by_name.get(item["name"])
            if q_fr is not None:
                rec["qw_fay_riddell"] = q_fr
            if qw_ledger_by_name is not None:
                rec["qw_ledger"] = qw_ledger_by_name.get(item["name"])
            if qw_head_norm is not None and q_fr is not None:
                rec["qw_head_norm"] = qw_head_norm      # raw, pre-decode
                rec["qw_head_pred"] = qw_spec.decode(qw_head_norm, q_fr)

            # heat flux from the predicted temperature field in kelvin, with
            # the wall nodes taken from the true field
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
                # write both or neither. a qw_true without its partner would
                # break the arithmetic on the pair further down.
                rec["qw_true"], rec["qw_pred"] = qw_true, qw_pred
            except ValueError:
                # the mesh or the wall normal is degenerate here. keep the
                # relative L2 record and drop the heat flux.
                pass
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

    # scored against the ledger value the head trained on, and never against
    # the finite-difference value above
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
    for ch_name in ("rho", "T", "p"):
        key = f"frac_nonpos_{ch_name}"
        vals = [r[key] for r in per_case if key in r]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
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
                        "reconstructing it from density and temperature, which "
                        "drops the equation-of-state constraint")
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

    # the training range of each parameter, which the envelope-distance analysis
    # reads later
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

    # built on every run, so that every per-case record carries both the ledger
    # and the analytical heat flux, and not only the runs that train a head
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
        # the floor on the standard deviation guards against a degenerate
        # training split. a real spread is of order 0.1 to 1.
        qw_spec = QwSpec(args.qw_head, float(np.mean(logs)),
                         max(float(np.std(logs)), 1e-6))
        qw_encoded = {n: qw_spec.encode(qw_raw[n], fr_by_name[n]) for n in qw_raw}
        print(f"[priors] q_w head on {len(logs)}/{len(train_names)} train cases, "
              f"log-target mean={qw_spec.mean:.3f} std={qw_spec.std:.3f}")

    qw_weight = args.qw_loss_weight if qw_spec is not None else 0.0

    if args.eval_only is not None:
        rundir = Path(args.eval_only).resolve()
        stats = SU2NormStats.load(rundir / "norm_stats.pt")
        target_order = stats.targets            # the checkpoint sets the layout
        spec_path = rundir / "qw_spec.json"
        if spec_path.is_file():
            qw_spec = QwSpec(**json.loads(spec_path.read_text()))
            if fr_by_name is None:
                qw_raw, fr_by_name = build_qw_maps(
                    paths, case_params, load_qw_from_ledger(ledger))
        else:
            qw_spec = None
        # the shape of the model must match the checkpoint, so take it from the
        # arguments the run recorded
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
        # the layout in the checkpoint outranks the command-line defaults here
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
            # record every cheap scalar on every epoch. the run directory is the
            # only lasting record, and recovering a curve later costs a whole
            # session of accelerator quota.
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
            # rewritten each epoch, so an interrupted session still holds its curve
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

    # without training, args.slice_num, n_hidden, n_layers and n_head still hold
    # the command-line defaults rather than the shape of the model that ran.
    # overlay them from run_args, so the provenance record matches the checkpoint
    # that was loaded.
    args_out = dict(vars(args))
    if args.eval_only is not None:
        for k in ("slice_num", "n_hidden", "n_layers", "n_head"):
            args_out[k] = run_args[k]
    (out / "final_eval.json").write_text(json.dumps({
        "splits": {k: len(v) for k, v in splits.items()},
        # the names as well as the counts. without them a later run on a
        # refreshed dataset could not be lined up against this one case by case.
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
