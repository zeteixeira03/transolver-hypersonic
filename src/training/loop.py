"""
Training and evaluation loop for Transolver on SU2 and AirfRANS meshes.

The loss is the mean squared error over normalized targets. Given a surface
weight, it follows the recipe of the Transolver authors on AirfRANS instead:
the error over volume nodes plus that weight times the error over surface nodes.
A batch holds one mesh, because the node count differs from mesh to mesh.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.analytical import R_AIR
from src.data.su2 import (
    CASE_PARAM_ORDER,
    SU2Dataset,
    SU2NormStats,
    denormalize_targets,
)
from src.eval.metrics import relative_l2
from src.eval.sanity import compute_q_w_from_T, identify_wall_nodes

# ============================================================================================
#                                       training
# ============================================================================================

def collate_single(batch: list[dict]) -> dict:
    """Add a leading batch axis of size one to every tensor of a single item."""
    if len(batch) != 1:
        raise NotImplementedError("batch_size must be 1: meshes differ in node count")
    return {k: v.unsqueeze(0) if torch.is_tensor(v) and k != "case_params" else v
            for k, v in batch[0].items()}


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float,
    surface_weight: float = 0.0,
    qw_weight: float = 0.0,
) -> dict[str, float]:
    """One pass over the training split. Returns the mean losses and gradient norms.

    Above zero, ``surface_weight`` splits the error into volume and surface
    terms, and ``qw_weight`` adds the loss of the stagnation heat-flux head. A
    case without a heat flux in the ledger carries NaN there and drops out.
    """
    model.train()
    t0 = time.time()
    n, n_qw = 0, 0
    sums = {"loss": 0.0, "field_loss": 0.0, "qw_loss": 0.0, "grad_norm_mean": 0.0}
    max_gnorm = 0.0
    for batch in loader:
        x, y, pos = (batch[k].to(device) for k in ("x", "y", "pos"))
        optimizer.zero_grad()
        out = model(x, pos=pos)
        pred, qw_pred = out if qw_weight > 0 else (out, None)
        pred, y = pred.float(), y.float()
        if surface_weight > 0:
            surf = batch["surface"].to(device).unsqueeze(-1).expand_as(pred)
            err = (pred - y) ** 2
            field_loss = err[~surf].mean()
            if surf.any():
                field_loss = field_loss + surface_weight * err[surf].mean()
        else:
            field_loss = F.mse_loss(pred, y)
        loss = field_loss
        qw_loss = torch.zeros((), device=pred.device)
        if qw_weight > 0:
            qw_true = batch["qw"].to(device).float()
            mask = torch.isfinite(qw_true)
            if mask.any():
                qw_loss = F.mse_loss(qw_pred.float()[mask], qw_true[mask])
                loss = loss + qw_weight * qw_loss
                n_qw += 1
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at step {n}, case {batch.get('name', '?')}: "
                f"x in [{float(x.min()):.3g}, {float(x.max()):.3g}], "
                f"prediction in [{float(pred.min()):.3g}, {float(pred.max()):.3g}]"
            )
        loss.backward()
        # an infinite max_norm clips nothing and still returns the norm, which is
        # worth logging on every run
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip if grad_clip > 0 else float("inf")))
        optimizer.step()
        sums["loss"] += float(loss.detach())
        sums["field_loss"] += float(field_loss.detach())
        sums["qw_loss"] += float(qw_loss.detach())
        sums["grad_norm_mean"] += gnorm
        max_gnorm = max(max_gnorm, gnorm)
        n += 1
    out = {k: v / max(n, 1) for k, v in sums.items()}
    out.update(grad_norm_max=max_gnorm, n_steps=n, n_qw_steps=n_qw,
               epoch_time_s=time.time() - t0)
    return out


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
    qw_spec=None,
    fr_by_name: dict[str, float] | None = None,
    qw_ledger_by_name: dict[str, float] | None = None,
) -> tuple[dict, list[dict]]:
    """Score every case of a split at full mesh resolution.

    Each per-case record holds the relative L2 per channel in physical units,
    the fraction of nodes where density, temperature or pressure comes out
    nonpositive, and, for a free pressure channel, the median departure from
    ``p = rho R T``. It also holds a finite-difference stagnation heat flux read
    from the predicted and the true temperature fields, and, with ``qw_spec``,
    the output of the heat-flux head next to the value SU2 reported.

    Returns the split means and the per-case records.
    """
    model.eval()
    targets = stats.targets
    stats_dev = stats.to(device)
    per_case: list[dict] = []
    for i in range(len(dataset)):
        item = dataset[i]
        out = model(item["x"].unsqueeze(0).to(device), pos=item["pos"].unsqueeze(0).to(device))
        qw_head = None
        if qw_spec is not None:
            out, qw_batch = out
            qw_head = float(qw_batch.squeeze(0))
        pred = denormalize_targets(out.squeeze(0), stats_dev)
        y_true = item["y_raw"].to(device)
        rl2 = relative_l2(pred, y_true).cpu().numpy()
        params = {k: float(item["case_params"][j]) for j, k in enumerate(CASE_PARAM_ORDER)}
        rec = {"name": item["name"], "case_params": params,
               "rL2": {c: float(rl2[j]) for j, c in enumerate(targets)},
               "qw_true": None, "qw_pred": None}

        for j, c in enumerate(targets):
            if c in ("rho", "T", "p"):
                rec[f"frac_nonpos_{c}"] = float((pred[:, j] <= 0).float().mean())
        if "p" in targets:
            p_eos = pred[:, targets.index("rho")] * R_AIR * pred[:, targets.index("T")]
            p_ref = y_true[:, targets.index("p")].abs()
            rec["eos_viol"] = float(((pred[:, targets.index("p")] - p_eos).abs() / p_ref).median())

        name = item["name"]
        if fr_by_name is not None and name in fr_by_name:
            rec["qw_fay_riddell"] = fr_by_name[name]
        if qw_ledger_by_name is not None:
            rec["qw_ledger"] = qw_ledger_by_name.get(name)
        if qw_head is not None and fr_by_name is not None and name in fr_by_name:
            rec["qw_head_norm"] = qw_head
            rec["qw_head_pred"] = qw_spec.decode(qw_head, fr_by_name[name])

        xy = item["pos"].cpu().numpy()
        t_true = y_true[:, targets.index("T")].cpu().numpy()
        t_pred = pred[:, targets.index("T")].cpu().numpy()
        kw = dict(x=xy[:, 0], r=xy[:, 1], T_w=params["T_w"],
                  y_axis_skip=0.05 * params["R_n"], n_average=3)
        try:
            wall = identify_wall_nodes(t_true, params["T_w"])
            rec["qw_true"] = float(compute_q_w_from_T(T=t_true, wall_indices=wall, **kw)["q_w"])
            rec["qw_pred"] = float(compute_q_w_from_T(T=t_pred, wall_indices=wall, **kw)["q_w"])
        except ValueError:
            rec["qw_true"] = rec["qw_pred"] = None
        per_case.append(rec)

    return summarize(per_case, targets), per_case


def summarize(per_case: list[dict], targets: tuple[str, ...]) -> dict[str, float]:
    """Split means of the per-case records written by :func:`evaluate`."""
    rl2 = np.array([[r["rL2"][c] for c in targets] for r in per_case]).mean(axis=0)
    out = {f"rL2_{c}": float(rl2[j]) for j, c in enumerate(targets)}

    pairs = np.array([(r["qw_true"], r["qw_pred"]) for r in per_case if r["qw_true"] is not None])
    if len(pairs):
        rel = (pairs[:, 1] - pairs[:, 0]) / pairs[:, 0]
        out.update(qw_rel_err_mean=float(rel.mean()), qw_rel_err_median=float(np.median(rel)),
                   qw_abs_rel_err_median=float(np.median(np.abs(rel))), n_qw_cases=len(pairs))

    head = np.array([(r["qw_ledger"], r["qw_head_pred"]) for r in per_case
                     if r.get("qw_head_pred") is not None and r.get("qw_ledger")])
    if len(head):
        rel = (head[:, 1] - head[:, 0]) / head[:, 0]
        out.update(qw_head_rel_err_median=float(np.median(rel)),
                   qw_head_abs_rel_err_median=float(np.median(np.abs(rel))),
                   n_qw_head_cases=len(head))

    eos = [r["eos_viol"] for r in per_case if "eos_viol" in r]
    if eos:
        out["eos_viol_median"] = float(np.median(eos))
    for c in ("rho", "T", "p"):
        vals = [r[f"frac_nonpos_{c}"] for r in per_case if f"frac_nonpos_{c}" in r]
        if vals:
            out[f"frac_nonpos_{c}_mean"] = float(np.mean(vals))
    return out
