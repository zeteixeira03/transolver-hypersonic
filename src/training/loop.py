"""
Training and evaluation loop, written against AirfRANS first.

The loss follows the recipe the Transolver authors use on AirfRANS. It is the
mean squared error over volume points plus a weighted mean squared error over
surface points. The default weight of 1.0 matches the ``--weight 1`` setting of
thuml/Transolver.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.airfrans import AirfRANSDataset, NormStats, denormalize_targets


# ============================================================================================
#                                       config
# ============================================================================================
@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    surface_weight: float = 1.0
    batch_size: int = 1
    device: str = "cuda"
    log_every: int = 1
    val_every: int = 10
    amp: bool = True        # mixed precision, on a CUDA device only
    grad_clip: float = 1.0  # global L2 gradient-norm clip; 0 or less turns it off


# ============================================================================================
#                                       loss
# ============================================================================================
def mse_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    surface: torch.Tensor,
    surface_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The volume mean squared error plus ``surface_weight`` times the surface one.

    Both terms are computed over points.

    pred, target : (B, N, C)
    surface      : (B, N) boolean
    """
    err = (pred - target) ** 2
    surface = surface.unsqueeze(-1).expand_as(err)
    surface_err = err[surface]
    volume_err = err[~surface]
    loss_surf = surface_err.mean() if surface_err.numel() > 0 else err.new_tensor(0.0)
    loss_vol = volume_err.mean() if volume_err.numel() > 0 else err.new_tensor(0.0)
    total = loss_vol + surface_weight * loss_surf
    return total, loss_vol.detach(), loss_surf.detach()


# ============================================================================================
#                                   collate one sample per batch
# ============================================================================================
def collate_single(batch: list[dict]) -> dict:
    """Collate one item into a batch by adding the leading axis, to (1, N, C).

    A batch holds one mesh, because the node count differs from mesh to mesh.
    """
    if len(batch) != 1:
        raise NotImplementedError("batch_size must be 1; meshes have variable N.")
    item = batch[0]
    return {
        "x": item["x"].unsqueeze(0),
        "y": item["y"].unsqueeze(0),
        "y_raw": item["y_raw"].unsqueeze(0),
        "pos": item["pos"].unsqueeze(0),
        "surface": item["surface"].unsqueeze(0),
        "name": item["name"],
    }


# ============================================================================================
#                                       train one epoch
# ============================================================================================
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    n = 0
    sum_total = sum_vol = sum_surf = 0.0
    use_amp = cfg.amp and str(cfg.device).startswith("cuda")
    for batch in loader:
        x = batch["x"].to(cfg.device)
        y = batch["y"].to(cfg.device)
        surface = batch["surface"].to(cfg.device)
        pos = batch["pos"].to(cfg.device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x, pos=pos)
        # the loss runs in float32 even under mixed precision. a mean squared
        # error over normalized targets costs little and stays well conditioned.
        loss, loss_vol, loss_surf = mse_weighted(
            pred.float(), y.float(), surface, cfg.surface_weight
        )

        if not torch.isfinite(loss):
            name = batch.get("name", "<unknown>")
            xs = x.detach().float()
            ys = y.detach().float()
            ps = pred.detach().float()
            diag = (
                f"step={n} sim={name}\n"
                f"  x: finite={bool(torch.isfinite(xs).all())} "
                f"min={float(xs.min())} max={float(xs.max())} mean={float(xs.mean())}\n"
                f"  y: finite={bool(torch.isfinite(ys).all())} "
                f"min={float(ys.min())} max={float(ys.max())} mean={float(ys.mean())}\n"
                f"  pred: finite={bool(torch.isfinite(ps).all())} "
                f"min={float(ps[torch.isfinite(ps)].min()) if torch.isfinite(ps).any() else float('nan')} "
                f"max={float(ps[torch.isfinite(ps)].max()) if torch.isfinite(ps).any() else float('nan')}\n"
                f"  loss={float(loss.detach())}"
            )
            raise FloatingPointError(f"non-finite loss diagnostics:\n{diag}")

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        sum_total += float(loss.detach())
        sum_vol += float(loss_vol)
        sum_surf += float(loss_surf)
        n += 1

    return {
        "loss": sum_total / max(n, 1),
        "loss_vol": sum_vol / max(n, 1),
        "loss_surf": sum_surf / max(n, 1),
    }


# ============================================================================================
#                                       full-mesh evaluation
# ============================================================================================
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: AirfRANSDataset,
    stats: NormStats,
    device: str,
) -> dict[str, float]:
    """
    Evaluate every held-out simulation at full mesh resolution.

    The metric is the relative L2 per channel, in physical units. For channel c
    on simulation s it is

        || y_pred_c - y_true_c ||_2 / || y_true_c ||_2

    The return value is the mean over simulations.
    """
    model.eval()
    saved_subsample = dataset.subsample
    dataset.subsample = None
    try:
        per_sim = []
        per_sim_surf = []
        for i in range(len(dataset)):
            item = dataset[i]
            x = item["x"].unsqueeze(0).to(device)
            pos = item["pos"].unsqueeze(0).to(device)
            y_true = item["y_raw"].to(device)
            surface = item["surface"].to(device)

            pred_norm = model(x, pos=pos).squeeze(0)
            pred = denormalize_targets(pred_norm, stats.to(device))

            num = (pred - y_true).pow(2).sum(dim=0).sqrt()
            den = y_true.pow(2).sum(dim=0).sqrt().clamp_min(1e-12)
            per_sim.append((num / den).cpu())

            # only pressure, channel 2, carries a usable surface metric. the
            # no-slip condition holds both velocity components near zero at the
            # wall and the turbulent viscosity is near zero there too, so the
            # relative error would divide by an almost empty norm
            if surface.any():
                p_pred_s = pred[surface, 2]
                p_true_s = y_true[surface, 2]
                num_s = (p_pred_s - p_true_s).pow(2).sum().sqrt()
                den_s = p_true_s.pow(2).sum().sqrt().clamp_min(1e-12)
                per_sim_surf.append((num_s / den_s).cpu())

        rl2 = torch.stack(per_sim).mean(dim=0)
        out = {f"rL2_y{c}": float(rl2[c]) for c in range(rl2.shape[0])}
        if per_sim_surf:
            out["rL2_surf_p"] = float(torch.stack(per_sim_surf).mean())
        return out
    finally:
        dataset.subsample = saved_subsample
