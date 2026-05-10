"""
Phase 1 training and evaluation loop for AirfRANS.

Loss matches the upstream Transolver/AirfRANS recipe: MSE on volume points
plus a configurable weight on MSE over surface points. Default weight 1.0
mirrors thuml/Transolver's `--weight 1` setting.
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
    amp: bool = True  # mixed-precision on CUDA per CLAUDE.md default
    grad_clip: float = 1.0  # global L2 grad-norm clip, disables if <= 0


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
    Volume MSE plus surface_weight * surface MSE, computed point-wise.

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
#                                       collate (single-sample batches)
# ============================================================================================
def collate_single(batch: list[dict]) -> dict:
    """Collate for batch_size=1; just promote to (1, N, C)."""
    if len(batch) != 1:
        raise NotImplementedError("Phase 1 uses batch_size=1; meshes have variable N.")
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
        # loss in fp32 regardless of AMP, MSE on normalized targets is cheap
        loss, loss_vol, loss_surf = mse_weighted(
            pred.float(), y.float(), surface, cfg.surface_weight
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at step {n}: total={float(loss.detach())}; "
                "training diverged, consider lower lr or disabling amp"
            )

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
    Evaluate on each held-out simulation at full resolution (no subsampling).
    Reports per-channel relative L2 in physical units.

    Relative L2 per channel c on simulation s:
        rL2_c(s) = || y_pred_c - y_true_c ||_2 / || y_true_c ||_2

    Returns the mean over simulations.
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

            if surface.any():
                num_s = (pred[surface] - y_true[surface]).pow(2).sum(dim=0).sqrt()
                den_s = y_true[surface].pow(2).sum(dim=0).sqrt().clamp_min(1e-12)
                per_sim_surf.append((num_s / den_s).cpu())

        rl2 = torch.stack(per_sim).mean(dim=0)
        out = {f"rL2_y{c}": float(rl2[c]) for c in range(rl2.shape[0])}
        if per_sim_surf:
            rl2s = torch.stack(per_sim_surf).mean(dim=0)
            for c in range(rl2s.shape[0]):
                out[f"rL2_surf_y{c}"] = float(rl2s[c])
        return out
    finally:
        dataset.subsample = saved_subsample
