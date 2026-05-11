"""
Phase 1 acceptance figure: predicted vs ground-truth flow on one held-out
airfoil. Two panels: surface Cp(x) on the left, |U| field on the right.

Pressure in AirfRANS is stored as p / specific_mass (m^2/s^2). The
freestream pressure is zero in this representation, so the pressure
coefficient simplifies to

    Cp = (p / rho) / (0.5 * U_inf^2).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.airfrans import AirfRANSDataset, NormStats, denormalize_targets


# ============================================================================================
#                                       cp helper
# ============================================================================================
def surface_cp(
    pred_or_true: torch.Tensor,
    surface: torch.Tensor,
    inlet_velocity: torch.Tensor,
) -> np.ndarray:
    """
    Pressure coefficient on airfoil surface points.

    Parameters
    ----------
    pred_or_true : torch.Tensor
        Per-point target tensor (N, 4) in physical units. Column 2 is p/rho.
    surface : torch.Tensor
        Boolean mask of shape (N,) marking airfoil surface points.
    inlet_velocity : torch.Tensor
        Per-point inlet velocity (N, 2). Magnitude is uniform per simulation.

    Returns
    -------
    np.ndarray
        Cp values of length M = surface.sum().
    """
    if not surface.any():
        return np.array([])
    p_over_rho = pred_or_true[surface, 2].cpu().numpy()
    u_inf = inlet_velocity[surface].pow(2).sum(dim=-1).sqrt().mean().item()
    return p_over_rho / (0.5 * u_inf ** 2)


# ============================================================================================
#                                       acceptance figure
# ============================================================================================
@torch.no_grad()
def plot_acceptance(
    model: torch.nn.Module,
    dataset: AirfRANSDataset,
    stats: NormStats,
    device: str,
    sim_index: int = 0,
    save_path: Path | str | None = None,
) -> tuple[plt.Figure, dict[str, float]]:
    """
    Build the Phase 1 two-panel acceptance figure for one held-out
    simulation. Returns the figure and a dict of relative-L2 metrics shown
    in the panel titles.
    """
    model.eval()
    saved_subsample = dataset.subsample
    dataset.subsample = None
    try:
        item = dataset[sim_index]
    finally:
        dataset.subsample = saved_subsample

    x = item["x"].unsqueeze(0).to(device)
    pos = item["pos"].unsqueeze(0).to(device)
    y_true = item["y_raw"].to(device)
    surface = item["surface"].to(device)
    pos_np = item["pos"].cpu().numpy()
    inlet = item["x"][:, 2:4] * stats.x_std[2:4] + stats.x_mean[2:4]

    pred = denormalize_targets(model(x, pos=pos).squeeze(0), stats.to(device))

    # rel-L2 metrics for titles
    num = (pred - y_true).pow(2).sum(dim=0).sqrt()
    den = y_true.pow(2).sum(dim=0).sqrt().clamp_min(1e-12)
    rl2_full = (num / den).cpu().numpy()

    # surface rL2 only computed for pressure: u, v are ~0 by no-slip and
    # nu_t is ~0 on the wall, so per-channel rel-L2 there is meaningless
    if surface.any():
        p_pred_s = pred[surface, 2]
        p_true_s = y_true[surface, 2]
        num_s = (p_pred_s - p_true_s).pow(2).sum().sqrt()
        den_s = p_true_s.pow(2).sum().sqrt().clamp_min(1e-12)
        rl2_surf_p = float(num_s / den_s)
    else:
        rl2_surf_p = float("nan")

    # panel 1: surface Cp(x), split by surface-normal y-sign so upper and
    # lower surface plot as two separate single-valued curves instead of a
    # zigzag through both. surface normals are in raw arr cols 5..6.
    surf_mask = item["surface"].cpu().numpy()
    surface_xy = pos_np[surf_mask]
    raw = dataset.arrays[sim_index]
    normal_y = raw[surf_mask, 6]
    upper = normal_y >= 0
    lower = ~upper
    cp_true = surface_cp(y_true, surface, inlet.to(device))
    cp_pred = surface_cp(pred, surface, inlet.to(device))

    # panel 2: |U| field
    u_true = y_true[:, :2].pow(2).sum(dim=-1).sqrt().cpu().numpy()
    u_pred = pred[:, :2].pow(2).sum(dim=-1).sqrt().cpu().numpy()

    fig = plt.figure(figsize=(13, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    for mask, lbl_t, lbl_p, style_t, style_p in [
        (upper, "true upper", "pred upper", "k-", "r--"),
        (lower, "true lower", "pred lower", "k:", "r-."),
    ]:
        if mask.any():
            order = np.argsort(surface_xy[mask, 0])
            ax1.plot(surface_xy[mask, 0][order], cp_true[mask][order], style_t, lw=1.2, label=lbl_t)
            ax1.plot(surface_xy[mask, 0][order], cp_pred[mask][order], style_p, lw=1.0, label=lbl_p)
    ax1.invert_yaxis()
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("Cp")
    ax1.set_title(f"surface Cp, surf rL2(p) = {rl2_surf_p:.3f}")
    ax1.grid(alpha=0.3)
    ax1.legend()

    vmin = min(u_true.min(), u_pred.min())
    vmax = max(u_true.max(), u_pred.max())
    ax2 = fig.add_subplot(1, 4, 3)
    ax2.scatter(pos_np[:, 0], pos_np[:, 1], c=u_true, s=0.3, vmin=vmin, vmax=vmax, cmap="viridis")
    ax2.set_aspect("equal")
    ax2.set_title("|U| ground truth")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax3 = fig.add_subplot(1, 4, 4)
    sc = ax3.scatter(pos_np[:, 0], pos_np[:, 1], c=u_pred, s=0.3, vmin=vmin, vmax=vmax, cmap="viridis")
    ax3.set_aspect("equal")
    ax3.set_title(f"|U| prediction, full rL2(u,v) = {rl2_full[0]:.3f}, {rl2_full[1]:.3f}")
    ax3.set_xlabel("x")
    fig.colorbar(sc, ax=ax3, fraction=0.04)

    fig.suptitle(f"AirfRANS held-out: {item['name']}")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    metrics = {f"rL2_y{c}": float(rl2_full[c]) for c in range(4)}
    metrics["rL2_surf_p"] = rl2_surf_p
    return fig, metrics
