"""Plot helpers for sphere-cone fields and the AirfRANS acceptance figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colorbar import Colorbar
from matplotlib.path import Path as Polygon
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.tri import Triangulation

from src.data.airfrans import AirfRANSDataset, NormStats, denormalize_targets
from src.eval.metrics import relative_l2

# ============================================================================================
#                                   sphere-cone field plots
# ============================================================================================


def masked_triangulation(x: np.ndarray, r: np.ndarray, outline: np.ndarray) -> Triangulation:
    """Delaunay triangulation, masking triangles whose centroid lies inside ``outline``."""
    tri = Triangulation(x, r)
    centroids = np.c_[x[tri.triangles].mean(axis=1), r[tri.triangles].mean(axis=1)]
    tri.set_mask(Polygon(outline).contains_points(centroids))
    return tri


def decade_ticks(cb: Colorbar) -> None:
    """Put a log colour bar's ticks at 1 and 3 of every decade, written as plain numbers."""
    cb.locator = LogLocator(base=10, subs=(1.0, 3.0))
    cb.formatter = FuncFormatter(lambda v, _: f"{v:g}")
    cb.update_ticks()
    cb.minorticks_off()


# ============================================================================================
#                                       cp helper
# ============================================================================================
def surface_cp(
    pred_or_true: torch.Tensor,
    surface: torch.Tensor,
    inlet_velocity: torch.Tensor,
) -> np.ndarray:
    """
    Pressure coefficient at the airfoil points, ``(p / rho) / (0.5 U_inf^2)``.

    AirfRANS stores pressure over density (column 2 of the (N, 4) targets) with
    zero freestream pressure.
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
    Surface pressure coefficient and velocity magnitude for one held-out simulation.

    Returns the figure and the relative L2 values in its titles.
    """
    model.eval()
    item = dataset[sim_index]

    x = item["x"].unsqueeze(0).to(device)
    pos = item["pos"].unsqueeze(0).to(device)
    y_true = item["y_raw"].to(device)
    surface = item["surface"].to(device)
    pos_np = item["pos"].cpu().numpy()
    inlet = item["x"][:, 2:4] * stats.x_std[2:4] + stats.x_mean[2:4]

    pred = denormalize_targets(model(x, pos=pos).squeeze(0), stats.to(device))

    rl2_full = relative_l2(pred, y_true).cpu().numpy()
    rl2_surf_p = (float(relative_l2(pred[surface, 2:3], y_true[surface, 2:3])[0])
                  if surface.any() else float("nan"))

    # the sign of the normal's y component (raw column 6) splits upper from lower
    surf_mask = item["surface"].cpu().numpy()
    surface_xy = pos_np[surf_mask]
    raw = dataset.arrays[sim_index]
    normal_y = raw[surf_mask, 6]
    upper = normal_y >= 0
    lower = ~upper
    cp_true = surface_cp(y_true, surface, inlet.to(device))
    cp_pred = surface_cp(pred, surface, inlet.to(device))

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
