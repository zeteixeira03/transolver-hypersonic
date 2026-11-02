"""Draw the README figures from the committed records and the local data.

Most figures need only ``results/``. The field comparison also needs the case
tensors in ``data/raw/sweep/cases`` and the ensemble in
``data/processed/ensemble_v3``. The normalization figure needs the tensors.

Usage:
    python scripts/figures.py                 # every figure into data/samples/
    python scripts/figures.py --only slice_count verdict_error
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from src.analytical import R_AIR
from src.data.sampler import FS_BOX, GEOM_BOX, us_standard_atmosphere
from src.data.su2 import TARGET_ORDER, load_case_npz
from src.eval.plots import decade_ticks, masked_triangulation
from src.eval.uncertainty import GUARD_DIST, decide, envelope_distance
from src.geometry.sphere_cone import body_outline

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "data/samples"

# ============================================================================================
#                                           style
# ============================================================================================

INK = "#1f2430"
MUTED = "#6b7280"
GRID = "#e5e7eb"

TIERS = {
    "test_interp":    ("interpolation", "#3b5b92"),
    "test":           ("family holdout", "#86a6d4"),
    "ood_cone_high":  ("high cone angle", "#c8553d"),
    "ood_mach_high":  ("high Mach", "#e0a030"),
    "ood_nose_large": ("large nose", "#2f8f7f"),
    "ood_nose_small": ("small nose", "#8e5ea2"),
}
VERDICT = {"trust": "#2e7d4f", "warn": "#c98a1b", "refuse": "#b23a3a"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def log_colorbar(fig: plt.Figure, mappable, ax: plt.Axes, label: str) -> None:
    """Horizontal colour bar under ``ax`` with ticks at 1 and 3 of every decade."""
    cb = fig.colorbar(mappable, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05)
    decade_ticks(cb)
    cb.set_label(label, color=INK)
    cb.ax.tick_params(colors=MUTED)


def mean_error(rl2: dict[str, float]) -> float:
    """Mean relative L2 over the four predicted channels."""
    return float(np.mean([rl2[c] for c in TARGET_ORDER]))


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"wrote data/samples/{name}.png")


def eval_records() -> list[dict]:
    """Per-case records of the published ensemble on the test tiers, validation left out."""
    recs = json.loads((RESULTS / "ensemble/per_case.json").read_text())
    return [r for r in recs if r["split"] in TIERS]


# ============================================================================================
#                                       field comparison
# ============================================================================================

def field_comparison() -> None:
    """SU2 (upper half) against the ensemble mean (lower half) on a held-out shape.

    The case is the family-holdout case at that tier's median error.
    """
    import torch

    from src.eval.uncertainty import load_ensemble, predict_on_nodes

    holdout = sorted((r for r in eval_records() if r["split"] == "test"),
                     key=lambda r: mean_error(r["rL2_ens"]))
    rec = holdout[len(holdout) // 2]
    case = load_case_npz(ROOT / "data/raw/sweep/cases" / f"{rec['name']}.npz")
    p = rec["case_params"]
    ens = load_ensemble(sorted((ROOT / "data/processed/ensemble_v3").glob("run_*")))
    coords = np.c_[case["x"], case["r"]]
    with torch.no_grad():
        pred = predict_on_nodes(ens, coords, p).mean(dim=0).numpy()
    truth = np.c_[[case[c] for c in TARGET_ORDER]].T
    rl2 = np.linalg.norm(pred - truth, axis=0) / np.linalg.norm(truth, axis=0)
    print(f"{rec['name']}: recomputed mean relative L2 {rl2.mean():.4f}, "
          f"record {mean_error(rec['rL2_ens']):.4f}")

    x, r = case["x"] * 1e3, case["r"] * 1e3
    outline = body_outline(p["R_n"], p["theta_c_deg"], p["R_b"], p["R_s"]) * 1e3
    upper = masked_triangulation(x, r, outline)
    lower = masked_triangulation(x, -r, outline)
    R_b = p["R_b"] * 1e3
    x_base = float(x.max())

    fields = [("T", "temperature [K]", "inferno"), ("rho", "density [kg/m³]", "viridis")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), gridspec_kw={"wspace": 0.08})
    for ax, (name, label, cmap) in zip(axes[:2], fields):
        i = TARGET_ORDER.index(name)
        lo = max(float(truth[:, i].min()), 1e-12)
        norm = LogNorm(lo, float(truth[:, i].max()))
        levels = np.geomspace(norm.vmin, norm.vmax, 64)
        kw = dict(levels=levels, norm=norm, cmap=cmap, extend="both")
        cf = ax.tricontourf(upper, np.clip(truth[:, i], lo, None), **kw)
        ax.tricontourf(lower, np.clip(pred[:, i], lo, None), **kw)
        log_colorbar(fig, cf, ax, label)

    # the velocity components are zero at the wall, so only the two positive
    # fields get a relative error
    i_t, i_rho = TARGET_ORDER.index("T"), TARGET_ORDER.index("rho")
    err_t = np.abs(pred[:, i_t] - truth[:, i_t]) / truth[:, i_t]
    err_rho = np.abs(pred[:, i_rho] - truth[:, i_rho]) / truth[:, i_rho]
    norm = LogNorm(1e-3, 1.0)
    levels = np.geomspace(1e-3, 1.0, 64)
    cf = axes[2].tricontourf(upper, np.clip(err_t, 1e-3, 1.0), levels=levels,
                             norm=norm, cmap="magma", extend="both")
    axes[2].tricontourf(lower, np.clip(err_rho, 1e-3, 1.0), levels=levels,
                        norm=norm, cmap="magma", extend="both")
    log_colorbar(fig, cf, axes[2], "relative error of the ensemble")

    tags = [("SU2", "ensemble"), ("SU2", "ensemble"), ("temperature", "density")]
    for ax, (top, bottom) in zip(axes, tags):
        ax.fill(outline[:, 0], outline[:, 1], color="#d1d5db", zorder=3)
        ax.axhline(0, color="white", lw=0.8, zorder=4)
        ax.set_xlim(x_base - 1.25 * R_b, x_base + 0.02 * R_b)
        ax.set_ylim(-1.25 * R_b, 1.25 * R_b)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.5, 1.02, top, transform=ax.transAxes, color=INK, ha="center", va="bottom")
        ax.text(0.5, -0.02, bottom, transform=ax.transAxes, color=INK, ha="center", va="top")
    save(fig, "field_comparison")


# ============================================================================================
#                                        design box
# ============================================================================================

def altitude_km(T_inf: np.ndarray, p_inf: np.ndarray) -> np.ndarray:
    """Altitude that gives this freestream pressure in the standard atmosphere."""
    grid = np.linspace(30.0, 80.0, 2001)
    p_grid = np.array([us_standard_atmosphere(h)[1] for h in grid])
    return np.interp(np.log(p_inf), np.log(p_grid[::-1]), grid[::-1])


def design_box() -> None:
    """Every solved case in geometry space and in flight space, with the core box drawn."""
    con = sqlite3.connect(RESULTS / "ledger.db")
    rows = con.execute("select group_name, R_n, theta_c_deg, mach, T_inf, p_inf "
                       "from cases where status = 'done'").fetchall()
    groups = {
        "core": ("core box", "#9ca3af"),
        "loop": ("active learning", INK),
        "cone_high": TIERS["ood_cone_high"],
        "mach_high": TIERS["ood_mach_high"],
        "nose_large": TIERS["ood_nose_large"],
        "nose_small": TIERS["ood_nose_small"],
    }
    fig, (ax_g, ax_f) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.28})
    for key, (label, color) in groups.items():
        sel = np.array([r[1:] for r in rows if r[0] == key], dtype=float)
        if not len(sel):
            continue
        style = dict(s=14, color=color, alpha=0.75, lw=0, label=f"{label} ({len(sel)})")
        if key == "loop":
            style.update(s=22, facecolor="none", edgecolor=color, lw=0.9, alpha=1.0)
            style.pop("color")
        ax_g.scatter(sel[:, 0] * 1e3, sel[:, 1], **style)
        ax_f.scatter(sel[:, 2], altitude_km(sel[:, 3], sel[:, 4]), **style)

    def box(ax, x, y):
        ax.add_patch(plt.Rectangle((x[0], y[0]), x[1] - x[0], y[1] - y[0], fill=False,
                                   ec=INK, lw=1.0, ls="--", zorder=0))

    box(ax_g, [1e3 * v for v in GEOM_BOX["R_n"]], GEOM_BOX["theta_c"])
    box(ax_f, FS_BOX["mach"], FS_BOX["altitude"])
    ax_g.set_xscale("log")
    ax_g.set_xticks([4, 10, 20, 50, 100])
    ax_g.set_xticklabels(["4", "10", "20", "50", "100"])
    ax_g.set_xlabel("nose radius [mm]")
    ax_g.set_ylabel("cone half-angle [degrees]")
    ax_f.set_xlabel("Mach number")
    ax_f.set_ylabel("altitude [km]")
    ax_f.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), labelcolor=INK)
    save(fig, "design_box")


# ============================================================================================
#                                     slice count and error
# ============================================================================================

def slice_count() -> None:
    """Error per tier against the number of physics-attention slices, one run per point."""
    counts = (8, 16, 32, 64)
    finals = [json.loads((RESULTS / f"slice_ablation/run_m{m}/final_eval.json").read_text())
              ["final"] for m in counts]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ends = []
    for key, (label, color) in TIERS.items():
        y = [mean_error({c: f[key][f"rL2_{c}"] for c in TARGET_ORDER}) for f in finals]
        inside = key in ("test_interp", "test")
        ax.plot(counts, y, color=color, lw=2.2 if inside else 1.6, marker="o", ms=4,
                ls="-" if inside else (0, (4, 2)))
        ends.append([np.log10(y[-1]), label, color])
    # 0.035 decades is about one label height
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.035)
    for log_y, label, color in ends:
        ax.annotate(label, (counts[-1], 10 ** log_y), xytext=(8, 0),
                    textcoords="offset points", va="center", color=color, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts)
    ax.set_xticklabels([str(m) for m in counts])
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.set_yticklabels(["0.1", "0.2", "0.3", "0.4"])
    ax.minorticks_off()
    ax.set_xlim(7, 64 * 1.08)
    ax.set_xlabel("slices M")
    ax.set_ylabel("mean relative error")
    save(fig, "slice_count")


def envelope_distance_figure() -> None:
    """Per-case error against distance outside the training box, at M = 32."""
    run = RESULTS / "slice_ablation/run_m32"
    envelope = json.loads((run / "final_eval.json").read_text())["train_envelope"]
    per_case = json.loads((run / "per_case_eval.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs, ys = [], []
    for key, (label, color) in TIERS.items():
        d = np.array([envelope_distance(c["case_params"], envelope) for c in per_case[key]])
        e = np.array([mean_error(c["rL2"]) for c in per_case[key]])
        jitter = np.where(d == 0, np.random.default_rng(0).uniform(-0.012, 0.012, len(d)), 0)
        ax.scatter(d + jitter, e, s=16, color=color, alpha=0.8, lw=0, label=label)
        xs.append(d)
        ys.append(e)
    d, e = np.concatenate(xs), np.concatenate(ys)
    edges = np.array([0.0, 1e-9, 0.1, 0.25, 0.5, 0.8, 1.4])
    mids, meds = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (d >= lo) & (d < hi)
        if sel.sum() >= 3:
            mids.append(np.median(d[sel]))
            meds.append(np.median(e[sel]))
    ax.plot(mids, meds, color=INK, lw=1.4)
    ax.axvspan(GUARD_DIST, d.max() * 1.05, color=VERDICT["refuse"], alpha=0.06, lw=0)
    ax.axvline(GUARD_DIST, color=VERDICT["refuse"], lw=1.2)
    ax.annotate("refuse", (GUARD_DIST, 0.95), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", color=VERDICT["refuse"])
    ax.set_yscale("log")
    ax.set_yticks([0.05, 0.1, 0.2, 0.5])
    ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5"])
    ax.minorticks_off()
    ax.set_xlim(-0.04, d.max() * 1.05)
    ax.set_xlabel("distance outside the training box")
    ax.set_ylabel("mean relative error")
    ax.legend(loc="lower right", ncol=2, fontsize=9, labelcolor=INK, markerscale=1.3)
    save(fig, "envelope_distance")


# ============================================================================================
#                                    uncertainty and verdict
# ============================================================================================

def thresholds() -> dict[str, float]:
    return json.loads((RESULTS / "ensemble/thresholds.json").read_text())


def spread_calibration() -> None:
    """Ensemble spread against actual error, with the warn and refuse bands."""
    th = thresholds()
    recs = eval_records()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for key, (label, color) in TIERS.items():
        sel = [r for r in recs if r["split"] == key]
        ax.scatter([mean_error(r["spread"]) for r in sel], [mean_error(r["rL2_ens"]) for r in sel],
                   s=16, color=color, alpha=0.85, lw=0, label=label)
    lo, hi = 0.04, 0.42
    ax.axvspan(th["warn"], th["refuse"], color=VERDICT["warn"], alpha=0.07, lw=0)
    ax.axvspan(th["refuse"], hi, color=VERDICT["refuse"], alpha=0.07, lw=0)
    for x, name in ((lo, "trust"), (th["warn"], "warn"), (th["refuse"], "refuse")):
        ax.annotate(name, (x, 0.97), xycoords=("data", "axes fraction"), xytext=(5, 0),
                    textcoords="offset points", color=VERDICT[name], va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.05, 0.1, 0.2, 0.3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticks([0.05, 0.1, 0.2, 0.5])
    ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5"])
    ax.minorticks_off()
    ax.set_xlim(lo, hi)
    ax.set_xlabel("disagreement between the five members")
    ax.set_ylabel("mean relative error")
    ax.legend(loc="lower right", ncol=2, fontsize=9, labelcolor=INK, markerscale=1.3)
    save(fig, "spread_calibration")


def verdict_error() -> None:
    """Actual error of every test case, grouped by the verdict the rule gives it."""
    th = thresholds()
    recs = eval_records()
    verdicts = [decide(r["dist"], r["kn"], mean_error(r["spread"]), th) for r in recs]
    print("verdict counts:", {v: verdicts.count(v) for v in VERDICT})
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, v in enumerate(VERDICT):
        sel = [r for r, d in zip(recs, verdicts) if d == v]
        e = np.array([mean_error(r["rL2_ens"]) for r in sel])
        colors = [TIERS[r["split"]][1] for r in sel]
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(e)), e, s=14, c=colors, alpha=0.85, lw=0)
        q50, q90 = np.percentile(e, [50, 90])
        ax.plot([i - 0.32, i + 0.32], [q50, q50], color=VERDICT[v], lw=2.6)
        ax.plot([i - 0.2, i + 0.2], [q90, q90], color=VERDICT[v], lw=1.2, ls=(0, (3, 2)))
    ax.set_xticks(range(len(VERDICT)))
    ax.set_xticklabels([f"{v}\n{verdicts.count(v)} cases" for v in VERDICT])
    for tick, v in zip(ax.get_xticklabels(), VERDICT):
        tick.set_color(VERDICT[v])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_yscale("log")
    ax.set_yticks([0.05, 0.1, 0.2, 0.5])
    ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5"])
    ax.minorticks_off()
    ax.set_ylabel("mean relative error")
    handles = [plt.Line2D([], [], color=MUTED, lw=2.6, label="median"),
               plt.Line2D([], [], color=MUTED, lw=1.2, ls=(0, (3, 2)),
                          label="90th percentile")]
    handles += [plt.Line2D([], [], marker="o", ls="", color=c, label=lab, ms=5)
                for lab, c in TIERS.values()]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9,
              labelcolor=INK)
    save(fig, "verdict_error")


# ============================================================================================
#                                       normalization
# ============================================================================================

def normalization_study(n_cases: int = 20) -> None:
    """Node values of three fields after plain and after log standardization."""
    paths = sorted((ROOT / "data/raw/sweep/cases").glob("case_*.npz"))[:n_cases]
    cases = [load_case_npz(p) for p in paths]
    rho = np.concatenate([c["rho"] for c in cases])
    T = np.concatenate([c["T"] for c in cases])
    fields = [("density", rho), ("temperature", T), ("pressure", rho * R_AIR * T)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.0), sharey=True, sharex="col",
                             gridspec_kw={"hspace": 0.12, "wspace": 0.08})
    for j, (name, v) in enumerate(fields):
        for i, w in enumerate((v, np.log10(v))):
            z = (w - w.mean()) / w.std()
            ax = axes[i, j]
            ax.hist(z, bins=120, color=INK if i else MUTED, lw=0)
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(name, color=INK, fontsize=10)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("standard deviations from the mean")
    axes[0, 0].set_ylabel("nodes\nstandardized")
    axes[1, 0].set_ylabel("nodes\nlog10, then standardized")
    save(fig, "normalization_study")


FIGURES = {
    "field_comparison": field_comparison,
    "design_box": design_box,
    "slice_count": slice_count,
    "envelope_distance": envelope_distance_figure,
    "spread_calibration": spread_calibration,
    "verdict_error": verdict_error,
    "normalization_study": normalization_study,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="+", choices=list(FIGURES), help="figures to draw")
    args = p.parse_args()
    for name in args.only or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    main()
