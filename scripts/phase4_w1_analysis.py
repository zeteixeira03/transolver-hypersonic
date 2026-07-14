"""Slice-count ablation tables and the envelope-distance figure.

Consumes the ``final_eval.json`` + ``per_case_eval.json`` pairs written by
``scripts/phase4_train_su2.py`` (one run dir per slice count) and emits:

1. the ablation table, mean rel-L2 per tier as a function of slice count M,
   with the per-tier optimum flagged (does the best M shift core vs OOD?)
2. the tier table for one chosen M: per-channel rel-L2 across the three
   tiers (interpolation, family holdout, OOD slabs)
3. the error-vs-envelope-distance figure with the guard threshold marked

Envelope distance is the L-inf normalized box-exceedance over the case
parameters: for each parameter with train range [lo, hi], the distance is
max(0, (v - hi) / (hi - lo), (lo - v) / (hi - lo)), and the case distance is
the max over parameters. Cases inside the training box sit at 0. The guard
flag combines this with the Knudsen floor: flag any input with distance
above the threshold or Kn > 0.01 (the continuum-validity limit of the
ground truth itself).

Usage
-----
    python scripts/phase4_w1_analysis.py data/processed/w1/run_m* \
        --fig-out data/samples/phase4_w1_envelope.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytical import knudsen_number
from src.data.su2 import CASE_PARAM_ORDER, TARGET_ORDER


KN_MAX = 0.01                                 # continuum floor, matches src.data.sampler
TIER_LABELS = {
    "test_interp": "interpolation",
    "test": "family holdout",
}


# ============================================================================================
#                                       loading
# ============================================================================================

def load_run(run_dir: Path) -> dict:
    final = json.loads((run_dir / "final_eval.json").read_text())
    per_case = json.loads((run_dir / "per_case_eval.json").read_text())
    return {
        "dir": run_dir,
        "slice_num": int(final["args"]["slice_num"]),
        "final": final["final"],
        "envelope": final["train_envelope"],
        "per_case": per_case,
    }


def mean_rl2(split_metrics: dict) -> float:
    return float(np.mean([split_metrics[f"rL2_{c}"] for c in TARGET_ORDER]))


def ood_splits(run: dict) -> list[str]:
    return [s for s in run["final"] if s.startswith("ood_")]


def pooled_ood_mean(run: dict) -> float:
    """Per-case mean rel-L2 pooled over every OOD slab present."""
    vals = [np.mean([r["rL2"][c] for c in TARGET_ORDER])
            for s in ood_splits(run) for r in run["per_case"][s]]
    return float(np.mean(vals))


# ============================================================================================
#                                       envelope distance
# ============================================================================================

def envelope_distance(params: dict[str, float], envelope: dict[str, list[float]]) -> float:
    """L-inf normalized box-exceedance of one case over the train envelope."""
    d = 0.0
    for k in CASE_PARAM_ORDER:
        lo, hi = envelope[k]
        width = hi - lo
        if width <= 0:
            continue                          # constant parameter (T_w)
        d = max(d, (params[k] - hi) / width, (lo - params[k]) / width)
    return max(d, 0.0)


def per_case_table(run: dict) -> list[dict]:
    """Flatten per-case records across eval splits, adding distance, Kn, tier."""
    rows = []
    for split, recs in run["per_case"].items():
        if split == "val":
            continue                          # model-selection split, not an eval tier
        for r in recs:
            p = r["case_params"]
            rows.append({
                "name": r["name"],
                "split": split,
                "mean_rL2": float(np.mean([r["rL2"][c] for c in TARGET_ORDER])),
                "dist": envelope_distance(p, run["envelope"]),
                "kn": knudsen_number(p["mach"], p["T_inf"], p["p_inf"], p["R_n"]),
            })
    return rows


# ============================================================================================
#                                       tables
# ============================================================================================

def ablation_table(runs: list[dict]) -> str:
    """Markdown: mean rel-L2 per tier vs slice count, optimum starred per column."""
    runs = sorted(runs, key=lambda r: r["slice_num"])
    slabs = sorted({s for r in runs for s in ood_splits(r)})
    cols = ["test_interp", "test", *slabs]
    rows = []
    for r in runs:
        row = {"M": r["slice_num"]}
        for c in cols:
            row[c] = mean_rl2(r["final"][c]) if c in r["final"] else None
        row["ood_pooled"] = pooled_ood_mean(r)
        rows.append(row)

    header = ["M", "interp", "family-holdout", *[s.removeprefix("ood_") for s in slabs],
              "OOD pooled"]
    keys = ["M", "test_interp", "test", *slabs, "ood_pooled"]
    best = {k: min((row[k] for row in rows if row[k] is not None), default=None)
            for k in keys[1:]}
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for row in rows:
        cells = [str(row["M"])]
        for k in keys[1:]:
            if row[k] is None:
                cells.append("-")
            else:
                star = "**" if row[k] == best[k] else ""
                cells.append(f"{star}{row[k]:.3f}{star}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def tier_table(run: dict) -> str:
    """Markdown: per-channel rel-L2 across tiers for one slice count."""
    order = ["test_interp", "test", *sorted(ood_splits(run))]
    lines = ["| tier | " + " | ".join(TARGET_ORDER) + " | mean | n | qw med abs err |",
             "|" + "---|" * (len(TARGET_ORDER) + 4)]
    for s in order:
        if s not in run["final"]:
            continue
        m = run["final"][s]
        label = TIER_LABELS.get(s, s.removeprefix("ood_") + " (OOD)")
        qw = m.get("qw_abs_rel_err_median")
        lines.append(
            f"| {label} | "
            + " | ".join(f"{m[f'rL2_{c}']:.3f}" for c in TARGET_ORDER)
            + f" | {mean_rl2(m):.3f} | {len(run['per_case'][s])} | "
            + (f"{qw:+.0%} |".replace("+", "") if qw is not None else "- |")
        )
    return "\n".join(lines)


# ============================================================================================
#                                       envelope figure
# ============================================================================================

def envelope_figure(run: dict, fig_out: Path, guard_factor: float = 2.0) -> float:
    """Scatter mean rel-L2 vs envelope distance; return the guard threshold.

    The guard threshold is the smallest envelope distance at which the
    binned median error exceeds ``guard_factor`` times the family-holdout
    median. Cases at distance 0 (inside the box) anchor the baseline.
    """
    rows = per_case_table(run)
    in_box = [r for r in rows if r["split"] in ("test", "test_interp")]
    base = float(np.median([r["mean_rL2"] for r in in_box]))

    # binned medians over distance for the trend line and the threshold
    out_rows = sorted((r for r in rows if r["dist"] > 0), key=lambda r: r["dist"])
    guard_dist = None
    trend_x, trend_y = [0.0], [base]
    if out_rows:
        edges = np.quantile([r["dist"] for r in out_rows],
                            np.linspace(0, 1, min(6, max(2, len(out_rows) // 4 + 1))))
        for lo, hi in zip(edges[:-1], edges[1:]):
            binned = [r["mean_rL2"] for r in out_rows if lo <= r["dist"] <= hi]
            if not binned:
                continue
            trend_x.append(float((lo + hi) / 2))
            trend_y.append(float(np.median(binned)))
        # guard threshold: linear interpolation of the first trend crossing
        thresh = guard_factor * base
        for (x0, y0), (x1, y1) in zip(zip(trend_x, trend_y), zip(trend_x[1:], trend_y[1:])):
            if y0 <= thresh < y1:
                guard_dist = x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)
                break

    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = sorted({r["split"] for r in rows})
    for g in groups:
        pts = [r for r in rows if r["split"] == g]
        label = TIER_LABELS.get(g, g.removeprefix("ood_") + " (OOD)")
        ax.scatter([r["dist"] for r in pts], [r["mean_rL2"] for r in pts],
                   s=18, alpha=0.7, label=label)
    ax.plot(trend_x, trend_y, "k--", lw=1, label="binned median")
    ax.axhline(guard_factor * base, color="gray", lw=0.8, ls=":",
               label=f"{guard_factor:.0f}x holdout median")
    if guard_dist is not None:
        ax.axvline(guard_dist, color="crimson", lw=1.2,
                   label=f"guard threshold d = {guard_dist:.3f}")
    ax.set_xlabel("envelope distance (L-inf normalized box exceedance)")
    ax.set_ylabel("per-case mean rel-L2")
    ax.set_title(f"Error growth outside the training envelope (M = {run['slice_num']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)
    return guard_dist if guard_dist is not None else float("nan")


# ============================================================================================
#                                       main
# ============================================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="slice-count ablation + envelope analysis")
    p.add_argument("runs", nargs="+", help="run dirs, one per slice count")
    p.add_argument("--fig-out", default="data/samples/phase4_w1_envelope.png")
    p.add_argument("--envelope-m", type=int, default=None,
                   help="slice count for the tier table + figure (default: best family-holdout)")
    p.add_argument("--guard-factor", type=float, default=2.0)
    args = p.parse_args()

    runs = [load_run(Path(d)) for d in args.runs]
    print("## Slice-count ablation (mean rel-L2 per tier)\n")
    print(ablation_table(runs))

    if args.envelope_m is not None:
        chosen = next(r for r in runs if r["slice_num"] == args.envelope_m)
    else:
        chosen = min(runs, key=lambda r: mean_rl2(r["final"]["test"]))
    print(f"\n## Tier table (M = {chosen['slice_num']})\n")
    print(tier_table(chosen))

    guard = envelope_figure(chosen, Path(args.fig_out), args.guard_factor)
    print(f"\nenvelope figure -> {args.fig_out}")
    print(f"guard threshold: envelope distance {guard:.2f} or Kn > {KN_MAX}")

    kn_near = [r for r in per_case_table(chosen) if r["kn"] > 0.5 * KN_MAX]
    if kn_near:
        print(f"{len(kn_near)} eval cases sit above Kn {0.5 * KN_MAX} "
              f"(max {max(r['kn'] for r in kn_near):.4f}); continuum margin is thin there")


if __name__ == "__main__":
    main()
