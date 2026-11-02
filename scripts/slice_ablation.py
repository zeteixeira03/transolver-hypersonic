"""Slice-count ablation tables and the guard threshold on envelope distance.

Reads ``final_eval.json`` and ``per_case_eval.json`` from one
``scripts/train.py`` run directory per slice count M. Prints the mean relative
L2 per tier against M, the error per channel and tier for one M, and the guard
threshold: the envelope distance where the binned median error passes twice
that of the cases inside the box.

Usage
-----
    python scripts/slice_ablation.py results/slice_ablation/run_m*
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.analytical import knudsen_number
from src.data.sampler import KN_MAX
from src.data.su2 import TARGET_ORDER
from src.eval.metrics import channel_mean, tier_label
from src.eval.uncertainty import envelope_distance

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


def ood_splits(run: dict) -> list[str]:
    return [s for s in run["final"] if s.startswith("ood_")]


def pooled_ood_mean(run: dict) -> float:
    """Mean relative L2 per case, pooled over every extrapolation slab present."""
    vals = [np.mean([r["rL2"][c] for c in TARGET_ORDER])
            for s in ood_splits(run) for r in run["per_case"][s]]
    return float(np.mean(vals))


# ============================================================================================
#                                       envelope distance
# ============================================================================================

def per_case_table(run: dict) -> list[dict]:
    """Flatten the per-case records of every evaluation split into one list.

    Each row gains its envelope distance, its Knudsen number, and its tier.
    """
    rows = []
    for split, recs in run["per_case"].items():
        if split == "val":
            continue                          # the split that selects the model, not a tier
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
    """Render the mean relative L2 per tier against slice count, as Markdown.

    The best value in each column comes out in bold.
    """
    runs = sorted(runs, key=lambda r: r["slice_num"])
    slabs = sorted({s for r in runs for s in ood_splits(r)})
    cols = ["test_interp", "test", *slabs]
    rows = []
    for r in runs:
        row = {"M": r["slice_num"]}
        for c in cols:
            row[c] = channel_mean(r["final"][c], "rL2_") if c in r["final"] else None
        row["ood_pooled"] = pooled_ood_mean(r)
        rows.append(row)

    header = ["M", "interpolation", "family holdout",
              *[s.removeprefix("ood_") for s in slabs], "pooled extrapolation"]
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
    """Render the relative L2 per channel across every tier, for one slice count."""
    order = ["test_interp", "test", *sorted(ood_splits(run))]
    lines = ["| tier | " + " | ".join(TARGET_ORDER)
             + " | mean | n | median q_w error |",
             "|" + "---|" * (len(TARGET_ORDER) + 4)]
    for s in order:
        if s not in run["final"]:
            continue
        m = run["final"][s]
        label = tier_label(s)
        qw = m.get("qw_abs_rel_err_median")
        lines.append(
            f"| {label} | "
            + " | ".join(f"{m[f'rL2_{c}']:.3f}" for c in TARGET_ORDER)
            + f" | {channel_mean(m, 'rL2_'):.3f} | {len(run['per_case'][s])} | "
            + (f"{qw:+.0%} |".replace("+", "") if qw is not None else "- |")
        )
    return "\n".join(lines)


# ============================================================================================
#                                       guard threshold
# ============================================================================================

def guard_threshold(run: dict, guard_factor: float = 2.0) -> float:
    """The smallest envelope distance where the binned median error passes
    ``guard_factor`` times the median of the cases inside the box.

    The binned medians are joined by straight lines, and the threshold is where
    that line first crosses. Returns nan when it never does.
    """
    rows = per_case_table(run)
    base = float(np.median([r["mean_rL2"] for r in rows if r["split"] in ("test", "test_interp")]))
    out_rows = sorted((r for r in rows if r["dist"] > 0), key=lambda r: r["dist"])
    if not out_rows:
        return float("nan")
    trend_x, trend_y = [0.0], [base]
    edges = np.quantile([r["dist"] for r in out_rows],
                        np.linspace(0, 1, min(6, max(2, len(out_rows) // 4 + 1))))
    for lo, hi in zip(edges[:-1], edges[1:]):
        binned = [r["mean_rL2"] for r in out_rows if lo <= r["dist"] <= hi]
        if binned:
            trend_x.append(float((lo + hi) / 2))
            trend_y.append(float(np.median(binned)))
    thresh = guard_factor * base
    for (x0, y0), (x1, y1) in zip(zip(trend_x, trend_y), zip(trend_x[1:], trend_y[1:])):
        if y0 <= thresh < y1:
            return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)
    return float("nan")


# ============================================================================================
#                                       main
# ============================================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="slice-count ablation + envelope analysis")
    p.add_argument("runs", nargs="+", help="run directories, one per slice count")
    p.add_argument("--envelope-m", type=int, default=None,
                   help="slice count for the tier table and the guard threshold; "
                        "the default takes the best family holdout")
    p.add_argument("--guard-factor", type=float, default=2.0)
    args = p.parse_args()

    runs = [load_run(Path(d)) for d in args.runs]
    print("## Slice-count ablation (mean relative L2 per tier)\n")
    print(ablation_table(runs))

    if args.envelope_m is not None:
        chosen = next(r for r in runs if r["slice_num"] == args.envelope_m)
    else:
        chosen = min(runs, key=lambda r: channel_mean(r["final"]["test"], "rL2_"))
    print(f"\n## Tier table (M = {chosen['slice_num']})\n")
    print(tier_table(chosen))

    guard = guard_threshold(chosen, args.guard_factor)
    print(f"\nguard threshold: envelope distance {guard:.3f} or Kn > {KN_MAX}")

    kn_near = [r for r in per_case_table(chosen) if r["kn"] > 0.5 * KN_MAX]
    if kn_near:
        print(f"{len(kn_near)} evaluation cases sit above Kn {0.5 * KN_MAX} "
              f"(largest {max(r['kn'] for r in kn_near):.4f}); "
              f"the continuum margin is thin there")


if __name__ == "__main__":
    main()
