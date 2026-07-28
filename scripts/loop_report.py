"""Before/after report for one active-learning loop iteration.

Compares two deep ensembles trained on the same split with the same recipe,
differing only in whether the loop-acquired SU2 cases were in the training
set (v2 without, v3 with). Both ensembles' per-case records come from
``scripts/ensemble_uq.py``, so this script is pure analysis and reruns in
seconds.

Measuring spread or error *at the acquired points* would be close to
tautological: those cases sit in v3's train split and were never in v2's, so
v3 is guaranteed to look better there. That is fitting, not generalization,
and this report deliberately does not use it as evidence. The three things
it does measure are all held out:

1. Paired tier comparison. Identical eval cases in both ensembles, so the
   per-case difference is paired and a signed-rank test applies.
2. Neighborhood generalization. Held-out error near the acquired points
   versus far from them, in normalized six-dimensional parameter distance.
   A working loop should show up here first: adding data at a point should
   help most in its neighborhood.
3. No-regression. How many held-out cases got worse, and by how much.

Distances run over the six free design axes (nose radius, cone half-angle,
the two aspect ratios, Mach, altitude), each divided by its core-box width,
so a distance of 1.0 means one full box width. Altitude is recovered from
the freestream pressure by inverting the atmosphere model.

Usage
-----
    python scripts/loop_report.py \
        --before data/processed/ensemble_v2/ensemble_per_case.json \
        --after data/processed/ensemble_v3/ensemble_per_case.json \
        --ledger data/raw/sweep/ledger.db \
        --out data/processed/loop/loop_report
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ensemble_uq import chan_mean, tier_label, tiers_present
from scripts.train import load_case_geom_ids, load_case_groups, split_paths
from src.data.sampler import FS_BOX, GEOM_BOX, us_standard_atmosphere
from src.data.su2 import list_case_npzs


# ============================================================================================
#                                       constants
# ============================================================================================

# the six free design axes, in the order the distance metric uses them, each
# paired with the core-box interval that normalizes it
SCAN_SPANS = {
    "R_n": GEOM_BOX["R_n"],
    "theta_c": GEOM_BOX["theta_c"],
    "R_b_ratio": GEOM_BOX["R_b_ratio"],
    "R_s_ratio": GEOM_BOX["R_s_ratio"],
    "mach": FS_BOX["mach"],
    "altitude": FS_BOX["altitude"],
}

# in-distribution tiers; the loop acquired inside the core box, so the OOD
# slabs are all far from every acquired point by construction and carry no
# neighborhood signal
NEIGHBORHOOD_TIERS = ("test", "test_interp")

_ALT_GRID = np.linspace(30.0, 80.0, 2001)
_P_GRID = np.array([us_standard_atmosphere(h)[1] for h in _ALT_GRID])


# ============================================================================================
#                                   parameter geometry
# ============================================================================================

def altitude_from_pressure(p_inf: float) -> float:
    """Geometric altitude in km whose standard-atmosphere pressure is ``p_inf``.

    Parameters
    ----------
    p_inf : float
        Freestream static pressure in Pa.

    Returns
    -------
    float
        Altitude in km, by interpolation on a precomputed monotone grid.
    """
    # pressure falls monotonically with altitude, so reverse both for np.interp
    return float(np.interp(p_inf, _P_GRID[::-1], _ALT_GRID[::-1]))


def scan_coords(params: dict[str, float]) -> np.ndarray:
    """Box-normalized six-axis coordinates of one case.

    Parameters
    ----------
    params : dict
        Case parameters keyed as in ``CASE_PARAM_ORDER``.

    Returns
    -------
    numpy.ndarray
        Length-6 vector; 0 and 1 are the core-box edges on each axis. OOD
        cases fall outside [0, 1], which is the intended behavior.
    """
    raw = {
        "R_n": params["R_n"],
        "theta_c": params["theta_c_deg"],
        "R_b_ratio": params["R_b"] / params["R_n"],
        "R_s_ratio": params["R_s"] / params["R_b"],
        "mach": params["mach"],
        "altitude": altitude_from_pressure(params["p_inf"]),
    }
    return np.array([(raw[a] - lo) / (hi - lo) for a, (lo, hi) in SCAN_SPANS.items()])


def load_loop_points(ledger: Path) -> np.ndarray:
    """Normalized coordinates of every converged loop-acquired case.

    Parameters
    ----------
    ledger : pathlib.Path
        SQLite ledger written by the generation pipeline.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_loop, 6).
    """
    cols = "R_n, theta_c_deg, R_b, R_s, mach, T_inf, p_inf, T_w"
    with sqlite3.connect(ledger) as con:
        rows = con.execute(
            f"select {cols} from cases where group_name = 'loop' and status = 'done'"
        ).fetchall()
    if not rows:
        raise SystemExit(f"no converged loop cases in {ledger}")
    keys = [c.strip() for c in cols.split(",")]
    return np.array([scan_coords(dict(zip(keys, r))) for r in rows])


def nearest_loop_distance(recs: list[dict], loop_pts: np.ndarray) -> np.ndarray:
    """Distance from each record to its closest acquired point."""
    X = np.array([scan_coords(r["case_params"]) for r in recs])
    return np.linalg.norm(X[:, None, :] - loop_pts[None, :, :], axis=2).min(axis=1)


def pre_loop_train_points(workdir: Path, seed: int = 0) -> np.ndarray:
    """Normalized coordinates of the training cases that predate the loop.

    Gives the distance bins a scale reference: if held-out cases sit farther
    from every acquired point than from the data the model already had, a
    local effect has little room to show up.

    Parameters
    ----------
    workdir : pathlib.Path
        Dataset workdir with case npzs and ledger.db.
    seed : int
        Split seed, matching the one both ensembles trained under.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_train_pre_loop, 6).
    """
    ledger = workdir / "ledger.db"
    paths = list_case_npzs(workdir)
    splits = split_paths(paths, load_case_groups(ledger), load_case_geom_ids(ledger),
                         0.1, 0.1, seed, 0.1)
    cols = "R_n, theta_c_deg, R_b, R_s, mach, T_inf, p_inf, T_w"
    keys = [c.strip() for c in cols.split(",")]
    with sqlite3.connect(ledger) as con:
        params = {f"case_{cid:04d}": dict(zip(keys, row)) for cid, *row
                  in con.execute(f"select case_id, {cols} from cases "
                                 f"where status = 'done' and group_name != 'loop'")}

    def case_name(p: Path) -> str:
        return p.parent.name if p.name == "case.npz" else p.stem

    names = [case_name(p) for p in splits["train"]]
    return np.array([scan_coords(params[n]) for n in names if n in params])


# ============================================================================================
#                                       pairing
# ============================================================================================

def pair_records(before: list[dict], after: list[dict]) -> list[dict]:
    """Match the two ensembles' records by case name and check tier agreement.

    Parameters
    ----------
    before : list of dict
        Per-case records of the control ensemble.
    after : list of dict
        Per-case records of the treatment ensemble.

    Returns
    -------
    list of dict
        One entry per shared case with ``err_before``, ``err_after``,
        ``spread_before``, ``spread_after``, the split, and the case params.
    """
    by_name = {r["name"]: r for r in before}
    paired = []
    for r in after:
        b = by_name.get(r["name"])
        if b is None:
            continue
        if b["split"] != r["split"]:
            raise SystemExit(f"{r['name']}: split {b['split']} before, {r['split']} "
                             f"after; the two ensembles do not share a split")
        paired.append({
            "name": r["name"],
            "split": r["split"],
            "case_params": r["case_params"],
            "err_before": chan_mean(b["rL2_ens"]),
            "err_after": chan_mean(r["rL2_ens"]),
            "spread_before": chan_mean(b["spread"]),
            "spread_after": chan_mean(r["spread"]),
        })
    n_only = len(before) - len(paired), len(after) - len(paired)
    if any(n_only):
        print(f"[pair] {n_only[0]} before-only and {n_only[1]} after-only cases dropped")
    return paired


def boot_ci(x: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% interval for the mean of ``x``."""
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ============================================================================================
#                                       tables
# ============================================================================================

def tier_table(paired: list[dict]) -> str:
    """Markdown: per-tier mean error before and after, with a signed-rank test."""
    lines = ["| tier | n | before | after | delta | 95% CI | p |",
             "|" + "---|" * 7]
    for split in tiers_present(paired):
        rows = [r for r in paired if r["split"] == split]
        b = np.array([r["err_before"] for r in rows])
        a = np.array([r["err_after"] for r in rows])
        d = a - b
        lo, hi = boot_ci(d)
        p = wilcoxon(d).pvalue if len(rows) >= 6 else float("nan")
        lines.append(f"| {tier_label(split)} | {len(rows)} | {b.mean():.3f} "
                     f"| {a.mean():.3f} | {d.mean():+.4f} "
                     f"| [{lo:+.4f}, {hi:+.4f}] | {p:.3f} |")
    return "\n".join(lines)


def neighborhood_table(paired: list[dict], loop_pts: np.ndarray,
                       train_pts: np.ndarray | None = None,
                       n_bins: int = 3) -> tuple[str, np.ndarray, np.ndarray]:
    """Markdown: held-out error by distance to the nearest acquired point.

    Parameters
    ----------
    paired : list of dict
        Paired records, filtered to the in-distribution tiers by the caller.
    loop_pts : numpy.ndarray
        Normalized coordinates of the acquired cases.
    train_pts : numpy.ndarray or None
        Normalized coordinates of the pre-loop training cases, used only to
        report how far the acquired points sit relative to data the model
        already had.
    n_bins : int
        Number of equal-count distance bins.

    Returns
    -------
    tuple
        The markdown table, the per-case distances, and the per-case deltas.
    """
    dist = nearest_loop_distance(paired, loop_pts)
    b = np.array([r["err_before"] for r in paired])
    a = np.array([r["err_after"] for r in paired])
    d = a - b

    edges = np.quantile(dist, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    lines = ["| distance to nearest acquired point | n | before | after | delta | 95% CI |",
             "|" + "---|" * 6]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if not m.any():
            continue
        clo, chi = boot_ci(d[m])
        lines.append(f"| {lo:.3f} to {hi:.3f} | {int(m.sum())} | {b[m].mean():.3f} "
                     f"| {a[m].mean():.3f} | {d[m].mean():+.4f} "
                     f"| [{clo:+.4f}, {chi:+.4f}] |")
    rho = spearmanr(dist, d)
    lines.append(f"\nSpearman(distance, delta) = {rho.statistic:.2f} "
                 f"(p = {rho.pvalue:.3f}, n = {len(dist)}). A working loop makes "
                 f"this positive: the improvement shrinks with distance.")
    if train_pts is not None:
        ref = nearest_loop_distance(paired, train_pts)
        lines.append(
            f"\nScale: the median held-out case sits {np.median(dist):.3f} from the "
            f"nearest of the {len(loop_pts)} acquired cases but only "
            f"{np.median(ref):.3f} from the nearest of the {len(train_pts)} cases "
            f"already in training. Twenty-six points is sparse cover for a "
            f"six-dimensional box, so most held-out cases have no acquired point "
            f"close enough for a local effect to reach them."
        )
    return "\n".join(lines), dist, d


def regression_table(paired: list[dict], n_worst: int = 5) -> str:
    """Markdown: how many held-out cases got worse, and the worst of them."""
    d = np.array([r["err_after"] - r["err_before"] for r in paired])
    worse = int((d > 0).sum())
    lines = [f"{worse} of {len(d)} held-out cases got worse ({worse / len(d):.0%}); "
             f"mean delta {d.mean():+.4f}, median {np.median(d):+.4f}.",
             "",
             "| case | tier | before | after | delta |",
             "|" + "---|" * 5]
    for i in np.argsort(-d)[:n_worst]:
        r = paired[i]
        lines.append(f"| {r['name']} | {tier_label(r['split'])} | {r['err_before']:.3f} "
                     f"| {r['err_after']:.3f} | {d[i]:+.4f} |")
    return "\n".join(lines)


# ============================================================================================
#                                       figure
# ============================================================================================

def neighborhood_figure(dist: np.ndarray, delta: np.ndarray, fig_out: Path,
                        n_bins: int = 6) -> None:
    """Per-case error change against distance to the nearest acquired point."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(dist, delta, s=18, alpha=0.6, color="steelblue",
               label="held-out case")
    edges = np.quantile(dist, np.linspace(0, 1, n_bins + 1))
    mids, meds = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist <= hi)
        if m.any():
            mids.append(0.5 * (lo + hi))
            meds.append(np.median(delta[m]))
    ax.plot(mids, meds, "k--", lw=1.2, marker="o", ms=4, label="binned median")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("normalized 6D distance to the nearest acquired case")
    ax.set_ylabel("change in per-case mean rel-L2 (after minus before)")
    ax.set_title("Where the acquired cases helped")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)


# ============================================================================================
#                                       main
# ============================================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="active-learning loop before/after report")
    p.add_argument("--before", default="data/processed/ensemble_v2/ensemble_per_case.json")
    p.add_argument("--after", default="data/processed/ensemble_v3/ensemble_per_case.json")
    p.add_argument("--workdir", default="data/raw/sweep",
                   help="dataset workdir with case npzs and ledger.db")
    p.add_argument("--out", default="data/processed/loop/loop_report")
    p.add_argument("--fig-out", default="data/samples/loop_neighborhood.png")
    args = p.parse_args()

    workdir = Path(args.workdir)
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())
    paired = pair_records(before, after)
    loop_pts = load_loop_points(workdir / "ledger.db")
    train_pts = pre_loop_train_points(workdir)
    print(f"[loop] {len(loop_pts)} acquired cases, {len(train_pts)} pre-loop "
          f"training cases, {len(paired)} paired eval cases")

    held_out = [r for r in paired if r["split"] != "val"]
    in_dist = [r for r in paired if r["split"] in NEIGHBORHOOD_TIERS]

    print("\n## Paired tier comparison (mean rel-L2 of the ensemble mean)\n")
    print(tier_table(paired))
    print("\n## Neighborhood generalization (in-distribution held-out cases)\n")
    nb_table, dist, delta = neighborhood_table(in_dist, loop_pts, train_pts)
    print(nb_table)
    print("\n## No-regression check (all held-out tiers)\n")
    print(regression_table(held_out))

    neighborhood_figure(dist, delta, Path(args.fig_out))
    print(f"\nneighborhood figure -> {args.fig_out}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "paired_per_case.json").write_text(json.dumps(paired, indent=2))
    print(f"paired records -> {out / 'paired_per_case.json'}")


if __name__ == "__main__":
    main()
