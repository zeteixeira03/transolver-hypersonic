"""Physics-prior ablation: low-data split construction and the results table.

The surrogate carries three physics priors. Each one is switched off in turn
and priced against a baseline that keeps all three. The training split is small
on purpose, because a prior earns its place when data is scarce. On the full
dataset these rows would sit too close together to separate.

Cells
-----
``base``        pressure reconstructed, log10 on density and temperature
``free_p``      pressure as a free fifth channel, no equation-of-state constraint
``plain_norm``  plain standardization on every channel
``qw_direct``   a second head that predicts log10 of the wall heat flux
``qw_resid``    a second head that predicts log10 of q_w over the Fay-Riddell value

``qw_resid`` is measured against ``qw_direct`` and not against ``base``, because
``qw_direct`` is the control for the second head itself. Comparing against
``base`` would price the head and the Fay-Riddell anchor together.

Splits
------
Only the training split is small. The evaluation tiers take their exact case
lists from an existing run, so every cell is scored on the same held-out cases
at full size.

The training split is drawn from the core group alone, stratified by geometry.
It excludes the active-learning cases. Ensemble uncertainty selected those
adaptively, so they are not an independent and identically distributed sample
of the parameter box and would confound a comparison between priors.

The split file lives in ``configs/`` rather than under ``data/processed/``,
because training runs from a fresh clone and ``data/processed/`` is not
committed.

Usage
-----
    python scripts/prior_ablation.py make-splits \
        --reference data/processed/ensemble_v3/run_m32_v3_s0/per_case_eval.json \
        --out configs/prior_splits_100.json

    python scripts/prior_ablation.py table data/processed/prior_ablation/run_* \
        --out data/processed/prior_ablation/table.md
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.cfd.ledger import read_cases
from src.data.su2 import TARGET_ORDER

CELL_ORDER = ("base", "free_p", "plain_norm", "qw_direct", "qw_resid")
CELL_LABELS = {
    "base": "baseline (reconstructed p, log rho and T)",
    "free_p": "free p channel (no equation-of-state constraint)",
    "plain_norm": "plain standardization (no log)",
    "qw_direct": "q_w head, direct",
    "qw_resid": "q_w head, Fay-Riddell residual",
}


# ============================================================================================
#                                       split construction
# ============================================================================================

def build_splits(
    reference_per_case: Path,
    ledger: Path,
    n_train: int,
    seed: int,
) -> dict[str, list[str]]:
    """Take the evaluation tiers from a reference run and draw a small training split.

    Parameters
    ----------
    reference_per_case : Path
        ``per_case_eval.json`` from the run whose validation, test and
        interpolation membership is copied without change.
    ledger : Path
        The sweep ledger, read for the group and geometry ids.
    n_train : int
        Size of the small training split.
    seed : int
        Seed for the stratified draw. It is written into the file, so the split
        can be rebuilt.

    Returns
    -------
    dict
        ``{split_name: [case_name, ...]}`` for train, val, test and
        test_interp. The extrapolation slabs are absent, because each one is a
        whole ledger group and needs no draw.
    """
    per_case = json.loads(reference_per_case.read_text())
    splits = {name: [r["name"] for r in per_case[name]]
              for name in ("val", "test", "test_interp")}
    held = {n for names in splits.values() for n in names}

    by_geom: dict[int, list[str]] = defaultdict(list)
    for case_id, r in read_cases(ledger).items():
        name = f"case_{case_id:04d}"
        if r["status"] == "done" and r["group_name"] == "core" and name not in held:
            by_geom[int(r["geom_id"])].append(name)

    pool_size = sum(len(v) for v in by_geom.values())
    if pool_size < n_train:
        raise SystemExit(f"train pool has {pool_size} cases, need {n_train}")

    # take one case per geometry in turn, so a small draw still covers the
    # family of shapes instead of filling up from the geometries with the most
    # cases
    rng = random.Random(seed)
    geoms = sorted(by_geom)
    rng.shuffle(geoms)
    for g in geoms:
        rng.shuffle(by_geom[g])
    train: list[str] = []
    depth = 0
    while len(train) < n_train:
        for g in geoms:
            if depth < len(by_geom[g]):
                train.append(by_geom[g][depth])
                if len(train) == n_train:
                    break
        depth += 1
    splits["train"] = sorted(train)
    return splits


# ============================================================================================
#                                       loading runs
# ============================================================================================

def parse_run_dir(run_dir: Path) -> tuple[str, int]:
    """Split ``run_<cell>_s<seed>`` into its cell name and seed."""
    stem = run_dir.name
    if not stem.startswith("run_") or "_s" not in stem:
        raise ValueError(f"cannot parse cell/seed from {stem!r}; "
                         f"expected run_<cell>_s<seed>")
    body, _, seed = stem[len("run_"):].rpartition("_s")
    return body, int(seed)


def tier_mean_rl2(records: list[dict], channels: tuple[str, ...]) -> float | None:
    """Mean over cases of the per-case mean relative L2 across ``channels``."""
    if not records:
        return None
    return float(np.mean([
        np.mean([rec["rL2"][c] for c in channels]) for rec in records
    ]))


def qw_abs_rel_median(records: list[dict], key: str, truth_key: str) -> float | None:
    """Median absolute relative error of a heat-flux estimate against its own truth.

    Two heat-flux numbers exist per case and they are not interchangeable. The
    estimate read off the temperature field is scored against the same estimator
    applied to the true field. The head is scored against the ledger value it
    trained on. The caller names the truth key, so the two never cross. Crossing
    them would inflate the head error by the factor between the estimators.
    """
    rel = [abs(r[key] - r[truth_key]) / abs(r[truth_key])
           for r in records
           if r.get(key) is not None and r.get(truth_key)]
    return float(np.median(rel)) if rel else None


def summarize_run(run_dir: Path) -> dict:
    """Collect the tier metrics of one run directory into a single record."""
    per_case = json.loads((run_dir / "per_case_eval.json").read_text())
    final = json.loads((run_dir / "final_eval.json").read_text())

    ood = [r for name, recs in per_case.items()
           if name.startswith("ood_") for r in recs]
    in_dist = per_case.get("test_interp", []) + per_case.get("test", [])
    out = {
        "n_train": final["splits"]["train"],
        "interp": tier_mean_rl2(per_case["test_interp"], TARGET_ORDER),
        "family": tier_mean_rl2(per_case["test"], TARGET_ORDER),
        "ood": tier_mean_rl2(ood, TARGET_ORDER),
        "qw_posthoc": qw_abs_rel_median(in_dist, "qw_pred", "qw_true"),
        "qw_head": qw_abs_rel_median(in_dist, "qw_head_pred", "qw_ledger"),
    }
    if per_case["test"] and "p" in per_case["test"][0]["rL2"]:
        out["rL2_p"] = tier_mean_rl2(per_case["test"], ("p",))
    eos = [r["eos_viol"] for r in in_dist if "eos_viol" in r]
    out["eos_viol"] = float(np.median(eos)) if eos else 0.0
    # the worst of the channels that physics requires to stay positive. a
    # negative freestream density moves the relative L2 little, because those
    # nodes carry small magnitudes. without this column the table would rank an
    # unusable field above a valid one.
    nonpos = [max(r.get(f"frac_nonpos_{c}", 0.0) for c in ("rho", "T", "p"))
              for r in in_dist
              if any(f"frac_nonpos_{c}" in r for c in ("rho", "T", "p"))]
    out["nonpos"] = float(np.mean(nonpos)) if nonpos else None
    return out


# ============================================================================================
#                                       table
# ============================================================================================

def _cell(values: list[float | None]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"


def render_table(by_cell: dict[str, list[dict]]) -> str:
    n_trains = {r["n_train"] for runs in by_cell.values() for r in runs}
    n_seeds = {len(runs) for runs in by_cell.values()}
    header = (
        f"Physics-prior ablation, n_train = {sorted(n_trains)}, "
        f"{sorted(n_seeds)} seed(s) per cell.\n"
        "Mean relative L2 over (rho, u, v, T), so every cell is scored on the same\n"
        "channels, pooled over the two in-distribution tiers; cells show\n"
        "mean +/- std across seeds.\n\n"
        "The two q_w columns are median |relative error| against different\n"
        "truths and are not comparable to each other. Post-hoc scores the\n"
        "finite-difference estimate on the predicted field against the same\n"
        "estimate on the true field. The head scores against the ledger's\n"
        "SU2-postprocessed value, which is what it was trained on. Read each\n"
        "column down its own cells, never across.\n\n"
    )
    lines = [
        "| prior cell | interpolation | family holdout | pooled extrapolation "
        "| q_w post-hoc | q_w head | equation-of-state violation | nonphysical |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cell in CELL_ORDER:
        runs = by_cell.get(cell)
        if not runs:
            continue
        lines.append(
            f"| {CELL_LABELS[cell]} "
            f"| {_cell([r['interp'] for r in runs])} "
            f"| {_cell([r['family'] for r in runs])} "
            f"| {_cell([r['ood'] for r in runs])} "
            f"| {_cell([r['qw_posthoc'] for r in runs])} "
            f"| {_cell([r['qw_head'] for r in runs])} "
            f"| {_cell([r['eos_viol'] for r in runs])} "
            f"| {_cell([r['nonpos'] for r in runs])} |"
        )
    extra = [c for c in by_cell if c not in CELL_ORDER]
    for cell in sorted(extra):
        runs = by_cell[cell]
        lines.append(
            f"| {cell} | {_cell([r['interp'] for r in runs])} "
            f"| {_cell([r['family'] for r in runs])} "
            f"| {_cell([r['ood'] for r in runs])} "
            f"| {_cell([r['qw_posthoc'] for r in runs])} "
            f"| {_cell([r['qw_head'] for r in runs])} "
            f"| {_cell([r['eos_viol'] for r in runs])} "
            f"| {_cell([r['nonpos'] for r in runs])} |"
        )
    return header + "\n".join(lines) + "\n"


# ============================================================================================
#                                       cli
# ============================================================================================

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    mk = sub.add_parser("make-splits", help="write the pinned low-data split file")
    mk.add_argument("--reference", required=True,
                    help="per_case_eval.json whose evaluation tiers are copied")
    mk.add_argument("--workdir", default="data/raw/sweep")
    mk.add_argument("--n-train", type=int, default=100)
    mk.add_argument("--seed", type=int, default=0)
    mk.add_argument("--out", required=True)

    tb = sub.add_parser("table", help="render the ablation table from run directories")
    tb.add_argument("run_dirs", nargs="+")
    tb.add_argument("--out", default=None,
                    help="write the table to this file as well as to the console")

    args = p.parse_args()

    if args.command == "make-splits":
        splits = build_splits(Path(args.reference), Path(args.workdir) / "ledger.db",
                              args.n_train, args.seed)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(splits, indent=2))
        for name, names in splits.items():
            print(f"[splits] {name}: {len(names)}")
        print(f"[splits] wrote {out}")
        return

    by_cell: dict[str, list[dict]] = defaultdict(list)
    for d in args.run_dirs:
        run_dir = Path(d)
        if not (run_dir / "per_case_eval.json").is_file():
            print(f"[skip] {run_dir}: no per_case_eval.json")
            continue
        cell, _ = parse_run_dir(run_dir)
        by_cell[cell].append(summarize_run(run_dir))
    if not by_cell:
        raise SystemExit("no usable run directories")

    table = render_table(by_cell)
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table)
        print(f"[table] wrote {args.out}")


if __name__ == "__main__":
    main()
