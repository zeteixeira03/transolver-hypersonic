"""Physics-prior ablation: low-data split construction and the results table.

Three priors that the surrogate normally carries are switched off one at a
time and priced against a baseline, on a deliberately small training split.
Priors are supposed to earn their place when data is scarce; on the full
dataset the rows would likely be indistinguishable.

Cells
-----
``base``        hard EoS (p reconstructed), log10 on rho and T, no q_w head
``free_p``      p emitted as a free fifth channel (EoS constraint relaxed)
``plain_norm``  plain standardization on every channel
``qw_direct``   auxiliary head predicting log10 q_w
``qw_resid``    auxiliary head predicting log10(q_w / q_FR)

The Fay-Riddell prior is measured by ``qw_resid`` against ``qw_direct``, not
against ``base``; ``qw_direct`` is the control for the auxiliary loss itself.

Splits
------
Only the train split is small. The evaluation tiers are pinned to an existing
run's exact case lists so every cell is scored on identical held-out cases at
full size. The train split is drawn from the core group alone, stratified by
geometry, and excludes the active-learning cases: those were chosen
adaptively by ensemble uncertainty, so they are not an iid sample of the
parameter box and would confound a comparison between priors.

The split file lives in ``configs/`` rather than under ``data/processed/``
because training runs on a cloned checkout and ``data/processed/`` is not
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
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.su2 import TARGET_ORDER


# common channels across every cell. the free-pressure arm predicts a fifth,
# and averaging over 5 channels against 4 would not be a like-for-like tier
# number, so the tier metric always uses these and p is reported on its own.
COMMON_CHANNELS = TARGET_ORDER

CELL_ORDER = ("base", "free_p", "plain_norm", "qw_direct", "qw_resid")
CELL_LABELS = {
    "base": "baseline (hard EoS, log rho/T)",
    "free_p": "free p channel (no EoS constraint)",
    "plain_norm": "plain standardization (no log)",
    "qw_direct": "q_w head, direct",
    "qw_resid": "q_w head, Fay-Riddell residual",
}
TIER_LABELS = {
    "test_interp": "interpolation",
    "test": "family holdout",
}
OOD_PREFIX = "ood_"


# ============================================================================================
#                                       split construction
# ============================================================================================

def build_splits(
    reference_per_case: Path,
    ledger: Path,
    n_train: int,
    seed: int,
) -> dict[str, list[str]]:
    """Pin the eval tiers to a reference run and subsample a small train split.

    Parameters
    ----------
    reference_per_case : Path
        ``per_case_eval.json`` from the run whose val/test/interp membership
        should be reused verbatim.
    ledger : Path
        Sweep ledger, read for group and geometry ids.
    n_train : int
        Size of the low-data train split.
    seed : int
        Seed for the stratified draw. Written into the file so the split is
        reproducible.

    Returns
    -------
    dict
        ``{split_name: [case_name, ...]}`` for train, val, test, test_interp.
        OOD slabs are left out; they are whole ledger groups and already
        deterministic.
    """
    per_case = json.loads(reference_per_case.read_text())
    splits = {name: [r["name"] for r in per_case[name]]
              for name in ("val", "test", "test_interp")}
    held = {n for names in splits.values() for n in names}

    con = sqlite3.connect(f"file:{ledger.as_posix()}?mode=ro&immutable=1", uri=True)
    rows = con.execute(
        "select case_id, geom_id from cases where status='done' and group_name='core'"
    ).fetchall()
    con.close()

    by_geom: dict[int, list[str]] = defaultdict(list)
    for case_id, geom_id in rows:
        name = f"case_{int(case_id):04d}"
        if name not in held:
            by_geom[int(geom_id)].append(name)

    pool_size = sum(len(v) for v in by_geom.values())
    if pool_size < n_train:
        raise SystemExit(f"train pool has {pool_size} cases, need {n_train}")

    # round-robin across geometries so a small draw still spans the shape
    # family rather than piling onto whichever geometries have most cases
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
    """Mean over cases of the per-case mean rel-L2 across ``channels``."""
    if not records:
        return None
    return float(np.mean([
        np.mean([rec["rL2"][c] for c in channels]) for rec in records
    ]))


def qw_abs_rel_median(records: list[dict], key: str, truth_key: str) -> float | None:
    """Median |relative error| of a q_w estimate against its own truth.

    The post-hoc estimate is scored against the finite-difference value on
    the true field; the head is scored against the ledger value it was
    trained on. Passing the truth key explicitly keeps those from being
    crossed, which silently inflates the head error by the factor between
    the two estimators.
    """
    rel = [abs(r[key] - r[truth_key]) / abs(r[truth_key])
           for r in records
           if r.get(key) is not None and r.get(truth_key)]
    return float(np.median(rel)) if rel else None


def summarize_run(run_dir: Path) -> dict:
    """Tier metrics for one run directory."""
    per_case = json.loads((run_dir / "per_case_eval.json").read_text())
    final = json.loads((run_dir / "final_eval.json").read_text())

    ood = [r for name, recs in per_case.items()
           if name.startswith(OOD_PREFIX) for r in recs]
    in_dist = per_case.get("test_interp", []) + per_case.get("test", [])
    out = {
        "n_train": final["splits"]["train"],
        "interp": tier_mean_rl2(per_case["test_interp"], COMMON_CHANNELS),
        "family": tier_mean_rl2(per_case["test"], COMMON_CHANNELS),
        "ood": tier_mean_rl2(ood, COMMON_CHANNELS),
        "qw_posthoc": qw_abs_rel_median(in_dist, "qw_pred", "qw_true"),
        "qw_head": qw_abs_rel_median(in_dist, "qw_head_pred", "qw_ledger"),
    }
    if per_case["test"] and "p" in per_case["test"][0]["rL2"]:
        out["rL2_p"] = tier_mean_rl2(per_case["test"], ("p",))
    eos = [r["eos_viol"] for r in in_dist if "eos_viol" in r]
    out["eos_viol"] = float(np.median(eos)) if eos else 0.0
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
        "Mean rel-L2 over (rho, u, v, T) so every cell is scored on the same\n"
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
        "| prior cell | interpolation | family holdout | OOD pooled | q_w post-hoc | q_w head | EoS violation |",
        "|---|---|---|---|---|---|---|",
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
            f"| {_cell([r['eos_viol'] for r in runs])} |"
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
            f"| {_cell([r['eos_viol'] for r in runs])} |"
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
                    help="per_case_eval.json whose eval tiers should be reused")
    mk.add_argument("--workdir", default="data/raw/sweep")
    mk.add_argument("--n-train", type=int, default=100)
    mk.add_argument("--seed", type=int, default=0)
    mk.add_argument("--out", required=True)

    tb = sub.add_parser("table", help="render the ablation table from run dirs")
    tb.add_argument("run_dirs", nargs="+")
    tb.add_argument("--out", default=None, help="write the table here as well as stdout")

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
