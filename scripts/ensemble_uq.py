"""Deep-ensemble uncertainty: spread calibration and the decision rule.

The script reads K run directories written by ``scripts/train.py``. Every member
must share a ``--seed`` and a slice count and differ only in ``--init-seed``. The
run at ``--init-seed 0`` counts as one member. The work happens in two stages.

Stage one is inference. It rebuilds the shared splits, runs every member over
every case in every evaluation tier, and writes ``ensemble_per_case.json``. That
file holds, per case, the relative L2 of the ensemble mean per channel, the
ensemble spread per channel, the relative L2 of each member, the envelope
distance, and the Knudsen number. The validation split is included here, because
it calibrates the thresholds. Spread is ``||std over members|| / ||truth||`` per
channel, which puts it in the same units as the relative L2 and makes the two
directly comparable. A graphics card speeds this up, and the model is small
enough at half a million parameters that a processor also finishes it.

Stage two is analysis. It is fast and it runs again from the JSON file alone
through ``--per-case``. It produces the table that compares the ensemble against
a single member, the calibration of spread against error as a Spearman rank
correlation per tier and a figure of binned medians, and the decision table.

The decision rule refuses an input when the envelope guard fails, meaning a
distance past ``--guard-dist`` or a Knudsen number above 0.01, and also when the
spread passes the refuse threshold. It warns when the spread passes the warn
threshold, and trusts otherwise. The default thresholds come from the spread
distribution on the validation split, at its 90th percentile and four times
that, and the script prints both. Look at the calibration figure first, then
override them with ``--warn-spread`` and ``--refuse-spread``.

Usage
-----
    python scripts/ensemble_uq.py data/processed/ablation/run_m32 \
        data/processed/ensemble/run_m32_s1 data/processed/ensemble/run_m32_s2 \
        data/processed/ensemble/run_m32_s3 data/processed/ensemble/run_m32_s4 \
        --workdir data/raw/kaggle_su2_stage --out data/processed/ensemble

    python scripts/ensemble_uq.py \
        --per-case data/processed/ensemble/ensemble_per_case.json
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
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import (
    OOD_GROUPS,
    load_case_geom_ids,
    load_case_groups,
    split_paths,
)
from scripts.slice_ablation import KN_MAX, TIER_LABELS, envelope_distance
from src.analytical import knudsen_number
from src.data.su2 import (
    CASE_PARAM_ORDER,
    N_CASE_PARAMS,
    POS_DIM,
    SU2Dataset,
    SU2NormStats,
    TARGET_DIM,
    TARGET_ORDER,
    denormalize_targets,
    list_case_npzs,
)
from src.models.transolver import Transolver


EVAL_SPLITS = ("val", "test", "test_interp", *(f"ood_{g}" for g in OOD_GROUPS))


# ============================================================================================
#                                       inference
# ============================================================================================

def load_member(run_dir: Path, device: str) -> Transolver:
    """Build one member from its recorded arguments and load ``best.pt``."""
    rec = json.loads((run_dir / "final_eval.json").read_text())["args"]
    model = Transolver(
        space_dim=POS_DIM, fun_dim=N_CASE_PARAMS, out_dim=TARGET_DIM,
        n_hidden=rec["n_hidden"], n_layers=rec["n_layers"],
        n_head=rec["n_head"], slice_num=rec["slice_num"],
    )
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def run_inference(run_dirs: list[Path], workdir: Path, device: str,
                  pinned_splits: Path | None = None) -> list[dict]:
    """Build the per-case ensemble record for every evaluation split.

    The validation split is included, because it calibrates the thresholds.
    """
    finals = [json.loads((d / "final_eval.json").read_text()) for d in run_dirs]
    args0 = finals[0]["args"]
    for f, d in zip(finals[1:], run_dirs[1:]):
        for k in ("slice_num", "seed", "val_frac", "test_frac", "interp_frac"):
            if f["args"][k] != args0[k]:
                raise SystemExit(f"{d}: {k}={f['args'][k]} differs from member 0 "
                                 f"({args0[k]}); not a valid ensemble")
    envelope = finals[0]["train_envelope"]
    stats = SU2NormStats.load(run_dirs[0] / "norm_stats.pt")
    stats_dev = stats.to(device)
    models = [load_member(d, device) for d in run_dirs]
    print(f"[infer] {len(models)} members, slice_num={args0['slice_num']}, "
          f"device={device}")

    ledger = workdir / "ledger.db"
    paths = list_case_npzs(workdir)
    splits = split_paths(paths, load_case_groups(ledger), load_case_geom_ids(ledger),
                         args0["val_frac"], args0["test_frac"], args0["seed"],
                         args0["interp_frac"])
    if pinned_splits is not None:
        pinned = json.loads(Path(pinned_splits).read_text())
        def _cn(p: Path) -> str:
            return p.parent.name if p.name == "case.npz" else p.stem
        by_name = {_cn(p): p for p in paths}
        for split_name, names in pinned.items():
            splits[split_name] = [by_name[n] for n in names if n in by_name]
        print(f"[infer] applied {len(pinned)} pinned split(s) from {pinned_splits}")

    recs: list[dict] = []
    for split in EVAL_SPLITS:
        if not splits[split]:
            continue
        ds = SU2Dataset(splits[split], stats, subsample=None)
        for i in range(len(ds)):
            item = ds[i]
            x = item["x"].unsqueeze(0).to(device)
            pos = item["pos"].unsqueeze(0).to(device)
            y = item["y_raw"].to(device)
            preds = torch.stack([
                denormalize_targets(m(x, pos=pos).squeeze(0), stats_dev) for m in models
            ])                                                     # (K, N, C)
            den = y.pow(2).sum(dim=0).sqrt().clamp_min(1e-12)      # (C,)
            rl2_members = (preds - y).pow(2).sum(dim=1).sqrt() / den
            rl2_ens = (preds.mean(dim=0) - y).pow(2).sum(dim=0).sqrt() / den
            spread = preds.std(dim=0).pow(2).sum(dim=0).sqrt() / den
            params = {k: float(item["case_params"][j])
                      for j, k in enumerate(CASE_PARAM_ORDER)}
            recs.append({
                "name": item["name"],
                "split": split,
                "case_params": params,
                "rL2_ens": {c: float(rl2_ens[j]) for j, c in enumerate(TARGET_ORDER)},
                "spread": {c: float(spread[j]) for j, c in enumerate(TARGET_ORDER)},
                "rL2_members": [
                    {c: float(rl2_members[k, j]) for j, c in enumerate(TARGET_ORDER)}
                    for k in range(len(models))
                ],
                "dist": envelope_distance(params, envelope),
                "kn": knudsen_number(params["mach"], params["T_inf"],
                                     params["p_inf"], params["R_n"]),
            })
        print(f"[infer] {split}: {len(splits[split])} cases")
    return recs


# ============================================================================================
#                                       analysis helpers
# ============================================================================================

def chan_mean(d: dict[str, float]) -> float:
    return float(np.mean([d[c] for c in TARGET_ORDER]))


def tiers_present(recs: list[dict]) -> list[str]:
    """Evaluation tiers in report order. The validation split is left out.

    That split only calibrates the thresholds and is not a tier.
    """
    order = [s for s in EVAL_SPLITS if s != "val"]
    present = {r["split"] for r in recs}
    return [s for s in order if s in present]


def tier_label(split: str) -> str:
    return TIER_LABELS.get(split, split.removeprefix("ood_") + " (extrapolation)")


# ============================================================================================
#                                       tables
# ============================================================================================

def ensemble_table(recs: list[dict]) -> str:
    """Render the accuracy of the ensemble against one member, per tier, as Markdown."""
    n_members = len(recs[0]["rL2_members"])
    lines = ["| tier | n | ensemble | member mean | best member | spread |",
             "|" + "---|" * 6]
    for split in tiers_present(recs):
        rows = [r for r in recs if r["split"] == split]
        ens = np.mean([chan_mean(r["rL2_ens"]) for r in rows])
        per_member = [np.mean([chan_mean(r["rL2_members"][k]) for r in rows])
                      for k in range(n_members)]
        spread = np.mean([chan_mean(r["spread"]) for r in rows])
        lines.append(f"| {tier_label(split)} | {len(rows)} | {ens:.3f} "
                     f"| {np.mean(per_member):.3f} | {min(per_member):.3f} "
                     f"| {spread:.3f} |")
    return "\n".join(lines)


def calibration_stats(recs: list[dict]) -> str:
    """Render the Spearman correlation of spread against error, as Markdown.

    One row per tier, then one pooled row over every tier.
    """
    lines = ["| tier | n | spearman |", "|---|---|---|"]
    pooled_s, pooled_e = [], []
    for split in tiers_present(recs):
        rows = [r for r in recs if r["split"] == split]
        s = [chan_mean(r["spread"]) for r in rows]
        e = [chan_mean(r["rL2_ens"]) for r in rows]
        pooled_s += s
        pooled_e += e
        rho = spearmanr(s, e).statistic if len(rows) >= 3 else float("nan")
        lines.append(f"| {tier_label(split)} | {len(rows)} | {rho:.2f} |")
    lines.append(f"| pooled | {len(pooled_s)} | "
                 f"{spearmanr(pooled_s, pooled_e).statistic:.2f} |")
    return "\n".join(lines)


def decision_table(recs: list[dict], warn: float, refuse: float,
                   guard_dist: float) -> str:
    """Render the measured error inside each decision bucket, as Markdown.

    The validation split is excluded, so the rows cover the tiers alone.
    """

    def decide(r: dict) -> str:
        if r["kn"] > KN_MAX or r["dist"] > guard_dist:
            return "refuse"
        s = chan_mean(r["spread"])
        if s >= refuse:
            return "refuse"
        if s >= warn:
            return "warn"
        return "trust"

    rows = [r for r in recs if r["split"] != "val"]
    lines = ["| decision | n | median err | p90 err | max err |",
             "|" + "---|" * 5]
    for bucket in ("trust", "warn", "refuse"):
        errs = [chan_mean(r["rL2_ens"]) for r in rows if decide(r) == bucket]
        if not errs:
            lines.append(f"| {bucket} | 0 | - | - | - |")
            continue
        lines.append(f"| {bucket} | {len(errs)} | {np.median(errs):.3f} "
                     f"| {np.quantile(errs, 0.9):.3f} | {max(errs):.3f} |")
    return "\n".join(lines)


# ============================================================================================
#                                       calibration figure
# ============================================================================================

def calibration_figure(recs: list[dict], fig_out: Path,
                       warn: float, refuse: float) -> None:
    """Plot the error of each case against its ensemble spread.

    Both axes are logarithmic, and a binned median runs through the cloud.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split in tiers_present(recs):
        rows = [r for r in recs if r["split"] == split]
        ax.scatter([chan_mean(r["spread"]) for r in rows],
                   [chan_mean(r["rL2_ens"]) for r in rows],
                   s=18, alpha=0.7, label=tier_label(split))

    rows = [r for r in recs if r["split"] != "val"]
    s = np.array([chan_mean(r["spread"]) for r in rows])
    e = np.array([chan_mean(r["rL2_ens"]) for r in rows])
    edges = np.quantile(s, np.linspace(0, 1, 7))
    mids, meds = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (s >= lo) & (s <= hi)
        if mask.sum() == 0:
            continue
        mids.append(np.sqrt(lo * hi) if lo > 0 else hi / 2)
        meds.append(np.median(e[mask]))
    ax.plot(mids, meds, "k--", lw=1, label="binned median")
    ax.axvline(warn, color="orange", lw=1.2, label=f"warn s = {warn:.3f}")
    ax.axvline(refuse, color="crimson", lw=1.2, label=f"refuse s = {refuse:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ensemble spread (channel-mean ||std|| / ||truth||)")
    ax.set_ylabel("per-case mean rel-L2 of the ensemble mean")
    ax.set_title("Deep-ensemble spread vs actual error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)


# ============================================================================================
#                                       main
# ============================================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="deep-ensemble uncertainty and the trust/warn/refuse rule")
    p.add_argument("runs", nargs="*",
                   help="member run directories; same --seed and M, "
                        "different --init-seed")
    p.add_argument("--workdir", default="data/raw/kaggle_su2_stage",
                   help="dataset directory holding the case tensors and ledger.db")
    p.add_argument("--out", default="data/processed/ensemble")
    p.add_argument("--per-case", default=None, metavar="JSON",
                   help="skip inference; analyze an existing ensemble_per_case.json")
    p.add_argument("--guard-dist", type=float, default=0.07,
                   help="envelope guard threshold on box exceedance")
    p.add_argument("--warn-spread", type=float, default=None,
                   help="spread above which the decision is warn; the default is "
                        "the 90th percentile on the validation split")
    p.add_argument("--refuse-spread", type=float, default=None,
                   help="spread above which the decision is refuse; the default is "
                        "four times the warn threshold")
    p.add_argument("--fig-out", default="data/samples/spread_calibration.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pinned-splits", default=None, metavar="JSON",
                   help="take the split membership from an earlier run, so that the "
                        "validation, test and interpolation tiers hold the same "
                        "cases across dataset versions")
    args = p.parse_args()

    if args.per_case is not None:
        recs = json.loads(Path(args.per_case).read_text())
    else:
        if len(args.runs) < 2:
            raise SystemExit("need at least 2 run directories, or a --per-case file")
        recs = run_inference([Path(d).resolve() for d in args.runs],
                             Path(args.workdir).resolve(), args.device,
                             pinned_splits=args.pinned_splits)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "ensemble_per_case.json").write_text(json.dumps(recs, indent=2))
        print(f"[infer] wrote {out / 'ensemble_per_case.json'}")

    val_spread = [chan_mean(r["spread"]) for r in recs if r["split"] == "val"]
    warn = args.warn_spread
    if warn is None:
        warn = float(np.quantile(val_spread, 0.9)) if val_spread else 0.05
    refuse = args.refuse_spread if args.refuse_spread is not None else 4.0 * warn
    print(f"\nthresholds: warn spread {warn:.4f}, refuse spread {refuse:.4f}, "
          f"guard d > {args.guard_dist}, Kn > {KN_MAX}")

    print("\n## Ensemble against single members (mean relative L2 per tier)\n")
    print(ensemble_table(recs))
    print("\n## Spread-error calibration (Spearman rank corr per tier)\n")
    print(calibration_stats(recs))
    print("\n## Trust/warn/refuse (eval tiers, actual error per bucket)\n")
    print(decision_table(recs, warn, refuse, args.guard_dist))

    calibration_figure(recs, Path(args.fig_out), warn, refuse)
    print(f"\ncalibration figure -> {args.fig_out}")


if __name__ == "__main__":
    main()
