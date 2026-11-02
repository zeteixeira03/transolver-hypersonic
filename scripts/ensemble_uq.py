"""Deep-ensemble uncertainty: spread calibration and the decision rule.

Inference runs every member over every evaluation tier and the validation
split and writes ``ensemble_per_case.json``. The members are run directories of
``scripts/train.py`` that share ``--seed`` and the slice count. Spread is
``||std over members|| / ||truth||`` per channel, in the units of the relative
L2. The analysis, which ``--per-case`` reruns from that file alone, prints the
ensemble against single members, the Spearman rank correlation of spread
against error per tier, and the decision table, and writes ``thresholds.json``.

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
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from src.analytical import knudsen_number
from src.cfd.ledger import read_cases
from src.data.sampler import KN_MAX
from src.data.su2 import (
    CASE_PARAM_ORDER,
    TARGET_ORDER,
    SU2Dataset,
    SU2NormStats,
    list_case_npzs,
    pin_splits,
    split_paths,
)
from src.eval.metrics import EVAL_SPLITS, channel_mean, relative_l2, tier_label, tiers_present
from src.eval.uncertainty import (
    GUARD_DIST,
    Ensemble,
    calibrate_thresholds,
    decide,
    envelope_distance,
    load_member,
    member_predictions,
    spread,
)

# ============================================================================================
#                                       inference
# ============================================================================================

@torch.no_grad()
def run_inference(run_dirs: list[Path], workdir: Path, device: str,
                  pinned_splits: Path | None = None) -> list[dict]:
    """Build the per-case ensemble record for every evaluation split.

    The validation split is included, because it calibrates the thresholds.
    """
    finals = [json.loads((d / "final_eval.json").read_text()) for d in run_dirs]
    args0 = finals[0]["args"]
    for f, d in zip(finals[1:], run_dirs[1:], strict=True):
        for k in ("slice_num", "seed", "val_frac", "test_frac", "interp_frac"):
            if f["args"][k] != args0[k]:
                raise SystemExit(f"{d}: {k}={f['args'][k]} differs from member 0 "
                                 f"({args0[k]}); not a valid ensemble")
    stats = SU2NormStats.load(run_dirs[0] / "norm_stats.pt")
    ens = Ensemble(models=[load_member(d, device) for d in run_dirs], stats=stats.to(device),
                   envelope=finals[0]["train_envelope"])
    print(f"[infer] {len(ens.models)} members, slice_num={args0['slice_num']}, device={device}")

    paths = list_case_npzs(workdir)
    rows = read_cases(workdir / "ledger.db")
    splits = split_paths(paths, {c: r["group_name"] for c, r in rows.items()},
                         {c: int(r["geom_id"]) for c, r in rows.items()},
                         args0["val_frac"], args0["test_frac"], args0["seed"],
                         args0["interp_frac"])
    if pinned_splits is not None:
        splits = pin_splits(splits, paths, json.loads(Path(pinned_splits).read_text()))

    recs: list[dict] = []
    for split in EVAL_SPLITS:
        if not splits[split]:
            continue
        ds = SU2Dataset(splits[split], stats, subsample=None)
        for i in range(len(ds)):
            item = ds[i]
            y = item["y_raw"].to(device)
            preds = member_predictions(ens, item["x"].unsqueeze(0).to(device),
                                       item["pos"].unsqueeze(0).to(device))   # (K, N, C)
            rl2_members = torch.stack([relative_l2(p, y) for p in preds])
            rl2_ens = relative_l2(preds.mean(dim=0), y)
            spread_c = spread(preds, y)
            params = {k: float(item["case_params"][j])
                      for j, k in enumerate(CASE_PARAM_ORDER)}
            recs.append({
                "name": item["name"],
                "split": split,
                "case_params": params,
                "rL2_ens": {c: float(rl2_ens[j]) for j, c in enumerate(TARGET_ORDER)},
                "spread": {c: float(spread_c[j]) for j, c in enumerate(TARGET_ORDER)},
                "rL2_members": [
                    {c: float(rl2_members[k, j]) for j, c in enumerate(TARGET_ORDER)}
                    for k in range(len(ens.models))
                ],
                "dist": envelope_distance(params, ens.envelope),
                "kn": knudsen_number(params["mach"], params["T_inf"],
                                     params["p_inf"], params["R_n"]),
            })
        print(f"[infer] {split}: {len(splits[split])} cases")
    return recs


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
        ens = np.mean([channel_mean(r["rL2_ens"]) for r in rows])
        per_member = [np.mean([channel_mean(r["rL2_members"][k]) for r in rows])
                      for k in range(n_members)]
        mean_spread = np.mean([channel_mean(r["spread"]) for r in rows])
        lines.append(f"| {tier_label(split)} | {len(rows)} | {ens:.3f} "
                     f"| {np.mean(per_member):.3f} | {min(per_member):.3f} "
                     f"| {mean_spread:.3f} |")
    return "\n".join(lines)


def calibration_stats(recs: list[dict]) -> str:
    """Render the Spearman correlation of spread against error, as Markdown.

    One row per tier, then one pooled row over every tier.
    """
    lines = ["| tier | n | spearman |", "|---|---|---|"]
    pooled_s, pooled_e = [], []
    for split in tiers_present(recs):
        rows = [r for r in recs if r["split"] == split]
        s = [channel_mean(r["spread"]) for r in rows]
        e = [channel_mean(r["rL2_ens"]) for r in rows]
        pooled_s += s
        pooled_e += e
        rho = spearmanr(s, e).statistic if len(rows) >= 3 else float("nan")
        lines.append(f"| {tier_label(split)} | {len(rows)} | {rho:.2f} |")
    lines.append(f"| pooled | {len(pooled_s)} | "
                 f"{spearmanr(pooled_s, pooled_e).statistic:.2f} |")
    return "\n".join(lines)


def decision_table(recs: list[dict], thresholds: dict[str, float]) -> str:
    """Render the measured error inside each decision bucket, as Markdown.

    The validation split is excluded, so the rows cover the tiers alone.
    """

    rows = [r for r in recs if r["split"] != "val"]
    lines = ["| decision | n | median err | p90 err | max err |",
             "|" + "---|" * 5]
    for bucket in ("trust", "warn", "refuse"):
        errs = [channel_mean(r["rL2_ens"]) for r in rows
                if decide(r["dist"], r["kn"], channel_mean(r["spread"]), thresholds) == bucket]
        if not errs:
            lines.append(f"| {bucket} | 0 | - | - | - |")
            continue
        lines.append(f"| {bucket} | {len(errs)} | {np.median(errs):.3f} "
                     f"| {np.quantile(errs, 0.9):.3f} | {max(errs):.3f} |")
    return "\n".join(lines)


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
    p.add_argument("--warn-spread", type=float, default=None,
                   help="spread above which the decision is warn; the default is "
                        "the 90th percentile on the validation split")
    p.add_argument("--refuse-spread", type=float, default=None,
                   help="spread above which the decision is refuse; the default is "
                        "four times the warn threshold")
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

    thresholds = calibrate_thresholds(
        [channel_mean(r["spread"]) for r in recs if r["split"] == "val"])
    if args.warn_spread is not None:
        thresholds = {"warn": args.warn_spread, "refuse": 4.0 * args.warn_spread}
    if args.refuse_spread is not None:
        thresholds["refuse"] = args.refuse_spread
    print(f"\nthresholds: warn spread {thresholds['warn']:.4f}, "
          f"refuse spread {thresholds['refuse']:.4f}, guard d > {GUARD_DIST}, Kn > {KN_MAX}")
    out = Path(args.per_case).parent if args.per_case else Path(args.out)
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    print("\n## Ensemble against single members (mean relative L2 per tier)\n")
    print(ensemble_table(recs))
    print("\n## Spread-error calibration (Spearman rank corr per tier)\n")
    print(calibration_stats(recs))
    print("\n## Trust/warn/refuse (eval tiers, actual error per bucket)\n")
    print(decision_table(recs, thresholds))


if __name__ == "__main__":
    main()
