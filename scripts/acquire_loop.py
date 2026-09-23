"""Active-learning acquisition scan over the core design box.

The script scores a Latin hypercube of candidate cases by the spread of the
deep ensemble, which the uncertainty study calibrated as a per-case error proxy.
It then selects a spread-out batch of the most uncertain candidates inside the
envelope. The ensemble names the places where it is least sure, and those places
become the next SU2 runs.

The candidates vary along the six free design axes: nose radius, cone half-angle,
the two ratios R_b/R_n and R_s/R_b, Mach number, and altitude. Freestream
temperature and pressure follow altitude through the atmosphere model, and the
wall temperature is fixed. A scan over all six axes, the two ratios included,
can find uncertainty anywhere in the design space rather than along one geometry
line.

Only candidates inside the envelope and inside the continuum regime are scored,
which means a box exceedance at or under the guard and a Knudsen number at or
under 0.01. The loop sharpens the model inside its validated domain. It does not
chase the extrapolation the guard already refuses.

One score costs seconds, because it meshes the candidate at training resolution
and runs every ensemble member. The scan therefore appends to a JSONL ledger and
picks up where it stopped when run again. Selection is a greedy max-min distance
walk in normalized parameter space over the highest-spread candidates, so the
batch does not collapse into one corner of the box.

Usage
-----
    python scripts/acquire_loop.py \
        --ensemble-dir data/processed/ensemble_v2 \
        --member-glob "run_m32_v2_s*" \
        --n-candidates 1200 --k 10 \
        --out data/processed/loop/acquisition
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from src.analytical import knudsen_number
from src.data.sampler import FS_BOX, GEOM_BOX, KN_MAX, case_params
from src.eval.uncertainty import (
    Ensemble,
    envelope_distance,
    load_ensemble,
    predict_on_nodes,
    spread,
)
from src.geometry.sphere_cone import mesh_nodes

# the six free design axes. four are geometric: nose radius, cone angle, and the
# ratios R_b/R_n and R_s/R_b. two are flight conditions: Mach number and
# altitude. freestream temperature and pressure follow altitude, and the wall
# temperature is fixed. all six vary here, so the scan reaches the ratio
# directions of the box and not only the line at R_b = 3 R_n, R_s = 0.1 R_b.
SCAN_AXES = ("R_n", "theta_c", "R_b_ratio", "R_s_ratio", "mach", "altitude")


def candidate_pool(n: int, seed: int) -> list[dict[str, float]]:
    """Draw a Latin hypercube over the six scan axes inside the core box."""
    lo = np.array([GEOM_BOX["R_n"][0], GEOM_BOX["theta_c"][0],
                   GEOM_BOX["R_b_ratio"][0], GEOM_BOX["R_s_ratio"][0],
                   FS_BOX["mach"][0], FS_BOX["altitude"][0]])
    hi = np.array([GEOM_BOX["R_n"][1], GEOM_BOX["theta_c"][1],
                   GEOM_BOX["R_b_ratio"][1], GEOM_BOX["R_s_ratio"][1],
                   FS_BOX["mach"][1], FS_BOX["altitude"][1]])
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    pts = qmc.scale(sampler.random(n), lo, hi)
    return [dict(zip(SCAN_AXES, row)) for row in pts]


def score_candidate(ens: Ensemble, cand: dict[str, float], device: str) -> dict:
    """Mesh one candidate, run the ensemble, and return its three scores.

    The scores are the ensemble spread, the distance outside the envelope, and
    the Knudsen number.
    """
    params = case_params(**cand)
    coords = mesh_nodes(params["R_n"], params["theta_c_deg"], params["R_b"], params["R_s"])
    members = predict_on_nodes(ens, coords, params, device)
    mean_spread = float(spread(members, members.mean(dim=0)).mean())
    dist = envelope_distance(params, ens.envelope)
    kn = knudsen_number(params["mach"], params["T_inf"], params["p_inf"], params["R_n"])
    return {
        "scan": cand, "params": params, "spread": mean_spread,
        "dist": dist, "kn": kn, "n_nodes": int(coords.shape[0]),
    }


def _normalize(recs: list[dict]) -> np.ndarray:
    """Scale the six scan axes to [0, 1], so the selection distance is unitless."""
    X = np.array([[r["scan"][a] for a in SCAN_AXES] for r in recs])
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (X - lo) / span


def select_batch(recs: list[dict], k: int, guard_dist: float,
                 pool_frac: float = 0.25) -> list[dict]:
    """Pick k candidates with high spread that also sit apart from one another.

    The function keeps the candidates inside the envelope and inside the
    continuum regime, then takes the top ``pool_frac`` of them by spread as the
    acquisition pool. It starts the batch at the single highest-spread candidate
    and adds, each time, the pool point farthest from everything already picked
    in normalized parameter space. The batch therefore covers the box rather
    than one uncertain corner of it.
    """
    eligible = [r for r in recs if r["dist"] <= guard_dist and r["kn"] <= KN_MAX]
    if len(eligible) < k:
        raise SystemExit(f"only {len(eligible)} in-envelope candidates, need >= {k}; "
                         f"raise --n-candidates")
    eligible.sort(key=lambda r: r["spread"], reverse=True)
    pool = eligible[: max(k, int(len(eligible) * pool_frac))]
    Xn = _normalize(pool)
    picked = [0]                                  # the highest-spread candidate
    while len(picked) < k:
        d = np.min([np.linalg.norm(Xn - Xn[p], axis=1) for p in picked], axis=0)
        d[picked] = -1.0
        picked.append(int(np.argmax(d)))
    return [pool[i] for i in picked]


def main() -> None:
    p = argparse.ArgumentParser(description="active-learning acquisition scan")
    p.add_argument("--ensemble-dir", default="data/processed/ensemble_v2")
    p.add_argument("--member-glob", default="run_m32_v2_s*")
    p.add_argument("--n-candidates", type=int, default=1200)
    p.add_argument("--k", type=int, default=10)
    # the committed batch came from a scan at 0.038. pass that value to
    # reproduce it. the default matches the guard the decision rule now uses,
    # which admits a wider candidate region.
    p.add_argument("--guard-dist", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/processed/loop/acquisition")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "scored.jsonl"

    ens = load_ensemble(list(Path(args.ensemble_dir).glob(args.member_glob)), args.device)
    pool = candidate_pool(args.n_candidates, args.seed)

    # resume: skip candidates already in the ledger (keyed by rounded scan tuple)
    def key(scan: dict) -> str:
        return ",".join(f"{scan[a]:.6g}" for a in SCAN_AXES)

    done: dict[str, dict] = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[key(r["scan"])] = r
        print(f"[resume] {len(done)} candidates already scored")

    recs = list(done.values())
    with open(ledger, "a") as f:
        for i, cand in enumerate(pool):
            if key(cand) in done:
                continue
            rec = score_candidate(ens, cand, args.device)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            recs.append(rec)
            if (i + 1) % 20 == 0:
                print(f"[scan] {i + 1}/{len(pool)} scored "
                      f"(latest spread {rec['spread']:.3f}, dist {rec['dist']:.3f})")

    batch = select_batch(recs, args.k, args.guard_dist)
    (out / "selected.json").write_text(json.dumps(
        [{"scan": r["scan"], "params": r["params"], "spread": r["spread"],
          "dist": r["dist"], "kn": r["kn"]} for r in batch], indent=2))

    print(f"\n[select] {args.k} loop candidates (spread-ranked, distance-spread):\n")
    print("| # | R_n mm | theta_c | R_b/R_n | R_s/R_b | Mach | altitude km "
          "| spread | distance | Kn |")
    print("|---|--------|---------|---------|---------|------|-------------"
          "|--------|----------|-----|")
    for i, r in enumerate(batch, 1):
        s = r["scan"]
        print(f"| {i} | {s['R_n']*1e3:.1f} | {s['theta_c']:.1f} | {s['R_b_ratio']:.2f} "
              f"| {s['R_s_ratio']:.3f} | {s['mach']:.1f} | {s['altitude']:.1f} "
              f"| {r['spread']:.3f} | {r['dist']:.3f} | {r['kn']:.4f} |")
    print(f"\n[select] wrote {out / 'selected.json'}")


if __name__ == "__main__":
    main()
