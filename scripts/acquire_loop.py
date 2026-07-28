"""Active-learning acquisition scan over the core design box.

Scores a Latin-hypercube pool of candidate cases by deep-ensemble spread (the
calibrated per-case error proxy from the UQ study) and selects a diverse batch
of the most-uncertain in-envelope cases to simulate next. This is the
acquisition half of one active-learning loop iteration: the ensemble points at
where it is least sure, and those points become the next SU2 runs.

The candidate axes are the six free design parameters: nose radius, cone
half-angle, the two aspect ratios R_b/R_n and R_s/R_b, Mach, and altitude
(freestream T and p follow altitude through the atmosphere model; wall
temperature is fixed). Scanning the full box, including the aspect-ratio
directions, lets the acquisition find uncertainty anywhere in the design
space rather than along a single geometry line. Only in-envelope (box
exceedance <= guard) and continuum (Kn <= 0.01) candidates are scored; the
loop sharpens the model inside its validated domain, not extrapolation the
guard already refuses.

Scoring meshes each candidate at training resolution and runs every ensemble
member, which costs seconds per candidate, so the scan writes a resumable JSONL
ledger and can be re-run to continue. Selection is greedy max-min distance in
normalized parameter space over the highest-spread candidates, so the batch does
not collapse into one corner of the box.

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
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from inference import (  # noqa: E402  -- app/ helpers, path-injected above
    Ensemble,
    _channel_spread,
    _ensemble_forward,
    envelope_distance,
    mesh_nodes,
)
from src.analytical import knudsen_number  # noqa: E402
from src.data.sampler import GEOM_BOX, FS_BOX, KN_MAX  # noqa: E402
from src.data.sampler import us_standard_atmosphere  # noqa: E402
from src.data.su2 import (  # noqa: E402
    CASE_PARAM_ORDER, N_CASE_PARAMS, POS_DIM, TARGET_ORDER,
    SU2NormStats,
)
from src.models.transolver import Transolver  # noqa: E402


T_WALL = 300.0  # K, isothermal cold wall, fixed across the dataset

# the six free design axes: four geometry (nose radius, cone angle, and the two
# aspect ratios R_b/R_n and R_s/R_b) plus Mach and altitude. T_inf and p_inf
# follow altitude through the atmosphere model; T_w is fixed. Scanning all six
# lets the acquisition reach the aspect-ratio directions of the box, not only
# the R_b = 3 R_n, R_s = 0.1 R_b line.
SCAN_AXES = ("R_n", "theta_c", "R_b_ratio", "R_s_ratio", "mach", "altitude")


def build_params(scan: dict[str, float]) -> dict[str, float]:
    """Expand the six scan axes into the full eight-parameter case vector."""
    T_inf, p_inf, _rho = us_standard_atmosphere(scan["altitude"])
    R_b = scan["R_b_ratio"] * scan["R_n"]
    R_s = scan["R_s_ratio"] * R_b
    return {
        "R_n": scan["R_n"], "theta_c_deg": scan["theta_c"], "R_b": R_b, "R_s": R_s,
        "mach": scan["mach"], "T_inf": T_inf, "p_inf": p_inf, "T_w": T_WALL,
    }


def load_ensemble_from(ensemble_dir: Path, member_glob: str, device: str) -> Ensemble:
    """Build an Ensemble from an arbitrary directory of member run dirs.

    Mirrors ``app.inference.load_ensemble`` but takes the member directories
    explicitly instead of the module-level dashboard constant, so the same
    scoring machinery can run against any ensemble version (here, v2).
    """
    dirs = sorted(ensemble_dir.glob(member_glob))
    if not dirs:
        raise SystemExit(f"no member dirs matching {member_glob} under {ensemble_dir}")
    models = []
    for run_dir in dirs:
        rec = json.loads((run_dir / "final_eval.json").read_text())["args"]
        model = Transolver(
            space_dim=POS_DIM, fun_dim=N_CASE_PARAMS, out_dim=len(TARGET_ORDER),
            n_hidden=rec["n_hidden"], n_layers=rec["n_layers"],
            n_head=rec["n_head"], slice_num=rec["slice_num"],
        )
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
        models.append(model.to(device).eval())
    stats = SU2NormStats.load(dirs[0] / "norm_stats.pt").to(device)
    envelope = json.loads((dirs[0] / "final_eval.json").read_text())["train_envelope"]
    print(f"[ensemble] {len(models)} members from {ensemble_dir}/{member_glob}")
    return Ensemble(models=models, stats=stats, envelope=envelope)


def candidate_pool(n: int, seed: int) -> list[dict[str, float]]:
    """LHS over the six scan axes within the core box."""
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
    """Mesh, run the ensemble, return spread + envelope distance + Kn."""
    params = build_params(cand)
    coords = mesh_nodes(params)
    members = _ensemble_forward(ens, coords, params, device)
    spread = _channel_spread(members)
    dist = envelope_distance(params, ens.envelope)
    kn = knudsen_number(params["mach"], params["T_inf"], params["p_inf"], params["R_n"])
    return {
        "scan": cand, "params": params, "spread": spread,
        "dist": dist, "kn": kn, "n_nodes": int(coords.shape[0]),
    }


def _normalize(recs: list[dict]) -> np.ndarray:
    """Min-max normalize the six scan axes to [0, 1] for distance selection."""
    X = np.array([[r["scan"][a] for a in SCAN_AXES] for r in recs])
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (X - lo) / span


def select_batch(recs: list[dict], k: int, guard_dist: float,
                 pool_frac: float = 0.25) -> list[dict]:
    """Greedy max-min-distance pick of k high-spread, in-envelope candidates.

    Restrict to in-envelope continuum candidates, take the top ``pool_frac`` by
    spread as the acquisition pool, then greedily add the point farthest (in
    normalized param space) from those already picked, seeding with the single
    highest-spread candidate. This spreads the batch across the box instead of
    clustering it in the single most-uncertain corner.
    """
    eligible = [r for r in recs if r["dist"] <= guard_dist and r["kn"] <= KN_MAX]
    if len(eligible) < k:
        raise SystemExit(f"only {len(eligible)} in-envelope candidates, need >= {k}; "
                         f"raise --n-candidates")
    eligible.sort(key=lambda r: r["spread"], reverse=True)
    pool = eligible[: max(k, int(len(eligible) * pool_frac))]
    Xn = _normalize(pool)
    picked = [0]                                  # highest-spread candidate seeds
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
    # the scan that produced the committed batch ran at 0.038, the guard value
    # current at the time; a tighter guard only narrows the candidate region
    p.add_argument("--guard-dist", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/processed/loop/acquisition")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "scored.jsonl"

    ens = load_ensemble_from(Path(args.ensemble_dir), args.member_glob, args.device)
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
    print("| # | R_n mm | theta_c | R_b/R_n | R_s/R_b | Mach | alt km | spread | dist | Kn |")
    print("|---|--------|---------|---------|---------|------|--------|--------|------|-----|")
    for i, r in enumerate(batch, 1):
        s = r["scan"]
        print(f"| {i} | {s['R_n']*1e3:.1f} | {s['theta_c']:.1f} | {s['R_b_ratio']:.2f} "
              f"| {s['R_s_ratio']:.3f} | {s['mach']:.1f} | {s['altitude']:.1f} "
              f"| {r['spread']:.3f} | {r['dist']:.3f} | {r['kn']:.4f} |")
    print(f"\n[select] wrote {out / 'selected.json'}")


if __name__ == "__main__":
    main()
