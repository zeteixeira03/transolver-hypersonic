"""Run the active-learning loop cases through SU2.

Takes the batch selected by ``scripts/acquire_loop.py``, appends the cases to
the sweep ledger tagged ``group_name='loop'`` (new case ids after the sweep,
each its own geometry so no warm-start clustering), and drives them through the
same two-pass SU2 recipe the sweep used by reusing the generation workers. Every
other ledger row is already terminal, so the workers solve exactly the loop
cases.

A loop case is judged the same way every sweep case was: it must complete and
clear the catastrophic-collapse thresholds; the three analytical gates are
recorded as advisory rel-errors, not a pass/fail bar. Failures are logged and
left in the ledger as failed, not rescued.

Run from the project root (Windows host auto-routes SU2 through WSL)::

    python scripts/run_loop_cases.py --workers 4

The loop tensors land in ``data/raw/sweep/case_0780/`` onward alongside the
sweep, so the training split picks them up (routed to train by group_name).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import multiprocessing as mp

from src.cfd import ledger as L
from scripts.generate_dataset import _build_parser, run_generation


LOOP_GROUP = "loop"


def append_loop_rows(db_path: Path, selected: list[dict],
                     first_case_id: int, first_geom_id: int) -> list[int]:
    """Insert the selected cases as pending loop rows. Idempotent by case_id.

    Each case gets its own geom_id (no warm-start clustering across the diverse
    acquired geometries) and restart_from NULL (cold solve). Returns the case
    ids that are pending after the call.
    """
    con = L.connect(db_path)
    ids = []
    try:
        with con:
            for i, rec in enumerate(selected):
                cid = first_case_id + i
                p = rec["params"]
                con.execute(
                    """INSERT OR IGNORE INTO cases
                       (case_id, ord, block, group_name, geom_id, restart_from,
                        R_n, theta_c_deg, R_b, R_s, mach, T_inf, p_inf, T_w)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (cid, cid, 0, LOOP_GROUP, first_geom_id + i, None,
                     p["R_n"], p["theta_c_deg"], p["R_b"], p["R_s"],
                     p["mach"], p["T_inf"], p["p_inf"], p["T_w"]),
                )
                ids.append(cid)
    finally:
        con.close()
    return ids


def report(db_path: Path, ids: list[int]) -> None:
    """Print the terminal state and advisory gate rel-errors for each loop case."""
    print("\n=== loop case outcomes ===")
    print("| case | status | M | R_n mm | theta_c | qw | p02 | standoff mm | gates |")
    print("|------|--------|---|--------|---------|----|----|-------------|-------|")
    for cid in ids:
        r = L.case_row(db_path, cid)
        if r is None:
            print(f"| {cid} | MISSING | | | | | | | |")
            continue
        gates = "clear" if r["checks_passed"] else ("advisory-miss" if r["status"] == "done" else "-")
        so = f"{r['standoff']*1e3:.2f}" if r["standoff"] is not None else "-"
        qw = f"{r['qw']:.3g}" if r["qw"] is not None else "-"
        p02 = f"{r['p02']:.3g}" if r["p02"] is not None else "-"
        print(f"| {cid} | {r['status']} | {r['mach']:.1f} | {r['R_n']*1e3:.1f} "
              f"| {r['theta_c_deg']:.1f} | {qw} | {p02} | {so} | {gates} |")


def main() -> None:
    p = argparse.ArgumentParser(description="run active-learning loop cases through SU2")
    p.add_argument("--selected", default="data/processed/loop/acquisition/selected.json")
    p.add_argument("--workdir", default="data/raw/sweep")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--first-case-id", type=int, default=780)
    p.add_argument("--first-geom-id", type=int, default=200)
    args = p.parse_args()

    workdir = Path(args.workdir).resolve()
    db_path = workdir / "ledger.db"
    selected = json.loads(Path(args.selected).read_text())
    print(f"[loop] {len(selected)} selected cases -> ledger {db_path}")

    ids = append_loop_rows(db_path, selected, args.first_case_id, args.first_geom_id)
    counts = L.status_counts(db_path)
    print(f"[loop] ledger now {dict(counts)}; loop case ids {ids[0]}..{ids[-1]}")

    # reuse the sweep generation workers: build a full args namespace from the
    # generate_dataset parser (all solver knobs at their sweep defaults) and run
    gen_args = _build_parser().parse_args([
        "--workdir", str(workdir), "--workers", str(args.workers),
    ])
    gen_args.workdir = str(workdir)
    gen_args.db = str(db_path)
    run_generation(gen_args)

    report(db_path, ids)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
