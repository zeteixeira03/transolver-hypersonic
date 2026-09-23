"""Run the active-learning loop cases through SU2.

The script takes the batch that ``scripts/acquire_loop.py`` selected and
appends it to the sweep ledger under ``group_name='loop'``. Each appended case
takes a case id past the end of the sweep and a geometry of its own, so no case
warm-starts from another. The generation workers then solve them through the
same two-pass SU2 recipe as the sweep. Every other row in the ledger has already
reached a terminal state, so the workers touch the loop cases alone.

A loop case is judged exactly as a sweep case was. It must finish, and it must
clear the thresholds that catch a collapsed field. The three analytical gates
are recorded as relative errors and advise rather than decide. A failure is
logged and left in the ledger as failed.

Run it from the project root. On a Windows host SU2 goes through the Windows
Subsystem for Linux without further setup::

    python scripts/run_loop_cases.py --workers 4

The tensors land in ``data/raw/sweep/case_0780/`` and onward, beside the sweep,
where ``group_name`` routes them into the training split.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from scripts.generate_dataset import _build_parser, run_generation
from src.cfd import ledger as L

LOOP_GROUP = "loop"


def append_loop_rows(db_path: Path, selected: list[dict],
                     first_case_id: int, first_geom_id: int) -> list[int]:
    """Insert the selected cases as pending loop rows.

    A second call adds nothing, because a row whose ``case_id`` exists is left
    alone. Each case takes its own ``geom_id`` and a NULL ``restart_from``, so
    every one solves cold. The acquired geometries are spread across the box and
    would gain nothing from a shared mesh. Returns the case ids now pending.
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
    """Print the final state and the advisory gate results for each loop case."""
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

    # reuse the sweep generation workers. the parser of generate_dataset builds a
    # complete argument namespace, which holds every solver setting at the value
    # the sweep used.
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
