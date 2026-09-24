"""
Resumable runner for SU2 jobs, with a one-line status.

A job solves one case with one recipe. A first-order job starts from the
freestream. A second-order job restarts from the solution of the job named in
``after`` and waits until that job is done. Jobs live in a SQLite table in the
work directory, so a killed run resumes where it stopped: ``run`` returns every
job left ``running`` to ``pending`` before it starts.

The ``plan-study`` command fills the table for the solver recipe study: six
representative cases, one shared first-order start each, and one second-order
job per recipe variant.

Usage
-----
    python scripts/run_cases.py plan-study --workdir ~/solver_study
    python scripts/run_cases.py run --workdir ~/solver_study --workers 7
    python scripts/run_cases.py status --workdir ~/solver_study
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from src.cfd.ledger import read_cases
from src.cfd.postprocess import measure_case
from src.cfd.runner import Case, detect_invocation, run_case
from src.geometry.sphere_cone import mesh_settings

CASE_KEYS = ("R_n", "theta_c_deg", "R_b", "R_s", "mach", "T_inf", "p_inf", "T_w")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    name      TEXT PRIMARY KEY,
    case_json TEXT NOT NULL,
    recipe    TEXT NOT NULL,
    after     TEXT,
    status    TEXT NOT NULL DEFAULT 'pending',
    started   REAL,
    finished  REAL,
    result    TEXT,
    error     TEXT
)
"""

# ============================================================================================
#                                       recipes
# ============================================================================================

# the stop rule of every study variant: the wall heat flux, the slowest quantity
# to settle, must change by less than 1e-6 per iteration on average over the last
# 500 iterations. the iteration count is only a ceiling.
HEATFLUX_STOP = {
    "CONV_FIELD": "TOTAL_HEATFLUX",
    "CONV_STARTITER": "500",
    "CONV_CAUCHY_ELEMS": "500",
    "CONV_CAUCHY_EPS": "1E-6",
}

# the first-order start of the old recipe, shared by every variant of a case
FIRST_ORDER = {"order": 1, "iter": 2500, "settings": {"CFL_ADAPT_PARAM": "( 0.5, 1.4, 0.1, 5.0 )"}}


def second_order(cfl_max: float = 5.0, mglevel: int = 2, **extra: str) -> dict:
    settings = {"CFL_ADAPT_PARAM": f"( 0.5, 1.4, 0.1, {cfl_max} )", "MGLEVEL": str(mglevel),
                **HEATFLUX_STOP, **extra}
    return {"order": 2, "iter": 16000, "settings": settings}


VARIANTS = {
    "v0_baseline": second_order(),
    "v1_cfl25": second_order(cfl_max=25),
    "v2_cfl100": second_order(cfl_max=100),
    "v3_limiter03": second_order(VENKAT_LIMITER_COEFF="0.3"),
    "v4_frozen_limiter": second_order(LIMITER_ITER="1000"),
    "v5_cfl100_ilu1": second_order(cfl_max=100, LINEAR_SOLVER_ILU_FILL_IN="1"),
    "v6_cfl100_nomg": second_order(cfl_max=100, mglevel=0),
}

# ============================================================================================
#                                       job table
# ============================================================================================

def connect(workdir: Path) -> sqlite3.Connection:
    con = sqlite3.connect(workdir / "jobs.db", timeout=30)
    con.row_factory = sqlite3.Row
    con.execute(SCHEMA)
    return con


def add_job(con: sqlite3.Connection, name: str, case: Case, recipe: dict,
            after: str | None = None) -> None:
    """Insert a job unless one with this name exists already."""
    con.execute("INSERT OR IGNORE INTO jobs (name, case_json, recipe, after) VALUES (?, ?, ?, ?)",
                (name, json.dumps({k: getattr(case, k) for k in CASE_KEYS}),
                 json.dumps(recipe), after))
    con.commit()


def ready_jobs(con: sqlite3.Connection) -> list[sqlite3.Row]:
    """Pending jobs whose first-order start, if they have one, is done."""
    return con.execute(
        "SELECT j.* FROM jobs j LEFT JOIN jobs a ON j.after = a.name "
        "WHERE j.status = 'pending' AND (j.after IS NULL OR a.status = 'done') "
        "ORDER BY j.after IS NOT NULL, j.name").fetchall()


# ============================================================================================
#                                       one job
# ============================================================================================

def solve(workdir: str, name: str, case_json: str, recipe_json: str, after: str | None,
          nprocs: int) -> dict:
    """Run one job in its own directory and measure the result. Runs in a worker process."""
    case = Case(**json.loads(case_json))
    recipe = json.loads(recipe_json)
    run_dir = Path(workdir) / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    if after is not None:
        start = Path(workdir) / after
        shutil.copyfile(start / "case.su2", run_dir / "case.su2")
        shutil.copyfile(start / "restart_flow.dat", run_dir / "solution_flow.dat")

    distro, env = detect_invocation()
    t0 = time.time()
    res = run_case(case, run_dir, mesh_kwargs=mesh_settings(case.R_n, case.R_b),
                   iter_max=recipe["iter"], muscl=recipe["order"] == 2,
                   restart_sol=after is not None, write_mesh=after is None,
                   settings=recipe["settings"], wsl_distro=distro, conda_env=env,
                   nprocs=nprocs)
    wall = time.time() - t0
    if res["returncode"] != 0:
        raise RuntimeError(f"SU2 returned {res['returncode']}, see {res['log']}")
    out = {"wall_s": wall, **read_history(run_dir / "history.csv")}
    if recipe["order"] == 2:
        m = measure_case(run_dir, case)
        out.update(qw=m["qw"], p02=m["p02"], standoff=m["standoff"],
                   checks={c.name: c.rel_err for c in m["summary"]["checks"]})
    for junk in ("flow.vtu", "solution_flow.dat", "surface_flow.csv"):
        (run_dir / junk).unlink(missing_ok=True)
    return out


def read_history(path: Path) -> dict:
    """Iterations run, the final density residual, and when the heat flux came within 1% of its final value."""
    with path.open() as f:
        rows = list(csv.reader(f))
    head = [h.strip().strip('"') for h in rows[0]]
    data = [[float(v) for v in r] for r in rows[1:] if r]
    out = {"iterations": len(data), "rms_rho": data[-1][head.index("rms[Rho]")]}
    if "HF" in head:
        hf = [r[head.index("HF")] for r in data]
        off = [i for i, q in enumerate(hf) if abs(q / hf[-1] - 1) > 0.01]
        out["hf_settled_iter"] = off[-1] + 1 if off else 0
    return out


# ============================================================================================
#                                       commands
# ============================================================================================

def plan_study(workdir: Path, ledger: Path) -> None:
    """Pick six representative cases from the sweep and queue every variant for each."""
    rows = [r for r in read_cases(ledger).values() if r["status"] == "done" and r["qw"]]
    core = sorted((r for r in rows if r["group_name"] == "core"), key=lambda r: r["mach"])
    picks = {
        "core_a": core[len(core) // 3],
        "core_b": core[2 * len(core) // 3],
        "cone_high": max((r for r in rows if r["group_name"] == "cone_high"),
                         key=lambda r: r["theta_c_deg"]),
        "mach_high": max(rows, key=lambda r: r["mach"]),
        "nose_small": min(rows, key=lambda r: r["R_n"]),
        "nose_large": max(rows, key=lambda r: r["R_n"]),
    }
    con = connect(workdir)
    reference = {}
    for label, r in picks.items():
        case = Case(**{k: r[k] for k in CASE_KEYS})
        tag = f"{label}_{r['case_id']:04d}"
        add_job(con, f"{tag}_start", case, FIRST_ORDER)
        for vname, recipe in VARIANTS.items():
            add_job(con, f"{tag}_{vname}", case, recipe, after=f"{tag}_start")
        reference[tag] = {k: r[k] for k in ("qw", "p02", "standoff")}
    (workdir / "reference.json").write_text(json.dumps(reference, indent=2))
    n = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    print(f"{n} jobs queued for {len(picks)} cases: {', '.join(reference)}")


def run(workdir: Path, workers: int, nprocs: int) -> None:
    con = connect(workdir)
    reset = con.execute("UPDATE jobs SET status = 'pending' WHERE status = 'running'").rowcount
    con.commit()
    if reset:
        print(f"[run] {reset} interrupted job(s) back to pending")
    running = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while True:
            for job in ready_jobs(con):
                if len(running) >= workers:
                    break
                if job["name"] in running.values():
                    continue
                con.execute("UPDATE jobs SET status = 'running', started = ? WHERE name = ?",
                            (time.time(), job["name"]))
                con.commit()
                fut = pool.submit(solve, str(workdir), job["name"], job["case_json"],
                                  job["recipe"], job["after"], nprocs)
                running[fut] = job["name"]
                print(f"[run] start {job['name']}", flush=True)
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                name = running.pop(fut)
                try:
                    result = fut.result()
                    con.execute("UPDATE jobs SET status = 'done', finished = ?, result = ? "
                                "WHERE name = ?", (time.time(), json.dumps(result), name))
                    print(f"[run] done  {name} in {result['wall_s'] / 3600:.2f} h", flush=True)
                except Exception as exc:  # noqa: BLE001 -- a failed job is logged, not fatal
                    con.execute("UPDATE jobs SET status = 'failed', finished = ?, error = ? "
                                "WHERE name = ?", (time.time(), str(exc)[:2000], name))
                    print(f"[run] FAIL  {name}: {exc}", flush=True)
                con.commit()
    print("[run] no job left to start")


def status(workdir: Path) -> str:
    """One line: job counts, then the fastest variant that passes, relative to the baseline."""
    con = connect(workdir)
    counts = dict(con.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())
    line = (f"{counts.get('done', 0)}/{sum(counts.values())} done, {counts.get('running', 0)} "
            f"running, {counts.get('pending', 0)} pending, {counts.get('failed', 0)} failed")
    ref_path = workdir / "reference.json"
    if not ref_path.exists():
        return line
    reference = json.loads(ref_path.read_text())
    done = {r["name"]: json.loads(r["result"]) for r in
            con.execute("SELECT name, result FROM jobs WHERE status = 'done'")}
    best = None
    for vname in VARIANTS:
        speedups, errs = [], []
        for tag, ref in reference.items():
            base, var = done.get(f"{tag}_v0_baseline"), done.get(f"{tag}_{vname}")
            if base and var:
                speedups.append(base["wall_s"] / var["wall_s"])
                errs.append(max(abs(var[k] / ref[k] - 1) for k in ("qw", "p02", "standoff")))
        if speedups and vname != "v0_baseline":
            s, e = min(speedups), max(errs)
            if e < 0.02 and (best is None or s > best[1]):
                best = (vname, s, e, len(speedups))
    if best:
        line += (f"; best so far {best[0]}, at least {best[1]:.1f}x faster on {best[3]} "
                 f"case(s), worst deviation {100 * best[2]:.1f}%")
    return line


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan-study", "run", "status"):
        s = sub.add_parser(name)
        s.add_argument("--workdir", required=True, type=lambda v: Path(os.path.expanduser(v)))
        if name == "plan-study":
            s.add_argument("--ledger", default="data/raw/sweep/ledger.db", type=Path)
        if name == "run":
            s.add_argument("--workers", type=int, default=7)
            s.add_argument("--nprocs", type=int, default=1, help="MPI processes per job")
    args = p.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.cmd == "plan-study":
        plan_study(args.workdir, args.ledger)
    elif args.cmd == "run":
        run(args.workdir, args.workers, args.nprocs)
    else:
        print(status(args.workdir))


if __name__ == "__main__":
    main()
