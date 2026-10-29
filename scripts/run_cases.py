"""
Resumable runner for the SU2 solve of dataset cases, with a one-line status.

A case is solved in three jobs. The first-order start runs from the
freestream. The fast job crosses the slow transient near the stagnation point
with a high CFL number and no multigrid. The settle job brings the solution
to rest, in rounds: after each round the wall heat flux is compared with its
value 4000 iterations earlier, and the job runs another round while it still
moves.

Jobs live in a SQLite table in the work directory. A job waits for the job
named in ``after`` and restarts from its solution. A killed run resumes where
it stopped: ``run`` returns every job left ``running`` to ``pending``, and a job
interrupted after SU2 wrote a restart file continues from that file.

Usage
-----
    python scripts/run_cases.py plan --workdir ~/sweep --cases 183 420
    python scripts/run_cases.py run --workdir ~/sweep --workers 7
    python scripts/run_cases.py status --workdir ~/sweep
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

import numpy as np

from src.cfd.ledger import read_cases
from src.cfd.postprocess import extract_surface, measure_case
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
#                                       recipe
# ============================================================================================

# the density residual never reaches this target, so the iteration count ends every job
NO_RESIDUAL_STOP = {"CONV_RESIDUAL_MINVAL": "-14"}

START = {"order": 1, "iter": 2500,
         "settings": {"CFL_ADAPT_PARAM": "( 0.5, 1.4, 0.1, 5.0 )", **NO_RESIDUAL_STOP}}

# with a CFL cap of 5 the heat flux near the stagnation point climbs for more
# than 30,000 iterations. a cap of 100 without multigrid crosses that in 8000,
# but leaves cases with an easy nose up to 15% off, which the settle job removes
FAST = {"order": 2, "iter": 8000,
        "settings": {"CFL_ADAPT_PARAM": "( 0.5, 1.4, 0.1, 100.0 )", "MGLEVEL": "0", **NO_RESIDUAL_STOP}}

SETTLE = {"order": 2, "iter": 8000, "rounds": 4, "keep_volume": True,
          "settings": {"CFL_ADAPT_PARAM": "( 0.5, 1.4, 0.1, 5.0 )", "MGLEVEL": "2",
                       "VENKAT_LIMITER_COEFF": "0.3", "OUTPUT_WRT_FREQ": "2000",
                       "WRT_SURFACE_OVERWRITE": "NO", **NO_RESIDUAL_STOP}}

# wall stations, in nose radii from the axis, where the heat flux is watched
STATIONS = (0.06, 0.1, 0.2, 0.3, 0.5)
SETTLED = 0.01     # largest relative change at a station over SETTLE_SPAN iterations
SETTLE_SPAN = 4000

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


def add_case(con: sqlite3.Connection, tag: str, case: Case) -> None:
    """Queue the three jobs of one case: ``tag_start``, ``tag_fast`` and ``tag_settle``."""
    add_job(con, f"{tag}_start", case, START)
    add_job(con, f"{tag}_fast", case, FAST, after=f"{tag}_start")
    add_job(con, f"{tag}_settle", case, SETTLE, after=f"{tag}_fast")


def ready_jobs(con: sqlite3.Connection) -> list[sqlite3.Row]:
    """Pending jobs whose predecessor, if they have one, is done.

    Jobs that continue a case come first, so cases finish one by one rather
    than every case getting its start before any is solved.
    """
    return con.execute(
        "SELECT j.* FROM jobs j LEFT JOIN jobs a ON j.after = a.name "
        "WHERE j.status = 'pending' AND (j.after IS NULL OR a.status = 'done') "
        "ORDER BY j.after IS NULL, j.name").fetchall()


# ============================================================================================
#                                       one job
# ============================================================================================

def solve(workdir: str, name: str, case_json: str, recipe_json: str, after: str | None,
          nprocs: int) -> dict:
    """Run one job in its own directory and measure the result. Runs in a worker process."""
    case = Case(**json.loads(case_json))
    recipe = json.loads(recipe_json)
    run_dir = Path(workdir) / name
    write_freq = int(recipe["settings"].get("OUTPUT_WRT_FREQ", 2000))
    if (run_dir / "restart_flow.dat").exists() and (run_dir / "history.csv").exists():
        keep_segment(run_dir, write_freq)
    else:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        if after is not None:
            start = Path(workdir) / after
            shutil.copyfile(start / "case.su2", run_dir / "case.su2")
            shutil.copyfile(start / "restart_flow.dat", run_dir / "solution_flow.dat")

    distro, env = detect_invocation()
    budget = recipe["iter"] * recipe.get("rounds", 1)
    t0 = time.time()
    while True:
        done = sum(len(f.read_text().splitlines()) - 1 for f in run_dir.glob("part_*/history.csv"))
        res = run_case(case, run_dir, mesh_kwargs=mesh_settings(case.R_n, case.R_b),
                       iter_max=recipe["iter"] - done % recipe["iter"], muscl=recipe["order"] == 2,
                       restart_sol=after is not None or done > 0,
                       write_mesh=after is None and done == 0,
                       settings=recipe["settings"], wsl_distro=distro, conda_env=env, nprocs=nprocs)
        if res["returncode"] != 0:
            raise RuntimeError(f"SU2 returned {res['returncode']}, see {res['log']}")
        out = read_history(run_dir)
        if "rounds" not in recipe:
            break
        out["drift"] = station_drift(run_dir, case.R_n)
        out["settled"] = out["drift"] is not None and out["drift"] < SETTLED
        if out["settled"] or out["iterations"] >= budget:
            break
        keep_segment(run_dir, write_freq)
    out["wall_s"] = time.time() - t0
    if recipe["order"] == 2:
        m = measure_case(run_dir, case)
        out.update(qw=m["qw"], p02=m["p02"], standoff=m["standoff"],
                   checks={c.name: c.rel_err for c in m["summary"]["checks"]})
    junk = ("solution_flow.dat", "surface_flow.csv") + (() if recipe.get("keep_volume") else ("flow.vtu",))
    for f in junk:
        (run_dir / f).unlink(missing_ok=True)
    return out


def keep_segment(run_dir: Path, write_freq: int) -> None:
    """Move a finished or interrupted segment's history and numbered surface files into ``part_N``.

    SU2 writes the restart file every ``write_freq`` iterations, so the history
    is cut back to the last multiple of it: the iterations after it are lost
    with the process and run again.
    """
    seg = run_dir / f"part_{len(list(run_dir.glob('part_*'))) + 1}"
    seg.mkdir()
    rows = (run_dir / "history.csv").read_text().splitlines()
    kept = (len(rows) - 1) // write_freq * write_freq
    (seg / "history.csv").write_text("\n".join(rows[:kept + 1]) + "\n")
    (run_dir / "history.csv").unlink()
    for f in run_dir.glob("surface_flow_*.vtu"):
        f.rename(seg / f.name)
    (run_dir / "restart_flow.dat").replace(run_dir / "solution_flow.dat")


def segments(run_dir: Path) -> list[Path]:
    """The ``part_N`` directories in order, then the run directory itself."""
    return [*sorted(run_dir.glob("part_*"), key=lambda d: int(d.name[5:])), run_dir]


def read_history(run_dir: Path) -> dict:
    """Iterations run and the final density residual, over every segment of the job."""
    data = []
    for seg in segments(run_dir):
        with (seg / "history.csv").open() as f:
            rows = list(csv.reader(f))
        head = [h.strip().strip('"') for h in rows[0]]
        data += [[float(v) for v in r] for r in rows[1:] if r]
    return {"iterations": len(data), "rms_rho": data[-1][head.index("rms[Rho]")]}


def heat_flux_at(surface_vtu: Path, R_n: float) -> np.ndarray:
    """Wall heat flux at ``STATIONS``."""
    s = extract_surface(surface_vtu)
    order = np.argsort(s["y"])
    return np.interp(STATIONS, s["y"][order] / R_n, np.abs(s["qw"][order]))


def station_drift(run_dir: Path, R_n: float) -> float | None:
    """Largest relative change of the wall heat flux at a station over the last ``SETTLE_SPAN`` iterations.

    None when no surface file was written that many iterations before the end.
    """
    snapshots, offset = {}, 0
    for seg in segments(run_dir):
        for f in seg.glob("surface_flow_*.vtu"):
            n = int(f.stem[-6:])  # SU2 numbers the file by the index of the last iteration
            snapshots[offset + n + n % 2] = f
        offset += len((seg / "history.csv").read_text().splitlines()) - 1
    earlier = snapshots.get(offset - SETTLE_SPAN)
    if earlier is None:
        return None
    now = heat_flux_at(run_dir / "surface_flow.vtu", R_n)
    return float(np.max(np.abs(now / heat_flux_at(earlier, R_n) - 1)))


# ============================================================================================
#                                       commands
# ============================================================================================

def plan(workdir: Path, ledger: Path, case_ids: list[int] | None, max_theta: float) -> None:
    """Queue cases of a sweep ledger: the listed ones, or every case up to a cone half-angle."""
    rows = read_cases(ledger)
    con = connect(workdir)
    for case_id in case_ids or sorted(rows):
        r = rows[case_id]
        if case_ids is None and r["theta_c_deg"] > max_theta:
            continue
        add_case(con, f"case_{case_id:04d}", Case(**{k: r[k] for k in CASE_KEYS}))
    n = con.execute("SELECT COUNT(*) FROM jobs WHERE name LIKE '%_settle'").fetchone()[0]
    print(f"{n} cases queued")


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
    """One line: job counts, then how many solved cases settled."""
    con = connect(workdir)
    counts = dict(con.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())
    solved = [json.loads(r[0]) for r in con.execute(
        "SELECT result FROM jobs WHERE status = 'done' AND name LIKE '%_settle'")]
    return (f"{counts.get('done', 0)}/{sum(counts.values())} jobs done, {counts.get('running', 0)} "
            f"running, {counts.get('pending', 0)} pending, {counts.get('failed', 0)} failed; "
            f"{len(solved)} cases solved, {sum(s['settled'] for s in solved)} settled")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "status"):
        s = sub.add_parser(name)
        s.add_argument("--workdir", required=True, type=lambda v: Path(os.path.expanduser(v)))
        if name == "plan":
            s.add_argument("--ledger", default="data/raw/sweep/ledger.db", type=Path)
            s.add_argument("--cases", type=int, nargs="*", help="case ids; default every case")
            s.add_argument("--max-theta", type=float, default=60.0,
                           help="largest cone half-angle queued when no case is listed, deg")
        if name == "run":
            s.add_argument("--workers", type=int, default=7)
            s.add_argument("--nprocs", type=int, default=1, help="MPI processes per job")
    args = p.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.cmd == "plan":
        plan(args.workdir, args.ledger, args.cases, args.max_theta)
    elif args.cmd == "run":
        run(args.workdir, args.workers, args.nprocs)
    else:
        print(status(args.workdir))


if __name__ == "__main__":
    main()
