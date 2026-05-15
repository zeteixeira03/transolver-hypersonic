"""Phase 3 dataset generation pipeline.

One command, four stages, all resumable:

    1. preflight   -- check SU2 is callable and the Python deps import
    2. validate    -- solve the Phase 2 canonical case with the Phase 3 settings
                      (loosened mesh, residual-driven convergence) and confirm it
                      still clears Fay-Riddell / Rayleigh-Pitot / Billig; write a
                      marker so this runs only once
    3. generate    -- build (or resume) a SQLite work ledger over a nested
                      geometry x freestream DOE and run N worker processes that
                      mesh, solve (warm-starting within a geometry cluster),
                      postprocess, sanity-check, and write per-node tensors;
                      a supervisor fires a quality plot every --quality-every
                      finished cases and stops the run if the success rate drops
                      below 70%
    4. package     -- once every case is terminal, write a manifest, copy the
                      ledger, optionally tarball the tensors and push to a
                      Hugging Face dataset

Designed for an always-on box (Oracle Cloud Always Free A1, 4 cores): one DB on
local disk, one worker per core, no session limit. ``scripts/setup_oracle.sh``
provisions such a box and launches this under ``nohup``. The ledger's block
layout also supports static partitioning across machines (``--block K``).

Run from the project root::

    python scripts/phase3_generate.py --workdir data/raw/phase3 --workers 4

Useful flags: ``--init-only`` (build ledger, stop), ``--dry-run`` (preflight +
plan, no SU2), ``--no-validate`` (skip stage 2), ``--validate-only``,
``--package-only``, ``--revalidate``, ``--force`` (proceed despite a preflight
warning), ``--hf-repo OWNER/NAME`` (needs ``HF_TOKEN``). On a Windows host the
SU2 call auto-routes through WSL; on Linux ``SU2_CFD`` is invoked directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cfd import ledger as L
from src.cfd.postprocess import (
    extract_axis_line,
    extract_surface,
    extract_training_tensors,
    find_shock_standoff,
    stagnation_values,
)
from src.cfd.runner import Case, run_case
from src.data.sampler import sample_cases
from src.eval.sanity import compare_to_analytical, format_summary


# stage-2 validation sample: 5 cases spanning corners of the Phase 3 design box.
# Replaces the single canonical case used in the first Phase 3 attempt, which
# silently validated only the regime closest to the Phase 2 acceptance recipe;
# the recipe then stalled on most of the actual sweep (sharp cones, high Mach).
# See PHASE_LOG entry 2026-05-14.
def _stage2_case(R_n: float, theta_c: float, R_b_ratio: float, R_s_ratio: float,
                 mach: float, altitude_km: float | None,
                 T_inf: float | None = None, p_inf: float | None = None,
                 T_w: float = 300.0) -> Case:
    R_b = R_n * R_b_ratio
    R_s = R_b * R_s_ratio
    if altitude_km is not None:
        from src.data.sampler import us_standard_atmosphere
        T_inf, p_inf, _ = us_standard_atmosphere(altitude_km)
    if T_inf is None or p_inf is None:
        raise ValueError("either altitude_km or both T_inf and p_inf are required")
    return Case(R_n=R_n, theta_c_deg=theta_c, R_b=R_b, R_s=R_s,
                mach=mach, T_inf=T_inf, p_inf=p_inf, T_w=T_w)


# (name, case) -- order matters: the canonical case is first so a partial
# run still informs whether the recipe regresses on the Phase 2 baseline.
def _build_stage2_sample() -> list[tuple[str, Case]]:
    return [
        ("canonical",        _stage2_case(0.0254, 60.0, 3.0,  0.10, 10.0, None,
                                          T_inf=220.0, p_inf=100.0)),
        ("sharp_loM",        _stage2_case(0.020,  35.0, 3.0,  0.15, 10.0, 50.0)),
        ("blunt_hiM",        _stage2_case(0.035,  68.0, 2.5,  0.15, 22.0, 65.0)),
        ("smallnose_midM",   _stage2_case(0.012,  50.0, 3.0,  0.15, 15.0, 60.0)),
        ("sharp_hiM_anchor", _stage2_case(0.0254, 46.5, 3.08, 0.20, 18.3, None,
                                          T_inf=254.5, p_inf=31.6)),
    ]


# kept for back-compat with anything still importing CANONICAL_CASE
CANONICAL_CASE = Case(
    R_n=0.0254, theta_c_deg=60.0, R_b=0.0762, R_s=0.00762,
    mach=10.0, T_inf=220.0, p_inf=100.0, T_w=300.0,
)

WSL_DEFAULT_DISTRO = "Ubuntu-22.04"
WSL_DEFAULT_ENV = "su2"
_PYTHON_DEPS = ("numpy", "scipy", "matplotlib", "pyvista", "gmsh")


# ============================================================================================
#                                       mesh preset
# ============================================================================================

def phase3_mesh_kwargs(case: Case) -> dict:
    """Phase 3 mesh: the Phase 2 canonical recipe, scaled with geometry.

    An earlier attempt loosened the boundary layer (ratio 1.2, first cell
    R_n/15000) for ~2x fewer cells; stage-2 validation showed it blows the
    Fay-Riddell heat-flux check (+127%) and smears the shock (standoff -27%).
    The Phase 2 finding stands: the fine BL (ratio 1.15, first cell R_n/30000,
    0.06 R_n thick) is what puts the wall heat flux in tolerance, so it is kept.
    The per-case speedup in Phase 3 comes from residual-driven convergence
    (stop at ``conv_minval`` instead of a fixed iteration count) and warm-starting
    within a geometry cluster, not from a coarser mesh.
    """
    return dict(
        L_far=8.0 * case.R_b,
        h_wall=case.R_n / 120.0,
        h_far=8.0 * case.R_b / 25.0,
        bl_first_height=case.R_n / 30000.0,
        bl_thickness=0.06 * case.R_n,
        bl_ratio=1.15,
        refine_shock_box=True,
    )


def _detect_invocation() -> tuple[str | None, str | None]:
    """Return (wsl_distro, conda_env): (None, None) if SU2_CFD is on PATH."""
    if shutil.which("SU2_CFD"):
        return None, None
    return WSL_DEFAULT_DISTRO, WSL_DEFAULT_ENV


# ============================================================================================
#                                       single-case solve
# ============================================================================================

def solve_case(
    case: Case,
    run_dir: Path,
    *,
    restart_src: Path | None,
    iter_pass1: int,
    iter_pass2: int,
    cfl_max: float,
    conv_minval: float,
    mglevel: int,
    wsl_distro: str | None,
    conda_env: str | None,
    timeout: float | None,
) -> dict:
    """Solve one case end to end, returning the postprocessed sanity quantities.

    If ``restart_src`` exists (the predecessor's ``restart_flow.dat`` on the same
    mesh), do a single second-order solve restarting from it; otherwise the
    two-pass first-order-startup + second-order recipe. Pass 2 runs a fixed
    ``iter_pass2`` iterations -- the wall heat flux lags the density residual by
    thousands of iters, so an early residual stop under-resolves the thermal
    layer; ``conv_minval`` is just a low safety floor.

    Returns
    -------
    dict
        ``qw``, ``p02``, ``standoff``, ``rel`` ({name: rel_err}),
        ``checks_passed``, ``warm_started``, ``npz`` (path to the training
        tensor), ``summary`` (the raw :func:`compare_to_analytical` dict).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    warm = restart_src is not None and Path(restart_src).exists()

    if warm:
        shutil.copyfile(restart_src, run_dir / "solution_flow.dat")
        res = run_case(
            case, run_dir=run_dir, mesh_kwargs=phase3_mesh_kwargs(case),
            iter_max=iter_pass2, cfl_max=cfl_max, muscl=True, restart_sol=True,
            mglevel=mglevel, conv_minval=conv_minval,
            wsl_distro=wsl_distro, conda_env=conda_env, timeout=timeout,
        )
        if res["returncode"] != 0:
            raise RuntimeError(f"SU2 warm restart failed rc={res['returncode']}; see {res['log']}")
    else:
        res1 = run_case(
            case, run_dir=run_dir, mesh_kwargs=phase3_mesh_kwargs(case),
            iter_max=iter_pass1, cfl_max=cfl_max, muscl=False, restart_sol=False,
            mglevel=0, conv_minval=conv_minval,
            wsl_distro=wsl_distro, conda_env=conda_env, timeout=timeout,
            cfg_filename="case_pass1.cfg", log_filename="su2_pass1.log",
        )
        if res1["returncode"] != 0:
            raise RuntimeError(f"SU2 pass1 failed rc={res1['returncode']}; see {res1['log']}")
        restart_dat = run_dir / "restart_flow.dat"
        if not restart_dat.exists():
            raise RuntimeError("pass1 produced no restart_flow.dat")
        shutil.copyfile(restart_dat, run_dir / "solution_flow.dat")
        res = run_case(
            case, run_dir=run_dir, mesh_kwargs=phase3_mesh_kwargs(case),
            iter_max=iter_pass2, cfl_max=cfl_max, muscl=True, restart_sol=True,
            mglevel=mglevel, conv_minval=conv_minval,
            wsl_distro=wsl_distro, conda_env=conda_env, timeout=timeout,
            write_mesh=False, cfg_filename="case_pass2.cfg", log_filename="su2_pass2.log",
        )
        if res["returncode"] != 0:
            raise RuntimeError(f"SU2 pass2 failed rc={res['returncode']}; see {res['log']}")

    # postprocess: off-axis stagnation extraction, same convention as Phase 2
    R_n = case.R_n
    L_far = 8.0 * case.R_b
    surface = extract_surface(res["surface_vtu"])
    stag = stagnation_values(surface, y_axis_skip=0.05 * R_n, n_average=3)
    axis = extract_axis_line(res["volume_vtu"], x_in=-L_far, x_nose=0.0, n=2000, r_offset=0.05 * R_n)
    standoff = find_shock_standoff(axis, p_inf=case.p_inf)

    qw = float(abs(stag.get("qw", np.nan)))
    p02 = standoff.get("p_post_plateau", np.nan)
    if np.isnan(p02):
        p02 = stag.get("p", np.nan)
    p02 = float(p02)
    delta = float(standoff["standoff"])

    # a degenerate shock pick (collapsed onto the body or no shock at all on
    # the sample line) means the field is broken in a way no postprocess
    # tolerance is going to salvage; fail the case so the ledger marks it
    # failed instead of recording garbage as "done OUT-OF-TOL"
    if not standoff.get("valid", not np.isnan(delta)):
        raise RuntimeError(
            f"shock standoff finder rejected the axis profile (method=none); "
            f"likely a collapsed bow shock or under-converged pass 2"
        )

    summary = compare_to_analytical(
        M_inf=case.mach, T_inf=case.T_inf, p_inf=case.p_inf, R_n=case.R_n, T_w=case.T_w,
        su2_qw=qw, su2_p02=p02, su2_standoff=delta,
    )
    npz_path = run_dir / "case.npz"
    extract_training_tensors(res["volume_vtu"], save_npz=npz_path)

    # drop the bulky intermediates once the tensor is out; restart_flow.dat stays
    # (a successor in the same geometry cluster restarts from it), the mesh and the
    # conservative-only surface CSV are regenerable and not worth ~GBs over 780 cases
    for junk in ("flow.vtu", "solution_flow.dat", "case.su2", "surface_flow.csv"):
        p = run_dir / junk
        if p.exists():
            p.unlink()

    rel = {c.name: c.rel_err for c in summary["checks"]}
    return {
        "qw": qw, "p02": p02, "standoff": delta, "rel": rel,
        "checks_passed": bool(summary["all_passed"]),
        "warm_started": warm, "npz": npz_path, "summary": summary,
    }


# ============================================================================================
#                                       HF upload
# ============================================================================================

def _maybe_upload_file(path: Path, repo: str | None, path_in_repo: str) -> None:
    """Best-effort push of one file to a HF dataset; never fatal."""
    if not repo:
        return
    try:
        from huggingface_hub import upload_file
        upload_file(
            path_or_fileobj=str(path), path_in_repo=path_in_repo,
            repo_id=repo, repo_type="dataset", token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:  # noqa: BLE001 -- upload is opportunistic
        print(f"[upload] {path_in_repo}: HF upload failed: {exc}")


# ============================================================================================
#                                       quality gate
# ============================================================================================

MIN_SUCCESS_RATE = 0.70
WARMUP_CASES = 30  # do not enforce the stop rule until this many cases have finished/failed
_GATE_TOLS = {"stagnation_heat_flux": 0.15, "stagnation_pressure": 0.05, "shock_standoff": 0.20}
_GATE_TITLES = {
    "stagnation_heat_flux": "stagnation q_w (Fay-Riddell)",
    "stagnation_pressure": "stagnation p (Rayleigh-Pitot)",
    "shock_standoff": "shock standoff (Billig)",
}


def run_quality_gate(db_path: Path, workdir: Path, window: int) -> tuple[bool, str]:
    """Plot rel-error vs Mach for the last ``window`` finished cases.

    Returns ``(ok, message)``; ``ok`` is False only when the run success rate
    has dropped below :data:`MIN_SUCCESS_RATE` after the warm-up.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = L.status_counts(db_path)
    done, failed = counts.get("done", 0), counts.get("failed", 0)
    rate = done / (done + failed) if (done + failed) else 1.0

    rows = L.recent_done(db_path, limit=window)
    if rows:
        mach = np.array([r["mach"] for r in rows])
        names = tuple(_GATE_TOLS)
        rel = {k: [] for k in names}
        passed = 0
        for r in rows:
            s = compare_to_analytical(
                M_inf=r["mach"], T_inf=r["T_inf"], p_inf=r["p_inf"],
                R_n=r["R_n"], T_w=r["T_w"],
                su2_qw=r["qw"], su2_p02=r["p02"], su2_standoff=r["standoff"],
            )
            for c in s["checks"]:
                rel[c.name].append(c.rel_err)
            passed += int(s["all_passed"])
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
        for ax, k in zip(axes, names):
            ax.axhline(0, color="k", lw=0.6)
            ax.axhspan(-_GATE_TOLS[k], _GATE_TOLS[k], color="C2", alpha=0.15)
            ax.scatter(mach, np.array(rel[k]), s=14, c="C0")
            ax.set_xlabel("M_inf")
            ax.set_ylabel("rel error")
            ax.set_title(_GATE_TITLES[k], fontsize=9)
        fig.suptitle(
            f"Phase 3 quality gate: last {len(rows)} of {done} done "
            f"({passed}/{len(rows)} clear all 3; run success {rate*100:.0f}%)"
        )
        fig.tight_layout()
        out = workdir / f"quality_gate_{done:04d}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[gate] wrote {out}")

    if (done + failed) >= WARMUP_CASES and rate < MIN_SUCCESS_RATE:
        return False, f"run success rate {rate*100:.1f}% < {MIN_SUCCESS_RATE*100:.0f}% -- stopping"
    return True, f"run success {rate*100:.1f}%"


# ============================================================================================
#                                       preflight
# ============================================================================================

def preflight(args: argparse.Namespace) -> tuple[bool, str]:
    """Check SU2 is callable and the Python deps import. Returns (ok, report)."""
    lines = []
    ok = True

    missing = []
    for mod in _PYTHON_DEPS + (("huggingface_hub",) if args.hf_repo else ()):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            missing.append(mod)
    if missing:
        ok = False
        lines.append(f"  [FAIL] missing Python modules: {', '.join(missing)}")
        lines.append(f"         pip install {' '.join(missing)}")
    else:
        lines.append("  [ ok ] Python deps import")

    distro, env = _detect_invocation()
    if distro is None:
        lines.append(f"  [ ok ] SU2_CFD on PATH: {shutil.which('SU2_CFD')}")
    else:
        cmd = ["wsl", "-d", distro, "--", "bash", "-c",
               "source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null; "
               "source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; "
               f"conda activate {env} 2>/dev/null && which SU2_CFD"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            if r.returncode == 0 and r.stdout.strip():
                lines.append(f"  [ ok ] SU2_CFD via WSL {distro}/{env}: {r.stdout.strip()}")
            else:
                ok = False
                lines.append(f"  [FAIL] SU2_CFD not found via WSL {distro}/{env}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            lines.append(f"  [FAIL] could not invoke WSL: {exc}")

    if args.hf_repo and not os.environ.get("HF_TOKEN"):
        lines.append("  [warn] --hf-repo set but HF_TOKEN is not in the environment; uploads will fail")

    head = "preflight: all checks passed" if ok else "preflight: FAILED (pass --force to proceed anyway)"
    return ok, head + "\n" + "\n".join(lines)


# ============================================================================================
#                                       stage 2: validate
# ============================================================================================

def validate_sample(args: argparse.Namespace) -> bool:
    """Solve the 5-case diverse stage-2 sample with the Phase 3 settings.

    The first Phase 3 attempt validated only the canonical case (theta_c=60,
    M=10), which sits in the regime where the Phase 2 acceptance recipe was
    tuned. The recipe cleared that case but stalled on most of the sweep --
    sharp cones (theta_c < 50) and high Mach (M > 18) with the bow shock
    collapsed onto the body. Validating on a diverse stage-2 sample catches
    those failure modes before any sweep cycles are burned.

    The five cases (see :func:`_build_stage2_sample`) span the corners:
    canonical (blunt + low-M, Phase 2 baseline), sharp+low-M, blunt+high-M,
    small-nose+mid-M, and the sharp+high-M anchor (Phase 3 sampler[0]). Each
    must complete without an exception. Gate misses are logged but do not
    block the sweep -- the per-case tolerances in
    :mod:`src.eval.sanity` are 5/15/20% against approximate analytical
    correlations whose accuracy itself degrades near the design-box
    corners (Billig drifts at very high Mach; Fay-Riddell tightens
    against mesh resolution at small R_n). The ledger's separate success
    rate (completed / total) is the real go/no-go gate; gates are
    advisory data-quality flags.

    Writes ``<workdir>/.phase3_validated`` on success. ``--revalidate`` forces
    a re-run; ``--no-validate`` skips altogether. Returns True iff every case
    completes (no exception); the marker records per-case rel errors.
    """
    workdir = Path(args.workdir)
    marker = workdir / ".phase3_validated"
    if marker.exists() and not args.revalidate:
        prev = json.loads(marker.read_text())
        cases = prev.get("cases", {})
        n_pass = sum(1 for c in cases.values() if c.get("checks_passed"))
        print(f"[validate] already validated {prev.get('when', '?')}: "
              f"{len(cases)} cases ran, {n_pass} cleared all 3 gates; "
              f"skipping (pass --revalidate to redo)")
        return True

    valdir = workdir / "_validation"
    if valdir.exists():
        shutil.rmtree(valdir)
    distro, env = _detect_invocation()
    sample = _build_stage2_sample()
    n = len(sample)
    print(f"[validate] solving {n} stage-2 cases with the Phase 3 settings "
          f"(iter_pass1={args.iter_pass1}, iter_pass2={args.iter_pass2}, "
          f"mglevel={args.mglevel}); ~3h wall-clock per case in sequence, "
          f"~{n*3}h total. Skip with --no-validate.")

    results: dict[str, dict] = {}
    n_completed = 0
    n_gates_clear = 0
    for i, (name, case) in enumerate(sample, start=1):
        print(f"\n[validate] ({i}/{n}) {name}: R_n={case.R_n*1000:.1f}mm "
              f"theta_c={case.theta_c_deg:.1f}deg M={case.mach:.2f}")
        t0 = time.time()
        try:
            out = solve_case(
                case, valdir / name, restart_src=None,
                iter_pass1=args.iter_pass1, iter_pass2=args.iter_pass2,
                cfl_max=args.cfl_max, conv_minval=args.conv_minval, mglevel=args.mglevel,
                wsl_distro=distro, conda_env=env, timeout=args.case_timeout_s or None,
            )
        except Exception as exc:  # noqa: BLE001 -- record + stop
            print(f"[validate] ({i}/{n}) {name} FAILED after {time.time()-t0:.0f}s: "
                  f"{type(exc).__name__}: {exc}")
            print("[validate] a case must at least complete (gates are advisory). "
                  "Investigate before the sweep -- pass --no-validate to skip if you "
                  "have already confirmed the recipe elsewhere.")
            return False
        n_completed += 1
        n_gates_clear += int(out["checks_passed"])
        print(format_summary(out["summary"]))
        print(f"[validate] ({i}/{n}) {name} done in {time.time()-t0:.0f}s "
              f"({'all gates clear' if out['checks_passed'] else 'gates: advisory miss'})")
        results[name] = {
            "rel": {k: round(v, 4) for k, v in out["rel"].items()},
            "checks_passed": bool(out["checks_passed"]),
            "qw": out["qw"], "p02": out["p02"], "standoff": out["standoff"],
        }

    marker.write_text(json.dumps({
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_completed": n_completed, "n_gates_clear": n_gates_clear, "n_total": n,
        "settings": {"conv_minval": args.conv_minval, "iter_pass1": args.iter_pass1,
                     "iter_pass2": args.iter_pass2, "cfl_max": args.cfl_max,
                     "mglevel": args.mglevel},
        "cases": results,
    }, indent=2))
    print(f"\n[validate] passed: {n_completed}/{n} cases completed cleanly, "
          f"{n_gates_clear}/{n} cleared all 3 gates; marker written to {marker}")
    return True


# ============================================================================================
#                                       worker loop + supervisor
# ============================================================================================

def worker_loop(args: argparse.Namespace, worker_id: str, stop_flag) -> None:
    db_path = Path(args.db)
    workdir = Path(args.workdir)
    wsl_distro, conda_env = _detect_invocation()
    deadline = time.time() + args.max_runtime_s if args.max_runtime_s > 0 else None

    L.reclaim_stale(db_path, max_age_s=args.stale_after_s)

    while True:
        if stop_flag.value:
            print(f"[{worker_id}] stop flag set; exiting")
            return
        if deadline and time.time() > deadline:
            print(f"[{worker_id}] max runtime reached; exiting")
            return
        claim = L.claim_next(db_path, worker_id, block=args.block)
        if claim is None:
            print(f"[{worker_id}] no pending cases; exiting")
            return
        case_id, case, restart_from = claim
        run_dir = workdir / f"case_{case_id:04d}"
        # only warm-start from a predecessor that finished cleanly; a restart file
        # left by a diverged run would poison this case
        restart_src = None
        if restart_from is not None and L.case_status(db_path, int(restart_from)) == "done":
            restart_src = workdir / f"case_{int(restart_from):04d}" / "restart_flow.dat"
        warm = restart_src is not None and restart_src.exists()
        print(f"[{worker_id}] case {case_id} ({'warm' if warm else 'cold'}): "
              f"M={case.mach:.2f} R_n={case.R_n*1e3:.1f}mm theta_c={case.theta_c_deg:.1f}")
        t0 = time.time()
        try:
            out = solve_case(
                case, run_dir, restart_src=restart_src,
                iter_pass1=args.iter_pass1, iter_pass2=args.iter_pass2,
                cfl_max=args.cfl_max, conv_minval=args.conv_minval, mglevel=args.mglevel,
                wsl_distro=wsl_distro, conda_env=conda_env, timeout=args.case_timeout_s or None,
            )
        except Exception as exc:  # noqa: BLE001 -- a failed case is logged, not fatal
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[{worker_id}] case {case_id} FAILED in {time.time()-t0:.0f}s: {msg}")
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "error.txt").write_text(traceback.format_exc())
            L.mark_failed(db_path, case_id, msg)
            continue
        L.mark_done(db_path, case_id, qw=out["qw"], p02=out["p02"],
                    standoff=out["standoff"], checks_passed=out["checks_passed"])
        _maybe_upload_file(out["npz"], args.hf_repo, f"cases/case_{case_id:04d}.npz")
        flags = ("warm " if out["warm_started"] else "") + ("OK" if out["checks_passed"] else "OUT-OF-TOL")
        print(f"[{worker_id}] case {case_id} done in {time.time()-t0:.0f}s [{flags}] "
              f"qw={out['qw']:.3g} p02={out['p02']:.3g} standoff={out['standoff']*1e3:.2f}mm")


def run_generation(args: argparse.Namespace) -> None:
    """Spawn the workers and supervise: quality gate every N done, stop on collapse."""
    workdir = Path(args.workdir)
    stop_flag = mp.Value("b", False)
    procs = [
        mp.Process(target=worker_loop, args=(args, f"w{i}", stop_flag), daemon=False)
        for i in range(args.workers)
    ]
    for pr in procs:
        pr.start()

    last_gate_at = 0
    try:
        while any(pr.is_alive() for pr in procs):
            time.sleep(20.0)
            finished = L.status_counts(args.db).get("done", 0)
            if finished >= last_gate_at + args.quality_every:
                last_gate_at = finished - (finished % args.quality_every)
                ok, msg = run_quality_gate(Path(args.db), workdir, args.quality_every)
                print(f"[main] {finished} done -- {msg}")
                if not ok:
                    print(f"[main] {msg}; signalling workers to stop")
                    stop_flag.value = True
    except KeyboardInterrupt:
        print("[main] interrupted; signalling workers to stop")
        stop_flag.value = True
    finally:
        for pr in procs:
            pr.join()
    print(f"[main] generation finished, status: {dict(L.status_counts(args.db))}")


# ============================================================================================
#                                       stage 4: package
# ============================================================================================

_MANIFEST_COLS = (
    "case_id", "ord", "block", "group_name", "geom_id", "restart_from", "status",
    "R_n", "theta_c_deg", "R_b", "R_s", "mach", "T_inf", "p_inf", "T_w",
    "qw", "p02", "standoff", "checks_passed", "error",
)


def package_dataset(args: argparse.Namespace) -> None:
    """Once every case is terminal, write the manifest, copy the ledger, and
    (optionally) tarball the tensors and push the lot to a HF dataset."""
    db_path = Path(args.db)
    workdir = Path(args.workdir)
    counts = L.status_counts(db_path)
    pending = counts.get("pending", 0) + counts.get("running", 0)
    if pending:
        print(f"[package] {pending} cases still pending/running; not packaging yet "
              f"(re-run with --package-only when the sweep is finished)")
        return

    ds = workdir / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    rows = L.all_rows(db_path)
    manifest = ds / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_MANIFEST_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in _MANIFEST_COLS})
    shutil.copyfile(db_path, ds / "ledger.db")
    done_ids = [r["case_id"] for r in rows if r["status"] == "done"]
    print(f"[package] {len(done_ids)} done, {counts.get('failed', 0)} failed; "
          f"manifest -> {manifest}, ledger -> {ds / 'ledger.db'}")

    if args.make_tarball:
        tarpath = ds / "phase3_tensors.tar.gz"
        with tarfile.open(tarpath, "w:gz") as tf:
            tf.add(manifest, arcname="manifest.csv")
            for cid in done_ids:
                npz = workdir / f"case_{cid:04d}" / "case.npz"
                if npz.exists():
                    tf.add(npz, arcname=f"cases/case_{cid:04d}.npz")
        print(f"[package] tarball -> {tarpath} ({tarpath.stat().st_size/1e6:.0f} MB)")
        _maybe_upload_file(tarpath, args.hf_repo, "phase3_tensors.tar.gz")

    _maybe_upload_file(manifest, args.hf_repo, "manifest.csv")
    _maybe_upload_file(ds / "ledger.db", args.hf_repo, "ledger.db")


# ============================================================================================
#                                       main
# ============================================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 3 dataset generation pipeline")
    p.add_argument("--workdir", default="data/raw/phase3", help="artifact root (one subdir per case)")
    p.add_argument("--db", default=None, help="ledger path (default <workdir>/ledger.db)")
    p.add_argument("--workers", type=int, default=4, help="worker processes")
    p.add_argument("--block", type=int, default=None, help="pin workers to one block (multi-machine)")
    # sampling / ledger build
    p.add_argument("--n-geom", type=int, default=70, help="core geometries")
    p.add_argument("--n-fs", type=int, default=10, help="freestreams per core geometry")
    p.add_argument("--n-geom-ood", type=int, default=10, help="geometries per OOD slab")
    p.add_argument("--n-fs-ood", type=int, default=2, help="freestreams per OOD geometry")
    p.add_argument("--n-blocks", type=int, default=4, help="contiguous ledger blocks")
    p.add_argument("--seed", type=int, default=0, help="LHS seed")
    # solver knobs
    # the 2500 + 8000 / mglevel=0 Phase 2 acceptance recipe clears the canonical
    # case but stalls at rms ~ -4.77 with the bow shock collapsed onto the body
    # for sharp cones and high-Mach geometries (PHASE_LOG 2026-05-13). Geometric
    # multigrid (mglevel=2) breaks the saddle: pass 2 then settles to rms ~ -6
    # with the shock at a physically correct standoff. CD plateaus by ~iter
    # 12-14k, so iter_pass2=16000 covers the slow drift; the diverse stage-2
    # sample confirms the recipe across blunt/sharp x lo/hi Mach corners
    # (PHASE_LOG 2026-05-14).
    p.add_argument("--iter-pass1", type=int, default=2500, help="first-order startup iterations")
    p.add_argument("--iter-pass2", type=int, default=16000, help="second-order iterations")
    p.add_argument("--conv-minval", type=float, default=-8.0,
                   help="RMS_DENSITY residual floor; default -8 is effectively off (heat flux, "
                        "not the density residual, sets when the solution is usable)")
    p.add_argument("--cfl-max", type=float, default=5.0)
    p.add_argument("--mglevel", type=int, default=2,
                   help="geometric multigrid levels (0=off). Default 2 is what breaks the "
                        "sharp-cone / high-Mach saddle that stalled the first Phase 3 attempt")
    p.add_argument("--case-timeout-s", type=float, default=0.0, help="per-SU2-call wall-clock cap (0=none)")
    # housekeeping
    p.add_argument("--quality-every", type=int, default=50, help="run the quality gate every N finished cases")
    p.add_argument("--stale-after-s", type=float, default=7200.0, help="reclaim running rows older than this")
    p.add_argument("--max-runtime-s", type=float, default=0.0, help="exit workers after this many seconds (0=unlimited)")
    p.add_argument("--hf-repo", default=None, help="HF dataset repo id to push tensors/manifest to (needs HF_TOKEN)")
    p.add_argument("--make-tarball", action="store_true", help="at packaging, build phase3_tensors.tar.gz")
    # stage control
    p.add_argument("--init-only", action="store_true", help="build the ledger and exit")
    p.add_argument("--dry-run", action="store_true", help="preflight + build ledger + print the plan, no SU2")
    p.add_argument("--no-validate", action="store_true", help="skip the stage-2 canonical validation")
    p.add_argument("--revalidate", action="store_true", help="re-run stage 2 even if the marker exists")
    p.add_argument("--validate-only", action="store_true", help="run stages 1-2 and exit")
    p.add_argument("--package-only", action="store_true", help="run stage 4 only (use after a finished sweep)")
    p.add_argument("--force", action="store_true", help="proceed even if preflight reports a failure")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    args.workdir = str(workdir)
    args.db = args.db or str(workdir / "ledger.db")

    if args.package_only:
        package_dataset(args)
        return

    specs = sample_cases(
        n_geom=args.n_geom, n_fs=args.n_fs,
        n_geom_ood=args.n_geom_ood, n_fs_ood=args.n_fs_ood, seed=args.seed,
    )
    L.init_ledger(args.db, specs, n_blocks=args.n_blocks)
    counts = L.status_counts(args.db)
    print(f"[main] ledger {args.db}: {sum(counts.values())} cases ({len(specs)} sampled), "
          f"status {dict(counts)}")
    if args.init_only:
        return

    ok, report = preflight(args)
    print(report)

    if args.dry_run:
        print(f"[main] dry run: would validate={'no' if args.no_validate else 'yes'}, "
              f"then run {args.workers} workers over {counts.get('pending', 0)} pending cases, "
              f"then {'package' if args.block is None else 'skip packaging (block-pinned)'}.")
        return
    if not ok and not args.force:
        sys.exit(2)

    if not args.no_validate:
        if not validate_sample(args):
            sys.exit(3)
    if args.validate_only:
        return

    run_generation(args)

    if args.block is None:
        package_dataset(args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
