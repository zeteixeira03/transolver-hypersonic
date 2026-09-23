"""Dataset generation pipeline.

One command runs four stages, and every stage resumes after an interruption:

    1. preflight   -- check that SU2 runs and that every Python dependency
                      imports
    2. validate    -- solve a small sample of cases with the sweep settings and
                      confirm that they still clear Fay-Riddell, Rayleigh-Pitot
                      and Billig. a marker file keeps this to one run.
    3. generate    -- build or resume a SQLite ledger over the nested sample of
                      geometries and freestreams, then run N worker processes.
                      each worker meshes, solves, warm-starts inside a geometry
                      cluster, reads the outputs, checks them, and writes the
                      per-node tensor. a supervisor draws a quality plot every
                      --quality-every finished cases and stops the run when the
                      success rate falls below 70%.
    4. package     -- once every case has reached a terminal state, write a
                      manifest, copy the ledger, and on request build a tarball
                      of the tensors and push it to a Hugging Face dataset

The pipeline targets a machine that stays on, such as an Oracle Cloud Always
Free A1 with four cores. It keeps one database on local disk and one worker per
core, and no session limit cuts it short. ``scripts/setup_oracle.sh`` provisions
such a machine and starts this under ``nohup``. The block column of the ledger
also lets several machines split the sweep between them, through ``--block K``.

Run it from the project root::

    python scripts/generate_dataset.py --workdir data/raw/sweep --workers 4

The flags worth knowing: ``--init-only`` builds the ledger and stops,
``--dry-run`` runs the preflight and prints the plan without calling SU2,
``--no-validate`` skips the validation stage, and ``--validate-only``,
``--package-only`` and ``--revalidate`` each run one stage alone. ``--force``
proceeds past a preflight warning. ``--hf-repo OWNER/NAME`` pushes the result
and needs ``HF_TOKEN`` in the environment. On a Windows host the SU2 call routes
through the Windows Subsystem for Linux. On Linux it calls ``SU2_CFD`` directly.
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


# the validation sample: 5 cases that reach the corners of the sweep design box.
# one case near the middle of the box cannot show whether the recipe holds at
# a sharp cone or at high Mach, and those are where it strains.
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


# pairs of (name, case). the order matters: the canonical case comes first, so
# even a partial run says whether the recipe still holds on the known baseline.
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


# the canonical case under its own name, for callers that want it alone
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

def sweep_mesh_kwargs(case: Case) -> dict:
    """Mesh settings for the sweep: the validated recipe, scaled with the geometry.

    The boundary layer must stay fine. It uses a growth ratio of 1.15, a first
    cell at R_n/30000 and a total thickness of 0.06 R_n, and that resolution is
    what holds the wall heat flux inside its tolerance. A coarser boundary layer
    halves the cell count and costs far more than it saves: at a ratio of 1.2 and
    a first cell at R_n/15000 the heat flux misses Fay-Riddell by 127% and the
    shock smears, with the standoff 27% low.

    The sweep gets its speed elsewhere. It stops on the residual at
    ``conv_minval`` rather than at a fixed iteration count, and it warm-starts
    inside a geometry cluster.
    """
    # a hard floor under the first cell height. R_n/30000 alone over-resolves
    # the boundary layer on a small nose, and the heat flux then reads 22 to
    # 26% above Fay-Riddell at R_n = 12 mm. the floor of 5e-7 m binds only
    # below R_n = 15 mm, so every larger nose keeps the full R_n/30000 spacing.
    bl_first = max(case.R_n / 30000.0, 5e-7)
    return dict(
        L_far=8.0 * case.R_b,
        h_wall=case.R_n / 120.0,
        h_far=8.0 * case.R_b / 25.0,
        bl_first_height=bl_first,
        bl_thickness=0.06 * case.R_n,
        bl_ratio=1.15,
        refine_shock_box=True,
    )


def _detect_invocation() -> tuple[str | None, str | None]:
    """Return the distribution and conda environment to run SU2 in, for this host.

    A SU2_CFD already on the executable search path returns ``(None, None)``,
    and the caller invokes it directly.
    """
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

    Where ``restart_src`` exists, it is the ``restart_flow.dat`` of a case on the
    same mesh, and one second-order solve restarts from it. Otherwise the case
    takes the two-pass recipe: a first-order pass to start, then a second-order
    pass.

    The second pass runs a fixed ``iter_pass2`` iterations. The wall heat flux
    settles thousands of iterations after the density residual does, so a stop
    driven by the residual leaves the thermal layer under-resolved.
    ``conv_minval`` sits far below that point and acts as a safety floor alone.

    Returns
    -------
    dict
        ``qw``, ``p02`` and ``standoff`` as measured values. ``rel`` as
        ``{name: relative error}``. ``checks_passed`` and ``warm_started`` as
        flags. ``npz`` as the path of the training tensor. ``summary`` as the
        full dict from :func:`compare_to_analytical`.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    warm = restart_src is not None and Path(restart_src).exists()

    if warm:
        shutil.copyfile(restart_src, run_dir / "solution_flow.dat")
        res = run_case(
            case, run_dir=run_dir, mesh_kwargs=sweep_mesh_kwargs(case),
            iter_max=iter_pass2, cfl_max=cfl_max, muscl=True, restart_sol=True,
            mglevel=mglevel, conv_minval=conv_minval,
            wsl_distro=wsl_distro, conda_env=conda_env, timeout=timeout,
        )
        if res["returncode"] != 0:
            raise RuntimeError(f"SU2 warm restart failed rc={res['returncode']}; see {res['log']}")
    else:
        res1 = run_case(
            case, run_dir=run_dir, mesh_kwargs=sweep_mesh_kwargs(case),
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
            case, run_dir=run_dir, mesh_kwargs=sweep_mesh_kwargs(case),
            iter_max=iter_pass2, cfl_max=cfl_max, muscl=True, restart_sol=True,
            mglevel=mglevel, conv_minval=conv_minval,
            wsl_distro=wsl_distro, conda_env=conda_env, timeout=timeout,
            write_mesh=False, cfg_filename="case_pass2.cfg", log_filename="su2_pass2.log",
        )
        if res["returncode"] != 0:
            raise RuntimeError(f"SU2 pass2 failed rc={res['returncode']}; see {res['log']}")

    # read the outputs. every stagnation quantity comes from off the axis, on the
    # same convention the single-case validation uses.
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

    # no valid shock pick means the field is broken, either with the shock
    # collapsed onto the body or with no shock on the sample line at all. no
    # tolerance on the output side can recover such a field. fail the case,
    # so that the ledger records a failure rather than a plausible number.
    if not standoff.get("valid", not np.isnan(delta)):
        raise RuntimeError(
            f"shock standoff finder rejected the axis profile (method=none); "
            f"likely a collapsed bow shock or under-converged pass 2"
        )

    summary = compare_to_analytical(
        M_inf=case.mach, T_inf=case.T_inf, p_inf=case.p_inf, R_n=case.R_n, T_w=case.T_w,
        su2_qw=qw, su2_p02=p02, su2_standoff=delta, theta_c_deg=case.theta_c_deg,
    )
    npz_path = run_dir / "case.npz"
    extract_training_tensors(res["volume_vtu"], case=case, save_npz=npz_path)

    # delete the large intermediate files once the tensor is written.
    # restart_flow.dat stays, because the next case in the geometry cluster
    # restarts from it. the mesh and the thin surface .csv file can both be
    # rebuilt, and together they would cost gigabytes across the sweep.
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
#                                   hugging face upload
# ============================================================================================

def _maybe_upload_file(path: Path, repo: str | None, path_in_repo: str) -> None:
    """Push one file to a Hugging Face dataset. A failure here never stops the run."""
    if not repo:
        return
    try:
        from huggingface_hub import upload_file
        upload_file(
            path_or_fileobj=str(path), path_in_repo=path_in_repo,
            repo_id=repo, repo_type="dataset", token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:  # noqa: BLE001 -- an upload never stops the run
        print(f"[upload] {path_in_repo}: upload failed: {exc}")


# ============================================================================================
#                                       quality gate
# ============================================================================================

MIN_SUCCESS_RATE = 0.70
WARMUP_CASES = 30  # the stop rule waits until this many cases end
# the thresholds that mark a case as broken rather than inaccurate. they sit far
# outside the bias the readers themselves carry: on a healthy case the median of
# the pressure plateau reads 5 to 10% low, and the standoff finder 10 to 15% low.
# they also sit clear of the collapsed-shock cluster, which shows a heat flux 100
# to 600% high, a stagnation pressure 50 to 95% low, and a standoff 45 to 80% low.
_CATASTROPHIC = {
    "qw_rel_max":  0.50,    # a heat flux 50% high means the boundary layer diverged
    "p02_rel_min": -0.25,   # a stagnation pressure 25% low means that plateau collapsed
    "sd_rel_min":  -0.30,   # a standoff 30% low means the bow shock sits on the body
}
MAX_CATASTROPHIC_RATE = 0.30  # halt if > this fraction of the last window are catastrophic
_GATE_TOLS = {"stagnation_heat_flux": 0.15, "stagnation_pressure": 0.05, "shock_standoff": 0.20}
_GATE_TITLES = {
    "stagnation_heat_flux": "stagnation q_w (Fay-Riddell)",
    "stagnation_pressure": "stagnation p (Rayleigh-Pitot)",
    "shock_standoff": "shock standoff (Billig)",
}


def run_quality_gate(db_path: Path, workdir: Path, window: int) -> tuple[bool, str]:
    """Plot the relative error against Mach number over the last ``window`` cases.

    Returns ``(ok, message)``. ``ok`` is False only once the success rate of the
    run falls below :data:`MIN_SUCCESS_RATE`, and only after the warm-up window.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = L.status_counts(db_path)
    done, failed = counts.get("done", 0), counts.get("failed", 0)
    rate = done / (done + failed) if (done + failed) else 1.0

    rows = L.recent_done(db_path, limit=window)
    catastrophic_n = 0
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
                theta_c_deg=r["theta_c_deg"],
            )
            err = {c.name: c.rel_err for c in s["checks"]}
            for c in s["checks"]:
                rel[c.name].append(c.rel_err)
            passed += int(s["all_passed"])
            if (err["stagnation_heat_flux"] > _CATASTROPHIC["qw_rel_max"]
                    or err["stagnation_pressure"] < _CATASTROPHIC["p02_rel_min"]
                    or err["shock_standoff"] < _CATASTROPHIC["sd_rel_min"]):
                catastrophic_n += 1
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
        for ax, k in zip(axes, names):
            ax.axhline(0, color="k", lw=0.6)
            ax.axhspan(-_GATE_TOLS[k], _GATE_TOLS[k], color="C2", alpha=0.15)
            ax.scatter(mach, np.array(rel[k]), s=14, c="C0")
            ax.set_xlabel("M_inf")
            ax.set_ylabel("rel error")
            ax.set_title(_GATE_TITLES[k], fontsize=9)
        fig.suptitle(
            f"Quality gate: last {len(rows)} of {done} done "
            f"({passed}/{len(rows)} clear all 3; run success {rate*100:.0f}%)"
        )
        fig.tight_layout()
        out = workdir / f"quality_gate_{done:04d}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[gate] wrote {out}")

    if (done + failed) >= WARMUP_CASES and rate < MIN_SUCCESS_RATE:
        return False, f"run success rate {rate*100:.1f}% < {MIN_SUCCESS_RATE*100:.0f}% -- stopping"
    if rows and len(rows) >= WARMUP_CASES:
        cat_rate = catastrophic_n / len(rows)
        if cat_rate > MAX_CATASTROPHIC_RATE:
            return False, (f"catastrophic gate-break rate {cat_rate*100:.0f}% over the "
                           f"last {len(rows)} done cases "
                           f"(> {MAX_CATASTROPHIC_RATE*100:.0f}%) -- stopping; the recipe "
                           f"is producing collapsed-shock solutions")
    return True, (f"run success {rate*100:.1f}% / catastrophic "
                  f"{catastrophic_n}/{len(rows) if rows else 0}")


# ============================================================================================
#                                       preflight
# ============================================================================================

def preflight(args: argparse.Namespace) -> tuple[bool, str]:
    """Check that SU2 runs and that every Python dependency imports.

    Returns ``(ok, report)``.
    """
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
        lines.append("  [ ok ] every Python dependency imports")

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
                lines.append(f"  [ ok ] SU2_CFD in {distro}/{env}: {r.stdout.strip()}")
            else:
                ok = False
                lines.append(f"  [FAIL] SU2_CFD not found in {distro}/{env}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            lines.append(f"  [FAIL] could not reach the Linux subsystem: {exc}")

    if args.hf_repo and not os.environ.get("HF_TOKEN"):
        lines.append("  [warn] --hf-repo set but HF_TOKEN is not in the environment; uploads will fail")

    head = "preflight: all checks passed" if ok else "preflight: FAILED (pass --force to proceed anyway)"
    return ok, head + "\n" + "\n".join(lines)


# ============================================================================================
#                                       stage 2: validate
# ============================================================================================

def validate_sample(args: argparse.Namespace) -> bool:
    """Solve the five-case validation sample with the sweep settings.

    The sample must span the corners of the design box. A recipe tuned near the
    middle can clear the canonical case at theta_c = 60 and Mach 10 and still
    stall across most of the sweep, with the bow shock collapsed onto the body
    at a sharp cone below 50 degrees or above Mach 18. Five cases catch that
    before the sweep spends any solver time.

    :func:`_build_stage2_sample` builds them: the blunt low-Mach canonical case,
    a sharp low-Mach case, a blunt high-Mach case, a small-nose case at middle
    Mach, and a sharp high-Mach anchor. Every one must run to completion without
    raising.

    A missed gate is logged and does not stop the sweep. The tolerances in
    :mod:`src.eval.sanity` are 5, 15 and 20% against approximate correlations
    that themselves lose accuracy at the corners of the box. Billig drifts at
    very high Mach, and Fay-Riddell grows sensitive to mesh resolution at a
    small nose radius. The success rate in the ledger, which is completed over
    total, decides whether the sweep proceeds. The gates flag data quality and
    advise.

    A success writes ``<workdir>/.validated``. ``--revalidate`` runs the stage
    again and ``--no-validate`` skips it. Returns True only when every case
    completes, and the marker file records the relative error of each one.
    """
    workdir = Path(args.workdir)
    marker = workdir / ".validated"
    only_case = getattr(args, "validate_case", None)
    if marker.exists() and not args.revalidate and only_case is None:
        prev = json.loads(marker.read_text())
        cases = prev.get("cases", {})
        n_pass = sum(1 for c in cases.values() if c.get("checks_passed"))
        print(f"[validate] already validated {prev.get('when', '?')}: "
              f"{len(cases)} cases ran, {n_pass} cleared all 3 gates; "
              f"skipping (pass --revalidate to redo, or --validate-case NAME "
              f"to re-run one case and merge results)")
        return True

    valdir = workdir / "_validation"
    distro, env = _detect_invocation()
    full_sample = _build_stage2_sample()
    if only_case is not None:
        names = [n for n, _ in full_sample]
        if only_case not in names:
            raise SystemExit(f"[validate] --validate-case '{only_case}' is not "
                             f"one of {names}")
        sample = [(n, c) for n, c in full_sample if n == only_case]
        case_valdir = valdir / only_case
        if case_valdir.exists():
            shutil.rmtree(case_valdir)
    else:
        sample = full_sample
        if valdir.exists():
            shutil.rmtree(valdir)
    n = len(sample)
    print(f"[validate] solving {n} stage-2 cases with the sweep settings "
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

    if only_case is not None and marker.exists():
        prev = json.loads(marker.read_text())
        merged = dict(prev.get("cases", {}))
        merged.update(results)
        n_total = len(merged)
        n_done = sum(1 for _ in merged)
        n_clear = sum(1 for c in merged.values() if c.get("checks_passed"))
        payload = {
            "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_completed": n_done, "n_gates_clear": n_clear, "n_total": n_total,
            "settings": {"conv_minval": args.conv_minval, "iter_pass1": args.iter_pass1,
                         "iter_pass2": args.iter_pass2, "cfl_max": args.cfl_max,
                         "mglevel": args.mglevel},
            "cases": merged,
            "last_updated_case": only_case,
        }
    else:
        payload = {
            "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_completed": n_completed, "n_gates_clear": n_gates_clear, "n_total": n,
            "settings": {"conv_minval": args.conv_minval, "iter_pass1": args.iter_pass1,
                         "iter_pass2": args.iter_pass2, "cfl_max": args.cfl_max,
                         "mglevel": args.mglevel},
            "cases": results,
        }
    marker.write_text(json.dumps(payload, indent=2))
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
        # restart only from a case that both finished without error and cleared the
        # three analytical gates. a case that finished outside tolerance
        # usually carries a collapsed shock, and a restart from it spreads
        # that collapsed field through the whole geometry cluster. of 39
        # broken cases measured in one sweep, 38 had warm-started this way.
        restart_src = None
        if restart_from is not None:
            prev = L.case_row(db_path, int(restart_from))
            if prev is not None and prev["status"] == "done" and prev["checks_passed"]:
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
    """Start the workers and supervise them.

    The supervisor runs the quality gate every N finished cases and stops the
    run when the success rate collapses.
    """
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
    """Package the finished sweep, once every case has reached a terminal state.

    The function writes the manifest and copies the ledger. On request it also
    builds a tarball of the tensors and pushes everything to a Hugging Face
    dataset.
    """
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
        tarpath = ds / "sweep_tensors.tar.gz"
        with tarfile.open(tarpath, "w:gz") as tf:
            tf.add(manifest, arcname="manifest.csv")
            for cid in done_ids:
                npz = workdir / f"case_{cid:04d}" / "case.npz"
                if npz.exists():
                    tf.add(npz, arcname=f"cases/case_{cid:04d}.npz")
        print(f"[package] tarball -> {tarpath} ({tarpath.stat().st_size/1e6:.0f} MB)")
        _maybe_upload_file(tarpath, args.hf_repo, "sweep_tensors.tar.gz")

    _maybe_upload_file(manifest, args.hf_repo, "manifest.csv")
    _maybe_upload_file(ds / "ledger.db", args.hf_repo, "ledger.db")


# ============================================================================================
#                                       main
# ============================================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="dataset generation pipeline")
    p.add_argument("--workdir", default="data/raw/sweep", help="artifact root (one subdir per case)")
    p.add_argument("--db", default=None, help="ledger path (default <workdir>/ledger.db)")
    p.add_argument("--workers", type=int, default=4, help="worker processes")
    p.add_argument("--block", type=int, default=None, help="pin workers to one block (multi-machine)")
    # sampling and ledger construction
    p.add_argument("--n-geom", type=int, default=70, help="core geometries")
    p.add_argument("--n-fs", type=int, default=10, help="freestreams per core geometry")
    p.add_argument("--n-geom-ood", type=int, default=10, help="geometries per OOD slab")
    p.add_argument("--n-fs-ood", type=int, default=2, help="freestreams per OOD geometry")
    p.add_argument("--n-blocks", type=int, default=4, help="contiguous ledger blocks")
    p.add_argument("--seed", type=int, default=0, help="LHS seed")
    # solver settings.
    # geometric multigrid at mglevel 2 is required across the box, not an
    # optimization. without it a sharp cone or a high-Mach geometry stalls with
    # the residual near -4.77 and the bow shock collapsed onto the body. with it
    # the second pass settles near -6 and the shock stands at a physical
    # distance. the drag coefficient reaches its plateau between iteration 12000
    # and 14000, so iter_pass2 = 16000 covers the slow drift after it. the
    # five-case sample confirms the recipe at the blunt, sharp, low-Mach and
    # high-Mach corners.
    p.add_argument("--iter-pass1", type=int, default=2500, help="first-order startup iterations")
    p.add_argument("--iter-pass2", type=int, default=16000, help="second-order iterations")
    p.add_argument("--conv-minval", type=float, default=-8.0,
                   help="RMS_DENSITY residual floor; default -8 is effectively off (heat flux, "
                        "not the density residual, sets when the solution is usable)")
    p.add_argument("--cfl-max", type=float, default=5.0)
    p.add_argument("--mglevel", type=int, default=2,
                   help="geometric multigrid levels (0=off). Default 2 is what breaks the "
                        "sharp-cone / high-Mach saddle that stalled the first sweep attempt")
    p.add_argument("--case-timeout-s", type=float, default=0.0, help="per-SU2-call wall-clock cap (0=none)")
    # housekeeping
    p.add_argument("--quality-every", type=int, default=50, help="run the quality gate every N finished cases")
    p.add_argument("--stale-after-s", type=float, default=7200.0, help="reclaim running rows older than this")
    p.add_argument("--max-runtime-s", type=float, default=0.0, help="exit workers after this many seconds (0=unlimited)")
    p.add_argument("--hf-repo", default=None, help="HF dataset repo id to push tensors/manifest to (needs HF_TOKEN)")
    p.add_argument("--make-tarball", action="store_true", help="at packaging, build sweep_tensors.tar.gz")
    # stage control
    p.add_argument("--init-only", action="store_true", help="build the ledger and exit")
    p.add_argument("--dry-run", action="store_true", help="preflight + build ledger + print the plan, no SU2")
    p.add_argument("--no-validate", action="store_true", help="skip the stage-2 canonical validation")
    p.add_argument("--revalidate", action="store_true", help="re-run stage 2 even if the marker exists")
    p.add_argument("--validate-only", action="store_true", help="run stages 1-2 and exit")
    p.add_argument("--validate-case", default=None,
                   help="run only the named stage-2 case (e.g. smallnose_midM); merges into existing marker")
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
