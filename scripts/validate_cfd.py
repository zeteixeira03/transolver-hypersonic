"""Validate the whole simulation pipeline on one canonical Mach 10 case.

The script meshes with gmsh, renders the SU2 configuration, runs SU2_CFD, reads
the outputs, compares three quantities against Fay-Riddell, Rayleigh-Pitot and
Billig, and writes an acceptance figure with a summary in JSON.

The case is a cold-wall re-entry body of the kind Anderson works through:

    R_n         = 0.0254 m   (1 inch)
    theta_c     = 60 deg
    R_b         = 0.0762 m
    R_s         = 0.00762 m
    M_inf       = 10
    T_inf       = 220 K
    p_inf       = 100 Pa     (near 60 km altitude)
    T_w         = 300 K      (isothermal cold wall)

Run it from the project root. Where SU2_CFD sits on the executable search path,
the script calls it directly. Otherwise it routes the call into the Windows
Subsystem for Linux, using the distribution "Ubuntu-22.04" and the conda
environment "su2".
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

# put the project root on the import path, so "from src..." resolves when this
# file runs as a script
from src.cfd.postprocess import (
    extract_axis_line,
    extract_surface,
    extract_training_tensors,
    find_shock_standoff,
    stagnation_values,
)
from src.cfd.runner import CANONICAL_CASE, Case, detect_invocation, run_case
from src.eval.sanity import compare_to_analytical, format_summary
from src.geometry.sphere_cone import mesh_settings

# ============================================================================================
#                                       canonical case
# ============================================================================================

CANONICAL = CANONICAL_CASE
DEFAULT_MESH_KWARGS = mesh_settings(CANONICAL.R_n, CANONICAL.R_b)


# ============================================================================================
#                                       helpers
# ============================================================================================

def make_acceptance_figure(
    case: Case,
    surface: dict,
    axis: dict,
    summary: dict,
    standoff: dict,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # sort the surface nodes by r, so a wall plot runs in one sweep from the
    # stagnation point at r = 0, up the sphere, along the cone and around the
    # shoulder
    x, y = surface["x"], surface["y"]
    order = np.argsort(y)
    x_s = x[order]
    y_s = y[order]
    s = np.cumsum(np.concatenate([[0.0], np.hypot(np.diff(x_s), np.diff(y_s))]))

    # ---- surface pressure against arc length ----
    ax = axes[0, 0]
    if "p" in surface:
        p_s = surface["p"][order]
        ax.plot(s, p_s, "k-", lw=1.0)
        ax.set_ylabel("wall pressure [Pa]")
        ax.set_xlabel("arc length from stagnation [m]")
        ax.set_title("surface pressure")
        ax.axhline(summary["ref_p02"], color="C3", ls="--", lw=1.0,
                   label=f"Rayleigh-Pitot p02 = {summary['ref_p02']:.3g} Pa")
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim(bottom=0)

    # ---- surface heat flux against arc length ----
    ax = axes[0, 1]
    if "qw" in surface:
        qw_s = surface["qw"][order]
        ax.plot(s, qw_s * 1e-6, "k-", lw=1.0)
        ax.set_ylabel("wall heat flux [MW/m^2]")
        ax.set_xlabel("arc length from stagnation [m]")
        ax.set_title("surface heat flux")
        ax.axhline(summary["ref_qw"] * 1e-6, color="C3", ls="--", lw=1.0,
                   label=f"Fay-Riddell q_w = {summary['ref_qw']:.3g} W/m^2")
        ax.legend(loc="best", fontsize=8)

    # ---- pressure along the axis line, which locates the shock ----
    ax = axes[1, 0]
    # frame about 0.5 R_n upstream of the body, where the shock stands
    x_axis = axis["x"]
    R_n = case.R_n
    mask = x_axis > -0.5 * R_n
    ax.plot(x_axis[mask] * 1e3, axis["p"][mask], "k-", lw=1.0)
    ax.axvline(standoff["x_shock"] * 1e3, color="C0", ls="--", lw=1.0,
               label=f"SU2 shock x = {standoff['x_shock']*1e3:.3f} mm")
    ax.axvline(-summary["ref_standoff"] * 1e3, color="C3", ls=":", lw=1.5,
               label=f"Billig delta = {summary['ref_standoff']*1e3:.3f} mm")
    ax.axhline(summary["ref_p02"], color="C3", ls="--", lw=0.8, alpha=0.5,
               label="Rayleigh-Pitot p02")
    ax.set_xlabel("axis x [mm]")
    ax.set_ylabel("pressure [Pa]")
    ax.set_yscale("log")
    ax.set_title("stagnation line pressure (offset r=0.05 R_n)")
    ax.legend(loc="best", fontsize=8)

    # ---- sanity check table ----
    ax = axes[1, 1]
    ax.axis("off")
    text = format_summary(summary)
    ax.text(0.0, 1.0, text, family="monospace", fontsize=8, va="top")
    ax.set_title("acceptance checks")

    fig.suptitle(
        f"Validation case: M={case.mach}, R_n={case.R_n*1000:.1f} mm, "
        f"theta_c={case.theta_c_deg} deg, T_inf={case.T_inf} K, p_inf={case.p_inf} Pa"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ============================================================================================
#                                       main driver
# ============================================================================================

def main():
    parser = argparse.ArgumentParser(description="canonical sphere-cone validation case")
    parser.add_argument("--run-dir", default="data/raw/cfd_validation",
                        help="directory for every file the run writes")
    parser.add_argument("--iter-pass1", type=int, default=3000,
                        help="iterations of the first-order pass")
    parser.add_argument("--iter-pass2", type=int, default=15000,
                        help="iterations of the second-order pass")
    parser.add_argument("--cfl-max", type=float, default=5.0)
    parser.add_argument("--nprocs", type=int, default=1)
    parser.add_argument("--skip-su2", action="store_true",
                        help="skip SU2_CFD and read the outputs of an existing run")
    parser.add_argument("--wsl-distro", default=None,
                        help="name the Linux distribution instead of detecting it")
    parser.add_argument("--conda-env", default=None,
                        help="name the conda environment instead of detecting it")
    parser.add_argument("--timeout", type=float, default=None,
                        help="wall-clock limit on one SU2 run, in seconds")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    auto_distro, auto_env = detect_invocation()
    wsl_distro = args.wsl_distro if args.wsl_distro is not None else auto_distro
    conda_env  = args.conda_env  if args.conda_env  is not None else auto_env

    print(f"[validate] run dir:    {run_dir}")
    print(f"[validate] invocation: distro={wsl_distro} env={conda_env}")
    print(f"[validate] case:       {CANONICAL}")

    if args.skip_su2:
        result: dict[str, Any] = {
            "run_dir": run_dir,
            "volume_vtu": run_dir / "flow.vtu",
            "surface_vtu": run_dir / "surface_flow.vtu",
            "surface_csv": run_dir / "surface_flow.csv",
            "log": run_dir / "su2.log",
            "returncode": 0,
        }
    else:
        # ---- first pass: first order, which converges from a cold start ----
        t0 = time.time()
        result_p1 = run_case(
            CANONICAL,
            run_dir=run_dir,
            mesh_kwargs=DEFAULT_MESH_KWARGS,
            iter_max=args.iter_pass1,
            cfl_max=args.cfl_max,
            muscl=False,
            restart_sol=False,
            wsl_distro=wsl_distro,
            conda_env=conda_env,
            nprocs=args.nprocs,
            timeout=args.timeout,
            cfg_filename="case_pass1.cfg",
            log_filename="su2_pass1.log",
        )
        elapsed_p1 = time.time() - t0
        print(f"[validate] SU2_CFD pass 1 (first-order, {args.iter_pass1} iters) "
              f"finished in {elapsed_p1:.1f} s (rc={result_p1['returncode']})")
        if result_p1["returncode"] != 0:
            print(f"[validate] pass 1 failed; see {result_p1['log']}")
            sys.exit(2)

        # the second pass reads solution_flow.dat, so rename the restart file
        # the first pass wrote
        restart_dat = run_dir / "restart_flow.dat"
        solution_dat = run_dir / "solution_flow.dat"
        if not restart_dat.exists():
            print("[validate] pass 1 produced no restart_flow.dat; aborting")
            sys.exit(2)
        shutil.copyfile(restart_dat, solution_dat)

        # ---- second pass: second order, restarted from the first ----
        t0 = time.time()
        result = run_case(
            CANONICAL,
            run_dir=run_dir,
            mesh_kwargs=DEFAULT_MESH_KWARGS,
            iter_max=args.iter_pass2,
            cfl_max=args.cfl_max,
            muscl=True,
            restart_sol=True,
            wsl_distro=wsl_distro,
            conda_env=conda_env,
            nprocs=args.nprocs,
            timeout=args.timeout,
            write_mesh=False,
            cfg_filename="case_pass2.cfg",
            log_filename="su2_pass2.log",
        )
        elapsed_p2 = time.time() - t0
        print(f"[validate] SU2_CFD pass 2 (second-order, {args.iter_pass2} iters) "
              f"finished in {elapsed_p2:.1f} s (rc={result['returncode']})")

    if result["returncode"] != 0:
        print(f"[validate] SU2_CFD failed with rc={result['returncode']}; see "
              f"{result['log']}")
        sys.exit(2)

    # ---- read the outputs ----
    # every stagnation quantity is taken off the axis rather than on it. the
    # wall node at r = 0 carries the singularity of the axisymmetric source
    # term, which inflates both pressure and heat flux there, and the accepted
    # practice is to skip a thin band above the axis.
    #
    # the stagnation pressure that Rayleigh-Pitot predicts is read as the
    # plateau behind the shock on a streamline offset from the axis. subsonic
    # shock-layer theory says the value at the body wall equals it, but the axis
    # artifact corrupts the wall value here, so the plateau is reported instead.
    R_n = CANONICAL.R_n
    L_far = DEFAULT_MESH_KWARGS["L_far"]
    y_skip = 0.05 * R_n
    r_offset = 0.05 * R_n

    surface = extract_surface(result["surface_vtu"])
    stag = stagnation_values(surface, y_axis_skip=y_skip, n_average=3)

    axis = extract_axis_line(result["volume_vtu"], x_in=-L_far, x_nose=0.0,
                             n=2000, r_offset=r_offset)
    standoff = find_shock_standoff(axis, p_inf=CANONICAL.p_inf)

    su2_qw = stag.get("qw", float("nan"))
    # the stagnation pressure is the median of the plateau behind the shock,
    # taken on the offset axis line
    su2_p02 = standoff.get("p_post_plateau", float("nan"))
    if np.isnan(su2_p02):
        su2_p02 = stag.get("p", float("nan"))
    summary = compare_to_analytical(
        M_inf=CANONICAL.mach,
        T_inf=CANONICAL.T_inf,
        p_inf=CANONICAL.p_inf,
        R_n=CANONICAL.R_n,
        T_w=CANONICAL.T_w,
        su2_qw=su2_qw,
        su2_p02=su2_p02,
        su2_standoff=standoff["standoff"],
    )
    print()
    print(format_summary(summary))

    # ---- persist ----
    out_json = run_dir / "cfd_validation_summary.json"
    payload = {
        "case": asdict(CANONICAL),
        "stagnation": stag,
        "standoff": standoff,
        "checks": [
            {
                "name": c.name, "su2": c.su2, "analytical": c.analytical,
                "rel_err": c.rel_err, "tolerance": c.tolerance, "passed": c.passed,
            }
            for c in summary["checks"]
        ],
        "all_passed": summary["all_passed"],
        "ref": {
            "qw": summary["ref_qw"],
            "p02": summary["ref_p02"],
            "standoff": summary["ref_standoff"],
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[validate] summary written to {out_json}")

    fig_path = run_dir / "cfd_validation.png"
    make_acceptance_figure(CANONICAL, surface, axis, summary, standoff, fig_path)
    print(f"[validate] acceptance figure written to {fig_path}")

    # ---- extract the training tensors, which also tests that path ----
    npz_path = run_dir / "case.npz"
    tensors = extract_training_tensors(result["volume_vtu"], save_npz=npz_path)
    print(f"[validate] training tensors written to {npz_path} "
          f"(N={len(tensors['x'])} nodes, fields={sorted(tensors)})")

    sys.exit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
