"""Render an SU2 cfg from the sphere-cone template and invoke SU2_CFD.

The high-level entry point :func:`run_case` takes a case-parameter dict and a
run directory and produces:

    <run_dir>/
        case.su2              # mesh (written here by mesh_sphere_cone)
        case.cfg              # rendered SU2 config
        flow.vtu              # volume solution (Paraview format)
        surface_flow.vtu      # surface solution
        surface_flow.csv      # surface scalar values per node
        restart_flow.dat      # SU2 restart
        history.csv           # convergence history
        su2.log               # stdout/stderr of SU2_CFD

The runner can invoke SU2_CFD natively (Linux/Mac, when SU2_CFD is on PATH)
or via WSL by setting ``wsl_distro`` and ``conda_env``. The Windows-host
development workflow is WSL+conda; production Phase 3 generation on Kaggle
runs Linux natively.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.geometry.sphere_cone import mesh_sphere_cone


# ============================================================================================
#                                 template substitution
# ============================================================================================

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "configs" / "sphere_cone_template.cfg"


def render_cfg(
    template_path: str | Path,
    output_path: str | Path,
    *,
    mach: float,
    T_inf: float,
    p_inf: float,
    T_w: float,
    mesh_filename: str,
    iter_max: int = 20000,
    cfl_max: float = 5.0,
    muscl: bool = True,
    restart_sol: bool = False,
    mglevel: int = 0,
    conv_minval: float = -8.0,
    volume_name: str = "flow",
    surface_name: str = "surface_flow",
    restart_name: str = "restart_flow",
) -> Path:
    """Render the SU2 cfg template with concrete case values.

    Returns the path to the written cfg.
    """
    text = Path(template_path).read_text()
    replacements = {
        "__MACH__":        f"{mach}",
        "__T_INF__":       f"{T_inf}",
        "__P_INF__":       f"{p_inf}",
        "__T_W__":         f"{T_w}",
        "__ITER__":        f"{iter_max}",
        "__CFL_MAX__":     f"{cfl_max}",
        "__MGLEVEL__":     f"{mglevel}",
        "__CONV_MINVAL__": f"{conv_minval}",
        "__MUSCL__":       "YES" if muscl else "NO",
        "__RESTART_SOL__": "YES" if restart_sol else "NO",
        "__MESH__":        mesh_filename,
        "__VOLUME__":      volume_name,
        "__SURFACE__":     surface_name,
        "__RESTART__":     restart_name,
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


# ============================================================================================
#                                       subprocess driver
# ============================================================================================

def _windows_to_wsl_path(p: Path) -> str:
    """Translate a Windows absolute path to a /mnt/<drive>/... WSL path."""
    s = str(p)
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return s.replace("\\", "/")


def run_su2(
    cfg_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    wsl_distro: str | None = None,
    conda_env: str | None = None,
    nprocs: int = 1,
    log_filename: str = "su2.log",
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Invoke SU2_CFD on a rendered cfg.

    Parameters
    ----------
    cfg_path : path
        Rendered SU2 cfg file. The run directory defaults to its parent.
    wsl_distro : str, optional
        If provided, SU2_CFD is invoked inside this WSL distribution; cfg/mesh
        paths are translated. Required on Windows hosts.
    conda_env : str, optional
        Conda env name to activate before running SU2_CFD (e.g. ``su2``).
    nprocs : int
        If > 1, run with ``mpirun -np <nprocs> SU2_CFD``. SU2 must be built
        with MPI support inside the chosen env.
    """
    cfg_path = Path(cfg_path)
    run_dir = Path(run_dir) if run_dir else cfg_path.parent
    log_path = run_dir / log_filename
    cfg_name = cfg_path.name

    if wsl_distro is None:
        # native invocation: assume cwd = run_dir, SU2_CFD on PATH
        if shutil.which("SU2_CFD") is None:
            raise RuntimeError("SU2_CFD not found on PATH; set wsl_distro or activate the env")
        if nprocs > 1:
            cmd = ["mpirun", "-np", str(nprocs), "SU2_CFD", cfg_name]
        else:
            cmd = ["SU2_CFD", cfg_name]
        with open(log_path, "w") as logf:
            return subprocess.run(
                cmd, cwd=run_dir, stdout=logf, stderr=subprocess.STDOUT,
                check=False, timeout=timeout,
            )

    # WSL invocation: translate paths and assemble a bash command
    wsl_run_dir = _windows_to_wsl_path(run_dir)
    if conda_env:
        env_prefix = (
            "source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null; "
            "source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; "
            f"conda activate {conda_env} && "
        )
    else:
        env_prefix = ""
    if nprocs > 1:
        su2_cmd = f"mpirun -np {nprocs} SU2_CFD {cfg_name}"
    else:
        su2_cmd = f"SU2_CFD {cfg_name}"

    bash_cmd = f"cd '{wsl_run_dir}' && {env_prefix}{su2_cmd}"
    full = ["wsl", "-d", wsl_distro, "--", "bash", "-c", bash_cmd]
    with open(log_path, "w") as logf:
        return subprocess.run(
            full, stdout=logf, stderr=subprocess.STDOUT, check=False, timeout=timeout,
        )


# ============================================================================================
#                                  high-level case driver
# ============================================================================================

@dataclass
class Case:
    """Sphere-cone hypersonic case parameters."""
    R_n: float          # nose radius                [m]
    theta_c_deg: float  # cone half-angle            [deg]
    R_b: float          # body max radius            [m]
    R_s: float          # shoulder rounding radius   [m]
    mach: float         # freestream Mach number
    T_inf: float        # freestream static T        [K]
    p_inf: float        # freestream static p        [Pa]
    T_w: float = 300.0  # isothermal wall T          [K]


def run_case(
    case: Case,
    run_dir: str | Path,
    *,
    template_path: str | Path = TEMPLATE_PATH,
    mesh_kwargs: dict | None = None,
    iter_max: int = 20000,
    cfl_max: float = 5.0,
    muscl: bool = True,
    restart_sol: bool = False,
    mglevel: int = 0,
    conv_minval: float = -8.0,
    wsl_distro: str | None = None,
    conda_env: str | None = None,
    nprocs: int = 1,
    timeout: float | None = None,
    write_mesh: bool = True,
    log_filename: str = "su2.log",
    cfg_filename: str = "case.cfg",
) -> dict:
    """Run a sphere-cone hypersonic case end-to-end.

    Steps: mesh (gmsh) -> cfg (template substitute) -> SU2_CFD -> return paths.

    Parameters
    ----------
    case : Case
        Case parameters.
    run_dir : path
        Directory for all run artifacts. Created if absent.
    mesh_kwargs : dict, optional
        Extra kwargs forwarded to :func:`mesh_sphere_cone` (e.g. ``L_far``,
        ``h_wall``, ``bl_first_height``).
    muscl : bool
        If True, run second-order with the configured slope limiter; if False,
        run first-order. First-order is the robust startup for hypersonic.
    restart_sol : bool
        If True, SU2 reads the existing ``solution_flow.dat`` for initial
        conditions. Used for the second-order stage of a two-pass solve.
    write_mesh : bool
        If False, expect ``case.su2`` to already exist in ``run_dir``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = run_dir / "case.su2"
    cfg_path = run_dir / cfg_filename

    geom = None
    if write_mesh:
        kwargs = mesh_kwargs or {}
        geom = mesh_sphere_cone(
            R_n=case.R_n,
            theta_c_deg=case.theta_c_deg,
            R_b=case.R_b,
            R_s=case.R_s,
            output_path=mesh_path,
            **kwargs,
        )

    render_cfg(
        template_path=template_path,
        output_path=cfg_path,
        mach=case.mach,
        T_inf=case.T_inf,
        p_inf=case.p_inf,
        T_w=case.T_w,
        mesh_filename=mesh_path.name,
        iter_max=iter_max,
        cfl_max=cfl_max,
        muscl=muscl,
        restart_sol=restart_sol,
        mglevel=mglevel,
        conv_minval=conv_minval,
    )

    result = run_su2(
        cfg_path,
        run_dir=run_dir,
        wsl_distro=wsl_distro,
        conda_env=conda_env,
        nprocs=nprocs,
        timeout=timeout,
        log_filename=log_filename,
    )

    return {
        "run_dir": run_dir,
        "mesh": mesh_path,
        "cfg": cfg_path,
        "volume_vtu": run_dir / "flow.vtu",
        "surface_vtu": run_dir / "surface_flow.vtu",
        "surface_csv": run_dir / "surface_flow.csv",
        "restart": run_dir / "restart_flow.dat",
        "log": run_dir / log_filename,
        "returncode": result.returncode,
        "geometry": geom,
    }
