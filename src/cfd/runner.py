"""Render an SU2 configuration from the sphere-cone template and run SU2_CFD.

The entry point :func:`run_case` takes a set of case parameters and a run
directory, and leaves behind:

    <run_dir>/
        case.su2              # mesh, written by mesh_sphere_cone
        case.cfg              # rendered SU2 configuration
        flow.vtu              # volume solution, Paraview format
        surface_flow.vtu      # surface solution
        surface_flow.csv      # surface scalar values per node
        restart_flow.dat      # SU2 restart file
        history.csv           # convergence history
        su2.log               # standard output and error of SU2_CFD

The runner reaches SU2_CFD two ways. On Linux and macOS it calls the binary
directly, which needs SU2_CFD on the executable search path. On a Windows host
it calls into the Windows Subsystem for Linux, which needs both ``wsl_distro``
and ``conda_env``. Dataset generation on Kaggle takes the direct path.
"""

from __future__ import annotations

import re
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
    settings: dict[str, str] | None = None,
) -> Path:
    """Substitute concrete case values into the SU2 configuration template.

    ``settings`` then sets any further SU2 option, as ``{"KEY": "value"}``: an
    existing ``KEY=`` line is replaced, and a missing one is appended.
    Returns the path of the written configuration file.
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
    for key, value in (settings or {}).items():
        line = f"{key}= {value}"
        text, n = re.subn(rf"(?m)^{re.escape(key)}\s*=.*$", line, text)
        if n == 0:
            text += f"\n{line}\n"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


# ============================================================================================
#                                       subprocess driver
# ============================================================================================

def _windows_to_wsl_path(p: Path) -> str:
    """Translate an absolute Windows path into its ``/mnt/<drive>/...`` form."""
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
    """Run SU2_CFD on a rendered configuration file.

    Parameters
    ----------
    cfg_path : path
        The rendered SU2 configuration file. The run directory defaults to the
        parent of this file.
    wsl_distro : str, optional
        Name of a Windows Subsystem for Linux distribution. When given, SU2_CFD
        runs inside it and the configuration and mesh paths are translated to
        their Linux form. A Windows host needs this.
    conda_env : str, optional
        Name of a conda environment to activate before SU2_CFD starts, such as
        ``su2``.
    nprocs : int
        Above 1, the command becomes ``mpirun -np <nprocs> SU2_CFD``. The SU2
        build in the chosen environment must carry Message Passing Interface
        support.
    """
    cfg_path = Path(cfg_path)
    run_dir = Path(run_dir) if run_dir else cfg_path.parent
    log_path = run_dir / log_filename
    cfg_name = cfg_path.name

    if wsl_distro is None:
        # direct call. the working directory is the run directory, and SU2_CFD
        # must sit on the executable search path
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

    # call through the Windows Subsystem for Linux: translate the paths, then
    # assemble one bash command
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
    R_n: float          # nose radius                     [m]
    theta_c_deg: float  # cone half-angle                 [deg]
    R_b: float          # maximum body radius             [m]
    R_s: float          # shoulder rounding radius        [m]
    mach: float         # freestream Mach number
    T_inf: float        # freestream static temperature   [K]
    p_inf: float        # freestream static pressure      [Pa]
    T_w: float = 300.0  # isothermal wall temperature     [K]


CANONICAL_CASE = Case(
    R_n=0.0254, theta_c_deg=60.0, R_b=0.0762, R_s=0.00762,
    mach=10.0, T_inf=220.0, p_inf=100.0, T_w=300.0,
)


def detect_invocation() -> tuple[str | None, str | None]:
    """Return the Linux distribution and conda environment that hold SU2.

    With ``SU2_CFD`` on the search path, as inside an active conda environment,
    this returns ``(None, None)`` and the caller runs it directly.
    """
    if shutil.which("SU2_CFD"):
        return None, None
    return "Ubuntu-22.04", "su2"


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
    settings: dict[str, str] | None = None,
) -> dict:
    """Run one sphere-cone hypersonic case from parameters to output files.

    Four steps: mesh with gmsh, substitute the configuration template, run
    SU2_CFD, and return the paths of everything written.

    Parameters
    ----------
    case : Case
        Case parameters.
    run_dir : path
        Directory for every file the run produces. Created if absent.
    mesh_kwargs : dict, optional
        Extra keyword arguments passed on to :func:`mesh_sphere_cone`, such as
        ``L_far``, ``h_wall`` and ``bl_first_height``.
    muscl : bool
        True runs second order with the configured slope limiter. False runs
        first order, which is what a hypersonic case needs from a cold start.
    restart_sol : bool
        True makes SU2 take its initial condition from an existing
        ``solution_flow.dat``. The second-order pass of a two-pass solve sets
        this.
    write_mesh : bool
        False expects ``case.su2`` to exist in ``run_dir`` already.
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
        settings=settings,
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
