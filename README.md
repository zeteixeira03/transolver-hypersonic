# transolver-hypersonic

Neural surrogate for steady-state flow fields around axisymmetric sphere-cone
re-entry capsules in hypersonic flight. Architecture is Transolver (Wu et al.,
ICML 2024, [arXiv:2402.02366](https://arxiv.org/abs/2402.02366)); ground truth
is SU2 axisymmetric laminar Navier-Stokes.

This is a work in progress. The repository will be filled out as phases land;
see `PHASE_LOG.md` for time-stamped progress.

## Status

| Phase | What | State |
|-------|------|-------|
| 1 | Stack validation on AirfRANS | done (2026-05-11) |
| 2 | SU2 single-case validation, three analytical sanity checks | done (2026-05-11) |
| 3 | Dataset generation (~700 core + 80 OOD sphere-cone cases) | next |
| 4 | Train the surrogate, slice-count ablation, OOD geometric extrapolation | not started |

## Phase 1 result

Transolver (0.52M parameters, 4 blocks, 32 slices, 8 heads, 128 hidden)
trained on the [AirfRANS](https://github.com/Extrality/AirfRANS) `scarce`
task (200 train, 200 test) for 200 epochs on a Kaggle T4 in 64 min. Per-channel
relative L2 on the test split:

| channel       | rel-L2 |
|---------------|--------|
| u             | 0.053  |
| v             | 0.124  |
| p / rho       | 0.094  |
| nu_t          | 0.102  |
| p on surface  | 0.072  |

The acceptance figure is `data/samples/phase1_acceptance.png`. The point of
this phase is not to compete with the AirfRANS Transolver; it is to confirm
that our vendored Physics-Attention, training loop, and evaluation path all
function end-to-end on a free-tier GPU. The headline analyses (slice-count
ablation, out-of-distribution geometric extrapolation) live in Phase 4 on the
SU2 dataset.

## Reproducing Phase 1

Open `notebooks/01_validate_stack.ipynb` on Kaggle. Attach
`zeteixeira/airfrans-dataset` as input, set Accelerator to GPU T4, Internet
on, then Run All. The notebook clones this repo, locates the AirfRANS data,
fits norm stats, trains, and saves the acceptance figure plus a checkpoint.

## Phase 2 result

A canonical 60-deg sphere-cone (1-inch nose, body radius 0.0762 m, shoulder
radius 0.00762 m) at Mach 10, T_inf = 220 K, p_inf = 100 Pa, isothermal cold
wall T_w = 300 K. SU2 v8.5 axisymmetric laminar Navier-Stokes, ideal gas,
Sutherland viscosity. Two-pass solve: a first-order MUSCL = NO startup
followed by a restart with Roe + Venkatakrishnan second-order. 66 611 mesh
nodes, 61 min wall-clock total.

| quantity              | SU2          | analytical                | rel err | tol |
|-----------------------|--------------|---------------------------|---------|-----|
| stagnation heat flux  | 1.003 MW/m^2 | 1.111 MW/m^2 (Fay-Riddell) | -9.79%  | 15% |
| stagnation pressure   | 12.35 kPa    | 12.92 kPa (Rayleigh-Pitot) | -4.41%  |  5% |
| shock standoff        | 4.27 mm      | 3.75 mm (Billig)           | +13.79% | 20% |

All three clear their stated tolerances. The acceptance figure is at
`data/samples/phase2_acceptance.png`; one full converged-case tensor
(`x, r, rho, u, v, T` per mesh node) is at `data/samples/phase2_canonical.npz`.

The phase produced the full pipeline that Phase 3 will exercise at scale:
parametric gmsh meshing in `src/geometry/`, SU2 template + subprocess driver
in `src/cfd/`, postprocess and three-correlation comparison in
`src/cfd/postprocess.py` and `src/eval/sanity.py`. `scripts/phase2_validate.py`
runs the canonical case end-to-end; `tests/` covers the analytical
correlations, the geometry math, and the postprocess.

## Reproducing Phase 2

Linux or WSL with SU2 v8.5 on PATH and `pyvista`, `gmsh`, `matplotlib`,
`numpy` available. Then:

```bash
python scripts/phase2_validate.py --iter-pass1 2500 --iter-pass2 8000
```

The driver writes mesh, two-stage cfg, SU2 logs, the volume and surface
VTUs, an acceptance figure, a JSON summary, and the training tensor to
`data/raw/phase2_validation/`.

## Headline analyses (Phase 4, not yet run)

1. **Slice-count ablation**: how Transolver's slice count `M` trades off
   against accuracy in shock-dominated flow, and whether the optimum `M`
   shifts between in-distribution and OOD sets.
2. **Out-of-distribution geometric extrapolation**: train on a core
   sphere-cone box, evaluate on extreme nose radii, extreme cone half-angles,
   and Mach above 25.

Which becomes the headline finding is decided once Phase 4 results are in.

## Scope

The surrogate predicts steady-state laminar flow over axisymmetric
sphere-cone capsules at zero angle of attack. Mach 8-25 for training, up
to Mach 30 OOD. Ideal gas, calorically perfect air, Sutherland viscosity,
isothermal cold wall at T_w = 300 K. No turbulence model, no real-gas
thermochemistry, no radiation, no ablation. The surrogate's value is speed
at acceptable accuracy, not beating SU2 on accuracy.

## Repository layout

```
src/
  analytical/__init__.py   # Fay-Riddell, Rayleigh-Pitot, Billig + Sutherland
  geometry/sphere_cone.py  # parametric sphere-cone + gmsh meshing
  cfd/runner.py            # SU2 cfg render and subprocess driver
  cfd/postprocess.py       # surface/axis-line readers and shock-standoff
  eval/sanity.py           # comparison vs analytical correlations
  eval/plots.py            # Phase 1 acceptance figure
  data/airfrans.py         # Phase 1 dataset, normalization, loading
  training/loop.py         # Phase 1 training and evaluation
  models/transolver.py     # vendored Transolver (MIT, thuml/Transolver)
configs/
  sphere_cone_template.cfg # SU2 axisymmetric laminar NS template
scripts/
  phase2_validate.py       # Phase 2 end-to-end driver
notebooks/
  01_validate_stack.ipynb  # Phase 1 Kaggle orchestrator
data/samples/              # committed example outputs (figures, one case .npz)
PHASE_LOG.md               # append-only progress log
```

## License

Code in `src/models/transolver.py` is adapted from
[thuml/Transolver](https://github.com/thuml/Transolver) (MIT, copyright 2024
THUML @ Tsinghua University) with the modifications noted in the file
header.
