# transolver-hypersonic

Neural surrogate for steady-state flow fields around axisymmetric sphere-cone
re-entry capsules in hypersonic flight. Architecture is Transolver (Wu et al.,
ICML 2024, [arXiv:2402.02366](https://arxiv.org/abs/2402.02366)); ground truth
is SU2 axisymmetric laminar Navier-Stokes.

Work in progress. The pipeline is validated and most of the dataset is
generated; training experiments are underway and results will appear here
once they firm up.

## Where things stand

The stack was validated end-to-end on AirfRANS first, then the SU2 pipeline
on a canonical case against three analytical correlations (both below). A
sweep over the sphere-cone design space has produced roughly 550 converged
cases so far: a core box plus out-of-distribution slabs at extreme nose
radii, extreme cone half-angles, and Mach above 25. The last corners of the
sweep are still filling in.

On the modeling side, a first Transolver trained on the dataset reproduces
held-out flow fields (`data/samples/phase4_w0_field.png` shows predicted vs
SU2 primitives on an unseen case), a slice-count ablation across
M in {8, 16, 32, 64} has run, and a five-member deep ensemble at the best
slice count is trained. Ensemble spread tracks per-case error closely enough
to drive a trust/warn/refuse rule, layered with an input-space envelope
guard that flags inputs outside the training box
(`data/samples/phase4_w2_calibration.png`). Quantitative results are held
back until the remaining out-of-distribution cases land and every checkpoint
is rescored against the complete set; partial numbers on a moving dataset
would not mean much.

## Evaluation questions

1. **Slice-count ablation.** Transolver's Physics-Attention assigns each mesh
   point to one of M learned slices and attends across slice tokens, O(M^2)
   instead of O(N^2). How does M trade off against accuracy in
   shock-dominated flow, and does the optimum M shift between
   in-distribution and out-of-distribution evaluation?
2. **Geometric extrapolation.** Train on a core sphere-cone box, evaluate on
   extreme nose radii, extreme cone half-angles, and Mach above the training
   range. Where does the surrogate degrade, and can that boundary be
   detected from the inputs alone?

## Validating the stack on AirfRANS

Transolver (0.52M parameters, 4 blocks, 32 slices, 8 heads, 128 hidden)
trained on the [AirfRANS](https://github.com/Extrality/AirfRANS) `scarce`
task (200 train, 200 test) for 200 epochs on a Kaggle T4 in 64 min.
Per-channel relative L2 on the test split:

| channel       | rel-L2 |
|---------------|--------|
| u             | 0.053  |
| v             | 0.124  |
| p / rho       | 0.094  |
| nu_t          | 0.102  |
| p on surface  | 0.072  |

The figure is `data/samples/phase1_acceptance.png`. The point was never to
compete with the published AirfRANS Transolver, only to confirm that the
vendored Physics-Attention, training loop, and evaluation path function
end-to-end on a free-tier GPU before pointing them at the SU2 data.

To reproduce: open `notebooks/01_validate_stack.ipynb` on Kaggle, attach
`zeteixeira/airfrans-dataset` as input, set Accelerator to GPU T4, Internet
on, then Run All.

## Validating the CFD pipeline

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

All three clear their stated tolerances. The figure is at
`data/samples/phase2_acceptance.png`; one full converged-case tensor
(`x, r, rho, u, v, T` per mesh node) is at `data/samples/phase2_canonical.npz`.

To reproduce, on Linux or WSL with SU2 v8.5 on PATH and `pyvista`, `gmsh`,
`matplotlib`, `numpy` available:

```bash
python scripts/phase2_validate.py --iter-pass1 2500 --iter-pass2 8000
```

The driver writes mesh, two-stage cfg, SU2 logs, the volume and surface
VTUs, a comparison figure, a JSON summary, and the training tensor to
`data/raw/phase2_validation/`.

## Dataset generation

`scripts/phase3_generate.py` sweeps the design box: geometry parameters
(R_n, theta_c, R_b, R_s) and freestream conditions (Mach, T_inf, p_inf via
altitude) sampled by Latin hypercube over a core box, plus dedicated slabs
past each box edge for out-of-distribution evaluation. Each case runs the
two-pass SU2 recipe above (with geometric multigrid, which the sharper cones
and higher Mach numbers need to converge) and is checked against the three
analytical correlations before acceptance. The runner is resumable: a SQLite
ledger tracks per-case status, so interrupted sessions pick up where they
stopped and failed cases are logged and skipped rather than rescued.

The raw sweep output is not committed. The converged cases (as
`case_*.npz` tensors plus the ledger) are published as the Kaggle dataset
`zeteixeira/su2-hypersonic-sphere-cone`, which is what the training
notebook consumes.

The model predicts four primitives (rho, u, v, T) at every mesh node.
Pressure is reconstructed as p = rho * R_specific * T at output time, so the
equation of state holds exactly and the network has no degree of freedom to
violate it. Per-field normalization stats are computed on the training split
only and persisted with each checkpoint.

## Scope

The surrogate predicts steady-state laminar flow over axisymmetric
sphere-cone capsules at zero angle of attack. Mach 8-25 for training, up
to Mach 30 out-of-distribution. Ideal gas, calorically perfect air,
Sutherland viscosity, isothermal cold wall at T_w = 300 K. No turbulence
model, no real-gas thermochemistry, no radiation, no ablation. The
surrogate's value is speed at acceptable accuracy, not beating SU2 on
accuracy.

## Repository layout

```
src/
  analytical/__init__.py   # Fay-Riddell, Rayleigh-Pitot, Billig + Sutherland
  geometry/sphere_cone.py  # parametric sphere-cone + gmsh meshing
  cfd/runner.py            # SU2 cfg render and subprocess driver
  cfd/postprocess.py       # surface/axis-line readers and shock-standoff
  cfd/ledger.py            # SQLite case ledger for the resumable sweep
  eval/sanity.py           # comparison vs analytical correlations
  eval/plots.py            # acceptance figures
  data/airfrans.py         # AirfRANS dataset, normalization, loading
  data/sampler.py          # design-box sampling for the sweep
  data/su2.py              # SU2 case dataset, splits, normalization
  training/loop.py         # training and evaluation loop
  models/transolver.py     # vendored Transolver (MIT, thuml/Transolver)
configs/
  sphere_cone_template.cfg # SU2 axisymmetric laminar NS template
scripts/
  phase2_validate.py       # single-case SU2 validation driver
  phase3_generate.py       # resumable dataset sweep runner
  phase4_train_su2.py      # training CLI (slice sweep, ensembles, eval-only)
  phase4_w1_analysis.py    # ablation tables + envelope-distance figure
  phase4_w2_ensemble.py    # deep-ensemble UQ and trust/warn/refuse
notebooks/
  01_validate_stack.ipynb  # AirfRANS validation, Kaggle orchestrator
  02_train_su2.ipynb       # SU2 training runs, Kaggle orchestrator
data/samples/              # committed example outputs (figures, one case .npz)
```

## License

Code in `src/models/transolver.py` is adapted from
[thuml/Transolver](https://github.com/thuml/Transolver) (MIT, copyright 2024
THUML @ Tsinghua University) with the modifications noted in the file
header.
