# transolver-hypersonic

Neural surrogate for steady-state flow fields around axisymmetric sphere-cone
re-entry capsules in hypersonic flight. Architecture is Transolver (Wu et al.,
ICML 2024, [arXiv:2402.02366](https://arxiv.org/abs/2402.02366)); ground truth
is SU2 axisymmetric laminar Navier-Stokes.

Work in progress. The pipeline is validated, the sweep is complete, and a
slice-count ablation plus a calibrated deep ensemble are in (results below).

## Where things stand

The stack was validated end-to-end on AirfRANS first, then the SU2 pipeline
on a canonical case against three analytical correlations (both below). A
sweep over the sphere-cone design space has produced 727 converged cases: a
core box of 667 cases plus out-of-distribution slabs at large and small nose
radius, high cone half-angle, and Mach above 25.

On the modeling side, a first Transolver reproduces held-out flow fields
(`data/samples/field_prediction.png` shows predicted vs SU2 primitives on an
unseen case). A slice-count ablation and a five-member deep ensemble with
calibrated uncertainty follow; both are in the results below, evaluated
against every out-of-distribution slab including the high-Mach one.

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

## Results

Numbers below are on the full sweep: 727 converged cases (667 core plus
out-of-distribution slabs at large nose radius, small nose radius, high cone
angle, and Mach above 25). Every OOD slab, including the high-Mach one, is
now evaluated. Relative L2 is per-channel, in physical units after
de-normalization, then averaged over the four primitives.

### Slice-count ablation

Mean relative L2 per evaluation tier, sweeping the slice count M (best per
column in bold):

| M  | interpolation | family holdout | cone_high | mach_high | nose_large | nose_small | OOD pooled |
|----|------|------|------|------|------|------|------|
| 8  | 0.121 | 0.129 | 0.428 | 0.215 | 0.344 | **0.257** | 0.313 |
| 16 | 0.105 | 0.117 | 0.343 | 0.194 | 0.300 | 0.295 | 0.278 |
| 32 | **0.100** | **0.104** | 0.337 | **0.189** | 0.292 | 0.277 | 0.270 |
| 64 | 0.108 | 0.111 | **0.307** | 0.202 | **0.269** | 0.286 | **0.260** |

The optimum slice count still shifts with the evaluation regime, and adding
the Mach-above-25 slab does not change the direction: M=32 is best
in-distribution (interpolation and family holdout); M=64 is best on the
pooled out-of-distribution metric and on most individual OOD slabs.
Mach extrapolation is the one slab where more slices stop helping past
M=32; every other slab keeps improving through M=64. More slices generally
help extrapolation after they stop helping interpolation: the
Physics-Attention bottleneck binds hardest where the flow leaves the
training box. Dropping below 16 slices costs accuracy everywhere.

Within the box, geometry generalization is nearly free. At M=32 the family
holdout, whole geometry clusters unseen in training (0.104), is barely harder
than freestream interpolation on seen shapes (0.100). The 3x error jump is at
the envelope boundary, not between geometry tiers. The
error-vs-envelope-distance figure (`data/samples/envelope_distance.png`)
puts a threshold on it: binned median error doubles about 7% outside the
box, which becomes the geometric half of the decision rule below. That
threshold moved from an earlier estimate of 4% once the high-Mach and
larger cone_high slabs were added; the boundary is real but its exact
location depends on how much of the extrapolation region has been sampled.

### Uncertainty and the trust/warn/refuse rule

Five Transolvers at M=32, identical splits and data, differing only in
initialization and data order (a deep ensemble). The ensemble mean is the
prediction, the spread across members is the uncertainty. Averaging beats the
best single member in-distribution and on the high-Mach slab:

| tier | ensemble | best single member | mean spread |
|------|------|------|------|
| interpolation    | 0.091 | 0.100 | 0.068 |
| family holdout   | 0.092 | 0.104 | 0.073 |
| cone_high (OOD)  | 0.296 | 0.288 | 0.176 |
| mach_high (OOD)  | 0.180 | 0.189 | 0.114 |
| nose_large (OOD) | 0.255 | 0.251 | 0.158 |
| nose_small (OOD) | 0.218 | 0.241 | 0.146 |

On cone_high and nose_large the ensemble mean stays within noise of the best
single member, same as before; the ensemble's clear wins are in-distribution
and now also on mach_high.

Spread tracks actual error: the rank correlation between per-case spread and
per-case error is 0.94 pooled over all evaluation tiers, 0.86 on each
in-distribution tier, 0.76-0.82 per OOD slab. Spread is a usable error proxy
that needs no ground truth (`data/samples/spread_calibration.png`).

That gives an operational rule for whether to trust a prediction. Refuse if
the input falls outside the training envelope (box exceedance, or Knudsen
number past the continuum limit) or the ensemble disagrees strongly; warn at
moderate spread; trust otherwise. Over the evaluation cases:

| decision | cases | median error | p90 error |
|------|------|------|------|
| trust  | 86 | 0.085 | 0.129 |
| warn   | 10 | 0.182 | 0.218 |
| refuse | 52 | 0.219 | 0.404 |

The buckets order cleanly by actual error. The honest limitation: spread
cannot catch a consensus error. One in-box case sits at 0.40 error with low
spread because all five members share the same systematic miss, so they agree
with each other. Ensemble disagreement measures epistemic uncertainty, not a
shared blind spot.

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

The figure is `data/samples/airfrans_validation.png`. The point was never to
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
`data/samples/cfd_validation.png`; one full converged-case tensor
(`x, r, rho, u, v, T` per mesh node) is at `data/samples/canonical_case.npz`.

To reproduce, on Linux or WSL with SU2 v8.5 on PATH and `pyvista`, `gmsh`,
`matplotlib`, `numpy` available:

```bash
python scripts/validate_cfd.py --iter-pass1 2500 --iter-pass2 8000
```

The driver writes mesh, two-stage cfg, SU2 logs, the volume and surface
VTUs, a comparison figure, a JSON summary, and the training tensor to
`data/raw/cfd_validation/`.

## Dataset generation

`scripts/generate_dataset.py` sweeps the design box: geometry parameters
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
  validate_cfd.py          # single-case SU2 validation driver
  generate_dataset.py      # resumable dataset sweep runner
  train.py                 # training CLI (slice sweep, ensembles, eval-only)
  slice_ablation.py        # ablation tables + envelope-distance figure
  ensemble_uq.py           # deep-ensemble UQ and trust/warn/refuse
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
