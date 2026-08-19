# transolver-hypersonic

Neural surrogate for steady-state flow fields around axisymmetric sphere-cone
re-entry capsules in hypersonic flight. Architecture is Transolver (Wu et al.,
ICML 2024, [arXiv:2402.02366](https://arxiv.org/abs/2402.02366)); ground truth
is SU2 axisymmetric laminar Navier-Stokes.

Work in progress. The pipeline is validated, the sweep is complete, and a
slice-count ablation, a calibrated deep ensemble, and one active-learning
iteration are in (results below).

## Where things stand

The stack was validated end-to-end on AirfRANS first, then the SU2 pipeline
on a canonical case against three analytical correlations (both below). A
sweep over the sphere-cone design space produced 727 converged cases: a core
box of 667 cases plus out-of-distribution slabs at large and small nose
radius, high cone half-angle, and Mach above 25. An active-learning pass
added 26 more, chosen where the ensemble disagreed with itself, bringing the
dataset to 753.

On the modeling side, a first Transolver reproduces held-out flow fields
(`data/samples/field_prediction.png` shows predicted vs SU2 primitives on an
unseen case). A slice-count ablation, a five-member deep ensemble with
calibrated uncertainty, the before/after on the active-learning cases, and an
ablation pricing the physics priors in a 100-case regime are all in the
results below, evaluated against every out-of-distribution slab including the
high-Mach one.

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

The dataset is 753 converged cases: a 727-case sweep (667 core plus
out-of-distribution slabs at large nose radius, small nose radius, high cone
angle, and Mach above 25) plus 26 cases chosen by the active-learning loop
described below. Every evaluation tier holds the same cases throughout, since
the loop cases are routed into training and never into a held-out tier. All
four out-of-distribution slabs are evaluated. Relative L2 is per-channel, in
physical units after de-normalization, then averaged over the four
primitives.

### Slice-count ablation

Mean relative L2 per evaluation tier, sweeping the slice count M (best per
column in bold). These four models predate the loop, so they train on the
727-case sweep alone:

| M  | interpolation | family holdout | cone_high | mach_high | nose_large | nose_small | OOD pooled |
|----|------|------|------|------|------|------|------|
| 8  | 0.121 | 0.129 | 0.428 | 0.215 | 0.344 | **0.257** | 0.313 |
| 16 | 0.105 | 0.117 | 0.343 | 0.194 | 0.300 | 0.295 | 0.278 |
| 32 | **0.100** | **0.104** | 0.337 | **0.189** | 0.292 | 0.277 | 0.270 |
| 64 | 0.108 | 0.111 | **0.307** | 0.202 | **0.269** | 0.286 | **0.260** |

The optimum slice count shifts with the evaluation regime. In-distribution
the answer is firm: M=32 wins both interpolation and family holdout, and it
is bracketed, since M=64 is worse on each. Out of distribution the grid says
less than it appears to. M=64 wins the pooled metric at 0.260, but on
individual slabs it takes two of four, cone_high and nose_large. Mach
extrapolation prefers M=32 by 0.013 over 18 cases and nose_small prefers M=8
over 8 cases; neither margin carries weight at those sample sizes. Pooled
OOD error also falls at every step of the grid, 0.313 down to 0.260, so the
out-of-distribution optimum is not bracketed and may sit past M=64. Finding
it would need M=96 and M=128, a full retrain each.

What the grid does support is that more slices help extrapolation after they
have stopped helping interpolation, the Physics-Attention bottleneck binding
hardest where the flow leaves the training box, and that dropping below 16
slices costs accuracy everywhere.

Within the box, geometry generalization is nearly free. At M=32 the family
holdout, whole geometry clusters unseen in training (0.104), is barely harder
than freestream interpolation on seen shapes (0.100). The 3x error jump is at
the envelope boundary, not between geometry tiers. The
error-vs-envelope-distance figure (`data/samples/envelope_distance.png`)
puts a threshold on it: binned median error doubles about 7% outside the
box, which becomes the geometric half of the decision rule below. The
boundary is real, but its measured location is not a fixed property of the
model. On an earlier, sparser out-of-distribution sample the same figure put
it at 3.8%, and it moved outward to 7% once the cone_high slab grew from 5
cases to 14 and the mach_high slab arrived with 18. Treat it as a threshold
calibrated against the extrapolation data on hand, not a constant.

### Uncertainty and the trust/warn/refuse rule

Five Transolvers at M=32, identical splits and data, differing only in
initialization and data order (a deep ensemble). The ensemble mean is the
prediction, the spread across members is the uncertainty. This is the model
the dashboard serves, trained on all 753 cases. Averaging beats the best
single member on every tier:

| tier | ensemble | best single member | mean spread |
|------|------|------|------|
| interpolation    | 0.084 | 0.092 | 0.064 |
| family holdout   | 0.083 | 0.094 | 0.069 |
| cone_high (OOD)  | 0.220 | 0.250 | 0.160 |
| mach_high (OOD)  | 0.162 | 0.170 | 0.103 |
| nose_large (OOD) | 0.276 | 0.277 | 0.153 |
| nose_small (OOD) | 0.208 | 0.223 | 0.128 |

The margin over the best member is largest where the flow leaves the training
box, apart from nose_large, where the two are indistinguishable.

Spread tracks actual error out of distribution: the rank correlation between
per-case spread and per-case error runs 0.74 to 0.86 across the four OOD
slabs, 0.72 pooled. In-distribution it is much weaker, 0.55 on interpolation
and 0.05 on the family holdout, because in-box errors are packed into a
narrow range and the ranking is mostly noise
(`data/samples/spread_calibration.png`). Spread is a usable error proxy for
detecting extrapolation, not for ranking cases the model already handles
well. The pooled 0.72 is inflated by the OOD tiers spanning a wide error
range and should not be quoted as if it held in-box.

That gives an operational rule for whether to trust a prediction. Refuse if
the input falls outside the training envelope (box exceedance, or Knudsen
number past the continuum limit) or the ensemble disagrees strongly; warn at
moderate spread; trust otherwise. Over the evaluation cases:

| decision | cases | median error | p90 error |
|------|------|------|------|
| trust  | 93 | 0.078 | 0.114 |
| warn   | 18 | 0.120 | 0.187 |
| refuse | 68 | 0.159 | 0.362 |

The buckets order cleanly by actual error. The honest limitation: spread
cannot catch a consensus error. One in-box case sits at 0.40 error with low
spread because all five members share the same systematic miss, so they agree
with each other. Ensemble disagreement measures epistemic uncertainty, not a
shared blind spot.

### Active-learning loop

One iteration of the obvious loop: score a Latin-hypercube pool over the six
free design axes by ensemble spread, pick a diverse batch of the most
uncertain in-envelope candidates, run them through SU2, retrain, compare.
Thirty cases were selected and run, 26 converged. The 13% failure rate is
above the sweep's 3.5%, which is expected: acquisition targets high-spread
regions and those are where the solver strains too. Failures were logged and
skipped, not rescued.

The control is a five-member ensemble trained on the 727 sweep cases; the
treatment is the same recipe with the 26 loop cases added. Both share a split
seed, so every evaluation tier holds the same cases and the per-case
difference is paired.

| tier | n | before | after | delta | 95% CI | p |
|------|---|--------|-------|-------|--------|---|
| interpolation    | 54 | 0.085 | 0.084 | -0.0013 | [-0.0035, +0.0008] | 0.35 |
| family holdout   | 65 | 0.090 | 0.083 | -0.0078 | [-0.0118, -0.0041] | <0.001 |
| cone_high (OOD)  | 14 | 0.236 | 0.220 | -0.0159 | [-0.0245, -0.0057] | 0.03 |
| mach_high (OOD)  | 18 | 0.171 | 0.162 | -0.0089 | [-0.0159, -0.0019] | 0.03 |
| nose_large (OOD) | 20 | 0.281 | 0.276 | -0.0048 | [-0.0144, +0.0041] | 0.52 |
| nose_small (OOD) | 8  | 0.244 | 0.208 | -0.0364 | [-0.0501, -0.0225] | 0.01 |

Measuring error at the acquired points themselves would prove nothing: those
cases are in the treatment's training set and were never in the control's, so
the treatment wins there by construction. The test that means something is
whether held-out error improved *more near the acquired points than far from
them*, in six-dimensional parameter distance normalized by the core-box
width:

| distance to nearest acquired case | n | before | after | delta |
|-----------------------------------|---|--------|-------|-------|
| 0.21 to 0.50 | 40 | 0.093 | 0.083 | -0.0097 |
| 0.50 to 0.59 | 39 | 0.084 | 0.081 | -0.0031 |
| 0.59 to 0.80 | 40 | 0.087 | 0.086 | -0.0017 |

The near bin improves about six times as much as the far bin, and the trend
is monotone (Spearman 0.31 between distance and error change, p = 0.001 over
119 cases). That is the loop mechanism doing what it should: data added where
the ensemble was unsure helps most in the neighborhood of what was added.
There is no global regression: 31% of held-out cases got worse, against a
mean change of -0.0075 (`data/samples/loop_neighborhood.png`).

Three caveats, none of them small. Twenty-six cases on a 481-case training
split is a 5% increase, so the absolute movement is a few thousandths of
relative L2 and this verifies a mechanism rather than delivering an accuracy
result. Twenty-six points is thin cover for a six-dimensional box: the median
held-out case sits 0.54 from the nearest acquired case but only 0.24 from the
nearest case already in training, so most of the evaluation set has no
acquired point near enough to be affected. And the loop inherits the blind
spot of its own acquisition signal, since spread-driven selection cannot see
consensus errors, the failure mode above. The out-of-distribution gains are
worth noting but not over-reading: acquisition ran in-box only, so nothing
targeted those slabs, and nose_small rests on 8 cases.

### Physics priors in the low-data regime

Three physics priors are built into the surrogate. Pressure is reconstructed
from the equation of state rather than predicted, density and temperature are
standardized in log10 space, and wall heat flux is available either as a
direct prediction or as a residual against the Fay-Riddell correlation. Each
was switched off in turn and priced against a baseline carrying all three.

Priors are supposed to earn their place when data is scarce, so the training
split is 100 cases drawn from the core box, stratified by geometry and
excluding the active-learning cases, which were selected adaptively and are
not an iid sample. The evaluation tiers are pinned to the same case lists
used everywhere else in this README and stay at full size, so only the
training set is small. Three seeds per cell, 500 epochs with cosine decay.
That schedule differs from the 350-epoch constant-rate recipe used for the
ensemble results above, so these rows are comparable to each other and not to
the numbers elsewhere on this page.

n_train = 100, mean +/- standard deviation over 3 seeds. Relative L2 is
averaged over (rho, u, v, T) so a cell predicting a fifth channel is scored
on the same four as the rest.

| prior cell | interpolation | family holdout | OOD pooled | q_w error | EoS violation | nonphysical |
|---|---|---|---|---|---|---|
| baseline (hard EoS, log rho/T) | 0.118 +/- 0.008 | 0.138 +/- 0.012 | 0.313 +/- 0.014 | -- | 0 | 0 |
| free p channel | 0.123 +/- 0.001 | 0.145 +/- 0.006 | 0.356 +/- 0.018 | -- | 0.0032 | 0 |
| plain standardization | 0.095 +/- 0.004 | 0.106 +/- 0.005 | 0.286 +/- 0.017 | -- | 0 | 0.023 |
| q_w head, direct | 0.128 +/- 0.003 | 0.153 +/- 0.004 | 0.327 +/- 0.018 | 0.075 +/- 0.004 | 0 | 0 |
| q_w head, Fay-Riddell residual | 0.151 +/- 0.002 | 0.171 +/- 0.007 | 0.331 +/- 0.005 | 0.061 +/- 0.006 | 0 | 0 |

`EoS violation` is the median relative departure from `p = rho R T`, which is
identically zero whenever pressure is reconstructed instead of predicted.
`nonphysical` is the fraction of held-out nodes where a channel that physics
requires to be positive (density, temperature, or reconstructed pressure)
comes out zero or negative, taking the worst such channel per case. `q_w error` is the median relative error of the heat-flux
head against the solver's own wall heat flux and exists only for the two
cells that have a head.

The normalization row is the one that does not say what the accuracy column
alone suggests. Plain standardization is more accurate on every tier and in
every seed, and the entire gap sits in density: 0.15 against 0.30 relative L2
on that channel, with the other three primitives indistinguishable. It also
predicts nonpositive density on about 2% of held-out nodes in every seed, and
temperature goes negative too, on a further 0.4%. In its worst single case
that reaches 37% of nodes in distribution and 61% once the extrapolation
slabs are included, which makes the reconstructed pressure negative and the
field useless downstream. A log-space channel
cannot do this at all, because exponentiating always lands positive. Relative
L2 barely notices, since the offending nodes sit in the thin freestream where
magnitudes are small and squared error is dominated by the shock layer. So
the log prior is not buying accuracy on this metric, it is buying a guarantee,
and a metric that only measures accuracy ranks a physically invalid field
first. The size of the accuracy penalty is also seed-dependent, ranging from
0.012 to 0.039 across the three seeds, while the nonphysical fraction holds
steady near 2%.

Hard equation-of-state reconstruction costs nothing worth having. Letting the
network predict pressure freely is no better in distribution, consistently
worse out of it, and buys a 0.32% median departure from the equation of
state. The constraint stays.

The Fay-Riddell residual head is a genuine trade rather than a free win.
Pairing by seed, it beats direct heat-flux prediction in three of three, by
19% on average, and the smallest of those margins is still wider than either
arm's seed-to-seed spread. It also degrades the flow field in three of three,
by more than the same spread. Anchoring the head to an analytical correlation
helps the quantity it anchors and pulls capacity away from the rest.

One caveat on how heat flux is measured here. The head is trained against the
solver's own wall heat flux and scored against it. The post-hoc estimate used
elsewhere in this README instead finite-differences the predicted temperature
field, and the two disagree by a median factor of about 27 on this dataset,
the finite-difference value being the smaller. That estimator is accurate on
the fine canonical mesh but underresolves the near-wall gradient at sweep
resolution. It remains a valid relative comparison, since predicted and true
fields pass through the same estimator, but it is not a physical heat flux
and the two numbers are not interchangeable.

Reproduce with `scripts/prior_ablation.py`, which builds the pinned split and
renders the table from the committed per-run records.

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

`scripts/acquire_loop.py` and `scripts/run_loop_cases.py` add cases the same
way, differing only in how the design points are chosen: by ensemble spread
rather than Latin hypercube. They carry `group_name='loop'` in the ledger,
which routes them into the training split and keeps them out of every
held-out tier.

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
  prior_splits_100.json    # pinned low-data split for the prior ablation
scripts/
  validate_cfd.py          # single-case SU2 validation driver
  generate_dataset.py      # resumable dataset sweep runner
  train.py                 # training CLI (slice sweep, ensembles, eval-only)
  slice_ablation.py        # ablation tables + envelope-distance figure
  ensemble_uq.py           # deep-ensemble UQ and trust/warn/refuse
  acquire_loop.py          # active-learning acquisition scan over the box
  run_loop_cases.py        # drives the acquired cases through SU2
  loop_report.py           # before/after report for the loop iteration
  prior_ablation.py        # physics-prior split builder and results table
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
