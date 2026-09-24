# transolver-hypersonic

A neural surrogate for the steady-state flow field around axisymmetric
sphere-cone re-entry capsules in hypersonic flight. The architecture is
Transolver (Wu et al., ICML 2024,
[arXiv:2402.02366](https://arxiv.org/abs/2402.02366)). The ground truth is SU2,
solved as axisymmetric laminar Navier-Stokes.

Work in progress. The pipeline is validated and the sweep is complete. A
slice-count ablation, a calibrated deep ensemble, one active-learning iteration
and a physics-prior ablation are all finished, and the results are below.

## Where things stand

The stack was validated on AirfRANS first, then the simulation pipeline was
validated on one canonical case against three analytical correlations. Both
sections appear below.

A sweep over the sphere-cone design space produced 727 converged cases across
140 distinct geometries: a core box of 667 cases, plus four thin slabs that sit
outside it at large nose radius, small nose radius, high cone angle, and Mach
number above 25. An active-learning pass added 26 more, chosen where the
ensemble disagreed with itself, which brings the dataset to 753.

A first Transolver reproduces held-out flow fields, and
`data/samples/field_prediction.png` shows its four primitives beside the
solver's on a case it never saw. The results below cover the slice-count
ablation, a five-member deep ensemble with calibrated uncertainty, the
before-and-after comparison on the active-learning cases, and an ablation that
prices the physics priors in a 100-case regime. Every extrapolation slab is
evaluated, the high-Mach one included.

## Evaluation questions

1. **Slice-count ablation.** The Physics-Attention block of Transolver assigns
   each mesh point to one of M learned slices and attends across the slice
   tokens, which costs O(M^2) instead of O(N^2). How does M trade against
   accuracy in shock-dominated flow, and does the best M move between
   evaluation inside the training box and evaluation outside it?
2. **Geometric extrapolation.** Train on a core sphere-cone box and evaluate on
   extreme nose radii, extreme cone half-angles, and Mach numbers above the
   training range. Where does the surrogate degrade, and can the inputs alone
   reveal that boundary?

## Results

The dataset holds 753 converged cases. The sweep contributed 727 of them, which
are 667 core cases and four extrapolation slabs at large nose radius, small nose
radius, high cone angle and Mach number above 25. The active-learning loop
contributed the other 26.

Every evaluation tier holds the same cases throughout, because the loop cases go
into training and never into a held-out tier. All four extrapolation slabs are
evaluated. The relative L2 is computed per channel in physical units, after the
normalization is undone, and then averaged over the four primitives.

### Slice-count ablation

The table gives the mean relative L2 per tier against the slice count M, with
the best value in each column in bold. These four models predate the loop, so
they train on the 727-case sweep alone.

| M  | interpolation | family holdout | cone_high | mach_high | nose_large | nose_small | pooled extrapolation |
|----|------|------|------|------|------|------|------|
| 8  | 0.121 | 0.129 | 0.428 | 0.215 | 0.344 | **0.257** | 0.313 |
| 16 | 0.105 | 0.117 | 0.343 | 0.194 | 0.300 | 0.295 | 0.278 |
| 32 | **0.100** | **0.104** | 0.337 | **0.189** | 0.292 | 0.277 | 0.270 |
| 64 | 0.108 | 0.111 | **0.307** | 0.202 | **0.269** | 0.286 | **0.260** |

The best slice count moves with the evaluation regime. Inside the box the answer
is firm. M = 32 wins both the interpolation tier and the family holdout, and it
is bracketed on each, because M = 64 is worse.

Outside the box the grid says less than it appears to. M = 64 wins the pooled
number at 0.260, and on the individual slabs it takes two of four, cone_high and
nose_large. Mach extrapolation prefers M = 32 by 0.013 over 18 cases, and
nose_small prefers M = 8 over 8 cases. Neither margin carries weight at those
sample sizes. The pooled error also falls at every step of the grid, from 0.313
to 0.260, so the best extrapolation slice count is not bracketed and may sit past
M = 64. Finding it needs M = 96 and M = 128, and a full retrain for each.

What the grid does support is narrower. More slices help extrapolation after
they have stopped helping interpolation, so the Physics-Attention bottleneck
binds hardest where the flow leaves the training box. Below 16 slices, accuracy
falls everywhere.

Inside the box, generalization across geometry costs almost nothing. At M = 32
the family holdout, which holds whole geometry clusters absent from training,
scores 0.104 against 0.100 for freestream interpolation on shapes the model has
seen. The error triples at the envelope boundary and not between geometry tiers.

The figure `data/samples/envelope_distance.png` puts a threshold on that
boundary: the binned median error doubles about 7% outside the box, and that
number becomes the geometric half of the decision rule below.

The boundary is real and its measured position is not a fixed property of the
model. On a sparser extrapolation sample the same figure placed it at 3.8%, and
it moved outward to 7% once the cone_high slab grew from 5 cases to 14 and the
mach_high slab arrived with 18. Read it as a threshold calibrated against the
extrapolation data on hand.

### Uncertainty and the trust/warn/refuse rule

Five Transolvers at M = 32 share their splits and their data and differ only in
their starting weights and their data order, which makes them a deep ensemble.
The mean over members is the prediction and the spread across members is the
uncertainty. This is the model the dashboard serves, and it trains on all 753
cases. The mean beats the best single member on every tier.

| tier | ensemble | best single member | mean spread |
|------|------|------|------|
| interpolation             | 0.084 | 0.092 | 0.064 |
| family holdout            | 0.083 | 0.094 | 0.069 |
| cone_high (extrapolation) | 0.220 | 0.250 | 0.160 |
| mach_high (extrapolation) | 0.162 | 0.170 | 0.103 |
| nose_large (extrapolation)| 0.276 | 0.277 | 0.153 |
| nose_small (extrapolation)| 0.208 | 0.223 | 0.128 |

The margin over the best member grows where the flow leaves the training box.
The one exception is nose_large, where the two are indistinguishable.

Outside the box the spread tracks the actual error. The rank correlation between
the spread of a case and its error runs from 0.74 to 0.86 across the four slabs,
and 0.72 when they are pooled. Inside the box it is far weaker, at 0.55 on
interpolation and 0.05 on the family holdout, because in-box errors pack into a
narrow range and the ranking is close to noise
(`data/samples/spread_calibration.png`). The spread detects extrapolation. It
does not rank cases the model already handles well. The pooled 0.72 is inflated
by the wide error range of the slabs and must not be quoted as an in-box number.

That gives an operational rule for whether to trust a prediction. Refuse when
the input falls outside the training envelope, either by box exceedance or by a
Knudsen number past the continuum limit, and also when the members disagree by a wide
margin. Warn at moderate spread. Trust otherwise. Over the evaluation cases:

| decision | cases | median error | 90th-percentile error |
|------|------|------|------|
| trust  | 93 | 0.078 | 0.114 |
| warn   | 18 | 0.120 | 0.187 |
| refuse | 68 | 0.159 | 0.362 |

The buckets order by actual error with no overlap. One limitation deserves a
plain statement: the spread cannot catch an error the members agree on. One case inside
the box sits at 0.40 error with a low spread, because all five members share the
same systematic miss. Ensemble disagreement measures what the members do not
know between them, and not a blind spot they hold in common.

### Active-learning loop

One iteration of the obvious loop. Score a Latin hypercube over the six free
design axes by ensemble spread, pick a spread-out batch of the most uncertain
candidates inside the envelope, run them through SU2, retrain, and compare.
Thirty cases were selected and run, and 26 converged.

The 13% failure rate sits above the 3.5% of the sweep, which is what the method
predicts: acquisition targets the regions of highest spread, and those are the
regions where the solver strains too. The failures were logged and skipped.

The control is a five-member ensemble trained on the 727 sweep cases. The
treatment is the same recipe with the 26 loop cases added. Both share a split
seed, so every evaluation tier holds the same cases and each difference is
paired.

| tier | n | before | after | delta | 95% interval | p |
|------|---|--------|-------|-------|--------|---|
| interpolation             | 54 | 0.085 | 0.084 | -0.0013 | [-0.0035, +0.0008] | 0.35 |
| family holdout            | 65 | 0.090 | 0.083 | -0.0078 | [-0.0118, -0.0041] | <0.001 |
| cone_high (extrapolation) | 14 | 0.236 | 0.220 | -0.0159 | [-0.0245, -0.0057] | 0.03 |
| mach_high (extrapolation) | 18 | 0.171 | 0.162 | -0.0089 | [-0.0159, -0.0019] | 0.03 |
| nose_large (extrapolation)| 20 | 0.281 | 0.276 | -0.0048 | [-0.0144, +0.0041] | 0.52 |
| nose_small (extrapolation)| 8  | 0.244 | 0.208 | -0.0364 | [-0.0501, -0.0225] | 0.01 |

Error measured at the acquired points themselves proves nothing. Those cases sit
in the training set of the treatment and were never in the control, so the
treatment must win there. The test that means something asks whether held-out
error improved *more near the acquired points than far from them*, in
six-dimensional parameter distance normalized by the width of the core box.

| distance to nearest acquired case | n | before | after | delta |
|-----------------------------------|---|--------|-------|-------|
| 0.21 to 0.50 | 40 | 0.093 | 0.083 | -0.0097 |
| 0.50 to 0.59 | 39 | 0.084 | 0.081 | -0.0031 |
| 0.59 to 0.80 | 40 | 0.087 | 0.086 | -0.0017 |

The near bin improves about six times as much as the far bin, and the trend runs
one way, with a Spearman correlation of 0.31 between distance and error change
at p = 0.001 over 119 cases. That is the mechanism of the loop at work: data
added where the ensemble was unsure helps most around what was added. No global
regression appears. Of the held-out cases 31% got worse, against a mean change
of -0.0075 (`data/samples/loop_neighborhood.png`).

Three caveats, and none of them is small. Twenty-six cases on a 481-case
training split is a 5% increase, so the absolute movement is a few thousandths
of relative L2 and this verifies a mechanism rather than delivering an accuracy
result. Twenty-six points are thin cover for a six-dimensional box: the median
held-out case sits 0.54 from the nearest acquired case and only 0.24 from the
nearest case already in training, so most of the evaluation set has no acquired
point near enough to feel the effect. And the loop inherits the blind spot of
its own acquisition signal, because spread-driven selection cannot see an error
the members agree on.

The gains outside the box are worth noting and not worth over-reading.
Acquisition ran inside the box alone, so nothing targeted those slabs, and
nose_small rests on 8 cases.

### Physics priors in the low-data regime

Three physics priors are built into the surrogate. Pressure is reconstructed
from the equation of state rather than predicted. Density and temperature are
standardized in log10 space. Wall heat flux is available either as a direct
prediction or as a residual against the Fay-Riddell correlation. Each prior was
switched off in turn and priced against a baseline that carries all three.

A prior earns its place when data is scarce, so the training split holds 100
cases drawn from the core box, stratified by geometry. It excludes the
active-learning cases, which an adaptive rule selected and which are not an independent
sample. The evaluation tiers are pinned to the same case lists used everywhere
else on this page and stay at full size, so only the training set is small.
Three seeds ran per cell, at 500 epochs with cosine decay. That schedule differs
from the 350-epoch constant-rate recipe behind the ensemble results above, so
these rows compare against each other and not against the numbers elsewhere.

At n_train = 100, mean and standard deviation over 3 seeds. The relative L2
averages over (rho, u, v, T), so a cell that predicts a fifth channel is scored
on the same four as the rest.

| prior cell | interpolation | family holdout | pooled extrapolation | q_w error | equation-of-state violation | nonphysical |
|---|---|---|---|---|---|---|
| baseline (reconstructed p, log rho and T) | 0.118 +/- 0.008 | 0.138 +/- 0.012 | 0.313 +/- 0.014 | -- | 0 | 0 |
| free p channel | 0.123 +/- 0.001 | 0.145 +/- 0.006 | 0.356 +/- 0.018 | -- | 0.0032 | 0 |
| plain standardization | 0.095 +/- 0.004 | 0.106 +/- 0.005 | 0.286 +/- 0.017 | -- | 0 | 0.023 |
| q_w head, direct | 0.128 +/- 0.003 | 0.153 +/- 0.004 | 0.327 +/- 0.018 | 0.075 +/- 0.004 | 0 | 0 |
| q_w head, Fay-Riddell residual | 0.151 +/- 0.002 | 0.171 +/- 0.007 | 0.331 +/- 0.005 | 0.061 +/- 0.006 | 0 | 0 |

The violation column holds the median relative departure from `p = rho R T`,
which is identically zero wherever pressure is reconstructed. The nonphysical
column holds the fraction of held-out nodes where a channel that physics
requires to stay positive comes out zero or negative, taking the worst such
channel per case. Those channels are density, temperature and the reconstructed
pressure. The heat-flux column holds the median relative error of the head
against the wall heat flux the solver itself reported, and it exists only for
the two cells that carry a head.

The normalization row does not say what its accuracy column alone suggests.
Plain standardization is more accurate on every tier and in every seed, and the
whole gap sits in density, at 0.15 against 0.30 relative L2 on that channel,
with the other three primitives indistinguishable.

It also predicts a nonpositive density on about 2% of held-out nodes in every
seed, and temperature goes negative on a further 0.4%. In its worst single case
that reaches 37% of nodes inside the box and 61% once the extrapolation slabs
join, which drives the reconstructed pressure negative and makes the field
useless downstream. A log-space channel cannot do this at all, because an
exponent always lands positive.

The relative L2 moves little, because the offending nodes sit in the thin
freestream where magnitudes are small and the squared error is dominated by the
shock layer. So the log prior buys a guarantee rather than accuracy, and a
metric that measures accuracy alone ranks a physically invalid field first. The
size of the accuracy penalty is also seed-dependent, from 0.012 to 0.039 across
the three seeds, while the nonphysical fraction holds near 2%.

Hard reconstruction of pressure costs nothing worth having. A network that
predicts pressure without that constraint is no better inside the box, worse outside it in every
tier, and buys a median departure of 0.32% from the equation of state. The
constraint stays.

The Fay-Riddell residual head is a trade and not a free win. Paired by seed it
beats direct prediction in three of three, by 19% on average, and the smallest
of those margins is wider than the seed-to-seed spread of either arm. It also
degrades the flow field in three of three, by more than that same spread. An
analytical anchor helps the quantity it anchors and pulls capacity away from the
rest.

One caveat on how heat flux is measured here. The head trains against the wall
heat flux the solver reported and is scored against it. The estimate quoted
elsewhere on this page instead takes a finite difference of the predicted
temperature field, and the two disagree by a median factor of about 27 on this
dataset, with the finite-difference value the smaller. That estimator is
accurate on the fine canonical mesh and underresolves the near-wall gradient at
the resolution of the sweep. It stays a valid relative comparison, because the
predicted field and the true field pass through the same estimator, and it is
not a physical heat flux. Do not read the two numbers as one quantity.

Every table in this section rebuilds from the per-run records committed in
`results/`. `python scripts/check_reproduces.py` regenerates them into
`results/reproduced.md` and fails if any row here differs.

## Validation on AirfRANS

A Transolver of 0.52M parameters, with 4 blocks, 32 slices, 8 heads and 128
hidden channels, trained on the [AirfRANS](https://github.com/Extrality/AirfRANS)
`scarce` task, 200 cases for training and 200 for test, over 200 epochs. It took
64 minutes on one Kaggle T4. Relative L2 per channel on the test split:

| channel       | relative L2 |
|---------------|--------|
| u             | 0.053  |
| v             | 0.124  |
| p / rho       | 0.094  |
| nu_t          | 0.102  |
| p on surface  | 0.072  |

The figure is `data/samples/airfrans_validation.png`. The aim was never to
compete with the published AirfRANS Transolver. It was to confirm that the
vendored Physics-Attention, the training loop and the evaluation path all
function end to end on a free-tier accelerator, before they were pointed at the
SU2 data.

To reproduce it, open `notebooks/01_validate_stack.ipynb` on Kaggle, attach
`zeteixeira/airfrans-dataset` as input, set the accelerator to T4, turn the
internet on, and run every cell.

## Validation of the simulation pipeline

One canonical 60-degree sphere-cone, with a 1-inch nose, a body radius of
0.0762 m and a shoulder radius of 0.00762 m, at Mach 10, T_inf = 220 K,
p_inf = 100 Pa, and an isothermal cold wall at T_w = 300 K. SU2 version 8.5
solved it as axisymmetric laminar Navier-Stokes, with an ideal gas and
Sutherland viscosity. The solve runs in two passes: a first-order start, then a
restart at second order with the Roe scheme and the Venkatakrishnan limiter. The
mesh carried 66611 nodes and both passes together took 61 minutes.

| quantity              | SU2          | analytical                | rel. error | limit |
|-----------------------|--------------|---------------------------|---------|-----|
| stagnation heat flux  | 1.003 MW/m^2 | 1.111 MW/m^2 (Fay-Riddell) | -9.79%  | 15% |
| stagnation pressure   | 12.35 kPa    | 12.92 kPa (Rayleigh-Pitot) | -4.41%  |  5% |
| shock standoff        | 4.27 mm      | 3.75 mm (Billig)           | +13.79% | 20% |

All three clear their stated limits. The figure is at
`data/samples/cfd_validation.png`, and one full converged tensor of
`x, r, rho, u, v, T` per mesh node is at `data/samples/canonical_case.npz`.

To reproduce it on Linux or the Windows Subsystem for Linux, with SU2 version
8.5 on the executable search path and `pyvista`, `gmsh`, `matplotlib` and
`numpy` available:

```bash
python scripts/validate_cfd.py --iter-pass1 2500 --iter-pass2 8000
```

The driver writes the mesh, the two-stage configuration, the SU2 logs, the
volume and surface files, a comparison figure, a summary in JSON, and the
training tensor, all into `data/raw/cfd_validation/`.

## Dataset generation

`scripts/generate_dataset.py` sweeps the design box. A Latin hypercube samples
the geometry parameters (R_n, theta_c, R_b, R_s) and the freestream conditions
(Mach, T_inf and p_inf through altitude) over a core box, and four dedicated
slabs sit past the box edges for extrapolation. Each case runs the two-pass
recipe above, with the geometric multigrid that the sharper cones and the higher
Mach numbers need to converge, and each one is checked against the three
analytical correlations before acceptance.

The runner resumes without loss. A SQLite ledger tracks the status of every case, so
an interrupted session picks up where it stopped, and a failed case is logged
and skipped rather than rescued.

`scripts/acquire_loop.py` and `scripts/run_loop_cases.py` add cases the same
way. They differ in how the design points are chosen: by ensemble spread rather
than by Latin hypercube. They carry `group_name='loop'` in the ledger, which
routes them into the training split and keeps them out of every held-out tier.

The raw sweep output is not committed. The converged cases, as `case_*.npz`
tensors with the ledger, are published as the Kaggle dataset
`zeteixeira/su2-hypersonic-sphere-cone`, which is what the training notebook
reads.

The model predicts four primitives, (rho, u, v, T), at every mesh node.
Pressure is reconstructed as `p = rho * R_specific * T` at output time, so the
equation of state holds exactly and the network has no degree of freedom with
which to break it. The normalization statistics are computed per field on the
training split alone and are saved beside each checkpoint.

## Scope

The surrogate predicts steady laminar flow over axisymmetric sphere-cone
capsules at zero angle of attack. Training covers Mach 8 to 25, and evaluation
reaches Mach 30 outside that range. The gas is ideal and calorically perfect
air, the viscosity follows Sutherland, and the wall is isothermal at 300 K.
There is no turbulence model, no real-gas thermochemistry, no radiation and no
ablation. The value of the surrogate is speed at acceptable accuracy, and not
accuracy above SU2.

## Repository layout

```
src/
  analytical/__init__.py   # Fay-Riddell, Rayleigh-Pitot, Billig, Sutherland
  geometry/sphere_cone.py  # parametric sphere-cone and gmsh meshing
  cfd/runner.py            # SU2 configuration rendering and solver driver
  cfd/postprocess.py       # surface and axis-line readers, shock standoff
  cfd/ledger.py            # SQLite case ledger for the resumable sweep
  eval/sanity.py           # comparison against the analytical correlations
  eval/metrics.py          # relative L2 and the names of the evaluation tiers
  eval/uncertainty.py      # deep ensemble, spread and the trust/warn/refuse rule
  eval/plots.py            # acceptance figures
  data/airfrans.py         # AirfRANS dataset, normalization, loading
  data/sampler.py          # design-box sampling for the sweep
  data/su2.py              # SU2 case dataset, splits, normalization
  training/loop.py         # training and evaluation loop, for SU2 and AirfRANS
  models/transolver.py     # vendored Transolver (MIT, thuml/Transolver)
configs/
  sphere_cone_template.cfg # SU2 axisymmetric laminar Navier-Stokes template
  prior_splits_100.json    # pinned low-data split for the prior ablation
scripts/
  validate_cfd.py          # single-case SU2 validation driver
  generate_dataset.py      # resumable dataset sweep runner
  train.py                 # training entry point, slice sweep, ensembles
  slice_ablation.py        # ablation tables and the envelope-distance figure
  ensemble_uq.py           # deep-ensemble uncertainty and the decision rule
  acquire_loop.py          # active-learning acquisition scan over the box
  run_loop_cases.py        # drives the acquired cases through SU2
  loop_report.py           # before-and-after report for the loop iteration
  prior_ablation.py        # physics-prior split builder and results table
  run_cases.py             # resumable SU2 job runner, used for the solver study
  check_reproduces.py      # rebuilds every results table and checks this README
notebooks/
  01_validate_stack.ipynb  # AirfRANS validation, Kaggle orchestrator
  02_train_su2.ipynb       # SU2 training runs, Kaggle orchestrator
data/samples/              # committed example outputs: figures and one case
results/                   # per-run evaluation records behind every results table
```

## License

Code in `src/models/transolver.py` is adapted from
[thuml/Transolver](https://github.com/thuml/Transolver), MIT licensed,
copyright 2024 THUML at Tsinghua University, with the differences noted in the
header of that file.
