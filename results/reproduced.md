# Slice-count ablation

## Slice-count ablation (mean relative L2 per tier)

| M | interpolation | family holdout | cone_high | mach_high | nose_large | nose_small | pooled extrapolation |
|---|---|---|---|---|---|---|---|
| 8 | 0.121 | 0.129 | 0.428 | 0.215 | 0.344 | **0.257** | 0.313 |
| 16 | 0.105 | 0.117 | 0.343 | 0.194 | 0.300 | 0.295 | 0.278 |
| 32 | **0.100** | **0.104** | 0.337 | **0.189** | 0.292 | 0.277 | 0.270 |
| 64 | 0.108 | 0.111 | **0.307** | 0.202 | **0.269** | 0.286 | **0.260** |

## Tier table (M = 32)

| tier | rho | u | v | T | mean | n | median q_w error |
|---|---|---|---|---|---|---|---|
| interpolation | 0.203 | 0.050 | 0.065 | 0.081 | 0.100 | 43 | 63% |
| family holdout | 0.206 | 0.038 | 0.084 | 0.089 | 0.104 | 45 | 57% |
| cone_high (extrapolation) | 0.605 | 0.317 | 0.183 | 0.245 | 0.337 | 14 | 100% |
| mach_high (extrapolation) | 0.427 | 0.074 | 0.128 | 0.129 | 0.189 | 18 | 71% |
| nose_large (extrapolation) | 0.540 | 0.169 | 0.223 | 0.236 | 0.292 | 20 | 98% |
| nose_small (extrapolation) | 0.492 | 0.157 | 0.223 | 0.236 | 0.277 | 8 | 1567% |

guard threshold: envelope distance 0.07 or Kn > 0.01
41 evaluation cases sit above Kn 0.005 (largest 0.0306); the continuum margin is thin there

# Uncertainty and decision rule

thresholds: warn spread 0.0770, refuse spread 0.3080, guard d > 0.07, Kn > 0.01

## Ensemble against single members (mean relative L2 per tier)

| tier | n | ensemble | member mean | best member | spread |
|---|---|---|---|---|---|
| family holdout | 65 | 0.083 | 0.102 | 0.094 | 0.069 |
| interpolation | 54 | 0.084 | 0.102 | 0.092 | 0.064 |
| cone_high (extrapolation) | 14 | 0.220 | 0.262 | 0.250 | 0.160 |
| mach_high (extrapolation) | 18 | 0.162 | 0.186 | 0.170 | 0.103 |
| nose_large (extrapolation) | 20 | 0.276 | 0.307 | 0.277 | 0.153 |
| nose_small (extrapolation) | 8 | 0.208 | 0.235 | 0.223 | 0.128 |

## Spread-error calibration (Spearman rank corr per tier)

| tier | n | spearman |
|---|---|---|
| family holdout | 65 | 0.05 |
| interpolation | 54 | 0.55 |
| cone_high (extrapolation) | 14 | 0.83 |
| mach_high (extrapolation) | 18 | 0.84 |
| nose_large (extrapolation) | 20 | 0.74 |
| nose_small (extrapolation) | 8 | 0.86 |
| pooled | 179 | 0.72 |

## Trust/warn/refuse (eval tiers, actual error per bucket)

| decision | n | median err | p90 err | max err |
|---|---|---|---|---|
| trust | 93 | 0.078 | 0.114 | 0.422 |
| warn | 18 | 0.120 | 0.187 | 0.199 |
| refuse | 68 | 0.159 | 0.362 | 0.492 |


# Active-learning loop

[loop] 26 acquired cases, 481 pre-loop training cases, 246 paired eval cases

## Paired tier comparison (mean rel-L2 of the ensemble mean)

| tier | n | before | after | delta | 95% interval | p |
|---|---|---|---|---|---|---|
| family holdout | 65 | 0.090 | 0.083 | -0.0078 | [-0.0118, -0.0041] | 0.000 |
| interpolation | 54 | 0.085 | 0.084 | -0.0013 | [-0.0035, +0.0008] | 0.346 |
| cone_high (extrapolation) | 14 | 0.236 | 0.220 | -0.0159 | [-0.0245, -0.0057] | 0.025 |
| mach_high (extrapolation) | 18 | 0.171 | 0.162 | -0.0089 | [-0.0159, -0.0019] | 0.027 |
| nose_large (extrapolation) | 20 | 0.281 | 0.276 | -0.0048 | [-0.0144, +0.0041] | 0.522 |
| nose_small (extrapolation) | 8 | 0.244 | 0.208 | -0.0364 | [-0.0501, -0.0225] | 0.008 |

## Neighborhood generalization (in-distribution held-out cases)

| distance to nearest acquired point | n | before | after | delta | 95% interval |
|---|---|---|---|---|---|
| 0.208 to 0.495 | 40 | 0.093 | 0.083 | -0.0097 | [-0.0144, -0.0054] |
| 0.495 to 0.592 | 39 | 0.084 | 0.081 | -0.0031 | [-0.0072, +0.0005] |
| 0.592 to 0.800 | 40 | 0.087 | 0.086 | -0.0017 | [-0.0056, +0.0019] |

Spearman(distance, delta) = 0.31 (p = 0.001, n = 119). A working loop makes this positive: the improvement shrinks with distance.

Scale: the median held-out case sits 0.541 from the nearest of the 26 acquired cases but only 0.242 from the nearest of the 481 cases already in training. Twenty-six points is sparse cover for a six-dimensional box, so most held-out cases have no acquired point close enough for a local effect to reach them.

## No-regression check (all held-out tiers)

56 of 179 held-out cases got worse (31%); mean delta -0.0075, median -0.0062.

| case | tier | before | after | delta |
|---|---|---|---|---|
| case_0712 | nose_large (extrapolation) | 0.424 | 0.455 | +0.0310 |
| case_0757 | cone_high (extrapolation) | 0.373 | 0.402 | +0.0290 |
| case_0641 | family holdout | 0.170 | 0.199 | +0.0289 |
| case_0646 | family holdout | 0.059 | 0.082 | +0.0230 |
| case_0342 | family holdout | 0.102 | 0.125 | +0.0226 |


# Physics priors

Physics-prior ablation, n_train = [100], [3] seed(s) per cell.
Mean relative L2 over (rho, u, v, T), so every cell is scored on the same
channels, pooled over the two in-distribution tiers; cells show
mean +/- std across seeds.

The two q_w columns are median |relative error| against different
truths and are not comparable to each other. Post-hoc scores the
finite-difference estimate on the predicted field against the same
estimate on the true field. The head scores against the ledger's
SU2-postprocessed value, which is what it was trained on. Read each
column down its own cells, never across.

| prior cell | interpolation | family holdout | pooled extrapolation | q_w post-hoc | q_w head | equation-of-state violation | nonphysical |
|---|---|---|---|---|---|---|---|
| baseline (reconstructed p, log rho and T) | 0.1183 +/- 0.0075 | 0.1383 +/- 0.0124 | 0.3132 +/- 0.0135 | 0.8943 +/- 0.0677 | -- | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| free p channel (no equation-of-state constraint) | 0.1225 +/- 0.0014 | 0.1448 +/- 0.0058 | 0.3555 +/- 0.0175 | 0.9836 +/- 0.0023 | -- | 0.0032 +/- 0.0002 | 0.0000 +/- 0.0000 |
| plain standardization (no log) | 0.0948 +/- 0.0037 | 0.1055 +/- 0.0046 | 0.2856 +/- 0.0173 | 1.0509 +/- 0.0090 | -- | 0.0000 +/- 0.0000 | 0.0225 +/- 0.0016 |
| q_w head, direct | 0.1278 +/- 0.0030 | 0.1525 +/- 0.0041 | 0.3271 +/- 0.0183 | 0.9899 +/- 0.0610 | 0.0750 +/- 0.0035 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| q_w head, Fay-Riddell residual | 0.1510 +/- 0.0022 | 0.1713 +/- 0.0068 | 0.3311 +/- 0.0047 | 0.9594 +/- 0.0233 | 0.0606 +/- 0.0055 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
