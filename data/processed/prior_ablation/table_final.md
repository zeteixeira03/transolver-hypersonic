Physics-prior ablation, n_train = [100], [3] seed(s) per cell.
Mean rel-L2 over (rho, u, v, T) so every cell is scored on the same
channels, pooled over the two in-distribution tiers; cells show
mean +/- std across seeds.

The two q_w columns are median |relative error| against different
truths and are not comparable to each other. Post-hoc scores the
finite-difference estimate on the predicted field against the same
estimate on the true field. The head scores against the ledger's
SU2-postprocessed value, which is what it was trained on. Read each
column down its own cells, never across.

| prior cell | interpolation | family holdout | OOD pooled | q_w post-hoc | q_w head | EoS violation | nonphysical |
|---|---|---|---|---|---|---|---|
| baseline (hard EoS, log rho/T) | 0.1183 +/- 0.0075 | 0.1383 +/- 0.0124 | 0.3132 +/- 0.0135 | 0.8943 +/- 0.0677 | -- | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| free p channel (no EoS constraint) | 0.1225 +/- 0.0014 | 0.1448 +/- 0.0058 | 0.3555 +/- 0.0175 | 0.9836 +/- 0.0023 | -- | 0.0032 +/- 0.0002 | 0.0000 +/- 0.0000 |
| plain standardization (no log) | 0.0948 +/- 0.0037 | 0.1055 +/- 0.0046 | 0.2856 +/- 0.0173 | 1.0509 +/- 0.0090 | -- | 0.0000 +/- 0.0000 | 0.0225 +/- 0.0016 |
| q_w head, direct | 0.1278 +/- 0.0030 | 0.1525 +/- 0.0041 | 0.3271 +/- 0.0183 | 0.9899 +/- 0.0610 | 0.0750 +/- 0.0035 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| q_w head, Fay-Riddell residual | 0.1510 +/- 0.0022 | 0.1713 +/- 0.0068 | 0.3311 +/- 0.0047 | 0.9594 +/- 0.0233 | 0.0606 +/- 0.0055 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
