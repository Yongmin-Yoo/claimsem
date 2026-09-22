# Final Train-Split Learned-Aggregation Evaluation

## Status

The audit completed and passed all integrity checks.

- Neural-training runs: 10 SSL-MLP and 10 Structural VICReg-GAT
- Clustering seeds per representation: 5
- Evaluation jobs: 22/22
- K-means runs: 110/110
- Paired-document bootstrap replicates: 2,000/2,000
- All K-means runs converged with all 30 clusters active
- CPC labels and TEST data were excluded from training and
  checkpoint selection

## Final TEST Results

| Method | Mean TEST NMI |
|---|---:|
| Uniform pooling | 0.345118692 |
| SSL-MLP + attention | 0.347182440 |
| Structural VICReg-GAT | 0.350546536 |
| **Balanced ROOTS** | **0.376691006** |

The learned models improve over uniform pooling on average:

- SSL-MLP minus Uniform: +0.002063748 NMI
- Structural VICReg-GAT minus Uniform: +0.005427844 NMI

## Primary Comparison

Balanced ROOTS minus Structural VICReg-GAT:

- Mean difference: +0.026144470 NMI
- Training-seed 95% CI: [0.025607057, 0.026681884]
- Paired-document bootstrap 95% CI:
  [0.021951984, 0.028013992]

The training-seed interval reflects learning randomness across
10 independently trained GAT models after averaging five
clustering runs per model. The bootstrap interval reflects paired
TEST-document resampling conditional on fixed checkpoints, PCA
transformations, and clustering outputs.

Balanced ROOTS has higher mean NMI under the evaluated protocol.
This result does not imply superiority for every possible learned
architecture, clustering run, or optimization budget.

## Optimization Qualification

SSL-MLP used at most 15 epochs and Structural VICReg-GAT used at
most 45 epochs. Six MLP runs selected epoch 15. Eight GAT runs
selected epoch 45, while two selected epoch 44.

These checkpoint selections do not establish full neural
optimization convergence. Longer training may change the learned
results, but additional training is not required to support the
stated comparison within the evaluated budget.

All 110 K-means runs converged. This clustering result is reported
separately from neural-training convergence.

## Methodological Scope

The deterministic inference architectures were recovered from
archived checkpoints. The original stochastic training source was
unavailable, so the train-split stochastic protocol was newly
specified and is not an exact reconstruction.

Earlier DEV-only three-seed scores, 25--27 second runtime
measurements, and attention statistics are not used as evidence
for the final train-split comparison.

## Artifact Policy

Git contains compact summaries, configurations, provenance
metadata, and hashes only. Checkpoints, graph caches, patent data,
CPC labels, raw representations, PCA binaries, predictions, and raw
bootstrap samples remain outside the repository.

External report package SHA-256:

`32f3ede28f2dc613595e84d6faa52df93b21954b719734a85dfda0843d63290e`
