# Final train-split learned-aggregation evaluation

## Status

The train-split learned-aggregation audit completed successfully.

- SSL-MLP + attention training runs: 10
- Structural VICReg-GAT training runs: 10
- Independent clustering seeds per representation: 5
- Evaluation jobs verified: 22/22
- Spherical K-means runs verified: 110/110
- Paired-document bootstrap replicates: 2,000/2,000
- All clustering runs converged.
- All 30 clusters were active in every run.
- CPC labels and TEST data were not used for training or checkpoint
  selection.

## Data and fixed evaluation protocol

- Train patents: 49,599
- Development patents: 9,855
- Test patents: 9,881
- Training seeds: 11, 17, 23, 29, 37, 42, 53, 61, 73, 89
- Clustering seeds: 17, 42, 73, 101, 137
- PCA: 768 to 128 dimensions, fitted separately on DEV for each
  representation
- PCA refitted on TEST: no
- Clustering: transductive spherical K-means on TEST
- Number of clusters: 30
- Primary metric: mean NMI across CPC section, class, and subclass

## Final TEST results

| Method | Mean NMI |
|---|---:|
| Uniform pooling | 0.345118692 |
| SSL-MLP + attention | 0.347182440 |
| Structural VICReg-GAT | 0.350546536 |
| **Balanced ROOTS** | **0.376691006** |

## Primary comparison

Balanced ROOTS minus Structural VICReg-GAT:

- Mean difference: **0.026144470 NMI**
- Between-training-run 95% CI:
  **[0.025607057, 0.026681884]**
- Paired-document bootstrap 95% CI:
  **[0.021951984, 0.028013992]**

Both intervals exclude zero. Under the fixed PatentSBERTa
representation, DEV-fitted PCA, and TEST clustering protocol, Balanced
ROOTS consistently outperformed the Structural VICReg-GAT.

The learned aggregators nevertheless improved over uniform pooling:

- SSL-MLP minus uniform pooling:
  +0.002063748 NMI
- Structural VICReg-GAT minus uniform pooling:
  +0.005427844 NMI

## Methodological qualification

The deterministic inference architectures were recovered from archived
checkpoints. The unavailable original stochastic training source was
not recovered. The train-split stochastic optimization protocol used in
this audit was newly specified and preregistered; it must not be
described as an exact reconstruction of the unavailable original
training implementation.

## Artifact policy

Large or sensitive artifacts are not stored in Git:

- model checkpoints;
- graph caches;
- patent records and CPC labels;
- raw learned representations;
- fitted PCA binary objects;
- TEST cluster predictions;
- raw bootstrap samples.

The repository contains only compact summaries, configurations,
provenance metadata, and SHA-256 digests.

Final external report package SHA-256:

`32f3ede28f2dc613595e84d6faa52df93b21954b719734a85dfda0843d63290e`
