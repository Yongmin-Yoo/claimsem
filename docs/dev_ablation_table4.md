# Verified DEV Table 4 ablation

This document records the verified development-set ablation results for
ClaimSem Table 4.

## Protocol

- Split: fixed development set
- Patents: 9,855
- CPC levels: section, class, subclass
- Clusters: 30
- Clustering seeds: 17, 42, 73
- Clustering backend: legacy GPU spherical K-means
- Maximum iterations: 100
- Tolerance: `1e-5`
- Primary representation dimension: 128
- PCA protocol: one frozen development-set PCA transform
- Full ClaimSem anchor Mean NMI: `0.368289425533`

The document-level control concatenates the patent claims in claim-number
order, applies masked-mean PatentSBERTa encoding with maximum length 512,
and then applies the same frozen 768-to-128-dimensional PCA transform.

## Results

| Variant | Sec. | Class | Subcl. | Mean NMI | Delta NMI | Mean Pur_p | Mean Pur_a |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full \textsc{ClaimSem} | 0.2632 | 0.3922 | 0.4494 | 0.3683 | +0.0000 | 0.4167 | 0.3478 |
| Uniform claim pooling | 0.2346 | 0.3559 | 0.4182 | 0.3362 | -0.0321 | 0.3931 | 0.3130 |
| No root emphasis | 0.2352 | 0.3579 | 0.4204 | 0.3378 | -0.0304 | 0.3941 | 0.3150 |
| No depth decay | 0.2626 | 0.3920 | 0.4495 | 0.3681 | -0.0002 | 0.4160 | 0.3465 |
| Root claims only | 0.2592 | 0.3881 | 0.4452 | 0.3642 | -0.0041 | 0.4156 | 0.3458 |
| First claim only | 0.2532 | 0.3817 | 0.4397 | 0.3582 | -0.0101 | 0.4108 | 0.3357 |
| Shuffled dependent depths | 0.2630 | 0.3918 | 0.4493 | 0.3680 | -0.0003 | 0.4161 | 0.3475 |
| Document-level encoding | 0.2570 | 0.3824 | 0.4401 | 0.3598 | -0.0085 | 0.4096 | 0.3325 |
| No PCA reduction | 0.2588 | 0.3883 | 0.4442 | 0.3638 | -0.0045 | 0.4149 | 0.3403 |

## Verification

- All nine Mean NMI values were verified against the saved predictions.
- All clustering runs converged.
- All 30 clusters remained active.
- The anchor was reproduced exactly:
  expected `0.368289425533`, measured `0.368289425533`.
- Corrected `root_claims_only` and `first_claim_only` features use explicit
  pre-accumulator claim filtering.
- CPC labels were read from the trusted record fields `section`, `class`,
  and `subclass`.

## Artifact policy

Small publication and provenance files are committed to GitHub. Raw/PCA
feature matrices and prediction files are stored in the private Hugging
Face dataset:

`https://huggingface.co/datasets/yongminyoo91/claimsem-dev-ablation-tables-5-6/tree/main/table4`

Intermediate shard partials, quarantined invalid outputs, and redundant
document-encoding chunks are intentionally excluded from the uploaded
bundle.
