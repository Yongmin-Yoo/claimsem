# Verified DEV sensitivity and robustness experiments

This directory documents the verified development-only experiments used for
ClaimSem Tables 5 and 6.

## Data policy

The experiments use 9,855 development patents. TEST labels and TEST
representations are not used.

## Table 5 protocol

Table 5 varies the root weight over `{1, 2, 4, 8, 12, 16}` and the
depth-decay coefficient over `{0, 0.05, 0.10, 0.20}`.

To isolate the effect of pooling, all 24 configurations use the same frozen
128-dimensional PCA transform fitted on the full ClaimSem development
representations. PCA is not refitted for each pooling configuration.

All configurations use spherical K-means with `K=30` and seeds
`{17, 42, 73}`.

The validated anchor is:

- pool: `root12_d010`
- mean NMI: `0.368289425533`

The highest full-development value is:

- pool: `root12_d020`
- mean NMI: `0.368636867127`

The difference from the selected `root12_d010` configuration is approximately
`0.000347`.

## Table 6 protocol

The full configuration `root12_d010` is evaluated with
`K in {20, 25, 30, 35, 40}`. All seeds converge and every requested cluster
is active.

## PCA provenance

A separately refitted PCA did not reproduce the historical DEV anchor.
Applying the original fitted DEV PCA model reproduced the historical anchor
features with maximum absolute difference below `1e-5`. Therefore the
verified sensitivity protocol uses the frozen original DEV PCA transform.

## Large artifacts

Patent records, raw representations, PCA representations, predictions, and
the fitted PCA model are not stored in Git. They are maintained in a private
Hugging Face Dataset pending redistribution-license review.
