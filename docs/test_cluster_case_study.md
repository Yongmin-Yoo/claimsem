# Deterministic TEST cluster case study for Tables 2–3

## Protocol

Tables 2 and 3 use the Full ClaimSem TEST partition produced with the fixed seed 42. Quantitative results in Table 1 remain averages over seeds 17, 42, and 73. The case study does not select the best-performing seed post hoc.

For each of the 30 clusters, subclass purity is defined as the fraction of patents belonging to the most frequent CPC subclass. Clusters containing fewer than 50 patents are excluded. Among the eligible clusters, the two highest-purity clusters and the lowest-purity cluster are selected automatically. Ties are resolved by larger cluster size and then by the zero-based internal cluster identifier. CPC labels are used only after clustering for cluster selection and description.

Representative patents are selected by cosine similarity to their assigned cluster centroid in the normalized 128-dimensional PCA space used by spherical K-means. No patent text or CPC label is manually inspected before cluster or patent selection.

## Verified results

- TEST patents: 9,881
- CPC cardinality: 9 sections, 121 classes, and 466 subclasses
- PCA features: 9,881 by 128
- Active clusters: 30
- Case-study seed: 42
- Selected clusters: C03, C01, and C04
- C03: 340 patents, subclass purity 0.738235, dominant subclass H01L
- C01: 222 patents, subclass purity 0.734234, dominant subclass G06Q
- C04: 335 patents, subclass purity 0.041791, dominant subclass F16B
- Saved labels exactly match nearest-centroid reassignment
- Independent copies of the seed-42 predictions are identical

## Large artifacts

The TEST PCA representations, complete seed-42 clustering artifact, and selected centroids are stored in the private Hugging Face dataset: https://huggingface.co/datasets/yongminyoo91/claimsem-test-case-study

Raw patent records and complete claim text are not redistributed.
