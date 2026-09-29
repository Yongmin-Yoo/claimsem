# Additional clustering baselines

This experiment adds four clustering baselines requested during internal review.

## Results

| Method | Section NMI | Class NMI | Subclass NMI | Mean NMI |
|---|---:|---:|---:|---:|
| TF-IDF + spherical K-means | 0.208890 | 0.314162 | 0.380350 | 0.301134 |
| TF-IDF + LSA-256 + spherical K-means | 0.212699 | 0.317553 | 0.381085 | 0.303779 |
| PatentSBERTa-V2 + agglomerative | 0.204722 | 0.303847 | 0.301258 | 0.269942 |
| MiniLM + agglomerative | 0.218035 | 0.215051 | 0.226545 | 0.219877 |

TF-IDF and LSA are fitted using the training split only. Spherical K-means
uses K=30 and seeds 17, 42, and 73. Agglomerative clustering uses cosine
distance, average linkage, and K=30 and is deterministic.

Large artifacts are stored in the private Hugging Face dataset:
https://huggingface.co/datasets/yongminyoo91/roots-additional-clustering-baselines

Raw patent records and full claim text are not redistributed.
