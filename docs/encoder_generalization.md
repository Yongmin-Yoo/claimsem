# Encoder Generalization

ClaimSem root-and-depth-aware aggregation was evaluated using fixed
patent splits, PCA dimensionality, spherical K-means configuration,
and seeds.

## Reported encoders

| Encoder | Uniform NMI | ClaimSem NMI | Delta | Relative gain |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 0.3147 | 0.3493 | +0.0347 | +11.0% |
| all-mpnet-base-v2 | 0.3391 | 0.3626 | +0.0235 | +6.9% |
| BERT-base-uncased | 0.2940 | 0.2991 | +0.0052 | +1.8% |
| PatentSBERTa-V2 | 0.3428 | 0.3755 | +0.0327 | +9.5% |

## Evaluation configuration

- Seeds: 17, 42, 73
- Clusters: 30
- PCA dimensionality: 128
- PCA fitting split: DEV
- Clustering: spherical K-means
- CPC granularities: section, class, subclass
- DEV patents: 9,855
- TEST patents: 9,881
- DEV claims: 160,048
- TEST claims: 161,661

SciBERT is retained as an exploratory artifact in the private
Hugging Face repository and is not included in the reported table.

PatentSBERTa numerical validation was intentionally excluded from the
release validation cell at the user's request.
