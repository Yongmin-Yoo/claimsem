# ClaimSem

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Google-Colab-orange.svg)](https://colab.research.google.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPU-red.svg)](https://pytorch.org/)

**Structure-aware claim aggregation for patent clustering**

ClaimSem is a lightweight method for constructing patent-level representations from individual patent claims. It encodes each claim separately with a frozen patent language model, emphasizes independent claims, attenuates dependent claims according to their dependency depth, reduces the resulting patent representations with PCA, and clusters them using spherical $K$-means.

Repository:

https://github.com/Yongmin-Yoo/claimsem

Author:

**Yongmin Yoo**

---

## Overview

Patent claims have different legal and structural roles. Independent claims define the basic scope of an invention, while dependent claims add further limitations to preceding claims. Treating all claims as interchangeable text units can therefore discard useful structural information.

ClaimSem addresses two limitations of conventional patent representation:

1. **Document-level truncation:** Concatenating all claims can exceed the maximum input length of a language model and remove later claims.
2. **Structure-agnostic aggregation:** Uniformly averaging separately encoded claims ignores the distinction between independent and dependent claims.

ClaimSem uses the following pipeline:

```text
Patent claims
→ claim-dependency validation
→ claim-depth calculation
→ individual frozen claim encoding
→ root- and depth-aware pooling
→ development-fitted PCA
→ L2 normalization
→ spherical K-means
→ patent clusters
```

ClaimSem does not require:

- a trained Depth-OT checkpoint
- patent-topic distributions
- topic-word distributions
- optimal transport plans
- a neural dependency encoder
- additional neural fine-tuning

---

## Method

### Patent claim-dependency graph

Let a patent $P_n$ contain claims


$$
\mathcal{C}_n
=
\{c_{n,1},\ldots,c_{n,M_n}\}.

$$

The claims form a directed dependency graph


$$
G_n
=
(\mathcal{C}_n,E_n).

$$

An edge $(c_i,c_j)\in E_n$ indicates that claim $c_j$ refers to and depends on claim $c_i$. Claims without valid antecedents are treated as root claims. A patent may contain multiple root claims when it has more than one independent claim.

### Claim dependency depth

The depth of claim $c$ is defined recursively as


$$
d(c)
=
\begin{cases}
0, & \operatorname{Pa}(c)=\varnothing,\\
1+\max\limits_{c'\in\operatorname{Pa}(c)}d(c'),
& \text{otherwise},
\end{cases}

$$

where $\operatorname{Pa}(c)$ is the set of valid parent claims referenced by claim $c$.

A root claim has depth zero. A dependent claim receives a depth determined by the longest valid path from a root claim.

### Individual claim encoding

Each claim is encoded independently with a frozen PatentSBERTa-V2 encoder.

Let


$$
H_c
=
[\mathbf{h}_{c1},\ldots,\mathbf{h}_{cL_c}]

$$

denote the contextual token representations of claim $c$, and let $m_{c\ell}$ denote its attention mask.

Masked mean pooling produces the claim representation


$$
\mathbf{e}_c
=
\frac{
\sum_{\ell=1}^{L_c}
m_{c\ell}\mathbf{h}_{c\ell}
}{
\sum_{\ell=1}^{L_c}
m_{c\ell}
}.

$$

Encoding claims separately prevents later claims from being removed because earlier claims consume the document-level input budget.

The pretrained encoder remains frozen throughout feature construction and clustering.

### Root- and depth-aware pooling

ClaimSem assigns each claim a weight based on its root status and dependency depth:


$$
w_c(\alpha,\lambda)
=
\alpha^{\mathbb{I}[d(c)=0]}
\exp\left(-\lambda d(c)\right),

$$

where:

- $\alpha$ controls the emphasis placed on root claims
- $\lambda$ controls the attenuation applied to deeper dependent claims

The patent-level representation is


$$
\mathbf{v}_n
=
\frac{
\sum_{c\in\mathcal{C}_n}
w_c(\alpha,\lambda)\mathbf{e}_c
}{
\sum_{c\in\mathcal{C}_n}
w_c(\alpha,\lambda)
}.

$$

Uniform claim pooling is recovered when $\alpha=1$ and $\lambda=0$.

### Dimensionality reduction

The pooled representation $\mathbf{v}_n$ has 768 dimensions. A PCA transform fitted on development representations projects it to 128 dimensions:


$$
\widetilde{\mathbf{r}}_n
=
W_{\mathrm{PCA}}
\left(
\mathbf{v}_n
-
\boldsymbol{\mu}_{\mathrm{PCA}}
\right).

$$

The reduced representation is normalized to unit length:


$$
\mathbf{r}_n
=
\frac{
\widetilde{\mathbf{r}}_n
}{
\left\lVert
\widetilde{\mathbf{r}}_n
\right\rVert_2
}.

$$

The PCA transform is fitted only on development representations. The same fixed transform is applied to test representations without refitting.

### Spherical clustering

The normalized patent representations are partitioned using spherical $K$-means.

The cluster assignment of patent $P_n$ is


$$
z_n
=
\underset{k\in\{1,\ldots,K\}}{\arg\max}
\;
\mathbf{r}_n^{\top}\boldsymbol{\nu}_k,

$$

where $\boldsymbol{\nu}_k$ is the unit-normalized centroid of cluster $k$.

The final configuration uses $K=30$ clusters.

---

## Final Configuration

| Component | Value |
|---|---:|
| Method | ClaimSem |
| Claim encoder | Frozen PatentSBERTa-V2 |
| Claim encoding | Individual claim encoding |
| Root weight $\alpha$ | 12.0 |
| Depth decay $\lambda$ | 0.1 |
| Original embedding dimension | 768 |
| PCA output dimension | 128 |
| Number of clusters | 30 |
| Clustering method | Spherical $K$-means |
| Clustering seeds | 17, 42, 73 |
| Clustering protocol | Transductive |
| Encoder fine-tuning | None |
| CPC labels used as encoder targets | No |
| Test CPC labels used for tuning | No |

The final configuration was selected on development data and frozen before test evaluation.

---

## Final Test Results

### Dataset statistics

| Split | Patents | Claims | CPC sections | CPC classes | CPC subclasses |
|---|---:|---:|---:|---:|---:|
| Development | 9,855 | 160,048 | 9 | 123 | 484 |
| Test | 9,881 | 161,661 | 9 | 121 | 466 |

### CPC alignment results

The results are averaged over spherical $K$-means seeds 17, 42, and 73.

| CPC level | Predicted-cluster purity | Label-wise inverse purity | NMI |
|---|---:|---:|---:|
| Section | $0.617009 \pm 0.010069$ | $0.188442 \pm 0.005584$ | $0.273401 \pm 0.004359$ |
| Class | $0.411328 \pm 0.006142$ | $0.366124 \pm 0.004701$ | $0.398139 \pm 0.003565$ |
| Subclass | $0.249030 \pm 0.008411$ | $0.512870 \pm 0.007868$ | $0.454984 \pm 0.004101$ |
| Mean | $0.425789 \pm 0.008153$ | $0.355812 \pm 0.004877$ | $0.375508 \pm 0.003952$ |

All 30 clusters remain active.

The maximum cluster share is


$$
0.055662 \pm 0.001145.

$$

### Seed-level results

| Seed | Mean NMI | Mean predicted-cluster purity | Mean label-wise inverse purity |
|---:|---:|---:|---:|
| 17 | 0.379336 | 0.430152 | 0.359444 |
| 42 | 0.370068 | 0.414364 | 0.348919 |
| 73 | 0.377121 | 0.432851 | 0.359073 |

---

## Evaluation Protocol

ClaimSem uses a transductive clustering protocol.

For each evaluation split:

1. Claim embeddings are pooled into patent representations.
2. The fixed development-fitted PCA transform is applied.
3. Spherical $K$-means is fitted to the unlabeled representations of the evaluation split.
4. Cluster assignments are generated without CPC labels.
5. CPC labels are accessed only after clustering.
6. NMI, predicted-cluster purity, and label-wise inverse purity are computed.

This setting should not be interpreted as inductive CPC classification. The method produces unsupervised patent clusters rather than CPC predictions.

Development CPC labels are used only for configuration selection. Test CPC labels are reserved for final evaluation.

---

## Repository Structure

```text
claimsem/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements-colab.txt
├── CITATION.cff
├── .gitignore
│
├── configs/
│   ├── final_claimsem.json
│   ├── dev_search.json
│   └── smoke_test.json
│
├── notebooks/
│   ├── 01_prepare_and_encode.ipynb
│   ├── 02_dev_selection_and_ablation.ipynb
│   ├── 03_final_test_evaluation.ipynb
│   └── 04_generate_paper_tables.ipynb
│
├── src/
│   └── claimsem/
│       ├── __init__.py
│       ├── config.py
│       ├── reproducibility.py
│       ├── data.py
│       ├── dependency.py
│       ├── encoder.py
│       ├── pooling.py
│       ├── reduction.py
│       ├── clustering.py
│       ├── metrics.py
│       ├── artifacts.py
│       ├── dev_search.py
│       └── test_evaluation.py
│
├── scripts/
│   ├── prepare_features.py
│   ├── run_dev_search.py
│   ├── run_ablation.py
│   ├── run_final_test.py
│   └── make_tables.py
│
├── tests/
│   ├── test_dependency.py
│   ├── test_pooling.py
│   ├── test_metrics.py
│   └── test_data_alignment.py
│
└── demo/
    ├── demo_records.json
    └── README.md
```

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Configuration loading and validation |
| `reproducibility.py` | Random seeds and environment logging |
| `data.py` | Patent records, CPC labels, and feature loading |
| `dependency.py` | Dependency validation and depth calculation |
| `encoder.py` | Frozen claim encoding and masked mean pooling |
| `pooling.py` | Root- and depth-aware patent representation |
| `reduction.py` | Development-only PCA fitting and transformation |
| `clustering.py` | Spherical $K$-means clustering |
| `metrics.py` | NMI, purity, inverse purity, and cluster balance |
| `artifacts.py` | Artifact, result, and manifest management |
| `dev_search.py` | Development search and ablation experiments |
| `test_evaluation.py` | Frozen final test evaluation |

---

## Google Colab

### Select a GPU runtime

In Google Colab, select:

```text
Runtime
→ Change runtime type
→ Hardware accelerator
→ T4 GPU
```

### Clone the repository

```python
!git clone https://github.com/Yongmin-Yoo/claimsem.git
%cd claimsem
```

### Install dependencies

```python
!pip install -q -r requirements-colab.txt
!pip install -q -e .
```

### Verify the environment

```python
import torch
import claimsem

print("ClaimSem import successful")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

### Mount Google Drive

```python
from google.colab import drive

drive.mount("/content/drive")
```

---

## Colab Notebooks

### 1. Prepare and encode

File:

```text
notebooks/01_prepare_and_encode.ipynb
```

Open in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yongmin-Yoo/claimsem/blob/main/notebooks/01_prepare_and_encode.ipynb)

This notebook:

- loads patent records
- validates dependency graphs
- calculates claim depths
- encodes claims individually
- saves resumable feature shards
- verifies record and feature alignment

### 2. Development selection and ablation

File:

```text
notebooks/02_dev_selection_and_ablation.ipynb
```

Open in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yongmin-Yoo/claimsem/blob/main/notebooks/02_dev_selection_and_ablation.ipynb)

This notebook:

- creates the fixed development partition
- evaluates candidate pooling settings
- fits development PCA models
- runs controlled ablations
- conducts sensitivity analysis
- saves the selected configuration

### 3. Final test evaluation

File:

```text
notebooks/03_final_test_evaluation.ipynb
```

Open in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yongmin-Yoo/claimsem/blob/main/notebooks/03_final_test_evaluation.ipynb)

This notebook:

- loads the frozen final configuration
- applies root- and depth-aware pooling
- applies the development-fitted PCA transform
- performs spherical $K$-means with three fixed seeds
- evaluates CPC alignment
- saves predictions, assignments, and metrics

### 4. Generate paper tables

File:

```text
notebooks/04_generate_paper_tables.ipynb
```

Open in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yongmin-Yoo/claimsem/blob/main/notebooks/04_generate_paper_tables.ipynb)

This notebook generates:

- CPC result tables
- ClaimSem ablation tables
- sensitivity tables
- seed-level result tables
- LaTeX-ready values

---

## Installation Outside Colab

Python 3.10 or later is recommended.

```bash
git clone https://github.com/Yongmin-Yoo/claimsem.git
cd claimsem

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
```

Run the tests:

```bash
pytest -q
```

---

## Data Format

A patent record should follow this structure:

```json
{
  "patent_id": "PATENT_000001",
  "claims": [
    {
      "claim_id": "1",
      "text": "A method comprising ...",
      "parent_ids": []
    },
    {
      "claim_id": "2",
      "text": "The method of claim 1, wherein ...",
      "parent_ids": ["1"]
    }
  ],
  "cpc": {
    "section": "G",
    "class": "G06",
    "subclass": "G06F"
  }
}
```

### Required fields

| Field | Type | Description |
|---|---|---|
| `patent_id` | String | Unique patent identifier |
| `claims` | List | Claims belonging to the patent |
| `claim_id` | String or integer | Claim identifier within the patent |
| `text` | String | Claim text |
| `parent_ids` | List | Referenced antecedent claim identifiers |

### Optional evaluation fields

| Field | Type | Description |
|---|---|---|
| `cpc.section` | String | CPC section label |
| `cpc.class` | String | CPC class label |
| `cpc.subclass` | String | CPC subclass label |

CPC labels are not required for representation construction or clustering.

---

## Dependency Validation

Before encoding, ClaimSem checks:

- duplicate patent identifiers
- duplicate claim identifiers
- missing or empty claim text
- self-referencing claims
- references to nonexistent claims
- cyclic dependency graphs
- patents without a valid root claim
- record-to-feature alignment
- patent order
- claim order
- embedding dimensions

Validation results are saved to:

```text
dependency_validation.json
```

The pipeline does not silently modify invalid dependency graphs without recording the modification.

---

## Encoder Configuration

The exact pretrained encoder identifier and revision must be specified in the configuration file.

Example:

```json
{
  "encoder": {
    "model_id": "MODEL_ID",
    "revision": "MODEL_REVISION",
    "frozen": true,
    "max_length": 512,
    "batch_size": 64,
    "mixed_precision": true
  }
}
```

The repository does not silently substitute a different encoder. Changing the encoder may change the reported results.

For efficient T4 inference, the encoder pipeline supports:

- PyTorch inference mode
- automatic mixed precision
- dynamic padding
- length-aware batching
- GPU masked mean pooling
- resumable shard processing
- FP32 feature storage

---

## Cached Feature Modes

ClaimSem supports two feature modes.

### Raw encoding mode

```json
{
  "feature_mode": "encode"
}
```

Raw claim text is passed through the configured frozen encoder.

### Cached feature mode

```json
{
  "feature_mode": "cached_shards"
}
```

Previously generated frozen claim representations are loaded from disk.

Cached representations should include metadata for:

- patent identifiers
- claim identifiers
- claim order
- embedding dimension
- encoder identifier
- encoder revision
- preprocessing configuration

A cache should not be used when its provenance cannot be verified.

---

## Configuration Files

### Final configuration

```text
configs/final_claimsem.json
```

The final configuration contains:

```json
{
  "method": "ClaimSem",
  "root_weight": 12.0,
  "depth_decay": 0.1,
  "pca_dim": 128,
  "n_clusters": 30,
  "seeds": [17, 42, 73],
  "clustering_mode": "transductive",
  "pca_fit_split": "dev"
}
```

### Development search

```text
configs/dev_search.json
```

This configuration defines:

- development tuning and holdout partitions
- candidate root weights
- candidate depth-decay coefficients
- candidate PCA dimensions
- candidate cluster counts
- evaluation metrics
- random seeds

### Smoke test

```text
configs/smoke_test.json
```

This configuration runs a small end-to-end test using the records in `demo/`.

---

## Command-Line Usage

### Prepare development features

```bash
python scripts/prepare_features.py \
    --config configs/final_claimsem.json \
    --split dev
```

### Prepare test features

```bash
python scripts/prepare_features.py \
    --config configs/final_claimsem.json \
    --split test
```

### Run development search

```bash
python scripts/run_dev_search.py \
    --config configs/dev_search.json
```

### Run ablations

```bash
python scripts/run_ablation.py \
    --config configs/dev_search.json
```

### Run final test evaluation

```bash
python scripts/run_final_test.py \
    --config configs/final_claimsem.json
```

### Generate paper tables

```bash
python scripts/make_tables.py \
    --results-dir artifacts/results \
    --output-dir artifacts/tables
```

---

## Ablation Study

The development ablation includes the following variants.

| Variant | Description |
|---|---|
| Full ClaimSem | Root weight 12, depth decay 0.1, PCA 128 |
| Uniform claim pooling | Root weight 1, depth decay 0 |
| No root emphasis | Root weight 1, depth decay 0.1 |
| No depth decay | Root weight 12, depth decay 0 |
| Root claims only | Uses only independent claims |
| First claim only | Uses only the first claim |
| Shuffled dependent depths | Permutes positive depths within each patent |
| Document-level encoding | Concatenates claims before encoding |
| No PCA reduction | Clusters the original 768-dimensional representations |

Sensitivity analysis supports the following values:

```text
Root weights:
1, 2, 4, 8, 12, 16

Depth-decay coefficients:
0.00, 0.05, 0.10, 0.20

Cluster counts:
20, 25, 30, 35, 40
```

Ablations and sensitivity analyses are conducted on development data.

---

## Output Artifacts

A complete run produces:

```text
artifacts/
├── cache/
│   ├── dev_claim_embeddings/
│   ├── test_claim_embeddings/
│   ├── dev_claim_metadata.parquet
│   └── test_claim_metadata.parquet
│
├── features/
│   ├── dev_claimsem_raw.npy
│   ├── test_claimsem_raw.npy
│   ├── dev_claimsem_pca128.npy
│   └── test_claimsem_pca128.npy
│
├── models/
│   ├── claimsem_pca128.joblib
│   └── pca_metadata.json
│
├── results/
│   ├── dev_search_ranking.csv
│   ├── ablation_results.csv
│   ├── sensitivity_results.csv
│   ├── test_3seed_predictions.npz
│   ├── test_3seed_assignments.csv
│   ├── test_3seed_metrics.csv
│   └── test_3seed_metrics.json
│
├── tables/
│   ├── cpc_results.tex
│   ├── claimsem_ablation.tex
│   └── claimsem_sensitivity.tex
│
└── manifests/
    ├── encoding_manifest.json
    ├── dev_run_manifest.json
    └── test_run_manifest.json
```

Large artifacts should not be committed to GitHub.

---

## Reproducibility

Each run records:

- Git commit hash
- configuration
- Python version
- PyTorch version
- CUDA version
- GPU model
- encoder identifier
- encoder revision
- random seeds
- patent count
- claim count
- CPC cardinalities
- input checksums
- output checksums
- PCA metadata
- clustering backend

The final test pipeline checks the following expected statistics:

```text
Test patents:       9,881
Test claims:        161,661
CPC sections:       9
CPC classes:        121
CPC subclasses:     466
Clusters:           30
Seeds:              17, 42, 73
```

The expected regression target is:

```text
Mean NMI:
0.375508 ± 0.003952

Mean predicted-cluster purity:
0.425789 ± 0.008153

Mean label-wise inverse purity:
0.355812 ± 0.004877
```

Small numerical differences may occur across CUDA, PyTorch, PCA, or clustering implementations.

---

## Testing

Run all tests:

```bash
pytest -q
```

Run an individual test:

```bash
pytest -q tests/test_dependency.py
```

The test suite covers:

- dependency-depth calculation
- multiple root claims
- multiple-parent claims
- invalid references
- dependency cycles
- uniform pooling
- root emphasis
- depth decay
- vectorized pooling
- NMI
- predicted-cluster purity
- label-wise inverse purity
- record-to-feature alignment

---

## Demo

The repository includes a small synthetic dataset:

```text
demo/demo_records.json
```

Run the smoke test:

```bash
python scripts/prepare_features.py \
    --config configs/smoke_test.json
```

```bash
python scripts/run_final_test.py \
    --config configs/smoke_test.json
```

The demo verifies the execution pipeline. It does not reproduce the paper results.

---

## Data Availability

Patent text and CPC labels may be subject to the terms of the original data provider. This repository does not redistribute restricted or licensed patent datasets.

The repository provides:

- an expected data schema
- preprocessing utilities
- a synthetic demonstration dataset
- dependency validation
- cached feature support
- clustering evaluation
- LaTeX table generation

Users are responsible for obtaining and using patent data in accordance with the applicable licenses and terms.

---

## Scope

ClaimSem is a patent representation and clustering method.

It does not:

- generate natural-language topic labels
- learn topic-word distributions
- induce a topic hierarchy
- use Depth-OT topic distributions
- require optimal transport
- fine-tune PatentSBERTa-V2
- use CPC labels as encoder training targets

The generated clusters should not be described as supervised CPC predictions.

---

## Limitations

1. Root weight and depth decay are selected using development labels and may not transfer unchanged to every patent collection.
2. The method assumes that claim-dependency references are available or can be extracted reliably.
3. The weighting rule is intentionally simple and may not capture every legal or semantic relation among claims.
4. The current evaluation uses one patent collection and CPC-based external metrics.
5. Spherical $K$-means requires the number of clusters to be specified.
6. The reported results use transductive clustering and should not be interpreted as inductive CPC classification.
7. The frozen patent encoder may inherit limitations from its pretraining data.

---

## Citation

If you use ClaimSem, please cite the software repository:

```bibtex
@software{yoo2026claimsem,
  author = {Yongmin Yoo},
  title  = {ClaimSem: Structure-Aware Claim Aggregation for Patent Clustering},
  year   = {2026},
  url    = {https://github.com/Yongmin-Yoo/claimsem}
}
```

A machine-readable citation file will be provided in:

```text
CITATION.cff
```

---

## License

See the `LICENSE` file for the software license.

Third-party models and datasets remain subject to their original licenses and terms.

---

## Author

**Yongmin Yoo**

Repository:

https://github.com/Yongmin-Yoo/claimsem

Issues:

https://github.com/Yongmin-Yoo/claimsem/issues
