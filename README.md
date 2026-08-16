# ClaimSem

**Structure-aware claim aggregation for patent clustering**

ClaimSem is a lightweight method for constructing patent-level representations from individual patent claims. It encodes each claim separately with a frozen patent language model, emphasizes independent claims, attenuates dependent claims according to their dependency depth, reduces the resulting patent representations with PCA, and clusters them using spherical \(K\)-means.

The repository provides a reproducible Google Colab pipeline for:

- validating patent claim-dependency graphs
- calculating claim dependency depths
- encoding claims individually with a frozen PatentSBERTa-V2 encoder
- constructing root- and depth-aware patent representations
- fitting PCA only on development representations
- running spherical \(K\)-means with fixed random seeds
- evaluating clusters against CPC labels
- conducting controlled ablations and robustness analyses
- generating LaTeX tables for the paper

ClaimSem does not require a trained Depth-OT model, topic distributions, optimal transport plans, or neural topic-model checkpoints.

---

## Method Overview

A patent \(P_n\) contains claims


\[
\mathcal{C}_n
=
\{c_{n,1},\ldots,c_{n,M_n}\}
\]

and a directed claim-dependency graph


\[
G_n=(\mathcal{C}_n,E_n).
\]

An edge \((c_i,c_j)\in E_n\) indicates that claim \(c_j\) depends on claim \(c_i\). Claims without valid antecedents are treated as root claims.

### Claim depth

The dependency depth of claim \(c\) is defined as


\[
d(c)=
\begin{cases}
0, & \operatorname{Pa}(c)=\varnothing,\\
1+\max_{c'\in\operatorname{Pa}(c)}d(c'), & \text{otherwise}.
\end{cases}
\]

Root claims have depth zero. A dependent claim receives a depth determined by the longest valid path from a root claim.

### Individual claim encoding

Each claim is encoded independently with a frozen PatentSBERTa-V2 encoder. Masked mean pooling produces a claim-level representation


\[
\mathbf{e}_c
=
\frac{
\sum_{\ell=1}^{L_c}
m_{c\ell}\mathbf{h}_{c\ell}
}{
\sum_{\ell=1}^{L_c}m_{c\ell}
}.
\]

Encoding claims separately avoids the loss of later claims caused by document-level input truncation.

### Root- and depth-aware pooling

ClaimSem assigns each claim the following weight:


\[
w_c(\alpha,\lambda)
=
\alpha^{\mathbb{I}[d(c)=0]}
\exp(-\lambda d(c)),
\]

where:

- \(\alpha\) controls independent-claim emphasis
- \(\lambda\) controls dependency-depth decay

The patent representation is


\[
\mathbf{v}_n
=
\frac{
\sum_{c\in\mathcal{C}_n}
w_c(\alpha,\lambda)\mathbf{e}_c
}{
\sum_{c\in\mathcal{C}_n}
w_c(\alpha,\lambda)
}.
\]

The final configuration uses:

| Component | Value |
|---|---:|
| Root weight \(\alpha\) | 12.0 |
| Depth decay \(\lambda\) | 0.1 |
| Claim embedding dimension | 768 |
| PCA output dimension | 128 |
| Number of clusters | 30 |
| Clustering seeds | 17, 42, 73 |
| Neural encoder training | None |
| CPC labels used as encoder targets | No |
| Test CPC labels used for tuning | No |

After pooling, the patent representations are projected to 128 dimensions using a PCA transform fitted on the development set. The reduced vectors are normalized to unit length and clustered with spherical \(K\)-means.

---

## Final Test Results

The final configuration was selected on the development data and frozen before test evaluation. The PCA transform was fitted on development representations and applied to the test representations without refitting.

The reported evaluation uses three spherical \(K\)-means seeds:

```text
17, 42, 73
```

### Dataset statistics

| Split | Patents | Claims | CPC sections | CPC classes | CPC subclasses |
|---|---:|---:|---:|---:|---:|
| Development | 9,855 | 160,048 | 9 | 123 | 484 |
| Test | 9,881 | 161,661 | 9 | 121 | 466 |

### CPC alignment

| CPC level | Purity \( \mathrm{Pur}_{\mathrm{p}} \) | Inverse purity \( \mathrm{Pur}_{\mathrm{a}} \) | NMI |
|---|---:|---:|---:|
| Section | \(0.617009 \pm 0.010069\) | \(0.188442 \pm 0.005584\) | \(0.273401 \pm 0.004359\) |
| Class | \(0.411328 \pm 0.006142\) | \(0.366124 \pm 0.004701\) | \(0.398139 \pm 0.003565\) |
| Subclass | \(0.249030 \pm 0.008411\) | \(0.512870 \pm 0.007868\) | \(0.454984 \pm 0.004101\) |
| Mean | \(0.425789 \pm 0.008153\) | \(0.355812 \pm 0.004877\) | \(0.375508 \pm 0.003952\) |

All 30 clusters remain active. The maximum topic share is


\[
0.055662 \pm 0.001145.
\]

These numbers are regression targets for the legacy-compatible reproduction pipeline. Small numerical differences can occur across CUDA, PyTorch, PCA, or clustering implementations.

---

## Evaluation Setting

ClaimSem uses a transductive clustering protocol.

For each evaluation split:

1. claim embeddings are pooled into patent representations
2. the fixed development-fitted PCA transform is applied
3. spherical \(K\)-means is fitted to the unlabeled representations of that split
4. CPC labels are accessed only after cluster assignments are produced
5. CPC labels are used only to compute evaluation metrics

This protocol differs from inductive classification or centroid transfer. In inductive mode, centroids learned on development data would be applied directly to test data. The paper results reported above use transductive clustering.

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
| `config.py` | Configuration loading, validation, and path resolution |
| `reproducibility.py` | Random seeds, environment logging, and deterministic settings |
| `data.py` | Patent records, CPC labels, claim metadata, and feature-shard loading |
| `dependency.py` | Dependency-graph validation and claim-depth calculation |
| `encoder.py` | Frozen PatentSBERTa-V2 inference and masked mean pooling |
| `pooling.py` | Root- and depth-aware patent representation construction |
| `reduction.py` | Development-only PCA fitting and fixed transformation |
| `clustering.py` | Spherical \(K\)-means and cluster-balance diagnostics |
| `metrics.py` | NMI, predicted-cluster purity, and label-wise inverse purity |
| `artifacts.py` | Saving models, features, predictions, summaries, and manifests |
| `dev_search.py` | Development search, held-out validation, and ablations |
| `test_evaluation.py` | Frozen final test evaluation |

---

## Google Colab Setup

### 1. Select a GPU runtime

In Google Colab, select:

```text
Runtime
→ Change runtime type
→ Hardware accelerator
→ T4 GPU
```

### 2. Mount Google Drive

```python
from google.colab import drive

drive.mount("/content/drive")
```

### 3. Clone the repository

Replace `<GITHUB_OWNER>` with the GitHub account or organization name.

```bash
!git clone https://github.com/<GITHUB_OWNER>/claimsem.git
%cd claimsem
```

### 4. Install dependencies

```bash
!pip install -q -r requirements-colab.txt
!pip install -q -e .
```

### 5. Verify the installation

```python
import torch
import claimsem

print("ClaimSem import successful")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

---

## Colab Notebooks

### 1. Prepare and encode

```text
notebooks/01_prepare_and_encode.ipynb
```

This notebook:

- mounts Google Drive
- loads patent records
- validates dependency graphs
- calculates claim depths
- encodes each claim independently
- saves resumable feature shards
- verifies patent and claim alignment

Open in Colab:

```text
https://colab.research.google.com/github/<GITHUB_OWNER>/claimsem/blob/main/notebooks/01_prepare_and_encode.ipynb
```

### 2. Development selection and ablation

```text
notebooks/02_dev_selection_and_ablation.ipynb
```

This notebook:

- creates or loads the fixed development partition
- evaluates candidate root and depth weights
- fits PCA on development representations
- runs controlled ClaimSem ablations
- conducts hyperparameter sensitivity analysis
- saves the selected configuration

Open in Colab:

```text
https://colab.research.google.com/github/<GITHUB_OWNER>/claimsem/blob/main/notebooks/02_dev_selection_and_ablation.ipynb
```

### 3. Final test evaluation

```text
notebooks/03_final_test_evaluation.ipynb
```

This notebook:

- loads the frozen final configuration
- applies root- and depth-aware pooling
- applies the saved development-fitted PCA model
- performs spherical \(K\)-means with seeds 17, 42, and 73
- evaluates CPC alignment
- saves predictions, assignments, metrics, and manifests

Open in Colab:

```text
https://colab.research.google.com/github/<GITHUB_OWNER>/claimsem/blob/main/notebooks/03_final_test_evaluation.ipynb
```

### 4. Generate paper tables

```text
notebooks/04_generate_paper_tables.ipynb
```

This notebook reads saved result files and produces:

- the CPC baseline table
- the ClaimSem ablation table
- the hyperparameter sensitivity table
- seed-level result tables
- LaTeX-ready mean and standard-deviation values

Open in Colab:

```text
https://colab.research.google.com/github/<GITHUB_OWNER>/claimsem/blob/main/notebooks/04_generate_paper_tables.ipynb
```

---

## Installation Outside Colab

Python 3.10 or later is recommended.

```bash
git clone https://github.com/<GITHUB_OWNER>/claimsem.git
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

The public demo uses JSON records. Private or licensed datasets may also be loaded from pickle files through the data adapter.

A patent record should contain the following fields:

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
| `patent_id` | string | Unique patent identifier |
| `claims` | list | Claims belonging to the patent |
| `claim_id` | string or integer | Claim identifier within the patent |
| `text` | string | Claim text |
| `parent_ids` | list | Referenced antecedent claim identifiers |

### Optional evaluation fields

| Field | Type | Description |
|---|---|---|
| `cpc.section` | string | CPC section label |
| `cpc.class` | string | CPC class label |
| `cpc.subclass` | string | CPC subclass label |

CPC labels are not required for representation construction or clustering. They are used only for development configuration selection and final evaluation.

---

## Dependency Validation

Before encoding, ClaimSem checks:

- duplicate patent identifiers
- duplicate claim identifiers
- missing claim text
- empty claim text
- self-referencing claims
- references to nonexistent claims
- cyclic dependency graphs
- patents without a valid root claim
- consistency between claim metadata and cached embeddings
- consistency between patent order and representation order

Validation results are saved as:

```text
dependency_validation.json
```

Invalid or unresolved references are reported explicitly. The public pipeline does not silently modify dependency graphs without recording the change.

---

## Encoder Configuration

ClaimSem requires the exact pretrained encoder identifier and model revision to be specified in the configuration file.

Example:

```json
{
  "encoder": {
    "model_id": "<EXACT_PATENTSBERTA_V2_MODEL_ID>",
    "revision": "<MODEL_REVISION>",
    "frozen": true,
    "max_length": 512,
    "batch_size": 64,
    "mixed_precision": true
  }
}
```

The repository does not silently substitute a different encoder. Using another model may change the reported results.

For efficient T4 inference, the encoder pipeline uses:

- `torch.inference_mode()`
- automatic mixed precision
- dynamic padding
- length-aware batching
- masked mean pooling on GPU
- shard-level caching
- resumable execution
- FP32 output storage

---

## Cached Feature Modes

ClaimSem supports two feature modes.

### Encode mode

```json
{
  "feature_mode": "encode"
}
```

Raw claim text is passed through the configured frozen encoder. This is the complete reproduction path.

### Cached-shard mode

```json
{
  "feature_mode": "cached_shards"
}
```

Previously generated frozen claim representations are loaded from disk. This mode is faster and is intended for development searches, ablations, and repeated clustering experiments.

Cached shards must include enough metadata to verify:

- patent identifier
- claim identifier
- claim order
- embedding dimension
- encoder model identifier
- encoder revision
- preprocessing configuration

A cache should not be used if its provenance cannot be verified.

---

## Configuration Files

### Final ClaimSem configuration

```text
configs/final_claimsem.json
```

This file stores the frozen paper configuration:

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

This file defines:

- the development tuning and holdout partition
- candidate root weights
- candidate depth-decay coefficients
- candidate PCA dimensions
- candidate cluster counts
- selection metrics
- fixed random seeds

### Smoke test

```text
configs/smoke_test.json
```

This file runs a small end-to-end check using the records in `demo/`.

---

## Command-Line Usage

### Prepare features

```bash
python scripts/prepare_features.py \
    --config configs/final_claimsem.json \
    --split dev
```

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

### Run the frozen final test evaluation

```bash
python scripts/run_final_test.py \
    --config configs/final_claimsem.json
```

### Generate LaTeX tables

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
| Shuffled dependent depths | Preserves roots but permutes positive depths within each patent |
| Document-level encoding | Concatenates claims before encoding |
| No PCA reduction | Clusters the original 768-dimensional pooled representations |

The repository also supports sensitivity analysis over:

```text
root weight:
1, 2, 4, 8, 12, 16

depth decay:
0.00, 0.05, 0.10, 0.20

number of clusters:
20, 25, 30, 35, 40
```

Ablations and sensitivity analyses are performed on development data. They are not used to redefine the final model after test evaluation.

---

## Output Artifacts

A complete run produces the following structure:

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

Large artifacts should not be committed to GitHub. Store them in Google Drive, an institutional repository, or a release archive.

---

## Reproducibility

Each run records:

- Git commit hash
- configuration file
- Python version
- PyTorch version
- CUDA version
- GPU model
- encoder model identifier
- encoder revision
- random seeds
- patent count
- claim count
- CPC cardinalities
- input-file checksums
- output-file checksums
- PCA metadata
- clustering backend
- clustering convergence information

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

If these values differ, the script emits a warning and records the discrepancy in the run manifest.

---

## Regression Target

For the frozen paper configuration, the expected three-seed test result is:

```text
Mean NMI:
0.375508 ± 0.003952

Mean predicted-cluster purity:
0.425789 ± 0.008153

Mean label-wise inverse purity:
0.355812 ± 0.004877
```

Exact bitwise reproduction is not guaranteed across different CUDA, BLAS, PCA, or clustering implementations. The test suite therefore uses configurable numerical tolerances.

---

## Testing

Run all tests:

```bash
pytest -q
```

Run an individual test module:

```bash
pytest -q tests/test_dependency.py
```

The tests cover:

### Dependency tests

- single-root chains
- multiple independent claims
- multiple-parent claims
- invalid references
- self-references
- dependency cycles
- longest-path depth calculation

### Pooling tests

- uniform pooling
- root emphasis
- depth decay
- multiple root claims
- single-claim patents
- vectorized pooling consistency
- numerical stability

### Metric tests

- NMI
- predicted-cluster purity
- label-wise inverse purity
- inactive clusters
- degenerate assignments

### Alignment tests

- patent order
- claim order
- record-to-feature alignment
- missing embeddings
- duplicate identifiers
- expected CPC cardinalities

---

## Demo

A small synthetic dataset is provided in:

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

The demo verifies the pipeline but does not reproduce the paper results.

---

## Data Availability

Patent text and CPC labels may be subject to the terms of the original data provider. This repository does not redistribute restricted or licensed patent datasets.

The repository provides:

- the expected record schema
- preprocessing utilities
- a synthetic demonstration dataset
- claim-dependency validation
- feature-cache support
- evaluation scripts
- table-generation scripts

Users are responsible for obtaining and using patent data in accordance with the applicable license and terms.

---

## Model Scope

ClaimSem is a patent representation and clustering method. It does not:

- generate natural-language topic labels
- learn topic-word distributions
- induce a topic hierarchy
- use Depth-OT topic distributions
- require an optimal transport solver
- fine-tune PatentSBERTa-V2
- use CPC labels as encoder training targets

The produced clusters can be evaluated against CPC categories, but they should not be described as supervised CPC predictions.

---

## Limitations

ClaimSem has several limitations.

1. The root weight and depth-decay coefficient are selected using development labels and may not transfer unchanged to every patent corpus.
2. The method assumes that valid claim-dependency references are available or can be extracted reliably.
3. The exponential weighting rule is intentionally simple and may not capture every legal or semantic relation among claims.
4. The reported evaluation uses one patent collection and CPC-based external metrics.
5. Spherical \(K\)-means requires the number of clusters to be specified in advance.
6. The final results use transductive clustering, so they should not be interpreted as inductive CPC classification performance.
7. PatentSBERTa-V2 may inherit biases or coverage limitations from its pretraining data.

---

## Citation

If you use ClaimSem, please cite:

```bibtex
@inproceedings{<CITATION_KEY>,
  title     = {<PAPER_TITLE>},
  author    = {<AUTHOR_NAMES>},
  booktitle = {<VENUE>},
  year      = {<YEAR>},
  url       = {<PAPER_URL>}
}
```

A machine-readable citation file is available in:

```text
CITATION.cff
```

---

## License

See the `LICENSE` file for the software license.

The license of this repository does not override the licenses or terms associated with:

- PatentSBERTa-V2
- Hugging Face models
- patent datasets
- CPC data
- third-party preprocessing resources

---

## Acknowledgments

This project uses pretrained patent-language representations and open-source scientific Python libraries. Please cite the original model, dataset, and software papers where applicable.

---

## Contact

For questions or reproducibility issues, open a GitHub issue:

```text
https://github.com/Yongmin-Yoo/claimsem/issues
```

Please include:

- the configuration file
- the run manifest
- the relevant log
- the Python and CUDA versions
- the GPU model
- the smallest example that reproduces the problem
