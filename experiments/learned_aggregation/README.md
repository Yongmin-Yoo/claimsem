# Learned Aggregation Audit

This directory records the learned-aggregation experiments for ROOTS
and the protocol introduced to address limitations in the original
comparison.

## Research question

The primary comparison is Balanced ROOTS versus Structural
VICReg-GAT under the same frozen PatentSBERTa-V2 claim embeddings,
preprocessing, PCA rule, clustering configuration, and test set.

The experiment is not designed to make ROOTS win. Its purpose is to
give the learned aggregators sufficient training data and optimization
budget, separate training randomness from clustering randomness, and
reassess the conclusion.

## Original protocol recovered from artifacts

The original learned-aggregation experiment used:

- Training data: USPTO-70k development split only
- Development documents: 9,855
- Self-supervised validation: 10% of the development split
- Seeds: 17, 42, and 73
- Maximum epochs: 15
- Minimum epochs: 5
- Early-stopping patience: 4
- Optimizer: AdamW
- Learning rate: 1e-3
- Weight decay: 1e-4
- Batch size: 256
- Hidden dimension: 192
- Projection dimension: 128
- Feature dropout: 0.1
- Edge dropout: 0.1
- Model dropout: 0.1
- InfoNCE temperature: 0.07
- PCA dimension: 128
- Number of clusters: 30
- Evaluation: transductive spherical K-means on test representations

The same numerical seed was used for neural training and K-means, so
training and clustering randomness were not independently reported.

For all three Structural VICReg-GAT runs, validation loss was still
improving at epoch 15. The original optimization budget may therefore
have stopped training prematurely.

## Recovered graph-cache construction

The graph-cache generation procedure was verified exactly against the
original development cache.

### Nodes

- One node represents one patent claim.
- Node order follows the claim order in each normalized patent record.
- Token embeddings are converted to float32 before mean pooling.
- The resulting 768-dimensional claim embedding is stored as float16.
- Root status is obtained from depth equal to zero.
- Dependency depth is obtained from the normalized record.

A smoke reproduction on the first development shard obtained exact
equality for all 1,727 claim embeddings. The maximum absolute
difference was 0.0.

### Edges

A normalized record stores each dependency as:

    (parent_claim_id, child_claim_id)

The graph cache stores both directions:

    parent to child: edge_type = 0
    child to parent: edge_type = 1

The smoke reproduction matched claim order and all graph edges for
100 out of 100 patents.

## Recovered Structural VICReg-GAT architecture

Checkpoint parameter names and shapes establish the following
architecture:

- Frozen claim embedding: 768 dimensions
- Root indicator: 1 dimension
- Normalized depth: 1 dimension
- Total node input: 770 dimensions
- Input projection: 770 to 192
- First GAT layer: 4 heads, 48 dimensions per head
- Second GAT layer: 1 head, 192 dimensions
- Directed edge feature: 2 dimensions
- Attention scorer: 194 to 96 to 1
- Attention-weighted sum over the original claim embeddings
- Projection head: 768 to 192 to 128
- Trainable parameters: 416,385

Settings not recoverable from artifacts must not be silently inferred.
A reconstructed implementation must first reproduce the original
development-only protocol before it is used for the new experiment.

## New train-split protocol

### Split roles

- Train: 49,599 patents for neural gradient updates
- Development: 9,855 patents for self-supervised validation, early
  stopping, limited hyperparameter selection, and PCA fitting
- Test: 9,881 patents for final clustering and CPC evaluation only

CPC labels must not be used for learned-aggregator training,
checkpoint selection, or hyperparameter selection.

PCA dimension 128 and K equal to 30 are fixed across methods. These
downstream settings were historically selected using development CPC
alignment and this fact must be disclosed.

### Primary methods

1. Uniform pooling
2. Balanced ROOTS
3. SSL-MLP with attention
4. Structural VICReg-GAT

CPC-development-tuned ROOTS is reported separately as a reference.

### Optimization audit

Training must save:

- Train loss by epoch
- Self-supervised validation loss by epoch
- Selected checkpoint and epoch
- VICReg invariance loss
- VICReg variance loss
- VICReg covariance loss
- Representation standard deviation or another collapse diagnostic
- Training time and peak memory

If validation loss is still improving at the maximum epoch, the
training budget must be increased before final evaluation.

Hyperparameter-selection runs must be separated from final
performance-measurement runs.

### Randomness

The initial final protocol uses:

- Ten independent neural training seeds
- Five K-means seeds per checkpoint
- A fixed and recorded PCA random state

Fifty training-seed-by-clustering-seed scores must not be treated as
fifty independent neural training runs.

For neural training seed s, the reported score is first averaged over
clustering seeds:

    m_s = mean over clustering seeds and CPC granularities

Final learned-model means and standard deviations are computed across
the ten values of m_s.

Fixed pooling has no neural training randomness. Its clustering
variation must be distinguished from the learned models' between-run
training variation.

## Primary statistical comparison

The primary comparison is:

    Balanced ROOTS minus Structural VICReg-GAT test mean NMI

Report:

- Mean difference
- 95 percent confidence interval over independent training runs
- Paired document bootstrap confidence interval where appropriate
- A clear distinction between training-seed uncertainty and
  test-document sampling uncertainty

A document bootstrap with fixed cluster assignments is conditional on
already trained and clustered outputs and must be described as such.

SSL-MLP is a secondary comparison.

## Interpretation policy

- If Balanced ROOTS is clearly better, restrict the claim to the
  sufficiently trained self-supervised baselines evaluated here.
- If the confidence interval includes zero, report that superiority
  was not established. Do not claim equivalence or non-inferiority.
- If a learned model is better, report it and frame ROOTS as a simple,
  competitive fixed prior rather than claiming learned aggregation is
  unnecessary.

## Efficiency reporting

Distinguish:

- Shared frozen claim encoding cost
- Graph-cache construction cost
- Neural training cost
- Aggregator inference cost
- Pooling cost
- PCA and clustering cost
- Hyperparameter-search cost

Zero neural training time does not mean zero total runtime.

## Data storage

Large records, embeddings, graph caches, predictions, and checkpoints
must not be committed to Git.

Use environment variables for private storage locations:

- CLAIMSEM_DATA_ROOT
- CLAIMSEM_ARTIFACT_ROOT
- HF_TOKEN

Never write Hugging Face tokens or private absolute paths into tracked
files.

## Artifact provenance

The original artifacts showed:

- Original learned models trained on development data only
- Structural VICReg-GAT validation loss improving through epoch 15
- Per-method PCA fitted on development representations
- No PCA refitting on test data
- Test-time transductive spherical K-means
- Separate fixed-pooling and learned-aggregation audit pipelines

The new experiment must recompute all primary methods in one
consistent pipeline rather than mixing numbers from separate audits.
