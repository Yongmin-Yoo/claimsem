# Learned-aggregation recovery notes

## Source-code status

The original learned-aggregation training source was not found in the
searched artifacts or the forty most recently modified Colab notebooks.
The original checkpoint files, training histories, PCA objects,
predictions, graph caches, and provenance records remain available.

The code in this directory therefore distinguishes between:

1. deterministic inference behavior recovered from archived outputs; and
2. newly specified train-split optimization code.

The latter must not be described as an exact reproduction of an
unavailable original training implementation.

## Structural VICReg-GAT

The parameterized architecture loads all three archived checkpoints
with `strict=True`.

- Trainable parameter count: 416,385
- Input claim embedding: 768 dimensions
- Structural input order: normalized depth followed by root indicator
- Original depth scale: maximum development depth, 27
- Input transformation: Linear, LayerNorm, GELU
- GAT layer 1: four heads of 48 dimensions
- GAT layer 2: one head of 192 dimensions
- GAT update: `ELU(GAT(x)) + x`
- Self-loops: enabled
- Directed edge feature: two-dimensional one-hot feature
- Attention network: 194 to 96 to 1 with Tanh
- Pooling: attention-weighted sum of the original 768-dimensional
  frozen claim embeddings
- Projection head: 768 to 192 to 128

Behavioral validation:

- Development attention-statistic mean relative error:
  approximately 3.58e-6
- Test attention-statistic mean relative error:
  approximately 4.27e-6

The two edge-feature column orientations were numerically
indistinguishable in aggregate diagnostics. The graph-builder convention
is retained: parent-to-child is column zero and child-to-parent is
column one.

The archived VICReg weights are:

- invariance: 25
- variance: 25
- covariance: 1

All three Structural VICReg-GAT runs selected epoch 15, which was the
maximum available epoch, while validation loss was still decreasing.

## SSL-MLP with attention

The parameterized architecture loads archived checkpoints with
`strict=True`.

- Trainable parameter count: 339,777
- Input transformation: Linear, LayerNorm, GELU
- Attention network: 192 to 96 to 1 with Tanh
- Pooling: attention-weighted sum of original claim embeddings
- Projection head: 768 to 192 to 128
- Development attention-statistic mean relative error:
  approximately 3.89e-6

The checkpoint contains `norm1` and `norm2`, but both remained exactly
at their initialization values and are not used by the behaviorally
recovered inference path. They are retained solely for strict checkpoint
compatibility.

For MLP seeds 17 and 73, the absolute minimum validation-loss epoch does
not equal the archived selected checkpoint. This is compatible with a
positive early-stopping `min_delta`, but the exact value is not treated
as recovered because the original source was unavailable.

## PCA environment warning

The archived PCA objects were created with scikit-learn 1.9.0. Loading
them under scikit-learn 1.6.1 produced an `InconsistentVersionWarning`.
Exact archived-prediction reproduction must use scikit-learn 1.9.0 or
refit all compared methods in one documented environment.

## Still unresolved from the original training implementation

- Exact stochastic view-generation procedure
- Placement of feature dropout during training
- Whether edge dropout means GAT attention dropout, explicit edge
  removal, or both
- Exact early-stopping minimum improvement threshold
- Exact minibatch sampler and augmentation RNG stream

These settings must not be silently inferred. The new train-split
protocol will define and record them explicitly before final runs.

## Newly specified full-train protocol

The unresolved original stochastic implementation is not silently
inferred. For the new train-split audit, stochastic views are explicitly
defined in `training_objectives.py`:

- independent claim-feature dropout at probability 0.1;
- projection-output dropout at probability 0.1;
- GAT attention-coefficient dropout at probability 0.1;
- no explicit edge deletion;
- VICReg weights 25/25/1 for Structural VICReg-GAT;
- symmetric in-batch NT-Xent/InfoNCE at temperature 0.07 for SSL-MLP.

These are preregistered experimental choices, not recovered historical
facts. Deterministic evaluation disables stochastic augmentation and
uses the behaviorally recovered inference paths.

The train-derived depth divisor is 55 and is frozen for train, DEV, and
TEST in the new experiment. The recovered historical DEV-only model
continues to use 27.

<!-- FINAL_TRAIN_SPLIT_AUDIT_START -->
## Final train-split audit outcome

The train-split audit completed ten SSL-MLP training runs and ten
Structural VICReg-GAT training runs. Checkpoints were selected only by
development self-supervised loss. TEST and CPC labels were not used for
training or checkpoint selection.

The final evaluation used five clustering seeds per representation and
a paired-document bootstrap with 2,000 replicates. Balanced ROOTS
outperformed Structural VICReg-GAT by 0.026144 NMI. The
between-training-run 95% confidence interval was [0.025607, 0.026682],
and the paired-document bootstrap interval was [0.021952, 0.028014].

This result does not change the source-code recovery qualification. The
deterministic model inference paths were recovered from archived
artifacts, but the original stochastic training implementation was not
recovered. The train-split stochastic protocol is newly specified and
must not be represented as an exact reconstruction of the unavailable
original implementation.
<!-- FINAL_TRAIN_SPLIT_AUDIT_END -->
