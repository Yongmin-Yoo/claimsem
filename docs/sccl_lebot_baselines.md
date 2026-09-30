# SCCL and LeBoT-style clustering baselines

This experiment evaluates document-level adaptations of SCCL and LeBoT on
the fixed USPTO-70k TEST split containing 9,881 patents.

## Results

| Method | Mean TEST NMI |
|---|---:|
| SCCL native clustering head | 0.282212 |
| SCCL representations + spherical K-means | 0.338877 |
| LeBoT-style (Qwen3-0.6B) | 0.089246 |

The main paper reports SCCL representations with spherical K-means because
this produces exactly 30 clusters under the matched clustering protocol.
The native SCCL result is retained for completeness.

The LeBoT-style experiment is an L4-efficient patent adaptation using
Qwen3-0.6B, 512-token truncation, MiniLM retrieval, and a 1,024-dimensional
bag-of-texts representation. It is not an exact reproduction of the
Gemma-2-9B configuration in the original paper.

## Data policy

Raw USPTO records, patent claim text, token caches, and training checkpoints
are not redistributed. Large prediction and representation artifacts are
stored in the associated Hugging Face dataset:

https://huggingface.co/datasets/yongminyoo91/roots-additional-clustering-baselines/tree/main/sccl_lebot
