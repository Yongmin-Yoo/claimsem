# Final Learned-Aggregation Comparison

| Method | Training seeds | Clustering seeds | Mean TEST NMI | SD |
|---|---:|---:|---:|---:|
| Uniform pooling | -- | 5 | 0.345119 | 0.001751 (clustering) |
| SSL-MLP + attention | 10 | 5 per model | 0.347182 | 0.000469 (training) |
| Structural VICReg-GAT | 10 | 5 per model | 0.350547 | 0.000751 (training) |
| **Balanced ROOTS** | **--** | **5** | **0.376691** | **0.002379 (clustering)** |

For learned methods, each training-seed score is first averaged
over five clustering seeds. The reported learned-method SD is the
sample SD across the resulting 10 training-run means. For fixed
pooling, SD is the population SD across five clustering seeds.
These SDs measure different sources of variation and should not be
compared as equivalent stability estimates.

Balanced ROOTS minus Structural VICReg-GAT is +0.026144 NMI.
The training-seed 95% CI is [0.025607, 0.026682], and the
paired-document bootstrap 95% CI is [0.021952, 0.028014].

The learned models improve over uniform pooling on average.
The primary conclusion is limited to the evaluated architectures,
protocol, and optimization budgets. Eight of 10 GAT runs selected
the maximum epoch, so full neural convergence is not claimed.
