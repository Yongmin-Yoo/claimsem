# Final learned-aggregation comparison

| Method | Training runs | Section NMI | Class NMI | Subclass NMI | Mean NMI | 95% CI for mean NMI |
|---|---:|---:|---:|---:|---:|---:|
| Uniform pooling | 1 | 0.2468 | 0.3628 | 0.4257 | 0.3451 | — |
| Balanced ROOTS | 1 | 0.2753 | 0.3977 | 0.4571 | 0.3767 | — |
| SSL-MLP + attention | 10 | 0.2487 | 0.3654 | 0.4274 | 0.3472 | [0.3468, 0.3475] |
| Structural VICReg-GAT | 10 | 0.2516 | 0.3687 | 0.4313 | 0.3505 | [0.3500, 0.3511] |

For learned aggregators, each training-run value is first averaged across five clustering seeds. The reported interval then represents between-training-run uncertainty.
