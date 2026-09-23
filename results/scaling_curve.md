# VQC scaling curve: accuracy vs parameter count

n=500 train / 125 test, baseline 0.4480. Exact parameter-shift gradients + L-BFGS, maxiter=30.

| layers | params | depth | loss | train acc | test acc | classical (nearest k) | gap |
|---|---|---|---|---|---|---|---|
| 2 | 9 | 11 | 0.9538 | 0.5451 | 0.5040 | 0.6440 (k=9) | -0.1400 |
| 4 | 15 | 21 | 0.8419 | 0.6934 | 0.6640 | 0.6960 (k=15) | -0.0320 |
| 8 | 27 | 41 | 0.8097 | 0.7214 | 0.5920 | 0.7640 (k=30) | -0.1720 |
| 12 | 39 | 61 | 0.7846 | 0.7515 | 0.6320 | 0.8200 (k=45) | -0.1880 |
| 16 | 51 | 81 | 0.7957 | 0.7275 | 0.5680 | 0.8200 (k=45) | -0.2520 |

Classical reference is logistic regression on the k most informative
order-3 Fourier basis functions. Note it selects those k using the
labels, an advantage the VQC does not have -- see roadmap caveat.
