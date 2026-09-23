# Capacity-matched comparison

Logistic regression on the k most informative order-3 Fourier basis functions, n=1000/250, baseline 0.4480.

| k basis functions | train acc | test acc | vs baseline |
|---|---|---|---|
| 3 | 0.6613 | 0.6800 | +0.2320 |
| 6 | 0.6914 | 0.6720 | +0.2240 |
| 9 | 0.7114 | 0.7040 | +0.2560 |
| 12 | 0.7415 | 0.7040 | +0.2560 |
| 15 | 0.7655 | 0.7280 | +0.2800 |
| 21 | 0.7715 | 0.7040 | +0.2560 |
| 30 | 0.8116 | 0.7200 | +0.2720 |
| 45 | 0.8457 | 0.7360 | +0.2880 |
| 60 | 0.8778 | 0.7920 | +0.3440 |
| 100 | 0.9178 | 0.8400 | +0.3920 |

## VQC reference (exact-gradient training)

| config | params | test acc |
|---|---|---|
| control | 9 | 0.5280 |
| reupload L=4 | 15 | 0.5920 |
