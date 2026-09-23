# VQC config search (n=1000 train / 250 test, 100 SPSA iterations)

Majority-class baseline: 0.4480

| feature map | reps | logit scale | params | loss | train acc | test acc | vs baseline |
|---|---|---|---|---|---|---|---|
| Z | 2 | 1.0 | 9 | 1.0147 | 0.5280 | 0.5440 | +0.0960 |
| ZZ | 2 | 1.0 | 9 | 1.0354 | 0.4180 | 0.4160 | -0.0320 |
| ZZ | 6 | 1.0 | 21 | 1.0536 | 0.4230 | 0.3960 | -0.0520 |
| Z | 6 | 1.0 | 21 | 1.0957 | 0.3870 | 0.3680 | -0.0800 |
