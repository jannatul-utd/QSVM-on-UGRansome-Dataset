# Data re-uploading test (n=1000/250, 300 SPSA iterations)

Majority baseline: 0.4480 | previous best VQC: 0.5440 | classical SVM: 0.9530

| config | params | depth | loss | train acc | test acc | vs baseline |
|---|---|---|---|---|---|---|
| control (1 upload) | 9 | 9 | 0.9914 | 0.5170 | 0.5080 | +0.0600 |
| reupload L=2 | 9 | 11 | 1.0381 | 0.4190 | 0.4400 | -0.0080 |
| reupload L=4 | 15 | 21 | 0.9498 | 0.6010 | 0.5600 | +0.1120 |
| reupload L=6 | 21 | 31 | 1.0471 | 0.4740 | 0.5000 | +0.0520 |
