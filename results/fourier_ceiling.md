# Fourier frequency ceiling diagnostic

Logistic regression on an order-K multivariate Fourier basis. The VQC's
function class is a subset of this, so each row is an UPPER BOUND on what
a VQC with L=K data uploads could reach.

Majority baseline: 0.4452 | tuned RBF SVM: 0.9440 | best VQC so far: 0.5440

| order K | basis size | train acc | test acc | vs baseline |
|---|---|---|---|---|
| 1 | 27 | 0.7572 | 0.7378 | +0.2926 |
| 2 | 125 | 0.9110 | 0.9001 | +0.4548 |
| 3 | 343 | 0.9494 | 0.9432 | +0.4980 |
| 4 | 729 | 0.9632 | 0.9480 | +0.5028 |
| 6 | 2197 | 0.9702 | 0.9440 | +0.4988 |
| 8 | 4913 | 0.9704 | 0.9424 | +0.4972 |
| 12 | 15625 | 0.9770 | 0.9456 | +0.5004 |
