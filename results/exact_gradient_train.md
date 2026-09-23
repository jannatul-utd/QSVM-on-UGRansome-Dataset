# Exact-gradient training (parameter-shift + L-BFGS)

n=1000 train / 250 test, baseline 0.4480.
Fourier upper bounds on the same data: order-1 = 0.7378, order-2 = 0.9001.

| config | params | loss | train acc | test acc | SPSA test acc | change |
|---|---|---|---|---|---|---|
| control (1 upload) | 9 | 0.9520 | 0.5400 | 0.5280 | 0.5080 | +0.0200 |
| reupload L=4 | 15 | 0.8693 | 0.6450 | 0.5920 | 0.5600 | +0.0320 |
