"""
Step 5 (run 4) - Is SPSA the bottleneck? Train with exact gradients.

The Fourier ceiling diagnostic showed the model's function class is NOT the
limit: an order-1 Fourier basis (the frequency content a single-upload
circuit already has) reaches 0.7378, and order 2-3 reaches 0.90-0.94. Yet
SPSA training tops out at 0.54.

That points at the optimizer. SPSA estimates the full gradient from ONE
random direction per iteration -- cheap, but very noisy. These circuits have
only 9-15 parameters, so the parameter-shift rule gives the EXACT analytic
gradient for 2*n_params forward passes, which L-BFGS can then use properly.

Parameter-shift: for a gate exp(-i*theta*P/2) with P a Pauli (our Ry gates),
  d<O>/dtheta = [ <O>(theta + pi/2) - <O>(theta - pi/2) ] / 2
Chained through softmax + cross-entropy:
  dL/dlogit_c = (softmax_c - onehot_c) / n

If the control circuit moves from ~0.51 toward ~0.74, the ceiling was SPSA.

Produces:
  results/exact_gradient_train.md
"""
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

RAW_PATH = "data/raw/ugransome.csv"
OUT_DIR = "results"
FEATURES = ["USD", "BTC", "Netflow_Bytes"]
CLASSES = ["S", "A", "SS"]
NUM_QUBITS = 3

N_TOTAL = 1250          # 1000 train / 250 test -- same split as the SPSA runs
TEST_FRAC = 0.2
MAX_ITER = 60
SEED = 0

# (label, n_layers, reupload, SPSA result for reference)
CONFIGS = [
    ("control (1 upload)", 2, False, 0.5080),
    ("reupload L=4", 4, True, 0.5600),
]

RNG = np.random.default_rng(SEED)


def load_and_prepare():
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]
    frac = N_TOTAL / len(df)
    parts = [df[df["Prediction"] == cls].sample(frac=frac, random_state=SEED)
             for cls in CLASSES]
    sampled = pd.concat(parts, ignore_index=True)
    X_log = np.log1p(sampled[FEATURES].to_numpy(dtype=float))
    X_scaled = MinMaxScaler(feature_range=(0, 2 * np.pi)).fit_transform(X_log)
    y = np.array([CLASSES.index(c) for c in sampled["Prediction"]])
    return train_test_split(X_scaled, y, test_size=TEST_FRAC,
                            stratify=y, random_state=SEED)


def build_circuit(n_layers, reupload):
    x = ParameterVector("x", NUM_QUBITS)
    theta = ParameterVector("\u03b8", NUM_QUBITS * (n_layers + 1))
    qc = QuantumCircuit(NUM_QUBITS)
    p = 0
    for layer in range(n_layers):
        if reupload or layer == 0:
            for q in range(NUM_QUBITS):
                qc.h(q)
                qc.p(2 * x[q], q)
        for q in range(NUM_QUBITS):
            qc.ry(theta[p], q)
            p += 1
        for q in range(NUM_QUBITS - 1):
            qc.cx(q, q + 1)
    for q in range(NUM_QUBITS):
        qc.ry(theta[p], q)
        p += 1
    names = [pr.name for pr in qc.parameters]
    assert all(n.startswith("x") for n in names[:NUM_QUBITS]), \
        f"unexpected parameter order: {names[:6]}"
    return qc, len(theta)


OBSERVABLES = [SparsePauliOp("ZII"), SparsePauliOp("IZI"), SparsePauliOp("IIZ")]


def forward_logits(estimator, circuit, X, weights):
    param_sets = np.hstack([X, np.tile(weights, (X.shape[0], 1))])
    pubs = [(circuit, obs, param_sets) for obs in OBSERVABLES]
    result = estimator.run(pubs).result()
    return np.stack([res.data.evs for res in result], axis=1)


def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits, y):
    p = softmax(logits)[np.arange(len(y)), y]
    return -np.mean(np.log(np.clip(p, 1e-12, 1.0)))


def accuracy(logits, y):
    return np.mean(logits.argmax(axis=1) == y)


def loss_and_exact_grad(weights, estimator, circuit, X, y, n_weights):
    """Cross-entropy loss plus its exact gradient via the parameter-shift rule."""
    logits = forward_logits(estimator, circuit, X, weights)
    loss = cross_entropy(logits, y)

    # dL/dlogit: (softmax - onehot) / n
    probs = softmax(logits)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    dL_dlogit = (probs - onehot) / len(y)          # (n, 3)

    grad = np.zeros(n_weights)
    for j in range(n_weights):
        shift = np.zeros(n_weights)
        shift[j] = np.pi / 2
        plus = forward_logits(estimator, circuit, X, weights + shift)
        minus = forward_logits(estimator, circuit, X, weights - shift)
        dlogit_dtheta = (plus - minus) / 2.0        # (n, 3)
        grad[j] = np.sum(dL_dlogit * dlogit_dtheta)

    return loss, grad


def run_config(X_train, y_train, X_test, y_test, n_layers, reupload):
    circuit, n_weights = build_circuit(n_layers, reupload)
    estimator = StatevectorEstimator()
    weights0 = RNG.uniform(0, 2 * np.pi, n_weights)

    history = []

    def fun(w):
        loss, grad = loss_and_exact_grad(w, estimator, circuit, X_train,
                                         y_train, n_weights)
        history.append(loss)
        return loss, grad

    start = time.perf_counter()
    res = minimize(fun, weights0, jac=True, method="L-BFGS-B",
                   options={"maxiter": MAX_ITER})
    elapsed = time.perf_counter() - start

    w = res.x
    tr = forward_logits(estimator, circuit, X_train, w)
    te = forward_logits(estimator, circuit, X_test, w)
    return {
        "n_params": n_weights,
        "loss": cross_entropy(tr, y_train),
        "train_acc": accuracy(tr, y_train),
        "test_acc": accuracy(te, y_test),
        "n_grad_evals": len(history),
        "time": elapsed,
    }


def main():
    X_train, X_test, y_train, y_test = load_and_prepare()
    baseline = np.bincount(y_test).max() / len(y_test)

    print(f"Train {len(y_train)} / Test {len(y_test)}  |  baseline {baseline:.4f}")
    print("Exact parameter-shift gradients + L-BFGS (vs SPSA's 1-direction estimate)")
    print(f"Fourier upper bounds: order-1 = 0.7378, order-2 = 0.9001\n")

    header = (f"{'config':>20} {'params':>7} {'loss':>8} {'train':>7} {'test':>7} "
              f"{'SPSA was':>9} {'change':>8} {'time':>7}")
    print(header)
    print("-" * len(header))

    results = []
    for label, n_layers, reupload, spsa_ref in CONFIGS:
        r = run_config(X_train, y_train, X_test, y_test, n_layers, reupload)
        r["label"] = label
        r["spsa_ref"] = spsa_ref
        results.append(r)
        print(f"{label:>20} {r['n_params']:>7} {r['loss']:>8.4f} "
              f"{r['train_acc']:>7.4f} {r['test_acc']:>7.4f} {spsa_ref:>9.4f} "
              f"{r['test_acc'] - spsa_ref:>+8.4f} {r['time']:>6.0f}s")

    with open(f"{OUT_DIR}/exact_gradient_train.md", "w") as f:
        f.write("# Exact-gradient training (parameter-shift + L-BFGS)\n\n")
        f.write(f"n=1000 train / 250 test, baseline {baseline:.4f}.\n")
        f.write("Fourier upper bounds on the same data: order-1 = 0.7378, order-2 = 0.9001.\n\n")
        f.write("| config | params | loss | train acc | test acc | SPSA test acc | change |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['label']} | {r['n_params']} | {r['loss']:.4f} | "
                    f"{r['train_acc']:.4f} | {r['test_acc']:.4f} | {r['spsa_ref']:.4f} | "
                    f"{r['test_acc'] - r['spsa_ref']:+.4f} |\n")

    print("\nIf test accuracy jumped substantially, SPSA was the bottleneck, not the circuit.")


if __name__ == "__main__":
    main()