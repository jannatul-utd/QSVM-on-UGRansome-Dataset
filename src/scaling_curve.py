"""
Step 5 (run 5) - VQC accuracy vs parameter count, against the classical control.

This is the headline experiment. The capacity-matched diagnostic established:
  - parameter count is the dominant constraint (classical needs ~100 params
    for 0.87; 9 params buys only 0.64)
  - the VQC trails its capacity-matched classical twin by ~10 points at both
    sizes measured so far

What is missing is the SHAPE of the VQC curve. Three outcomes, each a
different conclusion:
  converges toward classical  -> quantum catches up with scale
  stays parallel below it     -> a persistent, quantifiable variational deficit
  flattens                    -> the VQC stops benefiting from parameters

All configs use exact parameter-shift gradients + L-BFGS, so the optimizer is
held constant across the curve and cannot confound the comparison (SPSA could
not be used here -- its noise grows with parameter count, which is exactly the
axis being measured).

NOTE: this is a long job (~3-4 hours). Results are written to disk after EVERY
config, so killing it partway still leaves a usable partial curve.

Produces:
  results/scaling_curve.md   (rewritten after each config)
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

N_TOTAL = 625           # 500 train / 125 test -- halved to keep runtime sane
TEST_FRAC = 0.2
MAX_ITER = 30
SEED = 0

# ascending, so the cheap informative points land first
LAYER_COUNTS = [2, 4, 8, 12, 16]        # -> 9, 15, 27, 39, 51 parameters

# classical capacity-matched reference (order-3 Fourier, SelectKBest, n=1000/250)
CLASSICAL_CURVE = {3: 0.6080, 6: 0.6320, 9: 0.6440, 12: 0.6880, 15: 0.6960,
                   21: 0.7200, 30: 0.7640, 45: 0.8200, 60: 0.8320, 100: 0.8720}

RNG = np.random.default_rng(SEED)
OBSERVABLES = [SparsePauliOp("ZII"), SparsePauliOp("IZI"), SparsePauliOp("IIZ")]


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


def build_circuit(n_layers):
    """Data re-uploading circuit: encode + Ry layer + CX ladder, x n_layers,
    then a final Ry layer. Parameters = NUM_QUBITS * (n_layers + 1)."""
    x = ParameterVector("x", NUM_QUBITS)
    theta = ParameterVector("\u03b8", NUM_QUBITS * (n_layers + 1))
    qc = QuantumCircuit(NUM_QUBITS)
    p = 0
    for _ in range(n_layers):
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


def loss_and_grad(weights, estimator, circuit, X, y, n_weights):
    logits = forward_logits(estimator, circuit, X, weights)
    loss = cross_entropy(logits, y)

    probs = softmax(logits)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    dL_dlogit = (probs - onehot) / len(y)

    grad = np.zeros(n_weights)
    for j in range(n_weights):
        shift = np.zeros(n_weights)
        shift[j] = np.pi / 2
        plus = forward_logits(estimator, circuit, X, weights + shift)
        minus = forward_logits(estimator, circuit, X, weights - shift)
        grad[j] = np.sum(dL_dlogit * (plus - minus) / 2.0)
    return loss, grad


def run_config(X_train, y_train, X_test, y_test, n_layers):
    circuit, n_weights = build_circuit(n_layers)
    estimator = StatevectorEstimator()
    w0 = RNG.uniform(0, 2 * np.pi, n_weights)

    def fun(w):
        return loss_and_grad(w, estimator, circuit, X_train, y_train, n_weights)

    start = time.perf_counter()
    res = minimize(fun, w0, jac=True, method="L-BFGS-B",
                   options={"maxiter": MAX_ITER})
    elapsed = time.perf_counter() - start

    tr = forward_logits(estimator, circuit, X_train, res.x)
    te = forward_logits(estimator, circuit, X_test, res.x)
    return {
        "n_layers": n_layers,
        "n_params": n_weights,
        "depth": circuit.depth(),
        "loss": cross_entropy(tr, y_train),
        "train_acc": accuracy(tr, y_train),
        "test_acc": accuracy(te, y_test),
        "time": elapsed,
    }


def classical_at(n_params):
    """Nearest classical reference point, for side-by-side comparison."""
    k = min(CLASSICAL_CURVE, key=lambda kk: abs(kk - n_params))
    return k, CLASSICAL_CURVE[k]


def write_results(results, baseline):
    with open(f"{OUT_DIR}/scaling_curve.md", "w") as f:
        f.write("# VQC scaling curve: accuracy vs parameter count\n\n")
        f.write(f"n={int(N_TOTAL * (1 - TEST_FRAC))} train / "
                f"{int(N_TOTAL * TEST_FRAC)} test, baseline {baseline:.4f}. "
                f"Exact parameter-shift gradients + L-BFGS, maxiter={MAX_ITER}.\n\n")
        f.write("| layers | params | depth | loss | train acc | test acc | "
                "classical (nearest k) | gap |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in results:
            k, c = classical_at(r["n_params"])
            f.write(f"| {r['n_layers']} | {r['n_params']} | {r['depth']} | "
                    f"{r['loss']:.4f} | {r['train_acc']:.4f} | {r['test_acc']:.4f} | "
                    f"{c:.4f} (k={k}) | {r['test_acc'] - c:+.4f} |\n")
        f.write("\nClassical reference is logistic regression on the k most informative\n")
        f.write("order-3 Fourier basis functions. Note it selects those k using the\n")
        f.write("labels, an advantage the VQC does not have -- see roadmap caveat.\n")


def main():
    X_train, X_test, y_train, y_test = load_and_prepare()
    baseline = np.bincount(y_test).max() / len(y_test)

    print(f"Train {len(y_train)} / Test {len(y_test)}  |  baseline {baseline:.4f}")
    print(f"Exact gradients + L-BFGS (maxiter={MAX_ITER}), results saved after each config\n")

    header = (f"{'layers':>7} {'params':>7} {'depth':>6} {'loss':>8} {'train':>7} "
              f"{'test':>7} {'classical':>10} {'gap':>8} {'time':>8}")
    print(header)
    print("-" * len(header))

    results = []
    for n_layers in LAYER_COUNTS:
        r = run_config(X_train, y_train, X_test, y_test, n_layers)
        results.append(r)
        k, c = classical_at(r["n_params"])
        print(f"{r['n_layers']:>7} {r['n_params']:>7} {r['depth']:>6} {r['loss']:>8.4f} "
              f"{r['train_acc']:>7.4f} {r['test_acc']:>7.4f} {c:>10.4f} "
              f"{r['test_acc'] - c:>+8.4f} {r['time']:>7.0f}s")
        write_results(results, baseline)   # save after EVERY config

    print(f"\nWritten to {OUT_DIR}/scaling_curve.md")
    if len(results) >= 2:
        first, last = results[0], results[-1]
        print(f"VQC moved {first['test_acc']:.4f} ({first['n_params']} params) -> "
              f"{last['test_acc']:.4f} ({last['n_params']} params)")


if __name__ == "__main__":
    main()