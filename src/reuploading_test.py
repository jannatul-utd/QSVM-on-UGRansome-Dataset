"""
Step 5 (run 3) - Does data re-uploading raise the ceiling?

Background: the config search found the best architecture so far scores 0.544
(baseline 0.445, classical SVM 0.953). Diagnosis is that 3 qubits x 9
parameters is too small a function class -- not that the dataset is too small.

Data re-uploading (Perez-Salinas et al. 2020) feeds the SAME input features
into the circuit repeatedly, interleaved between trainable layers, instead of
once at the front. This sharply raises expressivity without adding qubits;
they proved a single qubit with re-uploading is a universal classifier.

The key comparison is CONTROL vs L=2: both have exactly 9 trainable
parameters, so any difference is attributable to re-uploading alone and not
to parameter count. L=4 and L=6 then ask whether depth pays off once
re-uploading is present (it did not without it -- but that test was run at a
starved iteration budget, so this one uses 300 iterations throughout).

Produces:
  results/reuploading_test.md
"""
import time
import numpy as np
import pandas as pd
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

N_TOTAL = 1250          # 1000 train / 250 test
TEST_FRAC = 0.2
N_ITERATIONS = 300      # raised from 100 to clear the under-training confound
SEED = 0

SPSA_ALPHA = 0.602
SPSA_GAMMA = 0.101
SPSA_C = 0.1
TARGET_FIRST_STEP = 0.2

# (label, n_layers, reupload?)  -- control has the SAME param count as L=2
CONFIGS = [
    ("control (1 upload)", 2, False),
    ("reupload L=2", 2, True),
    ("reupload L=4", 4, True),
    ("reupload L=6", 6, True),
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
    y_idx = np.array([CLASSES.index(c) for c in sampled["Prediction"]])

    return train_test_split(X_scaled, y_idx, test_size=TEST_FRAC,
                            stratify=y_idx, random_state=SEED)


def encode_layer(qc, x):
    """One ZFeatureMap-style encoding block: H then P(2x) per qubit."""
    for q in range(NUM_QUBITS):
        qc.h(q)
        qc.p(2 * x[q], q)


def build_circuit(n_layers, reupload):
    """
    n_layers trainable blocks. Each block = [encoding if re-uploading] +
    Ry rotations + linear CX ladder. A final Ry layer follows the last
    entanglement, matching RealAmplitudes' structure.
    Trainable parameters = NUM_QUBITS * (n_layers + 1).
    """
    x = ParameterVector("x", NUM_QUBITS)
    theta = ParameterVector("\u03b8", NUM_QUBITS * (n_layers + 1))
    qc = QuantumCircuit(NUM_QUBITS)
    p = 0

    for layer in range(n_layers):
        # control encodes once at the front; re-uploading encodes every layer
        if reupload or layer == 0:
            encode_layer(qc, x)
        for q in range(NUM_QUBITS):
            qc.ry(theta[p], q)
            p += 1
        for q in range(NUM_QUBITS - 1):
            qc.cx(q, q + 1)

    for q in range(NUM_QUBITS):
        qc.ry(theta[p], q)
        p += 1

    # Parameter binding below is positional, so verify the x's really do come
    # first in circuit.parameters before relying on it.
    names = [pr.name for pr in qc.parameters]
    assert all(n.startswith("x") for n in names[:NUM_QUBITS]), \
        f"unexpected parameter order: {names[:6]}"

    return qc, len(theta)


def forward_logits(estimator, circuit, observables, X, weights):
    param_sets = np.hstack([X, np.tile(weights, (X.shape[0], 1))])
    pubs = [(circuit, obs, param_sets) for obs in observables]
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


def calibrate_a(loss_fn, weights, n_weights, n_probes=5):
    mags = []
    for _ in range(n_probes):
        d = RNG.choice([-1, 1], size=n_weights)
        g = (loss_fn(weights + SPSA_C * d) - loss_fn(weights - SPSA_C * d))
        mags.append(np.linalg.norm(g / (2 * SPSA_C) * d))
    mean_mag = np.mean(mags)
    return (TARGET_FIRST_STEP / mean_mag if mean_mag > 1e-12 else 0.0), mean_mag


def run_config(X_train, y_train, X_test, y_test, n_layers, reupload):
    circuit, n_weights = build_circuit(n_layers, reupload)
    observables = [SparsePauliOp("ZII"), SparsePauliOp("IZI"), SparsePauliOp("IIZ")]
    estimator = StatevectorEstimator()

    def loss_fn(w):
        return cross_entropy(
            forward_logits(estimator, circuit, observables, X_train, w), y_train)

    weights = RNG.uniform(0, 2 * np.pi, n_weights)
    spsa_a, _ = calibrate_a(loss_fn, weights, n_weights)

    for k in range(1, N_ITERATIONS + 1):
        a_k = spsa_a / (k ** SPSA_ALPHA)
        c_k = SPSA_C / (k ** SPSA_GAMMA)
        d = RNG.choice([-1, 1], size=n_weights)
        g = (loss_fn(weights + c_k * d) - loss_fn(weights - c_k * d))
        weights = weights - a_k * (g / (2 * c_k) * d)

    tr = forward_logits(estimator, circuit, observables, X_train, weights)
    te = forward_logits(estimator, circuit, observables, X_test, weights)
    return {
        "n_params": n_weights,
        "depth": circuit.depth(),
        "loss": cross_entropy(tr, y_train),
        "train_acc": accuracy(tr, y_train),
        "test_acc": accuracy(te, y_test),
    }


def main():
    X_train, X_test, y_train, y_test = load_and_prepare()
    baseline = np.bincount(y_test).max() / len(y_test)
    print(f"Train {len(y_train)} / Test {len(y_test)}  |  baseline {baseline:.4f}  "
          f"|  {N_ITERATIONS} SPSA iterations")
    print(f"Reference points: previous best VQC 0.5440, classical SVM 0.9530\n")

    header = (f"{'config':>20} {'params':>7} {'depth':>6} {'loss':>8} "
              f"{'train':>7} {'test':>7} {'vs base':>8} {'time':>7}")
    print(header)
    print("-" * len(header))

    results = []
    for label, n_layers, reupload in CONFIGS:
        start = time.perf_counter()
        r = run_config(X_train, y_train, X_test, y_test, n_layers, reupload)
        r["label"] = label
        r["time"] = time.perf_counter() - start
        results.append(r)
        print(f"{label:>20} {r['n_params']:>7} {r['depth']:>6} {r['loss']:>8.4f} "
              f"{r['train_acc']:>7.4f} {r['test_acc']:>7.4f} "
              f"{r['test_acc'] - baseline:>+8.4f} {r['time']:>6.0f}s")

    with open(f"{OUT_DIR}/reuploading_test.md", "w") as f:
        f.write(f"# Data re-uploading test (n=1000/250, {N_ITERATIONS} SPSA iterations)\n\n")
        f.write(f"Majority baseline: {baseline:.4f} | previous best VQC: 0.5440 | "
                f"classical SVM: 0.9530\n\n")
        f.write("| config | params | depth | loss | train acc | test acc | vs baseline |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['label']} | {r['n_params']} | {r['depth']} | {r['loss']:.4f} | "
                    f"{r['train_acc']:.4f} | {r['test_acc']:.4f} | "
                    f"{r['test_acc'] - baseline:+.4f} |\n")

    ctrl = results[0]["test_acc"]
    reup = results[1]["test_acc"]
    print(f"\nMatched-parameter comparison (9 params each): "
          f"control {ctrl:.4f} vs re-upload {reup:.4f} -> {reup - ctrl:+.4f}")


if __name__ == "__main__":
    main()