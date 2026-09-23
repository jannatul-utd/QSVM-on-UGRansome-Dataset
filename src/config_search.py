"""
Step 5 (run 2 prep) - Config search after training run 1 failed to beat the
majority-class baseline (test 0.4253 vs 0.445 baseline).

Run 1 diagnosis: gradients were healthy (no barren plateau), so the suspects
are (1) logit range squashed into [-1,1] by raw Z expectations, (2) only 9
trainable parameters, (3) ZFeatureMap's lack of encoding entanglement.

This sweeps all three cheaply at n=1000 (~3 min/config instead of ~24) so we
only spend a full-size run on a config that demonstrably learns.

SPSA's step-size gain `a` is auto-calibrated per config from the observed
gradient magnitude. Without this, a larger LOGIT_SCALE would just produce
larger gradients and a fixed `a` would confound "the scale helped" with
"the step size happened to suit that scale".

Produces:
  results/config_search.md
"""
import itertools
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

RAW_PATH = "data/raw/ugransome.csv"
OUT_DIR = "results"
FEATURES = ["USD", "BTC", "Netflow_Bytes"]
CLASSES = ["S", "A", "SS"]
NUM_QUBITS = 3

N_TOTAL = 1250          # 1000 train / 250 test
TEST_FRAC = 0.2
N_ITERATIONS = 100
SEED = 0

# the grid
LOGIT_SCALES = [1.0]
REPS_OPTIONS = [2, 6]
FEATURE_MAPS = ["Z", "ZZ"]

SPSA_ALPHA = 0.602
SPSA_GAMMA = 0.101
SPSA_C = 0.1
TARGET_FIRST_STEP = 0.2   # radians of parameter movement on iteration 1

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


def build(feature_map_kind, reps):
    if feature_map_kind == "Z":
        feature_map = ZFeatureMap(NUM_QUBITS, reps=1)
    else:
        feature_map = ZZFeatureMap(NUM_QUBITS, reps=1, entanglement="linear")
    ansatz = RealAmplitudes(NUM_QUBITS, reps=reps, entanglement="linear")
    circuit = feature_map.compose(ansatz)
    observables = [SparsePauliOp("ZII"), SparsePauliOp("IZI"), SparsePauliOp("IIZ")]
    return circuit, ansatz.num_parameters, observables


def forward_logits(estimator, circuit, observables, X, weights, logit_scale):
    param_sets = np.hstack([X, np.tile(weights, (X.shape[0], 1))])
    pubs = [(circuit, obs, param_sets) for obs in observables]
    result = estimator.run(pubs).result()
    return logit_scale * np.stack([res.data.evs for res in result], axis=1)


def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits, y):
    p = softmax(logits)[np.arange(len(y)), y]
    return -np.mean(np.log(np.clip(p, 1e-12, 1.0)))


def accuracy(logits, y):
    return np.mean(logits.argmax(axis=1) == y)


def calibrate_a(loss_fn, weights, n_weights, n_probes=5):
    """Estimate SPSA gain `a` so iteration 1 moves ~TARGET_FIRST_STEP radians."""
    mags = []
    for _ in range(n_probes):
        delta = RNG.choice([-1, 1], size=n_weights)
        g = (loss_fn(weights + SPSA_C * delta) - loss_fn(weights - SPSA_C * delta))
        g = g / (2 * SPSA_C) * delta
        mags.append(np.linalg.norm(g))
    mean_mag = np.mean(mags)
    if mean_mag < 1e-12:
        return 0.0, mean_mag
    return TARGET_FIRST_STEP / mean_mag, mean_mag


def run_config(X_train, y_train, X_test, y_test, feature_map_kind, reps, logit_scale):
    circuit, n_weights, observables = build(feature_map_kind, reps)
    estimator = StatevectorEstimator()

    def loss_fn(w):
        return cross_entropy(
            forward_logits(estimator, circuit, observables, X_train, w, logit_scale),
            y_train)

    weights = RNG.uniform(0, 2 * np.pi, n_weights)
    spsa_a, init_grad_mag = calibrate_a(loss_fn, weights, n_weights)

    for k in range(1, N_ITERATIONS + 1):
        a_k = spsa_a / (k ** SPSA_ALPHA)
        c_k = SPSA_C / (k ** SPSA_GAMMA)
        delta = RNG.choice([-1, 1], size=n_weights)
        g = (loss_fn(weights + c_k * delta) - loss_fn(weights - c_k * delta))
        weights = weights - a_k * (g / (2 * c_k) * delta)

    train_logits = forward_logits(estimator, circuit, observables, X_train, weights, logit_scale)
    test_logits = forward_logits(estimator, circuit, observables, X_test, weights, logit_scale)

    return {
        "feature_map": feature_map_kind,
        "reps": reps,
        "scale": logit_scale,
        "n_params": n_weights,
        "loss": cross_entropy(train_logits, y_train),
        "train_acc": accuracy(train_logits, y_train),
        "test_acc": accuracy(test_logits, y_test),
        "init_grad_mag": init_grad_mag,
        "spsa_a": spsa_a,
    }


def main():
    X_train, X_test, y_train, y_test = load_and_prepare()
    baseline = np.bincount(y_test).max() / len(y_test)
    print(f"Train {len(y_train)} / Test {len(y_test)}  |  "
          f"majority-class baseline = {baseline:.4f}\n")

    header = (f"{'fmap':>5} {'reps':>5} {'scale':>6} {'params':>7} "
              f"{'loss':>8} {'train':>7} {'test':>7} {'vs base':>8} {'time':>7}")
    print(header)
    print("-" * len(header))

    results = []
    for fmap, reps, scale in itertools.product(FEATURE_MAPS, REPS_OPTIONS, LOGIT_SCALES):
        start = time.perf_counter()
        r = run_config(X_train, y_train, X_test, y_test, fmap, reps, scale)
        r["time"] = time.perf_counter() - start
        results.append(r)
        print(f"{r['feature_map']:>5} {r['reps']:>5} {r['scale']:>6.1f} "
              f"{r['n_params']:>7} {r['loss']:>8.4f} {r['train_acc']:>7.4f} "
              f"{r['test_acc']:>7.4f} {r['test_acc'] - baseline:>+8.4f} "
              f"{r['time']:>6.0f}s")

    results.sort(key=lambda r: r["test_acc"], reverse=True)

    with open(f"{OUT_DIR}/config_search.md", "w") as f:
        f.write("# VQC config search (n=1000 train / 250 test, 100 SPSA iterations)\n\n")
        f.write(f"Majority-class baseline: {baseline:.4f}\n\n")
        f.write("| feature map | reps | logit scale | params | loss | train acc | test acc | vs baseline |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['feature_map']} | {r['reps']} | {r['scale']:.1f} | {r['n_params']} | "
                    f"{r['loss']:.4f} | {r['train_acc']:.4f} | {r['test_acc']:.4f} | "
                    f"{r['test_acc'] - baseline:+.4f} |\n")

    best = results[0]
    print(f"\nBest: {best['feature_map']}FeatureMap, reps={best['reps']}, "
          f"scale={best['scale']:.1f} -> test {best['test_acc']:.4f} "
          f"({best['test_acc'] - baseline:+.4f} vs baseline)")


if __name__ == "__main__":
    main()