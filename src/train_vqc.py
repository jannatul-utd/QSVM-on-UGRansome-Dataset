"""
Step 5 - Train the VQC on the UGRansome dataset.

Architecture (per roadmap decisions):
  - Features: USD, BTC, Netflow_Bytes (k=3, from step 2)
  - Scaling: log1p then MinMax to [0, 2pi] (step 2)
  - Encoding: ZFeatureMap(3, reps=1) (step 3)
  - Ansatz: RealAmplitudes(3, reps=2, entanglement="linear") (step 4)
  - Measurement: Z on each of the 3 qubits -> 3 logits (one per class
    S/A/SS) -> softmax -> cross-entropy (step 5, "loss function" decision)
  - Optimizer: SPSA, constant 2 evals/iteration (step 5, "optimizer" decision)
  - Training set: n=5000 train / 1250 test, stratified (step 5, "training
    set size" decision, benchmarked at ~1200 samples/sec)

Produces:
  results/vqc_training_curve.png   -- loss + train/test accuracy per iteration
  results/vqc_metrics.md           -- final numbers
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

RAW_PATH = "data/raw/ugransome.csv"
OUT_DIR = "results"
FEATURES = ["USD", "BTC", "Netflow_Bytes"]
CLASSES = ["S", "A", "SS"]  # fixed order -> logit index 0/1/2
NUM_QUBITS = 3

N_TOTAL = 6250       # 5000 train / 1250 test, stratified
TEST_FRAC = 0.2
N_ITERATIONS = 150
SPSA_A = 0.15         # step-size gain (a-hat), tuned for this loss scale
SPSA_C = 0.1          # perturbation size
SPSA_ALPHA = 0.602    # Spall's standard decay exponents
SPSA_GAMMA = 0.101
SEED = 0

RNG = np.random.default_rng(SEED)


def load_and_prepare():
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]

        # stratified subsample to N_TOTAL rows (sample per class directly --
    # groupby(...).apply(sample) drops the grouping column under pandas 3.x)
    frac = N_TOTAL / len(df)
    parts = [df[df["Prediction"] == cls].sample(frac=frac, random_state=SEED)
             for cls in CLASSES]
    sampled = pd.concat(parts, ignore_index=True)

    X_raw = sampled[FEATURES].to_numpy(dtype=float)
    y_raw = sampled["Prediction"].to_numpy()

    # log1p then MinMax to [0, 2*pi] -- step 2 decision
    X_log = np.log1p(X_raw)
    scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
    X_scaled = scaler.fit_transform(X_log)

    y_idx = np.array([CLASSES.index(c) for c in y_raw])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_idx, test_size=TEST_FRAC, stratify=y_idx, random_state=SEED
    )
    return X_train, X_test, y_train, y_test


def build_qnn_pieces():
    feature_map = ZFeatureMap(NUM_QUBITS, reps=1)
    ansatz = RealAmplitudes(NUM_QUBITS, reps=2, entanglement="linear")
    circuit = feature_map.compose(ansatz)
    observables = [
        SparsePauliOp("Z" + "I" * (NUM_QUBITS - 1)),
        SparsePauliOp("I" + "Z" + "I"),
        SparsePauliOp("I" + "I" + "Z"),
    ]
    return circuit, feature_map.num_parameters, ansatz.num_parameters, observables


def forward_logits(estimator, circuit, n_input_params, observables, X, weights):
    """Batched forward pass: returns (n_samples, 3) logit array."""
    n = X.shape[0]
    weight_block = np.tile(weights, (n, 1))
    param_sets = np.hstack([X, weight_block])  # (n, n_input_params + n_weights)
    pubs = [(circuit, obs, param_sets) for obs in observables]
    result = estimator.run(pubs).result()
    logits = np.stack([res.data.evs for res in result], axis=1)  # (n, 3)
    return logits


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss(logits, y_idx):
    probs = softmax(logits)
    n = len(y_idx)
    correct_probs = probs[np.arange(n), y_idx]
    return -np.mean(np.log(np.clip(correct_probs, 1e-12, 1.0)))


def accuracy(logits, y_idx):
    preds = logits.argmax(axis=1)
    return np.mean(preds == y_idx)


def main():
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_prepare()
    print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")

    circuit, n_input_params, n_weights, observables = build_qnn_pieces()
    estimator = StatevectorEstimator()

    def loss_fn(weights):
        logits = forward_logits(estimator, circuit, n_input_params, observables, X_train, weights)
        return cross_entropy_loss(logits, y_train)

    # manual SPSA (Spall's standard calibration) so we can log every
    # iteration's loss/accuracy and watch for stalled gradients
    weights = RNG.uniform(0, 2 * np.pi, n_weights)

    history = {"iter": [], "loss": [], "train_acc": [], "test_acc": [], "grad_norm": []}

    start = time.perf_counter()
    for k in range(1, N_ITERATIONS + 1):
        a_k = SPSA_A / (k ** SPSA_ALPHA)
        c_k = SPSA_C / (k ** SPSA_GAMMA)

        delta = RNG.choice([-1, 1], size=n_weights)
        loss_plus = loss_fn(weights + c_k * delta)
        loss_minus = loss_fn(weights - c_k * delta)

        grad_estimate = (loss_plus - loss_minus) / (2 * c_k) * delta
        weights = weights - a_k * grad_estimate

        if k % 5 == 0 or k == 1:
            train_logits = forward_logits(estimator, circuit, n_input_params, observables, X_train, weights)
            test_logits = forward_logits(estimator, circuit, n_input_params, observables, X_test, weights)
            loss_val = cross_entropy_loss(train_logits, y_train)
            train_acc = accuracy(train_logits, y_train)
            test_acc = accuracy(test_logits, y_test)
            grad_norm = np.linalg.norm(grad_estimate)

            history["iter"].append(k)
            history["loss"].append(loss_val)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["grad_norm"].append(grad_norm)

            elapsed = time.perf_counter() - start
            print(f"iter {k:4d}  loss {loss_val:.4f}  train_acc {train_acc:.4f}  "
                  f"test_acc {test_acc:.4f}  |grad| {grad_norm:.4e}  ({elapsed:.1f}s elapsed)")

    total_time = time.perf_counter() - start
    print(f"\nTotal training time: {total_time/60:.1f} min")

    # final metrics
    train_logits = forward_logits(estimator, circuit, n_input_params, observables, X_train, weights)
    test_logits = forward_logits(estimator, circuit, n_input_params, observables, X_test, weights)
    final_train_acc = accuracy(train_logits, y_train)
    final_test_acc = accuracy(test_logits, y_test)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["iter"], history["loss"], color="#4C72B0")
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_xlabel("SPSA iteration")

    axes[1].plot(history["iter"], history["train_acc"], label="train", color="#4C72B0")
    axes[1].plot(history["iter"], history["test_acc"], label="test", color="#DD8452")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("SPSA iteration")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/vqc_training_curve.png", dpi=150)
    plt.close(fig)

    with open(f"{OUT_DIR}/vqc_metrics.md", "w") as f:
        f.write("# VQC training results\n\n")
        f.write(f"Train rows: {X_train.shape[0]}, Test rows: {X_test.shape[0]}\n\n")
        f.write(f"Iterations: {N_ITERATIONS}\n\n")
        f.write(f"Total training time: {total_time/60:.1f} min\n\n")
        f.write(f"Final train accuracy: {final_train_acc:.4f}\n\n")
        f.write(f"Final test accuracy: {final_test_acc:.4f}\n\n")
        f.write(f"Final loss: {history['loss'][-1]:.4f}\n\n")
        f.write(f"Gradient norm, first logged iter: {history['grad_norm'][0]:.4e}, "
                f"last logged iter: {history['grad_norm'][-1]:.4e}\n")

    print(f"\nFinal train accuracy: {final_train_acc:.4f}")
    print(f"Final test accuracy: {final_test_acc:.4f}")


if __name__ == "__main__":
    main()