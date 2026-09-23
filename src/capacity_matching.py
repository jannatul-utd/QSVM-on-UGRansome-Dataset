"""
Diagnostic - Is the VQC underperforming, or is 0.53 simply what 9 parameters buy?

The earlier Fourier sweep compared an 81-coefficient classical model against a
9-parameter circuit and concluded the circuit was underperforming. That was not
like-for-like. This redoes it capacity-matched: fit classical logistic
regression on the Fourier basis restricted to k basis functions, sweeping k
across the VQC's actual parameter counts.

Reference points to compare against (exact-gradient training, n=1000/250):
  control, 9 params  -> 0.5280
  L=4,    15 params  -> 0.5920

If classical at k~9 also lands near 0.53, the VQC is delivering what its size
buys, and the result is about parameter count rather than anything quantum --
with a clean scaling curve showing how many parameters the problem needs.

If classical at k~9 reaches 0.70, the circuit really is wasting its capacity,
and the next question is why (parameterization, or the bounded [-1,1] output).

Runs in seconds.

Produces:
  results/capacity_matched.md
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

RAW_PATH = "data/raw/ugransome.csv"
OUT_DIR = "results"
FEATURES = ["USD", "BTC", "Netflow_Bytes"]
CLASSES = ["S", "A", "SS"]

N_TOTAL = 625          # was 1250 — match the scaling curve's n         # 1000 train / 250 test -- SAME split as the VQC runs
TEST_FRAC = 0.2
FOURIER_ORDER = 3       # generous; selection decides what actually gets used
K_VALUES = [3, 6, 9, 12, 15, 21, 30, 45, 60, 100]
SEED = 0

# (label, n_params, test_acc) from exact-gradient training
VQC_REFERENCE = [("control", 9, 0.5280), ("reupload L=4", 15, 0.5920)]


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


def fourier_basis(X, order):
    n_samples, n_features = X.shape
    per_feature = []
    for j in range(n_features):
        cols = [np.ones(n_samples)]
        for k in range(1, order + 1):
            cols.append(np.cos(k * X[:, j]))
            cols.append(np.sin(k * X[:, j]))
        per_feature.append(np.stack(cols, axis=1))
    basis = per_feature[0]
    for j in range(1, n_features):
        basis = np.einsum("ni,nj->nij", basis, per_feature[j])
        basis = basis.reshape(n_samples, -1)
    return basis


def main():
    X_train, X_test, y_train, y_test = load_and_prepare()
    baseline = np.bincount(y_test).max() / len(y_test)

    B_train_full = fourier_basis(X_train, FOURIER_ORDER)
    B_test_full = fourier_basis(X_test, FOURIER_ORDER)

    print(f"Train {len(y_train)} / Test {len(y_test)}  |  baseline {baseline:.4f}")
    print(f"Order-{FOURIER_ORDER} Fourier basis: {B_train_full.shape[1]} functions available")
    print("Selecting the k most informative, then fitting logistic regression.\n")

    header = f"{'k (basis fns)':>14} {'train':>8} {'test':>8} {'vs base':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for k in K_VALUES:
        if k > B_train_full.shape[1]:
            continue
        sel = SelectKBest(f_classif, k=k).fit(B_train_full, y_train)
        Btr = sel.transform(B_train_full)
        Bte = sel.transform(B_test_full)

        clf = LogisticRegression(max_iter=5000, C=1.0)
        clf.fit(Btr, y_train)
        tr, te = clf.score(Btr, y_train), clf.score(Bte, y_test)
        results.append((k, tr, te))
        print(f"{k:>14} {tr:>8.4f} {te:>8.4f} {te - baseline:>+9.4f}")

    print("\nVQC reference (exact-gradient training, same split):")
    for label, n_params, acc in VQC_REFERENCE:
        # nearest classical k for comparison
        nearest = min(results, key=lambda r: abs(r[0] - n_params))
        print(f"  {label:>14} ({n_params} params): {acc:.4f}   "
              f"| classical at k={nearest[0]}: {nearest[2]:.4f}   "
              f"| gap {acc - nearest[2]:+.4f}")

    with open(f"{OUT_DIR}/capacity_matched.md", "w") as f:
        f.write("# Capacity-matched comparison\n\n")
        f.write(f"Logistic regression on the k most informative order-{FOURIER_ORDER} "
                f"Fourier basis functions, n=1000/250, baseline {baseline:.4f}.\n\n")
        f.write("| k basis functions | train acc | test acc | vs baseline |\n")
        f.write("|---|---|---|---|\n")
        for k, tr, te in results:
            f.write(f"| {k} | {tr:.4f} | {te:.4f} | {te - baseline:+.4f} |\n")
        f.write("\n## VQC reference (exact-gradient training)\n\n")
        f.write("| config | params | test acc |\n|---|---|---|\n")
        for label, n_params, acc in VQC_REFERENCE:
            f.write(f"| {label} | {n_params} | {acc:.4f} |\n")


if __name__ == "__main__":
    main()