"""
Diagnostic - What Fourier frequency order does this problem actually need?

Motivation: three architectural changes in a row (more ansatz reps, ZZ
encoding, data re-uploading at L=2) all failed to move the VQC above ~0.54.
Schuld, Sweke & Meyer (2021, arXiv:2008.08605) show a variational quantum
model with this encoding computes a trigonometric polynomial whose available
frequency spectrum is bounded by the number of data uploads L.

Features are scaled to [0, 2*pi], so frequency 1 = one oscillation across the
whole feature range. The tuned classical SVM needed gamma=100, i.e. structure
at ~1/10 of that range. If the problem genuinely requires high frequencies,
no amount of ansatz tuning will help -- only more uploads will.

This tests that directly and classically. The VQC's function class is a
SUBSET of the order-K Fourier basis, so logistic regression on that basis is
an UPPER BOUND on what a VQC with L=K uploads could achieve. If the upper
bound at K=2 is ~0.55, the VQC is not underperforming its architecture --
the architecture is the ceiling.

Runs in seconds (no quantum simulation at all).

Produces:
  results/fourier_ceiling.md
"""
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

RAW_PATH = "data/raw/ugransome.csv"
OUT_DIR = "results"
FEATURES = ["USD", "BTC", "Netflow_Bytes"]
CLASSES = ["S", "A", "SS"]

N_TOTAL = 6250          # 5000 train / 1250 test -- bigger, since this is cheap
TEST_FRAC = 0.2
ORDERS = [1, 2, 3, 4, 6, 8, 12]
SEED = 0


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
    """
    Full multivariate real Fourier basis up to `order` per feature, including
    cross-products between features (which is what entanglement buys a VQC).
    Per feature: [1, cos(x), sin(x), cos(2x), sin(2x), ...] -> 2*order+1 terms.
    Total basis size = (2*order+1)^n_features.
    """
    n_samples, n_features = X.shape
    per_feature = []
    for j in range(n_features):
        cols = [np.ones(n_samples)]
        for k in range(1, order + 1):
            cols.append(np.cos(k * X[:, j]))
            cols.append(np.sin(k * X[:, j]))
        per_feature.append(np.stack(cols, axis=1))

    # tensor product across features
    basis = per_feature[0]
    for j in range(1, n_features):
        basis = np.einsum("ni,nj->nij", basis, per_feature[j])
        basis = basis.reshape(n_samples, -1)
    return basis


def main():
    X_train, X_test, y_train, y_test = load_and_prepare()
    baseline = np.bincount(y_test).max() / len(y_test)

    print(f"Train {len(y_train)} / Test {len(y_test)}  |  baseline {baseline:.4f}")
    print("Fourier order K = upper bound on what a VQC with L=K uploads can express\n")

    header = f"{'order K':>8} {'basis size':>11} {'train':>8} {'test':>8} {'vs base':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for order in ORDERS:
        B_train = fourier_basis(X_train, order)
        B_test = fourier_basis(X_test, order)

        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(B_train, y_train)

        tr = clf.score(B_train, y_train)
        te = clf.score(B_test, y_test)
        results.append((order, B_train.shape[1], tr, te))
        print(f"{order:>8} {B_train.shape[1]:>11} {tr:>8.4f} {te:>8.4f} "
              f"{te - baseline:>+9.4f}")

    # reference: the tuned RBF SVM on the identical split
    svm = SVC(C=3000, gamma=100)
    svm.fit(X_train, y_train)
    svm_test = svm.score(X_test, y_test)
    print(f"\nReference -- tuned RBF SVM (C=3000, gamma=100) on same split: {svm_test:.4f}")
    print(f"Reference -- best VQC so far: 0.5440")

    with open(f"{OUT_DIR}/fourier_ceiling.md", "w") as f:
        f.write("# Fourier frequency ceiling diagnostic\n\n")
        f.write("Logistic regression on an order-K multivariate Fourier basis. The VQC's\n")
        f.write("function class is a subset of this, so each row is an UPPER BOUND on what\n")
        f.write("a VQC with L=K data uploads could reach.\n\n")
        f.write(f"Majority baseline: {baseline:.4f} | tuned RBF SVM: {svm_test:.4f} | "
                f"best VQC so far: 0.5440\n\n")
        f.write("| order K | basis size | train acc | test acc | vs baseline |\n")
        f.write("|---|---|---|---|---|\n")
        for order, size, tr, te in results:
            f.write(f"| {order} | {size} | {tr:.4f} | {te:.4f} | {te - baseline:+.4f} |\n")


if __name__ == "__main__":
    main()