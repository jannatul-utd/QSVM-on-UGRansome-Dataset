"""
Step 1 - EDA on the UGRansome dataset.

Produces:
  results/eda_summary.md   -- text findings
  results/class_balance.png
  results/mutual_info.png
  results/correlation_heatmap.png
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_PATH = "data/raw/ugransome.csv"
OUT_DIR = "results"


def main():
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]

    lines = []
    lines.append(f"# EDA summary\n")
    lines.append(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    # dtypes
    lines.append("## Dtypes\n")
    lines.append(df.dtypes.to_string() + "\n")

    # missing values
    na = df.isna().sum()
    lines.append("\n## Missing values\n")
    lines.append(f"Total missing cells: {na.sum()}\n")

    # duplicates
    dup = df.duplicated().sum()
    lines.append(f"\n## Duplicate rows: {dup}\n")

    # negative Time check (flagged by reference notebook)
    neg_time = (df["Time"] < 0).sum() if "Time" in df.columns else "n/a"
    lines.append(f"\n## Negative `Time` values: {neg_time}\n")

    # invalid ExpAddress == '1' (flagged by reference notebook)
    if "ExpAddress" in df.columns:
        invalid_exp = (df["ExpAddress"].astype(str) == "1").sum()
        lines.append(f"## `ExpAddress` == '1' rows: {invalid_exp}\n")

    # Bonet / NerisBonet spelling check (flagged by reference notebook)
    if "Threats" in df.columns:
        bonet = (df["Threats"] == "Bonet").sum()
        nerisbonet = (df["Threats"] == "NerisBonet").sum()
        lines.append(f"## `Threats` == 'Bonet': {bonet}, == 'NerisBonet': {nerisbonet} "
                     f"(likely misspellings of Botnet/NerisBotnet)\n")

    # cardinality
    lines.append("\n## Cardinality per column\n")
    card = df.nunique().sort_values()
    lines.append(card.to_string() + "\n")

    # target distribution
    lines.append("\n## Target (`Prediction`) distribution\n")
    counts = df["Prediction"].value_counts()
    pct = (counts / len(df) * 100).round(2)
    lines.append(pd.DataFrame({"count": counts, "pct": pct}).to_string() + "\n")

    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_title("Prediction class balance")
    ax.set_ylabel("count")
    for i, v in enumerate(counts):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/class_balance.png", dpi=150)
    plt.close(fig)

    # encode categoricals for correlation / mutual info against target
    enc_df = df.copy()
    encoders = {}
    for col in enc_df.columns:
        if not pd.api.types.is_numeric_dtype(enc_df[col]):
            le = LabelEncoder()
            enc_df[col] = le.fit_transform(enc_df[col].astype(str))
            encoders[col] = le

    target_col = "Prediction"
    X = enc_df.drop(columns=[target_col])
    y = enc_df[target_col]

    # mutual information of every column vs Prediction (catches leakage)
    mi = mutual_info_classif(X, y, discrete_features="auto", random_state=0)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    lines.append("\n## Mutual information vs `Prediction` (higher = more predictive)\n")
    lines.append(mi_series.to_string() + "\n")

    fig, ax = plt.subplots(figsize=(6, 5))
    mi_series.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.invert_yaxis()
    ax.set_title("Mutual information vs Prediction")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/mutual_info.png", dpi=150)
    plt.close(fig)

    # correlation heatmap among encoded features (redundancy / leakage check)
    corr = X.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature correlation (encoded)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # flag highly correlated pairs (potential redundancy)
    lines.append("\n## Highly correlated feature pairs (|corr| > 0.85, excluding target)\n")
    high_corr_pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if abs(v) > 0.85:
                high_corr_pairs.append((cols[i], cols[j], round(v, 3)))
    if high_corr_pairs:
        for a, b, v in high_corr_pairs:
            lines.append(f"- {a} <-> {b}: {v}\n")
    else:
        lines.append("(none above threshold)\n")

    with open(f"{OUT_DIR}/eda_summary.md", "w") as f:
        f.writelines(lines)

    print("".join(lines))


if __name__ == "__main__":
    main()
