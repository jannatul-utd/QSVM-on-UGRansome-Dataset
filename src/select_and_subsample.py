# src/select_and_subsample.py
import pandas as pd
from sklearn.model_selection import train_test_split

IN_PATH = "data/processed/ugransome_encoded.csv"
OUT_PATH = "data/processed/ugransome_subsampled.csv"

FEATURES = ["USD", "BTC", "Netflow_Bytes"]
TARGET = "Prediction"
N_TOTAL = 200  # total rows to keep across train+test — see note below

df = pd.read_csv(IN_PATH)[FEATURES + [TARGET]]

# stratified subsample: keeps the S/A/SS proportions from the full dataset
subsample, _ = train_test_split(
    df, train_size=N_TOTAL, stratify=df[TARGET], random_state=0
)

print("Subsample class balance:")
print(subsample[TARGET].value_counts(normalize=True).round(3))

subsample.to_csv(OUT_PATH, index=False)
print(f"Saved {subsample.shape[0]} rows to {OUT_PATH}")