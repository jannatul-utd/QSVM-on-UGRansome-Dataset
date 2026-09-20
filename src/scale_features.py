# src/scale_features.py — corrected
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

df = pd.read_csv("data/processed/ugransome_encoded.csv")
FEATURES = ["USD", "BTC", "Netflow_Bytes"]

# compress the long right tail before scaling
for col in FEATURES:
    df[col] = np.log1p(df[col])

scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
df[FEATURES] = scaler.fit_transform(df[FEATURES])

df.to_csv("data/processed/ugransome_scaled.csv", index=False)
joblib.dump(scaler, "data/processed/feature_scaler.pkl")

print(df[FEATURES].describe())