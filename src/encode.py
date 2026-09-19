import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

IN_PATH = "data/processed/ugransome.csv"
OUT_PATH = "data/processed/ugransome_encoded.csv"
MAP_PATH = "data/processed/label_maps.json"

df = pd.read_csv(IN_PATH)

label_maps = {}
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        # save the mapping so "0 -> TCP" etc. isn't lost
        label_maps[col] = {str(cls): int(i) for i, cls in enumerate(le.classes_)}

df.to_csv(OUT_PATH, index=False)
with open(MAP_PATH, "w") as f:
    json.dump(label_maps, f, indent=2)

print("Encoded columns:", list(label_maps.keys()))
print(df.dtypes)
print(f"Saved to {OUT_PATH}")