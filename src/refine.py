import pandas as pd

RAW_PATH = "data/raw/ugransome.csv"
OUT_PATH = "data/processed/ugransome.csv"

df = pd.read_csv(RAW_PATH)
print("Before:", df.shape)

# check exact values before touching them
print(df["Threats"].unique())

# 1. drop invalid Time rows
df = df[df["Time"] >= 0]

# 2. drop invalid ExpAddress placeholder rows
df = df[df["ExpAddress"] != "1"]

# 3. fix Threats misspellings (exact match, not substring)
df["Threats"] = df["Threats"].replace({
    "Bonet": "Botnet",
    "NerisBonet": "NerisBotnet",
})

df = df.reset_index(drop=True)
print("After:", df.shape)
print(df["Threats"].unique())

df.to_csv(OUT_PATH, index=False)
print(f"Saved to {OUT_PATH}")