# src/tune_svm.py
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/ugransome_encoded.csv")
X = StandardScaler().fit_transform(df[["USD", "BTC", "Netflow_Bytes"]])
y = df["Prediction"]

# use a moderate-size stratified sample for the search itself, for speed
X_search, _, y_search, _ = train_test_split(
    X, y, train_size=5000, stratify=y, random_state=0
)

param_grid = {
    "C": [10, 100, 300, 1000, 3000],
    "gamma": [1, 10, 30, 100, 300],
}
grid = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_search, y_search)
# src/tune_svm.py — add this after grid.fit(...)
import numpy as np

# a holdout set the grid search never touched
X_holdout, _, y_holdout, _ = train_test_split(
    X, y, train_size=5000, stratify=y, random_state=1  # different random_state, disjoint-ish sample
)

best_model = grid.best_estimator_
train_acc = best_model.score(X_search, y_search)
holdout_acc = best_model.score(X_holdout, y_holdout)

print(f"Train accuracy:   {train_acc:.4f}")
print(f"CV accuracy:      {grid.best_score_:.4f}")
print(f"Holdout accuracy: {holdout_acc:.4f}")
print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)