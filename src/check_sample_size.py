# check: how much does subsample size actually cost us?
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/ugransome_encoded.csv")
X = StandardScaler().fit_transform(df[["USD", "BTC", "Netflow_Bytes"]])
y = df["Prediction"]

train_sizes = [50, 100, 200, 500, 1000, 2000, 5000]

sizes, train_scores, test_scores = learning_curve(
    SVC(kernel="rbf", C=3000, gamma=100), X, y,
    train_sizes=[50, 100, 200, 500, 1000, 2000, 5000],
    cv=5, scoring="accuracy", random_state=0, shuffle=True
)

for n, test_acc in zip(sizes, test_scores.mean(axis=1)):
    print(f"n_train={n:5d}  mean_test_acc={test_acc:.4f}")