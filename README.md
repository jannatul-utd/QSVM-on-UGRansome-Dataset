# Implementing QSVM to Classify the UGRansome Dataset

Quantum SVM pipeline on the UGRansome1819 dataset: EDA → refine → feature map →
circuit/depth analysis → quantum kernel + QSVM (simulator, then one confirming
run on real IBM hardware) → classical SVM baseline → comparison.

Full roadmap / checklist (kept in sync as we work): https://claude.ai/artifact/SFSfufxw6ZGsisQtKib1Yh

## Data

Source: [UGRansome Dataset on Kaggle](https://www.kaggle.com/datasets/nkongolo/ugransome-dataset)
(not committed to this repo — see `.gitignore` — download `final(2).csv` from Kaggle
and place it at `data/raw/ugransome.csv` to reproduce).

149,043 rows x 14 columns. Target: `Prediction` (3 classes: S, A, SS).

## Structure

```
data/raw/         raw CSV (gitignored)
data/processed/   cleaned / subsampled data (gitignored, regenerate via src/)
notebooks/        exploratory notebooks
src/              pipeline scripts (eda, preprocessing, feature map, kernel, models)
results/          plots, metrics, saved kernel matrices
reference/        prior exploration notebook (LIDETA_reference.ipynb) kept for
                  reference on known data quirks -- not used directly in the pipeline
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Status

See the roadmap doc linked above for current progress.
