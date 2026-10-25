# SpecEn-Moll

Code and data for:

**SpecEn-Moll: a spectral-clustering-enhanced ensemble framework for data-constrained molluscicide discovery with ecotoxicology-informed molecular optimization**

SpecEn-Moll is a binary classifier (active / inactive) for molluscicidal compounds. It concatenates RDKit textual descriptors with two-dimensional spectral embeddings of molecular graphs and combines five base learners—random forest (RF), support vector machine (SVM), XGBoost (XGB), logistic regression (LR), and *k*-nearest neighbors (KNN)—through an adaptive ensemble fusion module (AEFM). Majority voting over the five learners is included as a non-learned ensemble baseline.

The AEFM is implemented as `models.specen_moll.SpecEnMoll` (hyphens are not valid in a Python class name). Display name in tables and figures: **SpecEn-Moll**.

## Repository layout

```
dataset/                      modeling compounds, hold-out split, PubChem list
models/                       RF/SVM/XGB/LR/KNN factories; SpecEnMoll (AEFM)
utils/                        descriptors, graphs, spectral embedding, preprocessing, SAR scripts
visualization/                comparison, radar, and table writers
theta/specen_moll/            serialized classifiers and nested-CV reports
theta/clustering/             KMeans and spectral clustering pickles
results/
  model_evaluation/           Table 1, Table S2, Table S3
  base_models/                per-model confusion matrices and radar plots
  ensemble/voting/            voting nested-CV metrics and radar
  ensemble/SpecEn-Moll/       SpecEn-Moll nested-CV metrics and radar
  comparison/                 nested-CV metric bar charts (Figure 4C / Tables S2–S3)
  comparison_holdout/         hold-out charts written by train.py
  prediction/                 Additional file 7 virtual screening
  SAR/  safe/  compound Bz1-Bz9/
train.py                      GridSearchCV base models and majority voting
train_specen_moll.py          AEFM training on stacked class probabilities
test.py                       apply saved models to dataset/test
prediction_experiment.py      unlabeled PubChem screening
models_test.py                hold-out evaluation of the five base models
setup.py                      optional Cython build of models.other_models
```

## Dataset

Canonical modeling table: `dataset/all compounds analysis.xlsx` (262 compounds; columns `Number`, `smiles`, `label`).

After `dropna()` on descriptor tables, 260 molecules remain. The modeling split is `sklearn.model_selection.train_test_split(..., test_size=0.3, random_state=42)`:

| Split | Path | Rows | Labels |
|---|---|---|---|
| Train | `dataset/train/` | 182 | see `dataset/split_manifest.csv` |
| Test (hold-out) | `dataset/test/` | 78 | 46 inactive / 32 active |

Each of `train/` and `test/` contains:

- `all compounds analysis.xlsx` — SMILES and activity (`lable`; historical spelling used by the training code)
- `text_feature.xlsx` — RDKit formula / InChI / descriptors / fingerprint strings
- `pc-gnn.pt` — PyTorch Geometric molecular graphs, row-aligned with the tables

Unlabeled screening list: `dataset/prediction/Additional file 7.csv`.

Species-level SAR tables in `dataset/` (`Oncomelania hupensis`, `Pomacea canaliculata`, `Deroceras reticulatum`, `Cornu aspersum`).

## Features

1. `utils/extract_rdkit_descriptors.py` (and `utils/pre-com.py`) — RDKit descriptors and fingerprints from SMILES → `text_feature.xlsx`
2. `utils/pc_gnn.py` — atom/bond graphs → `pc-gnn.pt`
3. `utils/spectral_embedding.py` — 2-D spectral embedding (`k=2`) of a similarity graph built from enhanced molecular features
4. `utils/data_preprocessing.py` — concatenate text features with `feature_0` / `feature_1`; `dropna`; `LabelEncoder` on object columns (unseen values → `-1`); `StandardScaler` on text columns fit on **train only**. Graph embedding columns are not scaled.

RF and XGB use the unscaled concatenated matrix. SVM, LR, and KNN use the scaled matrix.

## Models and weights

| Artifact | Location |
|---|---|
| RF, SVM, XGB, LR, KNN (`GridSearchCV` wrappers) | `theta/specen_moll/{RF,SVM,XGB,LR,KNN}_model.pkl` |
| SpecEn-Moll (TensorFlow checkpoint) | `theta/specen_moll/SpecEn-Moll.ckpt*` |
| Majority voting | implemented in `train.py` / `test.py` / `prediction_experiment.py` (no pickle); vote threshold ≥ 3 of 5 in `train.py`, ≥ 4 of 5 in `prediction_experiment.py` |
| Nested-CV mean ± SD reports (n = 5) | `theta/specen_moll/*_report_mean.txt` and `*_report_run1.txt` … `run5.txt` |
| Hold-out `classification_report` | `theta/specen_moll/*_report_holdout.txt` |
| KMeans / spectral clustering | `theta/clustering/{kmeans_clustering,spectral_clustering}.pkl` |

`train.py` and `train_specen_moll.py` fit on `dataset/train` and evaluate on `dataset/test` (the 182 / 78 hold-out). Manuscript and SI tables report **nested cross-validation** (mean ± SD, five independent runs) on combined features. Both sets of numbers are stored in the repository:

- Nested CV: `results/model_evaluation/`, `results/comparison/`, `results/ensemble/`, `theta/specen_moll/*_report_mean.txt`
- Hold-out: `*_report_holdout.txt`, `results/comparison_holdout/` (if `train.py` is executed)

## Nested-CV test-set metrics (Manuscript Table 1 / SI Tables S2–S3)

Mean ± SD, combined features, n = 5.

| Model | Accuracy | F1-score | AUC-ROC | Precision | Specificity | Recall |
|---|---|---|---|---|---|---|
| RF | 0.81 ± 0.03 | 0.72 ± 0.03 | 0.90 ± 0.02 | 0.86 ± 0.04 | 0.91 ± 0.03 | 0.72 ± 0.04 |
| SVM | 0.83 ± 0.05 | 0.81 ± 0.08 | 0.88 ± 0.02 | 0.83 ± 0.07 | 0.87 ± 0.06 | 0.78 ± 0.12 |
| XGB | 0.91 ± 0.02 | 0.89 ± 0.02 | 0.94 ± 0.01 | 0.95 ± 0.03 | 0.96 ± 0.03 | 0.91 ± 0.04 |
| LR | 0.86 ± 0.03 | 0.79 ± 0.02 | 0.87 ± 0.02 | 0.85 ± 0.03 | 0.89 ± 0.03 | 0.81 ± 0.02 |
| KNN | 0.80 ± 0.02 | 0.73 ± 0.04 | 0.84 ± 0.05 | 0.75 ± 0.06 | 0.82 ± 0.05 | 0.69 ± 0.03 |
| Voting | 0.87 ± 0.01 | 0.81 ± 0.05 | 0.91 ± 0.03 | 0.90 ± 0.07 | 0.93 ± 0.06 | 0.79 ± 0.06 |
| SpecEn-Moll | 0.89 ± 0.03 | 0.84 ± 0.05 | 0.95 ± 0.02 | 0.90 ± 0.07 | 0.93 ± 0.05 | 0.84 ± 0.04 |

LR test F1-score is **0.79 ± 0.02**. The Word cell in Table 1 / Table S3 is missing the decimal point (`079 ± 0.02`). SI Table S3 caption mentions AUC-ROC; the table body columns are Accuracy, F1-score, Precision, Specificity, and Recall. SpecEn-Moll is defined on the combined feature set only (N/A on textual-only and spectral-only columns in Table S2).

Train-set nested-CV metrics are in `results/model_evaluation/Table_S3.*`.

## Scripts

| Command | Role |
|---|---|
| `python train.py` | GridSearchCV on the five base models; majority voting; writes pickles under `theta/specen_moll/` and figures under `results/base_models/` and `results/comparison_holdout/` |
| `python train_specen_moll.py` | Trains SpecEnMoll on stacked base-model probabilities; writes `SpecEn-Moll.ckpt*` |
| `python test.py` | Loads saved weights; predicts `dataset/test` |
| `python prediction_experiment.py` | Screens `dataset/prediction/Additional file 7.csv`; writes `results/prediction/` |
| `python visualization/plot_comparison_manuscript.py` | Nested-CV comparison charts → `results/comparison/` |
| `python visualization/plot_ensemble_manuscript.py` | Nested-CV radar plots → `results/ensemble/` |
| `python visualization/write_model_evaluation_tables.py` | Table 1 / S2 / S3 → `results/model_evaluation/` |
| `python visualization/write_specen_moll_nested_cv_reports.py` | Mean and five-run reports → `theta/specen_moll/` |
| `python setup.py build_ext --inplace` | Optional Cython compile of `models.other_models` |

SAR / ECOTOX / Bz-series analyses: `utils/sar_*.py`, `utils/ecotox_substituent_clustering.py`, `utils/bz_series_activity_toxicity.py`. Inputs are under `dataset/`; outputs under `results/SAR/`, `results/safe/`, and `results/compound Bz1-Bz9/`.

## Environment

Python 3.8–3.11:

```
pip install -r requirements.txt
```

`requirements.txt`: numpy, pandas, scikit-learn, matplotlib, seaborn, rdkit, torch, torch-geometric, networkx, joblib, xgboost, tensorflow, openpyxl.

## Citation

Please cite the SpecEn-Moll article together with this repository.
