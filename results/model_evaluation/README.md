# Model evaluation tables

This folder is a machine-readable copy of the **manuscript and SI tables**. Numbers are nested cross-validation mean ± SD (runs = 5).

| File | Source |
|---|---|
| `Table1.csv` / `Table1.xlsx` | Manuscript Table 1 (test set) |
| `Table_S2.csv` / `Table_S2.xlsx` | SI Table S2 (textual / spectral clustering / combine features, test set) |
| `Table_S3.csv` / `Table_S3.xlsx` | SI Table S3 (combined features, train and test) |

Captions and notes are stored in the Excel sheets and match the Word files, with one transcription correction:

- LR test F1-score is **0.79 ± 0.02**. The Word cell is missing the decimal point (`079 ± 0.02`) in both Table 1 and Table S3.

SI Table S3’s caption mentions AUC-ROC, but the table body columns are Accuracy, F1-score, Precision, Specificity, and Recall. Test-set AUC-ROC is in Table 1 and Table S2 (SpecEn-Moll **0.95 ± 0.02**).

SpecEn-Moll is reported only on the combined feature set (N/A on textual-only and spectral-only columns in Table S2).

Regenerate:

```
python visualization/write_model_evaluation_tables.py
```
