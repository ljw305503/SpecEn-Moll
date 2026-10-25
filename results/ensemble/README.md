# Ensemble results (manuscript / SI)

Voting is the majority-vote baseline (≥3 of 5 base models positive). SpecEn-Moll is the adaptive ensemble fusion module (AEFM).

Radar axes and reports use **nested CV means** from SI Table S3 (combined features, runs = 5). Test AUC-ROC is from manuscript Table 1 / SI Table S2.

| Folder | Test Acc | Test F1 | Test AUC-ROC |
|---|---|---|---|
| `voting/` | 0.87 ± 0.01 | 0.81 ± 0.05 | 0.91 ± 0.03 |
| `SpecEn-Moll/` | 0.89 ± 0.03 | 0.84 ± 0.05 | 0.95 ± 0.02 |

Single-split hold-out ROC arrays previously stored here did not match the nested-CV AUC in the paper and were removed. `theta/specen_moll/SpecEn-Moll.ckpt` remains the uploaded weight file.

Regenerate:

```
python visualization/plot_ensemble_manuscript.py
```
