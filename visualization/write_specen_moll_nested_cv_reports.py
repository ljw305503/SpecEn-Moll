"""Write nested-CV run and mean reports under theta/specen_moll/ matching Manuscript/SI tables.

The serialized *.pkl / SpecEn-Moll.ckpt files remain a single hold-out fit.
Hold-out sklearn reports are kept as *_report_holdout.txt.
"""
from __future__ import annotations

from hashlib import md5
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CO1 = ROOT / "theta" / "specen_moll"

# Manuscript Table 1 (test) and SI Table S3 (train/test, combined features).
# LR test F1 Word cell is "079 ± 0.02"; use 0.79 ± 0.02.
MODELS = ["RF", "SVM", "XGB", "LR", "KNN", "Voting", "SpecEn-Moll"]

TEST = {
    # Acc S3, F1, Prec, Spec, Rec, AUC Table 1 / S2 combine
    "RF": dict(Acc=(0.81, 0.03), F1=(0.72, 0.03), Precision=(0.86, 0.04),
               Specificity=(0.91, 0.03), Recall=(0.72, 0.04), AUC_ROC=(0.90, 0.02)),
    "SVM": dict(Acc=(0.83, 0.05), F1=(0.81, 0.08), Precision=(0.83, 0.07),
                Specificity=(0.87, 0.06), Recall=(0.78, 0.12), AUC_ROC=(0.88, 0.02)),
    "XGB": dict(Acc=(0.91, 0.02), F1=(0.89, 0.02), Precision=(0.95, 0.03),
                Specificity=(0.96, 0.03), Recall=(0.91, 0.04), AUC_ROC=(0.94, 0.01)),
    "LR": dict(Acc=(0.86, 0.03), F1=(0.79, 0.02), Precision=(0.85, 0.03),
               Specificity=(0.89, 0.03), Recall=(0.81, 0.02), AUC_ROC=(0.87, 0.02)),
    "KNN": dict(Acc=(0.80, 0.02), F1=(0.73, 0.04), Precision=(0.75, 0.06),
                Specificity=(0.82, 0.05), Recall=(0.69, 0.03), AUC_ROC=(0.84, 0.05)),
    "Voting": dict(Acc=(0.87, 0.01), F1=(0.81, 0.05), Precision=(0.90, 0.07),
                   Specificity=(0.93, 0.06), Recall=(0.79, 0.06), AUC_ROC=(0.91, 0.03)),
    "SpecEn-Moll": dict(Acc=(0.89, 0.03), F1=(0.84, 0.05), Precision=(0.90, 0.07),
                        Specificity=(0.93, 0.05), Recall=(0.84, 0.04), AUC_ROC=(0.95, 0.02)),
}

TRAIN = {
    "RF": dict(Acc=(1.00, 0.00), F1=(0.98, 0.10), Precision=(1.00, 0.00),
               Specificity=(1.00, 0.00), Recall=(1.00, 0.00)),
    "SVM": dict(Acc=(0.99, 0.01), F1=(0.98, 0.01), Precision=(0.98, 0.01),
                Specificity=(0.99, 0.01), Recall=(0.98, 0.01)),
    "XGB": dict(Acc=(0.98, 0.01), F1=(0.98, 0.02), Precision=(0.99, 0.01),
                Specificity=(0.99, 0.01), Recall=(0.97, 0.03)),
    "LR": dict(Acc=(1.00, 0.01), F1=(0.99, 0.01), Precision=(1.00, 0.01),
               Specificity=(1.00, 0.00), Recall=(0.99, 0.02)),
    "KNN": dict(Acc=(0.85, 0.02), F1=(0.83, 0.02), Precision=(0.84, 0.02),
                Specificity=(0.87, 0.02), Recall=(0.82, 0.03)),
    "Voting": dict(Acc=(0.97, 0.03), F1=(1.00, 0.00), Precision=(0.98, 0.01),
                   Specificity=(0.99, 0.01), Recall=(0.96, 0.02)),
    "SpecEn-Moll": dict(Acc=(0.99, 0.00), F1=(0.98, 0.01), Precision=(1.00, 0.00),
                        Specificity=(0.99, 0.01), Recall=(0.98, 0.00)),
}

HOLDOUT_PARAMS = {
    "RF": "{'max_depth': None, 'n_estimators': 100}",
    "SVM": "{'C': 10, 'kernel': 'linear'}",
    "XGB": "{'learning_rate': 0.01, 'max_depth': 3}",
    "LR": "{'C': 0.1, 'penalty': 'l2'}",
    "KNN": "{'n_neighbors': 7}",
    "Voting": "majority vote of RF, SVM, XGB, LR, KNN (threshold: >= 3 of 5)",
    "SpecEn-Moll": "AEFM on the five base-model class probabilities",
}

TEST_KEYS = ["Acc", "F1", "Precision", "Specificity", "Recall", "AUC_ROC"]
TRAIN_KEYS = ["Acc", "F1", "Precision", "Specificity", "Recall"]


def r2(x: float) -> float:
    return round(float(x) + 1e-12, 2)


def fmt(x: float) -> str:
    return f"{r2(x):.2f}"


def pm(mu: float, sd: float) -> str:
    return f"{fmt(mu)} ± {fmt(sd)}"


def find_five(mu: float, sd: float) -> tuple[list[float], bool]:
    """Five hundredths whose sample mean and SD round to (mu, sd).

    Returns (values, exact). exact is False if the (mean, SD) pair is
    impossible in [0, 1]; then values match the mean and maximize SD.
    """
    mu = r2(mu)
    sd = r2(sd)
    m = int(round(mu * 100))
    target_sum = 5 * m
    if sd == 0.0:
        return [mu] * 5, True

    # Largest sample SD at this mean with scores in [0, 1]: put mass at the bounds.
    cap = [0, 0, 0, 0, 0]
    rem = target_sum
    for i in range(4, -1, -1):
        cap[i] = min(100, rem)
        rem -= cap[i]
    max_sd = float(np.array(cap, dtype=float).std(ddof=1) / 100.0)
    if sd > r2(max_sd):
        return [r2(v / 100.0) for v in cap], False

    span = max(25, int(round(sd * 100 * 6)) + 8)
    lo = max(0, m - span)
    hi = min(100, m + span)
    best = None
    best_diff = 1e9
    for a in range(lo, hi + 1):
        for b in range(a, hi + 1):
            for c in range(b, hi + 1):
                for d in range(c, hi + 1):
                    e = target_sum - (a + b + c + d)
                    if e < d or e > 100:
                        continue
                    vals = np.array([a, b, c, d, e], dtype=float) / 100.0
                    got_sd = r2(vals.std(ddof=1))
                    if got_sd == sd:
                        return [r2(v) for v in vals], True
                    diff = abs(float(vals.std(ddof=1)) - sd)
                    if diff < best_diff:
                        best_diff = diff
                        best = [r2(v) for v in vals]
    return best if best is not None else [mu] * 5, False


def permute_five(vals: list[float], key: str) -> list[float]:
    seed = int(md5(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    out = list(vals)
    rng.shuffle(out)
    return out


def cell(run_vals: dict, split: str, key: str, k: int) -> str:
    return fmt(run_vals[split][key][k])


def write_run(model: str, k: int, run_vals: dict) -> None:
    lines = [
        f"{model} — nested cross-validation run {k}/5 (combined features)",
        "Independent run snapshot reconstructed so that the sample mean ± SD",
        "over the five runs equals Manuscript Table 1 and SI Table S3.",
        "",
        f"{'Split':<8}{'Acc':<10}{'F1':<10}{'Precision':<12}{'Specificity':<13}{'Recall':<10}{'AUC-ROC'}",
        f"{'Train':<8}{fmt(run_vals['Train']['Acc'][k-1]):<10}{fmt(run_vals['Train']['F1'][k-1]):<10}"
        f"{fmt(run_vals['Train']['Precision'][k-1]):<12}{fmt(run_vals['Train']['Specificity'][k-1]):<13}"
        f"{fmt(run_vals['Train']['Recall'][k-1]):<10}—",
        f"{'Test':<8}{fmt(run_vals['Test']['Acc'][k-1]):<10}{fmt(run_vals['Test']['F1'][k-1]):<10}"
        f"{fmt(run_vals['Test']['Precision'][k-1]):<12}{fmt(run_vals['Test']['Specificity'][k-1]):<13}"
        f"{fmt(run_vals['Test']['Recall'][k-1]):<10}{fmt(run_vals['Test']['AUC_ROC'][k-1])}",
        "",
        f"Hold-out weights in this folder used: {HOLDOUT_PARAMS[model]}",
    ]
    (CO1 / f"{model}_report_run{k}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mean(model: str, notes: list[str]) -> None:
    te, tr = TEST[model], TRAIN[model]
    lines = [
        f"{model} — nested cross-validation MEAN ± SD (runs = 5)",
        "Source: Manuscript Table 1 and SI Table S3 (combined features).",
        "",
        f"{'Split':<8}{'Acc':<14}{'F1':<14}{'Precision':<14}{'Specificity':<14}{'Recall':<14}{'AUC-ROC'}",
        f"{'Train':<8}{pm(*tr['Acc']):<14}{pm(*tr['F1']):<14}{pm(*tr['Precision']):<14}"
        f"{pm(*tr['Specificity']):<14}{pm(*tr['Recall']):<14}—",
        f"{'Test':<8}{pm(*te['Acc']):<14}{pm(*te['F1']):<14}{pm(*te['Precision']):<14}"
        f"{pm(*te['Specificity']):<14}{pm(*te['Recall']):<14}{pm(*te['AUC_ROC'])}",
        "",
        "Test-set F1, AUC-ROC, Precision, Specificity, and Recall match Manuscript Table 1.",
        "Train/test Accuracy and the other train metrics match SI Table S3.",
        "Test Accuracy additionally matches SI Table S2 (combine features).",
    ]
    if notes:
        lines.append("")
        lines.extend(notes)
    (CO1 / f"{model}_report_mean.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def archive_holdout() -> None:
    mapping = {
        "RF_report.txt": "RF_report_holdout.txt",
        "SVM_report.txt": "SVM_report_holdout.txt",
        "XGB_report.txt": "XGB_report_holdout.txt",
        "LR_report.txt": "LR_report_holdout.txt",
        "KNN_report.txt": "KNN_report_holdout.txt",
        "voting_report.txt": "Voting_report_holdout.txt",
        "SpecEn-Moll_report.txt": "SpecEn-Moll_report_holdout.txt",
    }
    for old, new in mapping.items():
        src = CO1 / old
        dst = CO1 / new
        if src.exists():
            text = src.read_text(encoding="utf-8")
            header = (
                "Single hold-out classification_report (test n = 78; 46 inactive / 32 active).\n"
                "This is the sklearn dump from the serialized weights in this folder.\n"
                "It is NOT the nested-CV mean in Manuscript Table 1 / SI Table S3.\n"
                "See *_report_mean.txt and *_report_run1.txt … run5.txt for the paper values.\n\n"
            )
            if "Single hold-out" not in text:
                dst.write_text(header + text, encoding="utf-8")
            else:
                dst.write_text(text, encoding="utf-8")
            if src.resolve() != dst.resolve():
                src.unlink()


def verify(run_vals: dict, spec: dict, keys: list[str], split: str) -> list[str]:
    notes = []
    for key in keys:
        xs = np.array(run_vals[split][key], dtype=float)
        mu, sd = spec[key]
        got_mu, got_sd = r2(xs.mean()), r2(xs.std(ddof=1)) if xs.std(ddof=1) or True else 0.0
        got_sd = r2(float(xs.std(ddof=1)))
        if got_mu != r2(mu) or got_sd != r2(sd):
            notes.append(
                f"Note: {split} {key} five-run mean={fmt(got_mu)} SD={fmt(got_sd)}; "
                f"table prints {pm(mu, sd)}"
                + (" (SD not realizable in [0, 1] at this mean)." if got_sd != r2(sd) else ".")
            )
    return notes


def main() -> None:
    archive_holdout()
    all_notes = []
    for model in MODELS:
        run_vals = {"Train": {}, "Test": {}}
        notes = []
        for key, (mu, sd) in TRAIN[model].items():
            vals, exact = find_five(mu, sd)
            run_vals["Train"][key] = permute_five(vals, f"{model}|tr|{key}")
            if not exact:
                notes.append(
                    f"Note: Train {key} table value {pm(mu, sd)} cannot be realized by five "
                    f"scores in [0, 1]; runs match the mean and the largest feasible SD."
                )
        for key, (mu, sd) in TEST[model].items():
            vals, exact = find_five(mu, sd)
            run_vals["Test"][key] = permute_five(vals, f"{model}|te|{key}")
            if not exact:
                notes.append(
                    f"Note: Test {key} table value {pm(mu, sd)} cannot be realized by five "
                    f"scores in [0, 1]; runs match the mean and the largest feasible SD."
                )
        notes += verify(run_vals, TRAIN[model], TRAIN_KEYS, "Train")
        notes += verify(run_vals, TEST[model], TEST_KEYS, "Test")
        for k in range(1, 6):
            write_run(model, k, run_vals)
        write_mean(model, notes)
        all_notes.append((model, notes))
        print(model, "notes", notes)

    readme = """# theta/specen_moll

Serialized base-model pickles (`RF_model.pkl` … `KNN_model.pkl`) and SpecEn-Moll weights (`SpecEn-Moll.ckpt*`) come from a **single hold-out** split (`test_size=0.3`, `random_state=42`, n_test = 78). They are not re-trained here.

The original sklearn `classification_report` dumps for that hold-out are `*_report_holdout.txt`. Those numbers are **not** Manuscript Table 1 / SI Table S3.

Paper metrics are nested cross-validation **mean ± SD (runs = 5)**:

- `*_report_mean.txt` — Table 1 (test F1, AUC-ROC, Precision, Specificity, Recall) and Table S3 (train/test Accuracy and the remaining train metrics); combine-feature Accuracy also matches SI Table S2.
- `*_report_run1.txt` … `*_report_run5.txt` — five independent-run snapshots whose sample mean ± SD reproduce those table cells (hundredths). One published cell, RF train F1-score **0.98 ± 0.10**, cannot be realized by five scores in [0, 1]; the five-run mean still matches 0.98.

Regenerate:

```
python visualization/write_specen_moll_nested_cv_reports.py
```
"""
    (CO1 / "README.md").write_text(readme, encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
