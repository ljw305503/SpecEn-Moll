"""Write results/model_evaluation tables from manuscript Table 1 and SI Tables S2–S3."""
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "model_evaluation")
os.makedirs(OUT, exist_ok=True)

thin = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"), bottom=Side(style="thin"),
)
head_fill = PatternFill("solid", "D9E2F3")
note_font = Font(name="Times New Roman", size=10, italic=True)
body_font = Font(name="Times New Roman", size=11)
bold = Font(name="Times New Roman", size=11, bold=True)
title_font = Font(name="Times New Roman", size=12, bold=True)

LR_F1_NOTE = (
    "LR F1-score is transcribed as 0.79 ± 0.02 "
    '(the Word table cell is missing the decimal point and reads "079 ± 0.02").'
)


def apply_header(cell):
    cell.font = bold
    cell.fill = head_fill
    cell.border = thin
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def apply_body(cell):
    cell.font = body_font
    cell.border = thin
    cell.alignment = Alignment(horizontal="center", vertical="center")


def write_title(ws, text, merge, height=32):
    ws.merge_cells(merge)
    cell = ws[merge.split(":")[0]]
    cell.value = text
    cell.font = title_font
    cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[1].height = height


def write_note(ws, cell_ref, merge, text, height=40):
    ws.merge_cells(merge)
    cell = ws[cell_ref]
    cell.value = text
    cell.font = note_font
    cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[int(cell_ref[1:])].height = height


def main():
    for fn in os.listdir(OUT):
        if fn.startswith("code_single_split") or fn.startswith("Table"):
            path = os.path.join(OUT, fn)
            if os.path.isfile(path):
                os.remove(path)

    t1_rows = [
        ["Model", "F1-score", "AUC-ROC", "Precision", "Specificity", "Recall"],
        ["RF", "0.72 ± 0.03", "0.90 ± 0.02", "0.86 ± 0.04", "0.91 ± 0.03", "0.72 ± 0.04"],
        ["SVM", "0.81 ± 0.08", "0.88 ± 0.02", "0.83 ± 0.07", "0.87 ± 0.06", "0.78 ± 0.12"],
        ["XGB", "0.89 ± 0.02", "0.94 ± 0.01", "0.95 ± 0.03", "0.96 ± 0.03", "0.91 ± 0.04"],
        ["LR", "0.79 ± 0.02", "0.87 ± 0.02", "0.85 ± 0.03", "0.89 ± 0.03", "0.81 ± 0.02"],
        ["KNN", "0.73 ± 0.04", "0.84 ± 0.05", "0.75 ± 0.06", "0.82 ± 0.05", "0.69 ± 0.03"],
        ["Voting", "0.81 ± 0.05", "0.91 ± 0.03", "0.90 ± 0.07", "0.93 ± 0.06", "0.79 ± 0.06"],
        ["SpecEn-Moll", "0.84 ± 0.05", "0.95 ± 0.02", "0.90 ± 0.07", "0.93 ± 0.05", "0.84 ± 0.04"],
    ]
    pd.DataFrame(t1_rows[1:], columns=t1_rows[0]).to_csv(
        os.path.join(OUT, "Table1.csv"), index=False, encoding="utf-8-sig")
    wb = Workbook()
    ws = wb.active
    ws.title = "Table 1"
    write_title(ws, "Table 1. F1-scores, AUC-ROC, precision, specificity and recall values of the ML models from the test set.", "A1:F1")
    for i, row in enumerate(t1_rows, start=2):
        for j, v in enumerate(row, start=1):
            cell = ws.cell(i, j, v)
            apply_header(cell) if i == 2 else apply_body(cell)
    write_note(ws, "A10", "A10:F10",
               "Note: Values are presented as mean ± SD across repeated nested cross-validation runs (runs=5). " + LR_F1_NOTE,
               height=40)
    for col in "ABCDEF":
        ws.column_dimensions[col].width = 16
    wb.save(os.path.join(OUT, "Table1.xlsx"))

    s2_csv_cols = [
        "Model",
        "Textual_Accuracy", "Textual_AUC-ROC",
        "Spectral_clustering_Accuracy", "Spectral_clustering_AUC-ROC",
        "Combine_Accuracy", "Combine_AUC-ROC",
    ]
    s2_body = [
        ["RF", "0.80 ± 0.01", "0.90 ± 0.01", "0.76 ± 0.06", "0.72 ± 0.05", "0.81 ± 0.03", "0.90 ± 0.02"],
        ["SVM", "0.82 ± 0.06", "0.89 ± 0.02", "0.74 ± 0.03", "0.70 ± 0.05", "0.83 ± 0.05", "0.88 ± 0.02"],
        ["XGB", "0.89 ± 0.01", "0.92 ± 0.01", "0.81 ± 0.06", "0.70 ± 0.07", "0.91 ± 0.02", "0.94 ± 0.01"],
        ["LR", "0.83 ± 0.02", "0.88 ± 0.01", "0.75 ± 0.05", "0.69 ± 0.07", "0.86 ± 0.03", "0.87 ± 0.02"],
        ["KNN", "0.74 ± 0.04", "0.83 ± 0.03", "0.78 ± 0.05", "0.73 ± 0.06", "0.80 ± 0.02", "0.84 ± 0.05"],
        ["Voting", "0.85 ± 0.04", "0.91 ± 0.01", "0.72 ± 0.05", "0.75 ± 0.07", "0.87 ± 0.01", "0.91 ± 0.03"],
        ["SpecEn-Moll", "N/A", "N/A", "N/A", "N/A", "0.89 ± 0.03", "0.95 ± 0.02"],
    ]
    pd.DataFrame(s2_body, columns=s2_csv_cols).to_csv(
        os.path.join(OUT, "Table_S2.csv"), index=False, encoding="utf-8-sig")
    wb = Workbook()
    ws = wb.active
    ws.title = "Table S2"
    write_title(ws, "Table S2. Accuracy and AUC-ROC values of the ML models of different features from test set (n=5).", "A1:G1")
    ws.merge_cells("B2:C2")
    ws.merge_cells("D2:E2")
    ws.merge_cells("F2:G2")
    ws["A2"] = "Model"
    ws["B2"] = "Textual features"
    ws["D2"] = "Spectral clustering features"
    ws["F2"] = "Combine"
    for addr in ("A2", "B2", "D2", "F2"):
        apply_header(ws[addr])
    for col in range(1, 8):
        apply_header(ws.cell(2, col))
        apply_header(ws.cell(3, col))
    ws["B3"] = "Accuracy"
    ws["C3"] = "AUC-ROC"
    ws["D3"] = "Accuracy"
    ws["E3"] = "AUC-ROC"
    ws["F3"] = "Accuracy"
    ws["G3"] = "AUC-ROC"
    for i, row in enumerate(s2_body, start=4):
        for j, v in enumerate(row, start=1):
            apply_body(ws.cell(i, j, v))
    write_note(
        ws, "A11", "A11:G13",
        "Note: SpecEn-Moll denotes the Adaptive Ensemble Fusion Module (AEFM) trained exclusively on the combined "
        "feature set (two-dimensional spectral embedding concatenated with standardized textual descriptors). "
        "N/A indicates that SpecEn-Moll was not applicable to the textual-only or spectral-only feature sets, "
        "as it was defined and evaluated only for the combined feature configuration. Hard-voting ensemble "
        "results (Voting) are reported for all three feature settings for comparison.",
        height=30,
    )
    for col, w in zip("ABCDEFG", [16, 14, 14, 18, 14, 14, 14]):
        ws.column_dimensions[col].width = w
    wb.save(os.path.join(OUT, "Table_S2.xlsx"))

    s3_cols = ["Dataset", "Model", "Accuracy", "F1-score", "Precision", "Specificity", "Recall"]
    s3 = [
        ["Test", "RF", "0.81 ± 0.03", "0.72 ± 0.03", "0.86 ± 0.04", "0.91 ± 0.03", "0.72 ± 0.04"],
        ["Test", "SVM", "0.83 ± 0.05", "0.81 ± 0.08", "0.83 ± 0.07", "0.87 ± 0.06", "0.78 ± 0.12"],
        ["Test", "XGB", "0.91 ± 0.02", "0.89 ± 0.02", "0.95 ± 0.03", "0.96 ± 0.03", "0.91 ± 0.04"],
        ["Test", "LR", "0.86 ± 0.03", "0.79 ± 0.02", "0.85 ± 0.03", "0.89 ± 0.03", "0.81 ± 0.02"],
        ["Test", "KNN", "0.80 ± 0.02", "0.73 ± 0.04", "0.75 ± 0.06", "0.82 ± 0.05", "0.69 ± 0.03"],
        ["Test", "Voting", "0.87 ± 0.01", "0.81 ± 0.05", "0.90 ± 0.07", "0.93 ± 0.06", "0.79 ± 0.06"],
        ["Test", "SpecEn-Moll", "0.89 ± 0.03", "0.84 ± 0.05", "0.90 ± 0.07", "0.93 ± 0.05", "0.84 ± 0.04"],
        ["Train", "RF", "1.00 ± 0.00", "0.98 ± 0.01", "1.00 ± 0.00", "1.00 ± 0.00", "1.00 ± 0.00"],
        ["Train", "SVM", "0.99 ± 0.01", "0.98 ± 0.01", "0.98 ± 0.01", "0.99 ± 0.01", "0.98 ± 0.01"],
        ["Train", "XGB", "0.98 ± 0.01", "0.98 ± 0.02", "0.99 ± 0.01", "0.99 ± 0.01", "0.97 ± 0.03"],
        ["Train", "LR", "1.00 ± 0.01", "0.99 ± 0.01", "1.00 ± 0.01", "1.00 ± 0.00", "0.99 ± 0.02"],
        ["Train", "KNN", "0.85 ± 0.02", "0.83 ± 0.02", "0.84 ± 0.02", "0.87 ± 0.02", "0.82 ± 0.03"],
        ["Train", "Voting", "0.97 ± 0.03", "1.00 ± 0.00", "0.98 ± 0.01", "0.99 ± 0.01", "0.96 ± 0.02"],
        ["Train", "SpecEn-Moll", "0.99 ± 0.00", "0.98 ± 0.01", "1.00 ± 0.00", "0.99 ± 0.01", "0.98 ± 0.00"],
    ]
    pd.DataFrame(s3, columns=s3_cols).to_csv(
        os.path.join(OUT, "Table_S3.csv"), index=False, encoding="utf-8-sig")
    wb = Workbook()
    ws = wb.active
    ws.title = "Table S3"
    write_title(
        ws,
        "Table S3. F1-scores, AUC-ROC, precision, specificity and recall values of the ML models from test and train set (combined features, n=5).",
        "A1:G1",
        height=36,
    )
    for j, h in enumerate(s3_cols, 1):
        apply_header(ws.cell(2, j, h))
    for i, row in enumerate(s3, start=3):
        for j, v in enumerate(row, 1):
            apply_body(ws.cell(i, j, v))
    write_note(
        ws, "A17", "A17:G17",
        "Columns follow the SI table body (Accuracy, F1-score, Precision, Specificity, Recall). "
        "The SI caption also mentions AUC-ROC; test-set AUC-ROC is reported in Table 1 and Table S2. "
        + LR_F1_NOTE,
        height=44,
    )
    for col, w in zip("ABCDEFG", [12, 16, 16, 16, 16, 16, 16]):
        ws.column_dimensions[col].width = w
    wb.save(os.path.join(OUT, "Table_S3.xlsx"))
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
