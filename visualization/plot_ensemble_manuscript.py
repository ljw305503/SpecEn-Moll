"""Redraw results/ensemble radars from SI Table S3 (nested CV mean, n=5)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ensemble")

# Acc, F1, Precision, Specificity, Recall — SI Table S3 means
DATA = {
    "Voting": {
        "train": [0.97, 1.00, 0.98, 0.99, 0.96],
        "test": [0.87, 0.81, 0.90, 0.93, 0.79],
        "train_sd": [0.03, 0.00, 0.01, 0.01, 0.02],
        "test_sd": [0.01, 0.05, 0.07, 0.06, 0.06],
        "auc_test": (0.91, 0.03),
        "folder": "voting",
        "colors": {"train": "#FFD600", "test": "#F57F17"},
    },
    "SpecEn-Moll": {
        "train": [0.99, 0.98, 1.00, 0.99, 0.98],
        "test": [0.89, 0.84, 0.90, 0.93, 0.84],
        "train_sd": [0.00, 0.01, 0.00, 0.01, 0.00],
        "test_sd": [0.03, 0.05, 0.07, 0.05, 0.04],
        "auc_test": (0.95, 0.02),
        "folder": "SpecEn-Moll",
        "colors": {"train": "#FF5252", "test": "#B71C1C"},
    },
}
LABELS = ["Acc", "F1", "Precision", "Specificity", "Recall"]


def write_tables(name, rec, out_dir):
    rows = []
    for split, key, sdkey in (("Train", "train", "train_sd"), ("Test", "test", "test_sd")):
        for lab, mu, sd in zip(LABELS, rec[key], rec[sdkey]):
            rows.append({"Model": name, "Split": split, "Metric": lab, "Mean": mu, "SD": sd})
    rows.append({"Model": name, "Split": "Test", "Metric": "AUC-ROC",
                 "Mean": rec["auc_test"][0], "SD": rec["auc_test"][1]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "metrics_nested_CV.csv"), index=False)
    df.to_excel(os.path.join(out_dir, "metrics_nested_CV.xlsx"), index=False)
    lines = [
        f"{name} — nested cross-validation (mean ± SD, runs = 5)",
        "Source: manuscript Table 1 and SI Table S3 (combined features).",
        "",
        f"{'Split':<8}{'Acc':<14}{'F1':<14}{'Precision':<14}{'Specificity':<14}{'Recall':<14}{'AUC-ROC':<14}",
    ]
    def fmt(mus, sds, auc=None):
        cells = [f"{m:.2f} ± {s:.2f}" for m, s in zip(mus, sds)]
        cells.append(f"{auc[0]:.2f} ± {auc[1]:.2f}" if auc else "—")
        return cells
    tr = fmt(rec["train"], rec["train_sd"])
    te = fmt(rec["test"], rec["test_sd"], rec["auc_test"])
    lines.append(f"{'Train':<8}" + "".join(f"{c:<14}" for c in tr))
    lines.append(f"{'Test':<8}" + "".join(f"{c:<14}" for c in te))
    lines.append("")
    if name == "Voting":
        lines.append("Majority voting baseline: a compound is active when at least three")
        lines.append("of the five base classifiers (RF, SVM, XGB, LR, KNN) predict positive.")
    else:
        lines.append("SpecEn-Moll is the adaptive ensemble fusion module (AEFM) trained on")
        lines.append("combined textual descriptors and two-dimensional spectral embeddings.")
        lines.append("Test accuracy 0.89 ± 0.03; AUC-ROC 0.95 ± 0.02 (SI Table S2/S3).")
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def radar(name, rec):
    out_dir = os.path.join(ROOT, rec["folder"])
    os.makedirs(out_dir, exist_ok=True)
    train = np.array(rec["train"], dtype=float)
    test = np.array(rec["test"], dtype=float)
    n = len(LABELS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    train_c = np.concatenate((train, [train[0]]))
    test_c = np.concatenate((test, [test[0]]))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    tc, vc = rec["colors"]["train"], rec["colors"]["test"]
    ax.fill(angles, train_c, color=tc, alpha=0.2, label="Train set")
    ax.fill(angles, test_c, color=vc, alpha=0.2, label="Test set")
    ax.plot(angles, train_c, color=tc, linewidth=2.5, linestyle="-", marker="o",
            markersize=10, markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(angles, test_c, color=vc, linewidth=2.5, linestyle="--", marker="s",
            markersize=10, markeredgecolor="white", markeredgewidth=1.5)
    for i in range(n):
        ang = angles[i]
        tv, ev = train[i], test[i]
        if abs(tv - ev) < 0.05:
            ax.text(ang, (tv + ev) / 2, f"T:{tv:.2f}\nV:{ev:.2f}",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="gray", alpha=0.8))
        else:
            ax.text(ang - 0.08, tv, f"{tv:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=tc,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=tc, alpha=0.8))
            ax.text(ang + 0.08, ev, f"{ev:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=vc,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=vc, alpha=0.8))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(LABELS, fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 1.08)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=10)
    ax.grid(True, linestyle="--", linewidth=1.2, alpha=1)
    ax.set_title(f"{name}\nnested cross-validation (mean, n=5)",
                 fontsize=16, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=13,
              prop={"weight": "bold"}, framealpha=0.9, edgecolor="gray")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "radar.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "radar.svg"), bbox_inches="tight")
    plt.close(fig)
    write_tables(name, rec, out_dir)


if __name__ == "__main__":
    for name, rec in DATA.items():
        radar(name, rec)
    print("wrote", ROOT)
