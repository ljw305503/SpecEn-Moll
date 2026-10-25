"""Redraw results/base_models radars and reports from manuscript Table 1 / SI Table S3."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "base_models")
LABELS = ["Acc", "F1", "Precision", "Specificity", "Recall"]

HOLDOUT_PARAMS = {
    "RF": "{'max_depth': None, 'n_estimators': 100}",
    "SVM": "{'C': 10, 'kernel': 'linear'}",
    "XGB": "{'learning_rate': 0.01, 'max_depth': 3}",
    "LR": "{'C': 0.1, 'penalty': 'l2'}",
    "KNN": "{'n_neighbors': 7}",
}

DATA = {
    "RF": {
        "train": [1.00, 0.98, 1.00, 1.00, 1.00],
        "test": [0.81, 0.72, 0.86, 0.91, 0.72],
        "train_sd": [0.00, 0.01, 0.00, 0.00, 0.00],
        "test_sd": [0.03, 0.03, 0.04, 0.03, 0.04],
        "auc_test": (0.90, 0.02),
        "colors": {"train": "#7E1FA6", "test": "#2C004D"},
    },
    "SVM": {
        "train": [0.99, 0.98, 0.98, 0.99, 0.98],
        "test": [0.83, 0.81, 0.83, 0.87, 0.78],
        "train_sd": [0.01, 0.01, 0.01, 0.01, 0.01],
        "test_sd": [0.05, 0.08, 0.07, 0.06, 0.12],
        "auc_test": (0.88, 0.02),
        "colors": {"train": "#FB8C00", "test": "#E65100"},
    },
    "XGB": {
        "train": [0.98, 0.98, 0.99, 0.99, 0.97],
        "test": [0.91, 0.89, 0.95, 0.96, 0.91],
        "train_sd": [0.01, 0.02, 0.01, 0.01, 0.03],
        "test_sd": [0.02, 0.02, 0.03, 0.03, 0.04],
        "auc_test": (0.94, 0.01),
        "colors": {"train": "#43A047", "test": "#1B5E20"},
    },
    "LR": {
        "train": [1.00, 0.99, 1.00, 1.00, 0.99],
        "test": [0.86, 0.79, 0.85, 0.89, 0.81],
        "train_sd": [0.01, 0.01, 0.01, 0.00, 0.02],
        "test_sd": [0.03, 0.02, 0.03, 0.03, 0.02],
        "auc_test": (0.87, 0.02),
        "colors": {"train": "#1E88E5", "test": "#0D47A1"},
    },
    "KNN": {
        "train": [0.85, 0.83, 0.84, 0.87, 0.82],
        "test": [0.80, 0.73, 0.75, 0.82, 0.69],
        "train_sd": [0.02, 0.02, 0.02, 0.02, 0.03],
        "test_sd": [0.02, 0.04, 0.06, 0.05, 0.03],
        "auc_test": (0.84, 0.05),
        "colors": {"train": "#8E24AA", "test": "#4A148C"},
    },
}


def pm(mu, sd):
    return f"{mu:.2f} ± {sd:.2f}"


def write_report(name, rec, out_dir):
    lines = [
        f"{name} — nested cross-validation MEAN ± SD (runs = 5)",
        "",
        f"{'Split':<8}{'Acc':<14}{'F1':<14}{'Precision':<14}{'Specificity':<14}{'Recall':<14}{'AUC-ROC'}",
        f"{'Train':<8}{pm(rec['train'][0], rec['train_sd'][0]):<14}"
        f"{pm(rec['train'][1], rec['train_sd'][1]):<14}"
        f"{pm(rec['train'][2], rec['train_sd'][2]):<14}"
        f"{pm(rec['train'][3], rec['train_sd'][3]):<14}"
        f"{pm(rec['train'][4], rec['train_sd'][4]):<14}—",
        f"{'Test':<8}{pm(rec['test'][0], rec['test_sd'][0]):<14}"
        f"{pm(rec['test'][1], rec['test_sd'][1]):<14}"
        f"{pm(rec['test'][2], rec['test_sd'][2]):<14}"
        f"{pm(rec['test'][3], rec['test_sd'][3]):<14}"
        f"{pm(rec['test'][4], rec['test_sd'][4]):<14}{pm(*rec['auc_test'])}",
        "",
        f"Hold-out weights in this folder used: {HOLDOUT_PARAMS[name]}",
    ]
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def radar(name, rec):
    out_dir = os.path.join(ROOT, name)
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
    write_report(name, rec, out_dir)


if __name__ == "__main__":
    for name, rec in DATA.items():
        radar(name, rec)
    print("wrote", ROOT)
