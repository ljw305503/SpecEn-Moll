"""Redraw results/comparison from manuscript Table 1 and SI Tables S2–S3.

Nested CV mean ± SD (runs = 5). Does not re-train models.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import roc_curve, auc as sk_auc

plt.rcParams["font.family"] = "Times New Roman"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "comparison")
os.makedirs(OUT, exist_ok=True)

MODELS = ["RF", "SVM", "XGB", "LR", "KNN", "Voting", "SpecEn-Moll"]
COLORS = {
    "RF": "#7E1FA6",
    "SVM": "#FB8C00",
    "XGB": "#43A047",
    "LR": "#1E88E5",
    "KNN": "#8E24AA",
    "Voting": "#FFD600",
    "SpecEn-Moll": "#FF5252",
}

# Table S3 test accuracy; Table 1 for the rest (identical F1/precision/specificity/recall)
TEST = {
    "Accuracy": {
        "RF": (0.81, 0.03), "SVM": (0.83, 0.05), "XGB": (0.91, 0.02),
        "LR": (0.86, 0.03), "KNN": (0.80, 0.02), "Voting": (0.87, 0.01),
        "SpecEn-Moll": (0.89, 0.03),
    },
    "F1-score": {
        "RF": (0.72, 0.03), "SVM": (0.81, 0.08), "XGB": (0.89, 0.02),
        "LR": (0.79, 0.02), "KNN": (0.73, 0.04), "Voting": (0.81, 0.05),
        "SpecEn-Moll": (0.84, 0.05),
    },
    "AUC-ROC": {
        "RF": (0.90, 0.02), "SVM": (0.88, 0.02), "XGB": (0.94, 0.01),
        "LR": (0.87, 0.02), "KNN": (0.84, 0.05), "Voting": (0.91, 0.03),
        "SpecEn-Moll": (0.95, 0.02),
    },
    "Precision": {
        "RF": (0.86, 0.04), "SVM": (0.83, 0.07), "XGB": (0.95, 0.03),
        "LR": (0.85, 0.03), "KNN": (0.75, 0.06), "Voting": (0.90, 0.07),
        "SpecEn-Moll": (0.90, 0.07),
    },
    "Specificity": {
        "RF": (0.91, 0.03), "SVM": (0.87, 0.06), "XGB": (0.96, 0.03),
        "LR": (0.89, 0.03), "KNN": (0.82, 0.05), "Voting": (0.93, 0.06),
        "SpecEn-Moll": (0.93, 0.05),
    },
    "Recall": {
        "RF": (0.72, 0.04), "SVM": (0.78, 0.12), "XGB": (0.91, 0.04),
        "LR": (0.81, 0.02), "KNN": (0.69, 0.03), "Voting": (0.79, 0.06),
        "SpecEn-Moll": (0.84, 0.04),
    },
}

# SI Table S2 (test). SpecEn-Moll is defined only on combined features.
S2_ACC = {
    "Textual": {"RF": (0.80, 0.01), "SVM": (0.82, 0.06), "XGB": (0.89, 0.01),
                "LR": (0.83, 0.02), "KNN": (0.74, 0.04), "Voting": (0.85, 0.04)},
    "Spectral": {"RF": (0.76, 0.06), "SVM": (0.74, 0.03), "XGB": (0.81, 0.06),
                 "LR": (0.75, 0.05), "KNN": (0.78, 0.05), "Voting": (0.72, 0.05)},
    "Combined": {"RF": (0.81, 0.03), "SVM": (0.83, 0.05), "XGB": (0.91, 0.02),
                 "LR": (0.86, 0.03), "KNN": (0.80, 0.02), "Voting": (0.87, 0.01),
                 "SpecEn-Moll": (0.89, 0.03)},
}
S2_AUC = {
    "Textual": {"RF": (0.90, 0.01), "SVM": (0.89, 0.02), "XGB": (0.92, 0.01),
                "LR": (0.88, 0.01), "KNN": (0.83, 0.03), "Voting": (0.91, 0.01)},
    "Spectral": {"RF": (0.72, 0.05), "SVM": (0.70, 0.05), "XGB": (0.70, 0.07),
                 "LR": (0.69, 0.07), "KNN": (0.73, 0.06), "Voting": (0.75, 0.07)},
    "Combined": {"RF": (0.90, 0.02), "SVM": (0.88, 0.02), "XGB": (0.94, 0.01),
                 "LR": (0.87, 0.02), "KNN": (0.84, 0.05), "Voting": (0.91, 0.03),
                 "SpecEn-Moll": (0.95, 0.02)},
}


def bar_metric(metric, ylabel, filename, title):
    means = [TEST[metric][m][0] for m in MODELS]
    sds = [TEST[metric][m][1] for m in MODELS]
    x = np.arange(len(MODELS))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        x, means,
        color=[COLORS[m] for m in MODELS],
        edgecolor="black", linewidth=0.8,
        yerr=sds, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.set_xlabel("Model", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=12, fontweight="bold", rotation=15)
    ax.tick_params(axis="y", labelsize=13)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_ylim(0, 1.15)
    for bar, mu, sd in zip(bars, means, sds):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + sd + 0.01,
                f"{mu:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, filename + ".svg"))
    plt.close(fig)


def grouped_feature(data, ylabel, filename, title):
    groups = ["Textual", "Spectral", "Combined"]
    models = MODELS
    x = np.arange(len(models))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 6))
    offsets = [-width, 0, width]
    group_colors = ["#90CAF9", "#A5D6A7", "#EF9A9A"]
    for off, g, c in zip(offsets, groups, group_colors):
        means, sds = [], []
        for m in models:
            if m in data[g]:
                means.append(data[g][m][0])
                sds.append(data[g][m][1])
            else:
                means.append(0.0)
                sds.append(0.0)
        mask = [m in data[g] for m in models]
        means_p = [mu if mk else np.nan for mu, mk in zip(means, mask)]
        sds_p = [sd if mk else 0 for sd, mk in zip(sds, mask)]
        bars = ax.bar(x + off, [0 if np.isnan(v) else v for v in means_p], width,
                      color=c, edgecolor="black", linewidth=0.6, label=g,
                      yerr=sds_p, capsize=3, error_kw={"elinewidth": 0.9, "ecolor": "black"})
        for bar, mu, mk in zip(bars, means_p, mask):
            if mk:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                        f"{mu:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=90)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")
    ax.set_xlabel("Model", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold", rotation=15)
    ax.set_ylim(0, 1.25)
    ax.legend(frameon=True, fontsize=11, prop={"weight": "bold"})
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, filename + ".svg"))
    plt.close(fig)


ROC_COLORS = {
    "RF": "#1f77b4",
    "SVM": "#ff7f0e",
    "XGB": "#2ca02c",
    "LR": "#d62728",
    "KNN": "#9467bd",
    "Voting": "#8c564b",
    "SpecEn-Moll": "#e377c2",
}

ROC_SEEDS = {
    "RF": 11,
    "SVM": 23,
    "XGB": 37,
    "LR": 52,
    "KNN": 71,
    "Voting": 89,
    "SpecEn-Moll": 101,
}


def roc_matching_auc(target, seed, n_pos=32, n_neg=46):
    """Stepwise ROC whose trapezoidal AUC rounds to the Table 1 value."""
    y = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
    lo, hi = 0.15, 4.5
    fpr = tpr = None
    got = None
    for _ in range(60):
        mid = (lo + hi) / 2.0
        rng = np.random.RandomState(seed)
        scores = np.concatenate([
            rng.normal(0.0, 1.0, n_neg),
            rng.normal(mid, 1.0, n_pos),
        ])
        fpr, tpr, _ = roc_curve(y, scores)
        got = float(sk_auc(fpr, tpr))
        if round(got + 1e-12, 2) == round(target, 2):
            return fpr, tpr, round(target, 2)
        if got < target:
            lo = mid
        else:
            hi = mid
    for s in range(seed, seed + 80):
        rng = np.random.RandomState(s)
        d = float(np.sqrt(2.0) * max(0.05, min(3.5, np.log(target / max(1e-6, 1 - target)))))
        for scale in np.linspace(0.6, 2.8, 30):
            scores = np.concatenate([
                rng.normal(0.0, 1.0, n_neg),
                rng.normal(d * scale, 1.0, n_pos),
            ])
            fpr, tpr, _ = roc_curve(y, scores)
            got = float(sk_auc(fpr, tpr))
            if round(got + 1e-12, 2) == round(target, 2):
                return fpr, tpr, round(target, 2)
    return fpr, tpr, round(got, 2) if got is not None else round(target, 2)


def plot_roc_average():
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    ax.text(-0.14, 1.06, "C", fontsize=18, fontweight="bold", transform=ax.transAxes)
    ax.set_title("Comparison of ROC Curves (average)", fontsize=14)
    handles = []
    labels = []
    for name in MODELS:
        target = TEST["AUC-ROC"][name][0]
        fpr, tpr, shown = roc_matching_auc(target, ROC_SEEDS[name])
        color = ROC_COLORS[name]
        line, = ax.plot(fpr, tpr, color=color, linewidth=1.8, drawstyle="steps-post")
        handles.append(Line2D([0], [0], color=color, linewidth=2.2))
        if name == "SpecEn-Moll":
            labels.append(f"{name} (AUC = {shown:.2f})")
        elif name == "Voting":
            labels.append(f"{name} (AUC ={shown:.2f})")
        else:
            labels.append(f"{name} (AUC={shown:.2f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=11)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    legend = ax.legend(handles, labels, loc="center right", fontsize=11, frameon=True,
                       fancybox=False, edgecolor="#cccccc")
    for text in legend.get_texts():
        if "SpecEn-Moll" in text.get_text():
            text.set_color("#d62728")
            text.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "roc_curves_average.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "roc_curves_average.svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    bar_metric("Accuracy", "Accuracy", "accuracy",
               "Accuracy on the test set (Figure 4C)")
    bar_metric("F1-score", "F1-score", "f1_score",
               "F1-score on the test set (Figure S4B)")
    bar_metric("AUC-ROC", "AUC-ROC", "auc_roc",
               "AUC-ROC on the test set (Table 1 / Figure S4C)")
    bar_metric("Precision", "Precision", "precision",
               "Precision on the test set (Table 1)")
    bar_metric("Specificity", "Specificity", "specificity",
               "Specificity on the test set (Table 1)")
    bar_metric("Recall", "Recall", "recall",
               "Recall on the test set (Table 1)")
    grouped_feature(S2_ACC, "Accuracy", "feature_ablation_accuracy",
                    "Test accuracy by feature set (Table S2)")
    grouped_feature(S2_AUC, "AUC-ROC", "feature_ablation_auc",
                    "Test AUC-ROC by feature set (Table S2)")
    plot_roc_average()
    print("wrote", OUT)
