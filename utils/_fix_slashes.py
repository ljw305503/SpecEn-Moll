from pathlib import Path

DM = Path(__file__).resolve().parent

subs_der = [
    (
        'input_file = r"results/SAR/Deroceras reticulatum/Deroceras reticulatum analysis data.xlsx"',
        'input_file = str(DATASET_DIR / "Deroceras reticulatum compounds analysis data.xlsx")',
    ),
    (
        'output_file = r"results/SAR/Deroceras reticulatum/Deroceras reticulatum analysis data1.xlsx"',
        'output_file = str(RESULTS_DIR / "SAR" / "Deroceras reticulatum" / "Deroceras reticulatum analysis data1.xlsx")',
    ),
    (
        'plt.savefig(r"results/SAR/Deroceras reticulatum/clustering_analysis.png")',
        'plt.savefig(str(RESULTS_DIR / "SAR" / "Deroceras reticulatum" / "clustering_analysis.png"))',
    ),
    (
        'plt.savefig(r"results/SAR/Deroceras reticulatum/clustering_analysis.svg")',
        'plt.savefig(str(RESULTS_DIR / "SAR" / "Deroceras reticulatum" / "clustering_analysis.svg"))',
    ),
    (
        'plt.savefig(r"results/SAR/Deroceras reticulatum/dendrogram.png")',
        'plt.savefig(str(RESULTS_DIR / "SAR" / "Deroceras reticulatum" / "dendrogram.png"))',
    ),
    (
        'plt.savefig(r"results/SAR/Deroceras reticulatum/dendrogram.svg")',
        'plt.savefig(str(RESULTS_DIR / "SAR" / "Deroceras reticulatum" / "dendrogram.svg"))',
    ),
    (
        'plt.savefig(r"results/SAR/Deroceras reticulatum/cluster_structures.png")',
        'plt.savefig(str(RESULTS_DIR / "SAR" / "Deroceras reticulatum" / "cluster_structures.png"))',
    ),
    (
        'plt.savefig(r"results/SAR/Deroceras reticulatum/cluster_structures.svg")',
        'plt.savefig(str(RESULTS_DIR / "SAR" / "Deroceras reticulatum" / "cluster_structures.svg"))',
    ),
]

subs_safe = [
    (
        'input_file = r"results/safe/safe1-2.xlsx"',
        'input_file = str(RESULTS_DIR / "safe" / "safe1-2.xlsx")',
    ),
    (
        'output_prefix = r"results/safe/safe1_top_cluster"',
        'output_prefix = str(RESULTS_DIR / "safe" / "safe1_top_cluster")',
    ),
    (
        'input_file = r"results/safe/safe1-1.xlsx"',
        'input_file = str(RESULTS_DIR / "safe" / "safe1-1.xlsx")',
    ),
    (
        'output_file = r"results/safe/safe1-2.xlsx"',
        'output_file = str(RESULTS_DIR / "safe" / "safe1-2.xlsx")',
    ),
    (
        'output_prefix = r"results/safe/safe1_optimized"',
        'output_prefix = str(RESULTS_DIR / "safe" / "safe1_optimized")',
    ),
    (
        'cluster_file = r"results/safe/safe1-2.xlsx"',
        'cluster_file = str(RESULTS_DIR / "safe" / "safe1-2.xlsx")',
    ),
    (
        'output_prefix = r"results/safe/safe1-2_analysis-all"',
        'output_prefix = str(RESULTS_DIR / "safe" / "safe1-2_analysis-all")',
    ),
]

subs_all = [
    (
        'r"results/SAR/combing/all compounds analysis.xlsx"',
        'str(DATASET_DIR / "all compounds analysis.xlsx")',
    ),
]


def norm(t: str) -> str:
    t = t.replace("results/SAR/combing\\", "results/SAR/combing/")
    t = t.replace("results/SAR/Deroceras reticulatum\\", "results/SAR/Deroceras reticulatum/")
    t = t.replace("results/safe\\", "results/safe/")
    return t


def apply(name, pairs):
    p = DM / name
    t = norm(p.read_text(encoding="utf-8"))
    for a, b in pairs:
        if a not in t:
            print("MISSING", name, a)
        else:
            t = t.replace(a, b)
    p.write_text(t, encoding="utf-8")
    print("ok", name)


apply("sar_all_compounds.py", subs_all)
apply("sar_deroceras_reticulatum.py", subs_der)
apply("ecotox_substituent_clustering.py", subs_safe)
