"""One-off: retarget theta/DM analysis scripts to repo-relative paths and rename files."""
from pathlib import Path

DM = Path(__file__).resolve().parent
ROOT_INSERT = '''from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
RESULTS_DIR = REPO_ROOT / "results"
'''

# (old_name, new_name)
RENAMES = [
    ("c-1.py", "sar_all_compounds.py"),
    ("dl-1.py", "sar_oncomelania_hupensis.py"),
    ("fsl-sar.py", "sar_pomacea_canaliculata.py"),
    ("ky-1.py", "sar_deroceras_reticulatum.py"),
    ("hc.py", "bz_series_activity_toxicity.py"),
    ("safe4.py", "ecotox_substituent_clustering.py"),
]

# Longest prefixes first.
STRING_REPLACES = [
    (r"D:\BS222\FSL\sjk\WZ1\data\co\analysis", r"results/SAR/combing"),
    (r"D:\BS222\FSL\sjk\WZ1\data\Deroceras reticulatum\analysis", r"results/SAR/Deroceras reticulatum"),
    (r"D:\BS222\FSL\sjk\WZ1\word\20260420\data", r"results/compound Bz1-Bz9"),
    (r"D:\BS222\FSL\sjk\safe", r"results/safe"),
    ("/data_disk/ljw/code/wyf/sjk/dl/dl-1/gnn/oh-gnn.pt", "results/SAR/Oncomelania hupensis/gnn/oh-gnn.pt"),
    ("/data_disk/ljw/code/wyf/sjk/dl/dl-1/gnn", "results/SAR/Oncomelania hupensis/gnn"),
    ("/data_disk/ljw/code/wyf/sjk/dl/dl-1", "results/SAR/Oncomelania hupensis"),
    ("/data_disk/ljw/code/wyf/sjk/fsl/fsl-1/gnn/pc-gnn.pt", "results/SAR/Pomacea canaliculata/gnn/pc-gnn.pt"),
    ("/data_disk/ljw/code/wyf/sjk/fsl/fsl-1/gnn", "results/SAR/Pomacea canaliculata/gnn"),
    ("/data_disk/ljw/code/wyf/sjk/fsl/fsl-1", "results/SAR/Pomacea canaliculata"),
    ("/data_disk/ljw/code/wyf/TIMES.TTF", "Times New Roman"),
]


def inject_root(text: str) -> str:
    if "REPO_ROOT = Path(__file__)" in text:
        return text
    # After module docstring / encoding comment / first import block start.
    lines = text.splitlines(True)
    insert_at = 0
    i = 0
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1
    if i < len(lines) and "coding" in lines[i]:
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith('"""'):
        i += 1
        while i < len(lines) and '"""' not in lines[i]:
            i += 1
        if i < len(lines):
            i += 1
    insert_at = i
    return "".join(lines[:insert_at]) + "\n" + ROOT_INSERT + "".join(lines[insert_at:])


def retarget_live_assignments(name: str, text: str) -> str:
    """Point live (uncommented) I/O at pathlib roots where we introduced REPO_ROOT."""
    live = {
        "sar_all_compounds.py": [],
        "sar_oncomelania_hupensis.py": [
            (
                'input_file = "results/SAR/Oncomelania hupensis/Oncomelania hupensis compounds analysis data.csv"',
                'input_file = str(DATASET_DIR / "Oncomelania hupensis compounds analysis data.xlsx")',
            ),
            (
                'output_file_gnn = "results/SAR/Oncomelania hupensis/gnn/oh-gnn.pt"',
                'output_file_gnn = str(RESULTS_DIR / "SAR" / "Oncomelania hupensis" / "gnn" / "oh-gnn.pt")',
            ),
            (
                'output_image_folder = "results/SAR/Oncomelania hupensis/gnn"',
                'output_image_folder = str(RESULTS_DIR / "SAR" / "Oncomelania hupensis" / "gnn")',
            ),
            (
                'file_path = "results/SAR/Oncomelania hupensis/gnn/oh-gnn.pt"',
                'file_path = str(RESULTS_DIR / "SAR" / "Oncomelania hupensis" / "gnn" / "oh-gnn.pt")',
            ),
            (
                'df = pd.read_csv("results/SAR/Oncomelania hupensis/Oncomelania hupensis compounds analysis data.csv")',
                'df = pd.read_excel(DATASET_DIR / "Oncomelania hupensis compounds analysis data.xlsx")',
            ),
            (
                'plt.savefig(f"results/SAR/Oncomelania hupensis/{title_prefix.lower()}_{method}.svg")',
                'plt.savefig(str(RESULTS_DIR / "SAR" / "Oncomelania hupensis" / f"{title_prefix.lower()}_{method}.svg"))',
            ),
            (
                'plt.savefig(f"results/SAR/Oncomelania hupensis/{title_prefix.lower()}_{method}.png", dpi=900)',
                'plt.savefig(str(RESULTS_DIR / "SAR" / "Oncomelania hupensis" / f"{title_prefix.lower()}_{method}.png"), dpi=900)',
            ),
            (
                'plt.savefig("results/SAR/Oncomelania hupensis/sim_matrix_heatmap.svg")',
                'plt.savefig(str(RESULTS_DIR / "SAR" / "Oncomelania hupensis" / "sim_matrix_heatmap.svg"))',
            ),
            (
                'plt.savefig("results/SAR/Oncomelania hupensis/sim_matrix_heatmap.png")',
                'plt.savefig(str(RESULTS_DIR / "SAR" / "Oncomelania hupensis" / "sim_matrix_heatmap.png"))',
            ),
        ],
        "sar_pomacea_canaliculata.py": [
            (
                'input_file = "results/SAR/Pomacea canaliculata/Pomacea canaliculata compounds analysis data.csv"',
                'input_file = str(DATASET_DIR / "Pomacea canaliculata compounds analysis data.xlsx")',
            ),
            (
                'output_file_gnn = "results/SAR/Pomacea canaliculata/gnn/pc-gnn.pt"',
                'output_file_gnn = str(RESULTS_DIR / "SAR" / "Pomacea canaliculata" / "gnn" / "pc-gnn.pt")',
            ),
            (
                'output_image_folder = "results/SAR/Pomacea canaliculata/gnn"',
                'output_image_folder = str(RESULTS_DIR / "SAR" / "Pomacea canaliculata" / "gnn")',
            ),
            (
                'file_path = "results/SAR/Pomacea canaliculata/gnn/pc-gnn.pt"',
                'file_path = str(RESULTS_DIR / "SAR" / "Pomacea canaliculata" / "gnn" / "pc-gnn.pt")',
            ),
            (
                'df = pd.read_csv("results/SAR/Pomacea canaliculata/Pomacea canaliculata compounds analysis data.csv")',
                'df = pd.read_excel(DATASET_DIR / "Pomacea canaliculata compounds analysis data.xlsx")',
            ),
            (
                'plt.savefig(f"results/SAR/Pomacea canaliculata/{title_prefix.lower()}_{method}.svg")',
                'plt.savefig(str(RESULTS_DIR / "SAR" / "Pomacea canaliculata" / f"{title_prefix.lower()}_{method}.svg"))',
            ),
            (
                'plt.savefig(f"results/SAR/Pomacea canaliculata/{title_prefix.lower()}_{method}.png", dpi=900)',
                'plt.savefig(str(RESULTS_DIR / "SAR" / "Pomacea canaliculata" / f"{title_prefix.lower()}_{method}.png"), dpi=900)',
            ),
            (
                'plt.savefig("results/SAR/Pomacea canaliculata/sim_matrix_heatmap.svg")',
                'plt.savefig(str(RESULTS_DIR / "SAR" / "Pomacea canaliculata" / "sim_matrix_heatmap.svg"))',
            ),
            (
                'plt.savefig("results/SAR/Pomacea canaliculata/sim_matrix_heatmap.png")',
                'plt.savefig(str(RESULTS_DIR / "SAR" / "Pomacea canaliculata" / "sim_matrix_heatmap.png"))',
            ),
        ],
        "sar_deroceras_reticulatum.py": [
            (
                'input_file = r"dataset/Deroceras reticulatum compounds analysis data.xlsx"',
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
        ],
        "bz_series_activity_toxicity.py": [
            ('OUTPUT_DIR = r"results/compound Bz1-Bz9"',
             'OUTPUT_DIR = str(RESULTS_DIR / "compound Bz1-Bz9")'),
        ],
        "ecotox_substituent_clustering.py": [
            ('input_file = r"results/safe/safe1-2.xlsx"',
             'input_file = str(RESULTS_DIR / "safe" / "safe1-2.xlsx")'),
            ('output_prefix = r"results/safe/safe1_top_cluster"',
             'output_prefix = str(RESULTS_DIR / "safe" / "safe1_top_cluster")'),
            ('input_file = r"results/safe/safe1-1.xlsx"',
             'input_file = str(RESULTS_DIR / "safe" / "safe1-1.xlsx")'),
            ('output_file = r"results/safe/safe1-2.xlsx"',
             'output_file = str(RESULTS_DIR / "safe" / "safe1-2.xlsx")'),
            ('output_prefix = r"results/safe/safe1_optimized"',
             'output_prefix = str(RESULTS_DIR / "safe" / "safe1_optimized")'),
            ('cluster_file = r"results/safe/safe1-2.xlsx"',
             'cluster_file = str(RESULTS_DIR / "safe" / "safe1-2.xlsx")'),
            ('output_prefix = r"results/safe/safe1-2_analysis-all"',
             'output_prefix = str(RESULTS_DIR / "safe" / "safe1-2_analysis-all")'),
        ],
    }
    for a, b in live.get(name, []):
        text = text.replace(a, b)
    return text


def fix_font(text: str) -> str:
    old = '''font_path = "Times New Roman"
# 注册字体到 matplotlib
fm.fontManager.addfont(font_path)
matplotlib.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()'''
    new = "matplotlib.rcParams['font.family'] = 'Times New Roman'"
    text = text.replace(old, new)
    old2 = '''font_path = "Times New Roman"
fm.fontManager.addfont(font_path)
matplotlib.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()'''
    text = text.replace(old2, new)
    text = text.replace('# font_path = "Times New Roman"', '# font: Times New Roman via rcParams')
    text = text.replace("font_path = \"Times New Roman\"", "matplotlib.rcParams['font.family'] = 'Times New Roman'")
    return text


def fix_read_csv_xlsx(text: str) -> str:
    """Live GNN loaders now point at dataset xlsx."""
    text = text.replace(
        "df = pd.read_csv(input_file)",
        "df = pd.read_excel(input_file) if str(input_file).endswith(('.xlsx', '.xls')) else pd.read_csv(input_file)",
        1,
    )
    return text


def main():
    for old, new in RENAMES:
        src = DM / old
        text = src.read_text(encoding="utf-8")
        for a, b in STRING_REPLACES:
            text = text.replace(a, b)
            text = text.replace(a.replace("\\", "/"), b)
        text = inject_root(text)
        text = fix_font(text)
        # Canonical compound tables live in dataset/, not results/.
        for species_xlsx in (
            "Oncomelania hupensis compounds analysis data.xlsx",
            "Pomacea canaliculata compounds analysis data.xlsx",
            "Deroceras reticulatum compounds analysis data.xlsx",
            "all compounds analysis.xlsx",
        ):
            text = text.replace(
                f"results/SAR/Oncomelania hupensis/{species_xlsx}",
                f"dataset/{species_xlsx}",
            )
            text = text.replace(
                f"results/SAR/Pomacea canaliculata/{species_xlsx}",
                f"dataset/{species_xlsx}",
            )
            text = text.replace(
                f"results/SAR/combing/{species_xlsx}",
                f"dataset/{species_xlsx}",
            )
        text = text.replace(
            r'results/SAR/Deroceras reticulatum/Deroceras reticulatum analysis data.xlsx',
            'dataset/Deroceras reticulatum compounds analysis data.xlsx',
        )
        text = retarget_live_assignments(new, text)
        if new in ("sar_oncomelania_hupensis.py", "sar_pomacea_canaliculata.py"):
            text = fix_read_csv_xlsx(text)
        (DM / new).write_text(text, encoding="utf-8")
        src.unlink()
        print("renamed", old, "->", new)
    leftover = []
    for p in DM.glob("*.py"):
        t = p.read_text(encoding="utf-8", errors="ignore")
        if "D:\\BS222" in t or "/data_disk/" in t:
            leftover.append(p.name)
    print("leftover abs paths", leftover)


if __name__ == "__main__":
    main()
