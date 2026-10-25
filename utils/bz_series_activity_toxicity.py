
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
RESULTS_DIR = REPO_ROOT / "results"
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED, Lipinski
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = str(RESULTS_DIR / "compound Bz1-Bz9")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 0. 全局设置
# ============================================================
REFERENCE_COMPOUND = "Bz"
TOXICITY_DATA_TYPE = "toxicity_rate"  # "toxicity_rate" or "cell_viability"
ACTIVITY_Y_COLUMN = "Relative_activity_to_ref"
TOXICITY_Y_COLUMN = "Relative_toxicity_to_ref"


# ============================================================
# 1. 输入 Bz-Bz9 数据
# ============================================================
data = [
    {
        "Compound": "Bz",
        "SMILES": "ClC1=CC=CC(Cl)=C1/C=N/O",
        "Mean mortality rate": 79.17,
        "Mean toxicity rate": 70.00
    },
    {
        "Compound": "Bz1",
        "SMILES": "ClC1=CC=CC(Cl)=C1/C=N/NC(C2=CC(Br)=CC=C2)=O",
        "Mean mortality rate": 95.83,
        "Mean toxicity rate": 0.00
    },
    {
        "Compound": "Bz2",
        "SMILES": "ClC1=CC=CC(Cl)=C1/C=N/NC(OCC)=O",
        "Mean mortality rate": 50.00,
        "Mean toxicity rate": 26.67
    },
    {
        "Compound": "Bz3",
        "SMILES": "ClC1=CC=CC(Cl)=C1/C=N/N2CCN(C)CC2",
        "Mean mortality rate": 45.83,
        "Mean toxicity rate": 3.33
    },
    {
        "Compound": "Bz4",
        "SMILES": "ClC1=CC=CC(Cl)=C1/C=N/NC2=CC=CC=C2",
        "Mean mortality rate": 33.33,
        "Mean toxicity rate": 3.33
    },
    {
        "Compound": "Bz5",
        "SMILES": "ClC1=CC=CC(Cl)=C1/C=N/NC2=NC(Cl)=CC=C2",
        "Mean mortality rate": 29.17,
        "Mean toxicity rate": 0.00
    },
    {
        "Compound": "Bz6",
        "SMILES": "ClC1=CC=CC(Cl)=C1/C=N/NC2=NC=C(Br)C=C2",
        "Mean mortality rate": 45.83,
        "Mean toxicity rate": 3.33
    },
    {
        "Compound": "Bz7",
        "SMILES": "ClC1=CC=CC(Cl)=C1/C=N/C2=CC=C(C)C=C2",
        "Mean mortality rate": 37.50,
        "Mean toxicity rate": 3.33
    },
    {
        "Compound": "Bz8",
        "SMILES": "ClC1=CC=CC(Cl)=C1/C=N/C2=CC=C(F)C=C2",
        "Mean mortality rate": 41.67,
        "Mean toxicity rate": 0.00
    },
    {
        "Compound": "Bz9",
        "SMILES": "ClC1=CC=CC(Cl)=C1/C=N/CC2=CC=CC=C2",
        "Mean mortality rate": 0.00,
        "Mean toxicity rate": 0.00
    },
]


# ============================================================
# 2. LogS 估算函数
# ============================================================
def calculate_aromatic_proportion(mol):
    heavy_atoms = mol.GetNumHeavyAtoms()
    if heavy_atoms == 0:
        return np.nan

    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    return aromatic_atoms / heavy_atoms


def estimate_logs_esol_like(mol, logp, mw, rotb):
    aromatic_proportion = calculate_aromatic_proportion(mol)

    if np.isnan(aromatic_proportion):
        return np.nan

    logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotb - 0.74 * aromatic_proportion
    return logs


# ============================================================
# 3. 计算理化性质
# ============================================================

def calculate_descriptors(row):
    compound = row["Compound"]
    smiles = row["SMILES"]

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print(f"Warning: {compound} has invalid SMILES: {smiles}")
        return {
            "Compound": compound,
            "SMILES": smiles,
            "Mean mortality rate": row["Mean mortality rate"],
            "Mean toxicity rate": row["Mean toxicity rate"],
            "Valid_SMILES": False,
            "MW": np.nan,
            "cLogP": np.nan,
            "TPSA": np.nan,
            "HBD": np.nan,
            "HBA": np.nan,
            "RotB": np.nan,
            "MolMR": np.nan,
            "QED": np.nan,
            "LogS": np.nan,
            "HeavyAtomCount": np.nan,
            "RingCount": np.nan,
            "AromaticRingCount": np.nan,
            "FractionCSP3": np.nan,
            "LabuteASA": np.nan,
        }

    mw = Descriptors.MolWt(mol)
    clogp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    molmr = Crippen.MolMR(mol)
    qed_value = QED.qed(mol)
    logs = estimate_logs_esol_like(mol, clogp, mw, rotb)

    heavy_atom_count = mol.GetNumHeavyAtoms()
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    aromatic_ring_count = rdMolDescriptors.CalcNumAromaticRings(mol)
    fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    labute_asa = rdMolDescriptors.CalcLabuteASA(mol)

    return {
        "Compound": compound,
        "SMILES": smiles,
        "Mean mortality rate": row["Mean mortality rate"],
        "Mean toxicity rate": row["Mean toxicity rate"],
        "Valid_SMILES": True,
        "MW": mw,
        "cLogP": clogp,
        "TPSA": tpsa,
        "HBD": hbd,
        "HBA": hba,
        "RotB": rotb,
        "MolMR": molmr,
        "QED": qed_value,
        "LogS": logs,
        "HeavyAtomCount": heavy_atom_count,
        "RingCount": ring_count,
        "AromaticRingCount": aromatic_ring_count,
        "FractionCSP3": fraction_csp3,
        "LabuteASA": labute_asa,
    }


results = [calculate_descriptors(row) for row in data]
df = pd.DataFrame(results)


# ============================================================
# 4. 处理毒性数据
# ============================================================
if TOXICITY_DATA_TYPE == "toxicity_rate":
    df["Toxicity_rate"] = df["Mean toxicity rate"]
elif TOXICITY_DATA_TYPE == "cell_viability":
    df["Cell_viability"] = df["Mean toxicity rate"]
    df["Toxicity_rate"] = 100.0 - df["Cell_viability"]
else:
    raise ValueError("TOXICITY_DATA_TYPE must be 'toxicity_rate' or 'cell_viability'.")

# ============================================================
# 5. 计算相对活性和相对毒性
# ============================================================
if REFERENCE_COMPOUND not in df["Compound"].values:
    raise ValueError(f"Reference compound {REFERENCE_COMPOUND} not found in data.")

ref_activity = df.loc[df["Compound"] == REFERENCE_COMPOUND, "Mean mortality rate"].values[0]
ref_toxicity = df.loc[df["Compound"] == REFERENCE_COMPOUND, "Toxicity_rate"].values[0]
max_activity = df["Mean mortality rate"].max()
max_toxicity = df["Toxicity_rate"].max()

if ref_activity == 0:
    raise ValueError("Reference compound mean mortality rate is 0. Cannot normalize activity.")

if ref_toxicity == 0:
    print(
        "Warning: Reference compound toxicity is 0. "
        "Relative toxicity to reference cannot be calculated."
    )
    df["Relative_toxicity_to_ref"] = np.nan
else:
    df["Relative_toxicity_to_ref"] = df["Toxicity_rate"] / ref_toxicity

if max_activity == 0:
    df["Relative_activity_to_max"] = np.nan
else:
    df["Relative_activity_to_max"] = df["Mean mortality rate"] / max_activity

if max_toxicity == 0:
    df["Relative_toxicity_to_max"] = np.nan
else:
    df["Relative_toxicity_to_max"] = df["Toxicity_rate"] / max_toxicity

df["Relative_activity_to_ref"] = df["Mean mortality rate"] / ref_activity

# ============================================================
# 6. 活性-毒性综合指标
# ============================================================
df["Activity_toxicity_index"] = (
    df["Relative_activity_to_ref"] / df["Relative_toxicity_to_ref"]
)
df.replace([np.inf, -np.inf], np.nan, inplace=True)


# ============================================================
# 7. 整理 Table S1
# ============================================================
table_columns = [
    "Compound",
    "SMILES",
    "Mean mortality rate",
    "Mean toxicity rate",
    "Toxicity_rate",
    "Relative_activity_to_ref",
    "Relative_activity_to_max",
    "Relative_toxicity_to_ref",
    "Relative_toxicity_to_max",
    "Activity_toxicity_index",
    "MW",
    "cLogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotB",
    "MolMR",
    "QED",
    "LogS",
    "HeavyAtomCount",
    "RingCount",
    "AromaticRingCount",
    "FractionCSP3",
    "LabuteASA",
]

if TOXICITY_DATA_TYPE == "cell_viability":
    table_columns.insert(4, "Cell_viability")

table_df = df[table_columns].copy()
numeric_columns = [
    col for col in table_df.columns
    if col not in ["Compound", "SMILES"]
]
table_df[numeric_columns] = table_df[numeric_columns].round(3)
table_df.to_excel(
    os.path.join(OUTPUT_DIR, "Table_S1_activity_toxicity_physicochemical_properties.xlsx"),
    index=False
)
table_df.to_csv(
    os.path.join(OUTPUT_DIR, "Table_S1_activity_toxicity_physicochemical_properties.csv"),
    index=False
)

print("\nTable S1: activity, toxicity, and calculated physicochemical properties")
print(table_df)


# ============================================================
# 8. 相关性分析
# ============================================================
valid_df = df[df["Valid_SMILES"] == True].copy()
descriptor_for_correlation = [
    "MW",
    "cLogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotB",
    "MolMR",
    "QED",
    "LogS",
    "FractionCSP3",
    "LabuteASA",
]

correlation_results = []

for desc in descriptor_for_correlation:
    activity_pearson = valid_df[desc].corr(valid_df[ACTIVITY_Y_COLUMN], method="pearson")
    activity_spearman = valid_df[desc].corr(valid_df[ACTIVITY_Y_COLUMN], method="spearman")
    toxicity_pearson = valid_df[desc].corr(valid_df[TOXICITY_Y_COLUMN], method="pearson")
    toxicity_spearman = valid_df[desc].corr(valid_df[TOXICITY_Y_COLUMN], method="spearman")
    index_pearson = valid_df[desc].corr(valid_df["Activity_toxicity_index"], method="pearson")
    index_spearman = valid_df[desc].corr(valid_df["Activity_toxicity_index"], method="spearman")

    correlation_results.append({
        "Descriptor": desc,
        "Activity_Pearson_r": activity_pearson,
        "Activity_Spearman_r": activity_spearman,
        "Toxicity_Pearson_r": toxicity_pearson,
        "Toxicity_Spearman_r": toxicity_spearman,
        "Activity_toxicity_index_Pearson_r": index_pearson,
        "Activity_toxicity_index_Spearman_r": index_spearman,
    })

corr_df = pd.DataFrame(correlation_results)
corr_df = corr_df.round(3)
corr_df.to_excel(
    os.path.join(OUTPUT_DIR, "Descriptor_activity_toxicity_correlation.xlsx"),
    index=False
)
corr_df.to_csv(
    os.path.join(OUTPUT_DIR, "Descriptor_activity_toxicity_correlation.csv"),
    index=False
)

print("\nDescriptor-activity/toxicity correlation:")
print(corr_df)


# ============================================================
# 9. 通用散点图函数
# ============================================================
def make_scatter_plot(x_col, y_col, x_label, y_label, title, output_prefix):
    plt.figure(figsize=(6, 5))
    plt.scatter(valid_df[x_col], valid_df[y_col], s=80)

    for _, row in valid_df.iterrows():
        plt.annotate(
            row["Compound"],
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
            fontsize=9,
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}.png"), dpi=600)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}.svg"))
    plt.show()


# ============================================================
# 10. Figure 1A-B: 活性 vs cLogP / TPSA
# ============================================================
make_scatter_plot(
    x_col="cLogP",
    y_col=ACTIVITY_Y_COLUMN,
    x_label="cLogP",
    y_label=f"Relative activity to {REFERENCE_COMPOUND}",
    title="Relative activity vs cLogP",
    output_prefix="Figure_1A_relative_activity_vs_cLogP",
)
make_scatter_plot(
    x_col="TPSA",
    y_col=ACTIVITY_Y_COLUMN,
    x_label="TPSA",
    y_label=f"Relative activity to {REFERENCE_COMPOUND}",
    title="Relative activity vs TPSA",
    output_prefix="Figure_1B_relative_activity_vs_TPSA",
)


# ============================================================
# 11. Figure 1C-D: 毒性 vs cLogP / TPSA
# ============================================================
make_scatter_plot(
    x_col="cLogP",
    y_col=TOXICITY_Y_COLUMN,
    x_label="cLogP",
    y_label=f"Relative toxicity to {REFERENCE_COMPOUND}",
    title="Relative toxicity vs cLogP",
    output_prefix="Figure_1C_relative_toxicity_vs_cLogP",
)
make_scatter_plot(
    x_col="TPSA",
    y_col=TOXICITY_Y_COLUMN,
    x_label="TPSA",
    y_label=f"Relative toxicity to {REFERENCE_COMPOUND}",
    title="Relative toxicity vs TPSA",
    output_prefix="Figure_1D_relative_toxicity_vs_TPSA",
)


# ============================================================
# 12. Figure 2: Descriptor heatmap, Z-score 标准化
# ============================================================
heatmap_descriptors = [
    "cLogP",
    "TPSA",
    "MW",
    "HBA",
    "HBD",
    "RotB",
    "QED",
    "LogS",
    "Relative_activity_to_ref",
    "Relative_toxicity_to_ref",
    "Activity_toxicity_index",
]

heatmap_data = valid_df[heatmap_descriptors].copy()
heatmap_data.index = valid_df["Compound"]

scaler = StandardScaler()
heatmap_z = scaler.fit_transform(heatmap_data)

heatmap_z_df = pd.DataFrame(
    heatmap_z,
    index=heatmap_data.index,
    columns=heatmap_descriptors
)
heatmap_z_df.to_excel(
    os.path.join(OUTPUT_DIR, "Figure_2_descriptor_activity_toxicity_heatmap_Zscore_values.xlsx")
)

plt.figure(figsize=(11, 6))
im = plt.imshow(heatmap_z_df.values, aspect="auto")
plt.colorbar(im, label="Z-score")
plt.xticks(
    ticks=np.arange(len(heatmap_descriptors)),
    labels=heatmap_descriptors,
    rotation=45,
    ha="right"
)
plt.yticks(
    ticks=np.arange(len(heatmap_z_df.index)),
    labels=heatmap_z_df.index
)
plt.title("Descriptor, activity, and toxicity heatmap of Bz-Bz9")
plt.xlabel("Parameters")
plt.ylabel("Compounds")

for i in range(heatmap_z_df.shape[0]):
    for j in range(heatmap_z_df.shape[1]):
        plt.text(
            j,
            i,
            f"{heatmap_z_df.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=7
        )
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_2_descriptor_activity_toxicity_heatmap.png"), dpi=600)
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_2_descriptor_activity_toxicity_heatmap.pdf"))
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_2_descriptor_activity_toxicity_heatmap.svg"))
plt.show()


# ============================================================
# 13. Figure 3: 活性-毒性平衡图
# ============================================================
plt.figure(figsize=(6, 5))
plt.scatter(
    valid_df["Relative_toxicity_to_ref"],
    valid_df["Relative_activity_to_ref"],
    s=100
)

for _, row in valid_df.iterrows():
    plt.annotate(
        row["Compound"],
        (row["Relative_toxicity_to_ref"], row["Relative_activity_to_ref"]),
        textcoords="offset points",
        xytext=(5, 5),
        ha="left",
        fontsize=9
    )

plt.axvline(1.0, linestyle="--", linewidth=1)
plt.axhline(1.0, linestyle="--", linewidth=1)
plt.xlabel(f"Relative toxicity to {REFERENCE_COMPOUND}")
plt.ylabel(f"Relative activity to {REFERENCE_COMPOUND}")
plt.title("Activity-toxicity balance of Bz-Bz9")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_3_activity_toxicity_balance.png"), dpi=600)
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_3_activity_toxicity_balance.pdf"))
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_3_activity_toxicity_balance.svg"))
plt.show()


# ============================================================
# 14. 输出总结
# ============================================================
print("\nAnalysis completed.")
print(f"Output directory: {OUTPUT_DIR}")
print("Generated files:")
print("1. Table_S1_activity_toxicity_physicochemical_properties.xlsx")
print("2. Table_S1_activity_toxicity_physicochemical_properties.csv")
print("3. Descriptor_activity_toxicity_correlation.xlsx")
print("4. Descriptor_activity_toxicity_correlation.csv")
print("5. Figure_1A_relative_activity_vs_cLogP.png / .pdf / .svg")
print("6. Figure_1B_relative_activity_vs_TPSA.png / .pdf / .svg")
print("7. Figure_1C_relative_toxicity_vs_cLogP.png / .pdf / .svg")
print("8. Figure_1D_relative_toxicity_vs_TPSA.png / .pdf / .svg")
print("9. Figure_2_descriptor_activity_toxicity_heatmap.png / .pdf / .svg")
print("10. Figure_2_descriptor_activity_toxicity_heatmap_Zscore_values.xlsx")
print("11. Figure_3_activity_toxicity_balance.png / .pdf / .svg")