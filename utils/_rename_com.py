from pathlib import Path

src = Path(__file__).with_name("com.py")
t = src.read_text(encoding="utf-8")
marker = "# # # 带标签分子图提取"
i = t.find(marker)
if i < 0:
    raise SystemExit("marker not found")
rest = t[i:]
rest = rest.replace(
    '# font_path = "Times New Roman"\n'
    "# # 注册字体到 matplotlib\n"
    "# fm.fontManager.addfont(font_path)\n"
    "# matplotlib.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()",
    "# matplotlib.rcParams['font.family'] = 'Times New Roman'",
)
rest = rest.replace(
    '# font_path = "Times New Roman"\n'
    "# fm.fontManager.addfont(font_path)\n"
    "# plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()",
    "# plt.rcParams['font.family'] = 'Times New Roman'",
)
rest = rest.replace(
    'df = pd.read_csv("dataset/all compounds analysis.xlsx")',
    'df = pd.read_excel("dataset/all compounds analysis.xlsx")',
)

header = '''# Extract RDKit molecular descriptors and fingerprints from SMILES.
# Formula, InChI, InChIKey, ~208 RDKit descriptors, and fingerprint strings.
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.inchi import MolToInchi, MolToInchiKey
from rdkit import DataStructs

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"


def extract_rdkit_descriptors(input_file, output_file):
    """Read a SMILES table and write concatenated RDKit text features."""
    df1 = pd.read_excel(input_file)
    if "smiles" not in df1.columns:
        raise ValueError("输入表格中必须包含 smiles 列！")

    descriptor_names = [desc[0] for desc in Descriptors._descList]
    columns = ["MolFormula", "InChI", "InChIKey"] + descriptor_names + [
        "RDKFingerprint", "TopologicalFP", "MACCS",
        "AtomPairsFP", "TopologicalTorsionsFP",
        "MorganFP_2", "MorganFP_3", "molprint2D",
    ]
    results = []

    for idx, row in df1.iterrows():
        smiles = row["smiles"]
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("无效的 smiles")
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)
            mol_formula = CalcMolFormula(mol)
            inchi = MolToInchi(mol)
            inchi_key = MolToInchiKey(mol)
            descriptor_values = [Descriptors.__dict__[name](mol) for name in descriptor_names]
            rdk_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2).ToBitString()
            topo_fp = AllChem.RDKFingerprint(mol).ToBitString()
            maccs_fp = MACCSkeys.GenMACCSKeys(mol).ToBitString()
            atom_pairs_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol).ToBitString()
            topo_torsions_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol).ToBitString()
            morgan_fp_2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2).ToBitString()
            morgan_fp_3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3).ToBitString()
            molprint2D = AllChem.GetMorganFingerprint(mol, radius=2, useFeatures=True).GetNonzeroElements()
            molprint2D_str = ",".join(f"{k}:{v}" for k, v in molprint2D.items())
            results.append(
                [mol_formula, inchi, inchi_key]
                + descriptor_values
                + [rdk_fp, topo_fp, maccs_fp, atom_pairs_fp, topo_torsions_fp, morgan_fp_2, morgan_fp_3, molprint2D_str]
            )
        except Exception as e:
            print(f"SMILES {smiles} 计算失败，原因：{e}")
            results.append([np.nan] * len(columns))
        features_df1 = pd.DataFrame(results, columns=columns)

    df1 = pd.concat([df1, features_df1], axis=1)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df1.to_excel(output_file, index=False)
    print(f"特征提取完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    for split in ("train", "test"):
        extract_rdkit_descriptors(
            DATASET_DIR / split / "all compounds analysis.xlsx",
            DATASET_DIR / split / "text_feature.xlsx",
        )
'''

out = Path(__file__).with_name("extract_rdkit_descriptors.py")
out.write_text(header + "\n\n" + rest, encoding="utf-8")
src.unlink()
print("wrote", out)
