# #提取分子描述符和指纹特征（使用RDKit，从合物表格中依据化合物的smiles获取分子式，InChI，InChIKey，及其相应的分子描述符和分子指纹等208个分子描述符。）
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.inchi import MolToInchi, MolToInchiKey
from rdkit import DataStructs

# 加载数据
input_file = "../dataset/train/all compounds analysis.xlsx"  # 输入的化合物表格，需包含 SMILES 列
output_file = "../dataset/train/text_feature.xlsx"  # 输出文件路径

# 读取表格
df = pd.read_excel(input_file)
# 确保数据中包含 'smiles' 列
if 'smiles' not in df.columns:
    raise ValueError("输入表格中必须包含 'smiles' 列！")

# 准备存储所有特征的 DataFrame
descriptor_names = [desc[0] for desc in Descriptors._descList]
columns = ['Mol','MolFormula', 'InChI', 'InChIKey'] + descriptor_names + [
    'RDKFingerprint', 'TopologicalFP', 'MACCS',
    'AtomPairsFP', 'TopologicalTorsionsFP',
    'MorganFP_2', 'MorganFP_3',
    'molprint2D'
]
results = []
print(columns)
# 遍历 SMILES 计算特征
for idx, row in df.iterrows():
    smiles = row['smiles']
    try:
        # 转换 smiles 为分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("无效的 smiles")
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)
        norm_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        # 获取分子式、InChI、InChIKey
        mol_formula = CalcMolFormula(mol)
        inchi = MolToInchi(mol)
        inchi_key = MolToInchiKey(mol)

        # 计算 208 个描述符
        descriptor_values = [Descriptors.__dict__[name](mol) for name in descriptor_names]

        # 计算指纹特征
        rdk_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2).ToBitString()  # RDKFingerprint
        topo_fp = AllChem.RDKFingerprint(mol).ToBitString()  # TopologicalFP
        maccs_fp = MACCSkeys.GenMACCSKeys(mol).ToBitString()  # MACCS
        atom_pairs_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol).ToBitString()  # AtomPairsFP
        topo_torsions_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol).ToBitString()  # TopologicalTorsionsFP
        morgan_fp_2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2).ToBitString()  # MorganFP_2
        morgan_fp_3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3).ToBitString()  # MorganFP_3

        # 计算 molprint2D
        molprint2D = AllChem.GetMorganFingerprint(mol, radius=2, useFeatures=True).GetNonzeroElements()
        molprint2D_str = ",".join(f"{k}:{v}" for k, v in molprint2D.items())  # 转换为字符串
        # 将结果添加到列表中
        results.append(
            [mol,mol_formula, inchi, inchi_key] +
            descriptor_values +
            [
                rdk_fp, topo_fp, maccs_fp,
             atom_pairs_fp, topo_torsions_fp, morgan_fp_2, morgan_fp_3,
            molprint2D_str]
        )

    except Exception as e:
        print(f"SMILES {smiles} 计算失败，原因：{e}")
        results.append([np.nan] * len(columns))

    # 将结果保存到 DataFrame
    features_df = pd.DataFrame(results, columns=columns)

# 合并原始数据和特征
df = pd.concat([df, features_df], axis=1)

# 保存到 CSV 文件
df.to_excel(output_file, index=False)
print(f"特征提取完成，结果已保存到 {output_file}")

success_count = sum(1 for x in results if not pd.isna(x[0]))  # 假设第一个特征不为空则表示成功
fail_count = len(df) - success_count
print(f"\n=== 处理状态统计 ===")
print(f"成功解析的化合物数量: {success_count}")
print(f"解析失败的化合物数量: {fail_count}")
