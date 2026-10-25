import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.inchi import MolToInchi, MolToInchiKey
from rdkit import DataStructs

# 加载数据
input_file1 = "../dataset/train/fsl-1/Pomacea canaliculata compounds analysis data-1.csv"# 输入的化合物表格，需包含 SMILES 列
output_file1 = "../dataset/train/fsl-1/Pomacea canaliculata compounds analysis data-2.csv"# 输出文件路径
# 读取表格
df1 = pd.read_csv(input_file1)
# 确保数据中包含 'smiles' 列
if 'smiles' not in df1.columns:
    raise ValueError("输入表格中必须包含 'smiles' 列！")
# 准备存储所有特征的 DataFrame
descriptor_names = [desc[0] for desc in Descriptors._descList]
columns = ['MolFormula', 'InChI', 'InChIKey'] + descriptor_names + [
    'RDKFingerprint', 'TopologicalFP', 'MACCS',
    'AtomPairsFP', 'TopologicalTorsionsFP',
    'MorganFP_2', 'MorganFP_3', 'molprint2D'
]
results = []
# 遍历 SMILES 计算特征
for idx, row in df1.iterrows():
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
        rdk_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2).ToBitString()# RDKFingerprint
        topo_fp = AllChem.RDKFingerprint(mol).ToBitString()# TopologicalFP
        maccs_fp = MACCSkeys.GenMACCSKeys(mol).ToBitString()# MACCS
        atom_pairs_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol).ToBitString()# AtomPairsFP
        topo_torsions_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol).ToBitString()# TopologicalTorsionsFP
        morgan_fp_2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2).ToBitString()# MorganFP_2
        morgan_fp_3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3).ToBitString()# MorganFP_3
        # 计算 molprint2D
        molprint2D = AllChem.GetMorganFingerprint(mol, radius=2, useFeatures=True).GetNonzeroElements()
        molprint2D_str = ",".join(f"{k}:{v}" for k, v in molprint2D.items())# 转换为字符串
        # 将结果添加到列表中
        results.append(
            [mol_formula, inchi, inchi_key] +
            descriptor_values +
            [rdk_fp, topo_fp, maccs_fp, atom_pairs_fp, topo_torsions_fp, morgan_fp_2, morgan_fp_3, molprint2D_str])
    except Exception as e:
        print(f"SMILES {smiles} 计算失败，原因：{e}")
        results.append([np.nan] * len(columns))
    # 将结果保存到 DataFrame
    features_df1 = pd.DataFrame(results, columns=columns)
# 合并原始数据和特征
df1 = pd.concat([df1, features_df1], axis=1)
# 保存到 CSV 文件
df1.to_csv(output_file1, index=False)
print(f"特征提取完成，结果已保存到 {output_file1}")