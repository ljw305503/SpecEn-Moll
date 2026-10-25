# # 带标签分子图提取
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
from torch_geometric.data import Data
import networkx as nx
from rdkit.Chem import Draw
import os
import concurrent.futures
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载数据
# input_file = "../dataset/train/all compounds analysis.xlsx"# 输入的化合物表格，需包含 SMILES 列
# output_file_gnn = "../dataset/train/pc-gnn.pt"# 输出GNN格式
# output_image_folder = "../results/fsl/structures"# 输出GNN图像文件夹
def get_pc_gnn(input_file,output_file_gnn,output_image_folder):
    # 读取表格
    df = pd.read_excel(input_file)
    # 确保数据中包含 'smiles' 列
    if 'smiles' not in df.columns:
        raise ValueError("输入表格中必须包含 'smiles' 列！")
    # 用于存储GNN图数据的列表
    graph_data = []
    labels = []  # 修改变量名避免覆盖
    # 输出GNN图像文件夹
    os.makedirs(output_image_folder, exist_ok=True)
    # 原子类型映射字典
    ATOM_TYPE_MAPPING = {'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Br': 8, 'I': 9}
    for idx, row in df.iterrows():
        smiles = row['smiles']
        current_label = int(row['lable'])  # 使用临时变量存储当前标签
        try:
            # 分子对象处理
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                raise ValueError("Invalid SMILES")
            Chem.SanitizeMol(mol)

            # 构建分子图
            G = nx.Graph()

            # 添加节点特征
            for atom in mol.GetAtoms():
                features = [
                    ATOM_TYPE_MAPPING.get(atom.GetSymbol(), 0),  # 原子类型
                    atom.GetDegree(),  # 连接度
                    atom.GetImplicitValence()  # 隐式价
                ]
                G.add_node(atom.GetIdx(), features=features)

            # 添加边特征
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_type = bond.GetBondTypeAsDouble()
                G.add_edge(i, j, bond_type=bond_type)

            # 转换为PyG格式
            edge_index = []
            edge_attr = []
            for u, v, data in G.edges(data=True):
                # 双向边
                edge_index.extend([[u, v], [v, u]])
                # 边特征：键类型+芳香性
                edge_features = [data['bond_type'], bond.GetIsAromatic()]
                edge_attr.extend([edge_features, edge_features])

            # 构建数据对象
            data = Data(
                x=torch.tensor([n[1]['features'] for n in G.nodes(data=True)], dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                y=torch.tensor([current_label], dtype=torch.long)  # 使用当前标签
            )

            # 保存数据和标签
            graph_data.append(data)
            labels.append(current_label)  # 追加到标签列表
            # 生成带标签的分子图
            img = Draw.MolToImage(mol)
            img.save(os.path.join(output_image_folder, f"mol_{idx}_L1-{current_label}.png"))
        except Exception as e:
            print(f"处理失败：{smiles}，错误：{str(e)}")
            continue
    # 保存数据集
    torch.save(graph_data, output_file_gnn)
    # 生成统计报告
    stats = pd.DataFrame({
        '生成标签': labels,
    })
    print(f"\n总样本数：{len(df)}，有效样本：{len(graph_data)}")
    print(f"GNN数据已保存至：{output_file_gnn}")

# get_pc_gnn(input_file,output_file_gnn,output_image_folder)

def get_pc_gnn_1(input_file,output_file_gnn):
    # 读取表格
    df = pd.read_excel(input_file)
    # 确保数据中包含 'smiles' 列
    if 'smiles' not in df.columns:
        raise ValueError("输入表格中必须包含 'smiles' 列！")
    # 用于存储GNN图数据的列表
    graph_data = []
    # 原子类型映射字典
    ATOM_TYPE_MAPPING = {'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Br': 8, 'I': 9}
    for idx, row in df.iterrows():
        smiles = row['smiles']
        try:
            # 分子对象处理
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                raise ValueError("Invalid SMILES")
            Chem.SanitizeMol(mol)

            # 构建分子图
            G = nx.Graph()

            # 添加节点特征
            for atom in mol.GetAtoms():
                features = [
                    ATOM_TYPE_MAPPING.get(atom.GetSymbol(), 0),  # 原子类型
                    atom.GetDegree(),  # 连接度
                    atom.GetImplicitValence()  # 隐式价
                ]
                G.add_node(atom.GetIdx(), features=features)

            # 添加边特征
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_type = bond.GetBondTypeAsDouble()
                G.add_edge(i, j, bond_type=bond_type)

            # 转换为PyG格式
            edge_index = []
            edge_attr = []
            for u, v, data in G.edges(data=True):
                # 双向边
                edge_index.extend([[u, v], [v, u]])
                # 边特征：键类型+芳香性
                edge_features = [data['bond_type'], bond.GetIsAromatic()]
                edge_attr.extend([edge_features, edge_features])

            # 构建数据对象
            data = Data(
                x=torch.tensor([n[1]['features'] for n in G.nodes(data=True)], dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float)
            )
            graph_data.append(data)
        except Exception as e:
            print(f"处理失败：{smiles}，错误：{str(e)}")
            continue
    print(graph_data[0])
    torch.save(graph_data, output_file_gnn)
    return graph_data