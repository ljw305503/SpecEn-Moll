import torch
import os
import numpy as np
from sklearn.manifold import SpectralEmbedding
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import _pickle


def enhanced_feature_extractor(graph_data, smiles_list):
    """增强型特征提取：包含拓扑描述符、指纹特征"""
    feature_list = []
    for data, smiles in zip(graph_data, smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        # 基础图特征
        node_feat = data.x.mean(dim=0).numpy()
        edge_feat = data.edge_attr.mean(dim=0).numpy()
        # 拓扑描述符
        topo_feat = [
            Descriptors.NumAromaticRings(mol),  # 芳香环数量
            Descriptors.TPSA(mol),  # 极性表面积
            Descriptors.MolLogP(mol),  # 亲脂性
            Descriptors.NumRotatableBonds(mol),  # 可旋转键
            Descriptors.MolWt(mol),  # 分子量
            Descriptors.NumHDonors(mol),  # 氢键供体数量
            Descriptors.NumHAcceptors(mol),  # 氢键受体数量
            rdMolDescriptors.CalcNumAtoms(mol),  # 原子总数
            rdMolDescriptors.CalcNumHeteroatoms(mol)  # 杂原子数
        ]
        # RDK指纹特征（摩根指纹）
        fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=2048)
        fp_array = np.zeros((2048,))
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        # 合并特征
        combined_feat = np.concatenate([node_feat, edge_feat, topo_feat, fp_array])
        feature_list.append(combined_feat)
    return np.array(feature_list)

def dynamic_constraint_matrix(X):
        """动态约束矩阵生成"""
        k = int(np.sqrt(len(X)))  # 自适应近邻数
        # 构建相似度矩阵
        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
        sim_matrix = X_normalized @ X_normalized.T
        min=np.min(sim_matrix)
        max=np.max(sim_matrix)
        sim_matrix=(sim_matrix-min)/(max-min)
        # 自适应稀疏化
        # nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        # *nbrs.kneighbors_graph(X).toarray()
        return sim_matrix


def data_enhanced(graph_path,file_path):
    # file_path = "../dataset/train/pc-gnn.pt"
    try:
        # 尝试使用 weights_only=True 加载
        graph_data = torch.load(graph_path, weights_only=True)
    except _pickle.UnpicklingError:
        print("使用 weights_only=True 加载失败，尝试使用 weights_only=False 加载...")
        # 若失败，使用 weights_only=False 重新加载
        graph_data = torch.load(graph_path, weights_only=False)
    df = pd.read_excel(file_path)
    # 增强特征提取
    X = enhanced_feature_extractor(graph_data, df['smiles'])
    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # PCA降维，减少特征维度，去除噪声
    pca = PCA(n_components=0.95)  # 保留95%方差
    X_pca = pca.fit_transform(X_scaled)
    # y = np.array([data.y.item() for data in graph_data])
    return X_pca

def get_embedding(graph_path,file_path,k=2):
    X_pca= data_enhanced(graph_path,file_path)
    # 构建动态约束矩阵
    sim_matrix = dynamic_constraint_matrix(X_pca)
    # for i in range(sim_matrix.shape[0]):
    #     print(sim_matrix[i])
    # print(np.sum(sim_matrix))
    embedding = SpectralEmbedding(n_components=k, affinity='precomputed')
    X_emb = embedding.fit_transform(sim_matrix)
    # print(X_emb)
    return X_emb

if __name__ == "__main__":
        x_emb=get_embedding()
        print(x_emb.shape)
