# Extract RDKit molecular descriptors and fingerprints from SMILES.
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


# # # 带标签分子图提取
# import pandas as pd
# import numpy as np
# from rdkit import Chem
# import torch
# from torch_geometric.data import Data
# import networkx as nx
# from rdkit.Chem import Draw
# import os
# import concurrent.futures
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# # 加载数据
# input_file = "dataset/all compounds analysis.xlsx"  # 输入的化合物表格，需包含 SMILES 列
# output_file_gnn = "dataset/train/pc-gnn.pt"  # 输出GNN格式
# output_image_folder = "results/SAR/combing/gnn"  # 输出GNN图像文件夹

# # 读取表格
# df = pd.read_excel(input_file)

# # 确保数据中包含 'smiles' 列
# if 'smiles' not in df.columns:
#     raise ValueError("输入表格中必须包含 'smiles' 列！")

# # 用于存储GNN图数据的列表
# graph_data = []
# labels = []  # 修改变量名避免覆盖

# # 输出GNN图像文件夹
# os.makedirs(output_image_folder, exist_ok=True)

# # 原子类型映射字典
# ATOM_TYPE_MAPPING = {'C':1, 'N':2, 'O':3, 'F':4, 'P':5, 'S':6, 'Cl':7, 'Br':8, 'I':9}

# for idx, row in df.iterrows():
#     smiles = row['smiles']
#     current_label = int(row['lable'])  # 使用临时变量存储当前标签

#     try:
#         # 分子对象处理
#         mol = Chem.MolFromSmiles(smiles)
#         if not mol:
#             raise ValueError("Invalid SMILES")
#         Chem.SanitizeMol(mol)
        
#         # 构建分子图
#         G = nx.Graph()
        
#         # 添加节点特征
#         for atom in mol.GetAtoms():
#             features = [
#                 ATOM_TYPE_MAPPING.get(atom.GetSymbol(), 0),  # 原子类型
#                 atom.GetDegree(),                            # 连接度
#                 atom.GetImplicitValence()                    # 隐式价
#             ]
#             G.add_node(atom.GetIdx(), features=features)
        
#         # 添加边特征
#         for bond in mol.GetBonds():
#             i = bond.GetBeginAtomIdx()
#             j = bond.GetEndAtomIdx()
#             bond_type = bond.GetBondTypeAsDouble()
#             G.add_edge(i, j, bond_type=bond_type)
        
#         # 转换为PyG格式
#         edge_index = []
#         edge_attr = []
#         for u, v, data in G.edges(data=True):
#             # 双向边
#             edge_index.extend([[u, v], [v, u]])  
#             # 边特征：键类型+芳香性
#             edge_features = [data['bond_type'], bond.GetIsAromatic()]
#             edge_attr.extend([edge_features, edge_features])
        
#         # 构建数据对象
#         data = Data(
#             x = torch.tensor([n[1]['features'] for n in G.nodes(data=True)], dtype=torch.float),
#             edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
#             edge_attr = torch.tensor(edge_attr, dtype=torch.float),
#             y = torch.tensor([current_label], dtype=torch.long)  # 使用当前标签
#         )
        
#         # 保存数据和标签
#         graph_data.append(data)
#         labels.append(current_label)  # 追加到标签列表

#         # 生成带双标签的分子图
#         img = Draw.MolToImage(mol)
#         img.save(os.path.join(output_image_folder, f"mol_{idx}_L1-{current_label}.png"))

#     except Exception as e:
#         print(f"处理失败：{smiles}，错误：{str(e)}")
#         continue

# # 保存数据集
# torch.save(graph_data, output_file_gnn)

# # 生成统计报告
# stats = pd.DataFrame({
#     '生成标签': labels,
# })

# print(f"\n总样本数：{len(df)}，有效样本：{len(graph_data)}")
# print(f"GNN数据已保存至：{output_file_gnn}")









# import torch
# import numpy as np
# from sklearn.cluster import SpectralClustering, KMeans
# from sklearn.metrics import adjusted_rand_score, silhouette_score
# from sklearn.neighbors import NearestNeighbors
# from sklearn.manifold import TSNE
# from rdkit.Chem import Descriptors, rdMolDescriptors
# from rdkit.Chem import AllChem
# from rdkit import DataStructs
# import matplotlib.pyplot as plt
# import pandas as pd
# from rdkit import Chem
# from sklearn.metrics import silhouette_samples
# from sklearn.preprocessing import StandardScaler
# import umap
# from sklearn.metrics import pairwise_distances
# import seaborn as sns
# from torch_geometric.data import Data
# from torch.serialization import add_safe_globals
# from sklearn.decomposition import PCA  # 新增PCA用于特征降维
# from matplotlib.colors import ListedColormap
# import matplotlib
# import matplotlib.font_manager as fm
# from matplotlib.font_manager import FontProperties

# add_safe_globals([Data])

# # 重置matplotlib配置
# matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# matplotlib.rcParams['font.family'] = 'Times New Roman'

# def enhanced_feature_extractor(graph_data, smiles_list):
#     """增强型特征提取：包含拓扑描述符、指纹特征"""
#     feature_list = []
#     for data, smiles in zip(graph_data, smiles_list):
#         mol = Chem.MolFromSmiles(smiles)

#         # 基础图特征
#         node_feat = data.x.mean(dim=0).numpy()
#         edge_feat = data.edge_attr.mean(dim=0).numpy()

#         # 拓扑描述符
#         topo_feat = [
#             Descriptors.NumAromaticRings(mol),  # 芳香环数量
#             Descriptors.TPSA(mol),              # 极性表面积
#             Descriptors.MolLogP(mol),            # 亲脂性
#             Descriptors.NumRotatableBonds(mol),  # 可旋转键
#             Descriptors.MolWt(mol),             # 分子量
#             Descriptors.NumHDonors(mol),        # 氢键供体数量
#             Descriptors.NumHAcceptors(mol),     # 氢键受体数量
#             rdMolDescriptors.CalcNumAtoms(mol), # 原子总数
#             rdMolDescriptors.CalcNumHeteroatoms(mol) # 杂原子数
#         ]

#         # RDK指纹特征（摩根指纹）
#         fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=2048)
#         fp_array = np.zeros((2048,))
#         DataStructs.ConvertToNumpyArray(fp, fp_array)

#         # 合并特征
#         combined_feat = np.concatenate([node_feat, edge_feat, topo_feat, fp_array])
#         feature_list.append(combined_feat)

#     return np.array(feature_list)

# def dynamic_constraint_matrix(X, y, base_alpha=0.8):
#     """动态约束矩阵生成"""
#     # 计算标签覆盖率
#     label_coverage = np.sum(y != -1) / len(y)

#     # 动态调整参数，更灵活的alpha计算
#     alpha = base_alpha + 0.9 * label_coverage  # 调整系数
#     k = int(np.sqrt(len(X)))                   # 自适应近邻数

#     # 构建相似度矩阵
#     X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
#     sim_matrix = X_normalized @ X_normalized.T
#     # distances = pairwise_distances(X, metric='euclidean')
#     # sim_matrix = 1 / (1 + distances)

#     # 注入动态约束
#     labeled_mask = (y != -1)
#     for i in range(len(y)):
#         for j in range(len(y)):
#             if labeled_mask[i] and labeled_mask[j]:
#                 if y[i] == y[j]:
#                     sim_matrix[i,j] = min(1.0, sim_matrix[i,j] + alpha)
#                 else:
#                     sim_matrix[i,j] = max(0.0, sim_matrix[i,j] - alpha)

#     # 自适应稀疏化
#     nbrs = NearestNeighbors(n_neighbors=k).fit(X)
#     return sim_matrix * nbrs.kneighbors_graph(X).toarray()

# def supervised_tsne(X, y, perplexity=30):
#     """监督t-SNE实现（通过标签调整距离矩阵）"""
#     # 创建标签相似性矩阵
#     y_sim = np.array([y == y_i for y_i in y], dtype=float)

#     # 调整距离矩阵
#     distances = pairwise_distances(X)
#     adjusted_dist = distances * (1 - y_sim) + distances * 0.3 * y_sim

#     # 执行t-SNE
#     tsne = TSNE(n_components=2,
#                 perplexity=perplexity,
#                 metric='precomputed',
#                 random_state=50,
#                 init='random')
#     return tsne.fit_transform(adjusted_dist)

# def supervised_umap(X, y):
#     """监督UMAP实现"""
#     return umap.UMAP(n_components=2,
#                     n_neighbors=20,
#                     min_dist=0.1,

#                     target_metric='categorical',
#                     target_weight=0.8,
#                     random_state=42).fit_transform(X, y=y)

# def visualize_supervised_clusters(X, labels, true_labels, method='tsne', title_prefix=""):
#     """监督流形学习可视化"""
#     plt.figure(figsize=(18, 6))

#     # 执行监督降维
#     if method == 'tsne':
#         X_emb = supervised_tsne(X, true_labels)
#         title_suffix = "Supervised t-SNE"
#     elif method == 'umap':
#         X_emb = supervised_umap(X, true_labels)
#         title_suffix = "UMAP"
#     else:
#         raise ValueError("Unsupported method")

#     # 真实标签分布
#     plt.subplot(131)
#     scatter = plt.scatter(X_emb[:, 0], X_emb[:, 1], c=true_labels,
#                          cmap='cool', alpha=0.7, edgecolors='w')
#     cbar = plt.colorbar(scatter)
#     cbar.set_label('True Labels', fontsize=14)
#     cbar.ax.tick_params(labelsize=14)  # 调整颜色条刻度字体
#     plt.title(f"{title_prefix}True Labels ({title_suffix})", fontsize=14, fontweight='bold')
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)

#     # 聚类结果
#     plt.subplot(132)
#     scatter = plt.scatter(X_emb[:, 0], X_emb[:, 1], c=labels,
#                         cmap='rainbow', alpha=0.7, edgecolors='w')
#     cbar = plt.colorbar(scatter)
#     cbar.set_label('Cluster Labels', fontsize=14)
#     cbar.ax.tick_params(labelsize=14)  # 调整颜色条刻度字体
#     plt.title(f"{title_prefix}Clustering Result ({title_suffix})", fontsize=14, fontweight='bold')
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)

#     # 置信度可视化
#     plt.subplot(133)
#     sil_values = silhouette_samples(X, labels)
#      # 添加 vmin=0 和 vmax=1 限制颜色条范围
#     sc = plt.scatter(X_emb[:, 0], X_emb[:, 1], c=sil_values,
#                    cmap='coolwarm', alpha=0.7, edgecolors='w', vmin=0.1, vmax=0.5)
#     cbar = plt.colorbar(sc)
#     cbar.set_label('Silhouette Coefficient', fontsize=14)
#     cbar.ax.tick_params(labelsize=14)  # 调整颜色条刻度字体
#     plt.title(f"{title_prefix}Cluster Confidence ({title_suffix})", fontsize=14, fontweight='bold')
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)

#     plt.tight_layout()
#     plt.savefig(f"results/SAR/combing/gnn/{title_prefix.lower()}_{method}.svg")
#     plt.savefig(f"results/SAR/combing/gnn/{title_prefix.lower()}_{method}.png", dpi=900)
#     plt.show()

# def main():
#     file_path = "dataset/train/pc-gnn.pt"
#     try:
#         # 尝试使用 weights_only=True 加载
#         graph_data = torch.load(file_path, weights_only=True)
#     except _pickle.UnpicklingError:
#         print("使用 weights_only=True 加载失败，尝试使用 weights_only=False 加载...")
#         # 若失败，使用 weights_only=False 重新加载
#         graph_data = torch.load(file_path, weights_only=False)
#     df = pd.read_excel("dataset/all compounds analysis.xlsx")

#     # 增强特征提取
#     X = enhanced_feature_extractor(graph_data, df['smiles'])

#     # 特征标准化
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # PCA降维，减少特征维度，去除噪声
#     pca = PCA(n_components=0.95)  # 保留95%方差
#     X_pca = pca.fit_transform(X_scaled)

#     y = np.array([data.y.item() for data in graph_data])

#     # 构建动态约束矩阵
#     sim_matrix = dynamic_constraint_matrix(X_pca, y)

#     # 新增相似度矩阵可视化
#     plt.figure(figsize=(10, 8))
#     # 获取viridis颜色映射
#     viridis_cmap = plt.get_cmap('viridis')
#     # 创建一个新的颜色映射，调整透明度
#     new_cmap = ListedColormap(viridis_cmap(np.linspace(0, 1, 256)) * [1, 1, 1, 0.5])
#     sns.heatmap(sim_matrix[:20, :20], cmap=new_cmap)
#     plt.title("Similarity Matrix Subset", fontsize=14, fontweight='bold')
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.savefig("results/SAR/combing/gnn/sim_matrix_heatmap.svg")
#     plt.savefig("results/SAR/combing/gnn/sim_matrix_heatmap.png")
#     plt.close()  # 防止内存泄漏

#     # 优化谱聚类（调整参数）
#     spectral = SpectralClustering(
#         n_clusters=2,
#         affinity='precomputed',
#         assign_labels='cluster_qr',
#         random_state=42,
#         eigen_solver='lobpcg',
#         n_init=100,
#         gamma=0.1  # 调整相似性矩阵的影响
#     )
#     cluster_labels_spectral = spectral.fit_predict(sim_matrix)

#     # 尝试KMeans聚类
#     kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
#     cluster_labels_kmeans = kmeans.fit_predict(X_pca)

#     # 综合评估 - 谱聚类
#     labeled_idx = (y != -1)
#     if sum(labeled_idx) > 0:
#         ari_spectral = adjusted_rand_score(y[labeled_idx], cluster_labels_spectral[labeled_idx])
#         sil_score_spectral = silhouette_score(X_pca, cluster_labels_spectral)
#         print(f"谱聚类优化后评估指标：\nARI: {ari_spectral:.3f}\nSilhouette: {sil_score_spectral:.3f}")

#         # 综合评估 - KMeans
#         ari_kmeans = adjusted_rand_score(y[labeled_idx], cluster_labels_kmeans[labeled_idx])
#         sil_score_kmeans = silhouette_score(X_pca, cluster_labels_kmeans)
#         print(f"KMeans评估指标：\nARI: {ari_kmeans:.3f}\nSilhouette: {sil_score_kmeans:.3f}")

#     # 可视化 - 谱聚类
#     visualize_supervised_clusters(X_pca, cluster_labels_spectral, y, method='tsne', title_prefix="Spectral ")
#     visualize_supervised_clusters(X_pca, cluster_labels_spectral, y, method='umap', title_prefix="Spectral ")

#     # 可视化 - KMeans
#     visualize_supervised_clusters(X_pca, cluster_labels_kmeans, y, method='tsne', title_prefix="KMeans-")
#     visualize_supervised_clusters(X_pca, cluster_labels_kmeans, y, method='umap', title_prefix="KMeans-")

# if __name__ == "__main__":
#     import _pickle
#     main()












import torch
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
import matplotlib.font_manager as fm
from rdkit import DataStructs

# 设置中文字体
# plt.rcParams['font.family'] = 'Times New Roman'
#
# def enhanced_feature_extractor(graph_data, smiles_list):
#     """分子特征提取器（无监督版）"""
#     feature_list = []
#     for data, smiles in zip(graph_data, smiles_list):
#         mol = Chem.MolFromSmiles(smiles)
#
#         # 图结构特征
#         node_feat = data.x.mean(dim=0).numpy()
#         edge_feat = data.edge_attr.mean(dim=0).numpy()
#
#         # 分子描述符
#         topo_feat = [
#             Descriptors.NumAromaticRings(mol),
#             Descriptors.TPSA(mol),
#             Descriptors.MolLogP(mol),
#             Descriptors.NumRotatableBonds(mol),
#             rdMolDescriptors.CalcNumAtoms(mol),
#             rdMolDescriptors.CalcNumHeteroatoms(mol)
#         ]
#
#         # 分子指纹
#         fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=1024)
#         fp_array = np.zeros(1024)
#         DataStructs.ConvertToNumpyArray(fp, fp_array)
#
#         combined_feat = np.concatenate([node_feat, edge_feat, topo_feat, fp_array])
#         feature_list.append(combined_feat)
#
#     return np.array(feature_list)
#
# def auto_cluster(X, method='spectral', max_clusters=5):
#     """自动选择聚类数的改进方法"""
#     # 基于轮廓分析的聚类数选择
#     silhouette_scores = []
#     cluster_range = range(2, max_clusters+1)
#
#     for k in cluster_range:
#         if method == 'kmeans':
#             model = KMeans(n_clusters=k, n_init=10, random_state=42)
#         elif method == 'spectral':
#             model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors',
#                                      random_state=42, n_init=10)
#         labels = model.fit_predict(X)
#         silhouette_scores.append(silhouette_score(X, labels))
#
#     optimal_k = cluster_range[np.argmax(silhouette_scores)]
#
#     # 执行最终聚类
#     if method == 'kmeans':
#         return KMeans(n_clusters=optimal_k, n_init=10, random_state=42).fit(X)
#     return SpectralClustering(n_clusters=optimal_k, affinity='nearest_neighbors',
#                              random_state=42).fit(X)
#
# def visualize_dual(X, cluster_labels, true_labels, method='tsne'):
#     """双模式可视化：聚类椭圆+真实标签"""
#     # 降维处理
#     reducer = TSNE(n_components=2, random_state=42) if method == 'tsne' else umap.UMAP()
#     X_emb = reducer.fit_transform(X)
#
#     plt.figure(figsize=(18, 6))
#
#     # 子图1：聚类可视化
#     ax1 = plt.subplot(121)
#     colors = plt.cm.tab10(cluster_labels % 10)
#
#     # 绘制椭圆区域
#     gmm = GaussianMixture(n_components=len(np.unique(cluster_labels)),
#                          covariance_type='full', random_state=42)
#     gmm.fit(X_emb)
#
#     for i in range(gmm.n_components):
#         # 计算椭圆参数
#         mean = gmm.means_[i]
#         cov = gmm.covariances_[i]
#         eigvals, eigvecs = np.linalg.eigh(cov)
#         angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
#
#         # 绘制双层椭圆
#         for n_std in [1.5, 2.5]:
#             width = 2 * n_std * np.sqrt(eigvals[0])
#             height = 2 * n_std * np.sqrt(eigvals[1])
#             ellipse = Ellipse(mean, width, height, angle,
#                             edgecolor=colors[i], linestyle='--',
#                             facecolor='none', alpha=0.3, linewidth=2)
#             ax1.add_patch(ellipse)
#
#     ax1.scatter(X_emb[:,0], X_emb[:,1], c=colors, alpha=0.6, edgecolors='w', s=40)
#     ax1.set_title(f"Clustering Result ({method.upper()})", fontsize=14)
#
#     # 子图2：真实标签可视化
#     ax2 = plt.subplot(122)
#     colors = np.where(true_labels == 1, 'red', 'blue')
#     ax2.scatter(X_emb[:,0], X_emb[:,1], c=colors, alpha=0.6, edgecolors='w', s=40)
#
#     # 添加决策边界示意线
#     x_min, x_max = X_emb[:,0].min()-1, X_emb[:,0].max()+1
#     y_min, y_max = X_emb[:,1].min()-1, X_emb[:,1].max()+1
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
#                         np.linspace(y_min, y_max, 50))
#
#     # 简单线性分割示意
#     Z = np.where(xx > np.median(X_emb[:,0]), 1, 0)
#     ax2.contourf(xx, yy, Z, alpha=0.1, colors=['blue','red'])
#
#     ax2.set_title("True Label Distribution", fontsize=14)
#     plt.tight_layout()
#     plt.savefig(f"results/SAR/combing/gnn/dual_view_{method}.png", dpi=300)
#     plt.show()
#
# def main():
#     # 数据加载
#     try:
#         graph_data = torch.load("dataset/train/pc-gnn.pt",
#                                weights_only=True)
#     except Exception as e:
#         print(f"数据加载失败: {e}")
#         return
#
#     df = pd.read_excel("dataset/all compounds analysis.xlsx")
#
#     # 特征工程
#     X = enhanced_feature_extractor(graph_data, df['smiles'])
#     X = StandardScaler().fit_transform(X)
#     X = PCA(n_components=0.9).fit_transform(X)  # 更强的降维
#
#     # 获取真实标签
#     true_labels = df['label'].values
#
#     # 谱聚类分析
#     spectral = auto_cluster(X, method='spectral')
#     print(f"谱聚类轮廓系数: {silhouette_score(X, spectral.labels_):.3f}")
#
#     # K-means分析
#     kmeans = auto_cluster(X, method='kmeans')
#     print(f"K-means轮廓系数: {silhouette_score(X, kmeans.labels_):.3f}")
#
#     # 可视化对比
#     for method in ['tsne', 'umap']:
#         visualize_dual(X, spectral.labels_, true_labels, method)
#
# if __name__ == "__main__":
#     main()