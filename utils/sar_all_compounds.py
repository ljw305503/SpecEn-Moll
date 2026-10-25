
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
RESULTS_DIR = REPO_ROOT / "results"
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import numpy as np
# import networkx as nx
#
# # 设置全局字体为 Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
#
# # 读取Excel文件
# file_path = r'results/SAR/combing/all plant species analysis.xlsx'
# df = pd.read_excel(file_path, sheet_name='Sheet1')
#
# # 定义目标物种列表
# species_list = [
#     'Pomacea canaliculata',
#     'Oncomelania hupensis',
#     'Cornu aspersum',
#     'Deroceras reticulatum'
# ]
#
# # 获取每个物种的plants集合
# species_sets = {}
# for species in species_list:
#     plants = df[df['molluscs'] == species]['plants species'].dropna().unique()
#     species_sets[species] = set(plants)
#
# # 计算相似性矩阵（基于两两交集大小）
# similarity = np.zeros((4, 4))
# for i in range(4):
#     for j in range(i + 1, 4):
#         intersect = len(species_sets[species_list[i]] & species_sets[species_list[j]])
#         similarity[i][j] = similarity[j][i] = intersect
#
# # 使用力导向布局生成坐标
# G = nx.Graph()
# for i in range(4):
#     G.add_node(i)
# for i in range(4):
#     for j in range(i + 1, 4):
#         if similarity[i][j] > 0:
#             G.add_edge(i, j, weight=similarity[i][j])
#
# # 生成布局并缩放坐标
# pos = nx.spring_layout(G, weight='weight', seed=42, iterations=1000)
# coords = np.array([pos[i] for i in range(4)]) * 8  # 坐标缩放因子
#
# # 计算矩形尺寸（面积与数据量成正比）
# sizes = [len(species_sets[s]) for s in species_list]
# max_size = max(sizes)
# scale_factor = 10  # 控制整体大小
# rect_sizes = [np.sqrt(s / max_size) * scale_factor for s in sizes]
#
# # 定义颜色和透明度
# colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
# alpha = 0.6
#
# # 创建画布
# fig, ax = plt.subplots(figsize=(10, 8))
# ax.set_aspect('equal')
# ax.axis('off')
#
# # 绘制矩形
# for i, (x, y) in enumerate(coords):
#     size = rect_sizes[i]
#     rect = Rectangle((x - size / 2, y - size / 2), size, size,
#                      facecolor=colors[i], alpha=alpha,
#                      edgecolor='black', lw=1.5)
#     ax.add_patch(rect)
#
#     # 添加物种名称
#     ax.text(x, y, species_list[i],
#             ha='center', va='center',
#             fontsize=9, style='italic')
#
# # 设置坐标范围
# padding = 8
# ax.set_xlim(coords[:, 0].min() - padding, coords[:, 0].max() + padding)
# ax.set_ylim(coords[:, 1].min() - padding, coords[:, 1].max() + padding)
#
# plt.title('Species Plants Distribution Diagram', pad=20)
# plt.tight_layout()
# plt.savefig(r"results/SAR/combing/1.png")
# plt.savefig(r"results/SAR/combing/1.svg")
# plt.show()





# from sklearn.preprocessing import MinMaxScaler
# import numpy as np
# from scipy.stats import spearmanr
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
#
# # 设置全局字体为 Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
#
# # 读取 CSV 文件
# df = pd.read_csv(r"results/SAR/combing/all compounds analysis3.csv",encoding="utf-8")
# # df["Activity Classification 1"] = df["Activity Classification 1"].replace({0: "00", 1: "01"})
# # 目标列：Activity Classification 1
# target_col = "label"
#
# # 选取数值列（排除非数值列）
# numeric_cols = df.select_dtypes(include=['number']).columns
#
# # 排除不需要归一化的列
# exclude_cols = ['label', 'Number']
#
# # 进一步排除二进制列（只包含 0 和 1 的列）
# binary_cols = [col for col in numeric_cols if set(df[col].dropna().unique()).issubset({0, 1})]
#
# # 需要归一化的列 = 所有数值列 - 需要排除的列
# normalize_cols = [col for col in numeric_cols if col not in exclude_cols + binary_cols]
#
# # 初始化 MinMaxScaler
# scaler = MinMaxScaler()
#
# # 对选定的列进行归一化
# df[normalize_cols] = scaler.fit_transform(df[normalize_cols])
#
# print(df.head())  # 查看归一化后的数据
#
#
# # 计算所有列与目标列的 Spearman 相关系数
# correlations = {}
# for col in df.columns:
#     if col != target_col:  # 排除自身计算
#         corr, p_value = spearmanr(df[col], df[target_col])
#         correlations[col] = (corr, p_value)
#
# # 转换为 DataFrame 方便查看
# corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=["Spearman Corr", "p-value"])
#
# # 按相关系数排序
# corr_df = corr_df.sort_values(by="Spearman Corr", ascending=False)
#
# # # 打印结果
# # # print(corr_df)
# # sns.pairplot(df, hue="Activity Classification 1", vars=normalize_cols, diag_kind="kde")
# # plt.show()
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 仅显示最相关的前 20 个特征（去掉不需要的列）
# top_n = 20
# corr_df_top = corr_df.drop(index=["Activity Classification 2", "10mg"], errors="ignore").head(top_n)
#
# fig, ax1 = plt.subplots(figsize=(12, 8))
#
# # 颜色映射
# colors = sns.color_palette("Spectral", len(corr_df_top))
#
# # 绘制 Spearman 相关系数的条形图
# ax1.barh(corr_df_top.index, corr_df_top["Spearman Corr"], color=colors, alpha=0.7)
# ax1.set_xlabel("Spearman Correlation", fontsize=14)
# ax1.set_ylabel("Features", fontsize=14)
# ax1.set_title(f"Top {top_n} Spearman Correlations with label", fontsize=16)
# ax1.invert_yaxis()  # 让相关性最高的特征排在最上面
#
# # 添加数值标签（相关系数）
# for i, v in enumerate(corr_df_top["Spearman Corr"]):
#     ax1.text(v, i, f"{v:.2f}", ha="left", va="center", fontsize=12)
#
# # 创建第二个 y 轴用于 p-value
# ax2 = ax1.twiny()
# ax2.scatter(corr_df_top["p-value"], corr_df_top.index, color='black', label="p-value", marker='o', s=50)
# ax2.set_xlabel("p-value (log scale)", fontsize=14)
# ax2.set_xscale("log")  # 使用对数刻度，避免 p 值过大时挤在一起
#
# # 添加图例
# ax2.legend(loc="lower right")
#
# # 保存图片
# plt.savefig(r"results/SAR/combing/spearman_correlation_with_pvalue.png", dpi=800, bbox_inches="tight")
# plt.savefig(r"results/SAR/combing/spearman_correlation_with_pvalue.svg")
# # 显示图像
# plt.show()
#
# # 提取最正负相关的前5个特征
# positive_corr_df = corr_df[corr_df['Spearman Corr'] > 0]
# top_positive_features = positive_corr_df.index[:5]
#
# # negative_corr_df = corr_df[corr_df['Spearman Corr'] < 0]
# # top_negative_features = positive_corr_df.index[:5]
#
# if len(top_positive_features) == 0:
#     print("没有正相关特征，无法绘制")
# else:
#     plt.figure(figsize=(15, 6))
#     for i, feature in enumerate(top_positive_features):
#         plt.subplot(1, len(top_positive_features), i+1)
#         # 整理数据格式
#         temp_df = df[[feature, 'label']].melt(id_vars='label', value_vars=feature)
#         temp_df.rename(columns={'value': feature}, inplace=True)
#         # 绘制雨云图（小提琴图+散点+箱线图组合）
#         sns.violinplot(
#             data=temp_df, x='label', y=feature, hue='label',
#             palette={0: '#7DF9FF', 1: '#F8C8DC'},  # 修改这里的键为数值类型
#             split=True, inner=None
#         )
#         sns.swarmplot(
#             data=temp_df, x='label', y=feature, hue='label',
#             palette={0: 'blue', 1: 'red'},  # 同步修改这里
#             dodge=True, size=3
#         )
#         sns.boxplot(
#             data=temp_df, x='label', y=feature, hue='label',
#             palette={0: '#7DF9FF', 1: '#F8C8DC'},  # 同步修改这里
#             width=0.2, notch=True
#         )
#         plt.title(feature, fontsize=14)
#         plt.xlabel('label', fontsize=12)
#         plt.ylabel(feature, fontsize=12)
#         plt.legend([], frameon=False)  # 隐藏重复图例
#     plt.tight_layout()
#     plt.savefig(r"results/SAR/combing/raincloud_plots1.png", dpi=600, bbox_inches="tight")
#     plt.savefig(r"results/SAR/combing/raincloud_plots1.svg", bbox_inches="tight")
#     # plt.show()
#
#     # plt.savefig(r"results/SAR/combing/raincloud_plots2.png", dpi=600, bbox_inches="tight")
#     # plt.savefig(r"results/SAR/combing/raincloud_plots2.svg", bbox_inches="tight")
#     plt.show()








# # -*- coding: utf-8 -*-
# """
# 基于多策略优化的化合物聚类分析
# 功能：1. 组合指纹特征 2. 智能聚类算法选择 3. 多维度评估 4. 深度可视化
# """
#
# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import AllChem, Draw, DataStructs, rdMolDescriptors
# from sklearn.cluster import AgglomerativeClustering, DBSCAN
# from sklearn.metrics import silhouette_score, calinski_harabasz_score
# from scipy.cluster.hierarchy import dendrogram, linkage
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from joblib import Parallel, delayed
# from tqdm import tqdm
# import warnings
# import umap  # 新增UMAP导入
# from matplotlib.patches import Ellipse  # 新增椭圆绘制库
#
# warnings.filterwarnings('ignore')
# # 设置全局字体为 Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
#
#
# def get_combined_fingerprint(mol):
#     """生成组合分子指纹"""
#     # Morgan指纹（局部结构）
#     morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
#     # 拓扑指纹（全局结构）
#     topological = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=512)
#     # MACCS密钥（预定义官能团）
#     maccs = AllChem.GetMACCSKeysFingerprint(mol)
#
#     # 手动合并指纹
#     def _merge_fingerprints(*fps):
#         """合并多个ExplicitBitVect指纹"""
#         total_bits = sum(fp.GetNumBits() for fp in fps)
#         combined = DataStructs.ExplicitBitVect(total_bits)
#         offset = 0
#         for fp in fps:
#             for bit in fp.GetOnBits():
#                 combined.SetBit(bit + offset)
#             offset += fp.GetNumBits()
#         return combined
#
#     return _merge_fingerprints(morgan, topological, maccs)
#
#
# def calculate_similarity(i, j, fps):
#     """并行计算相似度"""
#     return DataStructs.TanimotoSimilarity(fps[i], fps[j])
#
#
# def main():
#     # ===================== 参数配置 =====================
#     input_file = str(DATASET_DIR / "all compounds analysis.xlsx")
#     output_file = r"results/SAR/combing/all compounds analysis_optimized1.xlsx"
#     fingerprint_params = {
#         'radius': 5,
#         'nBits': 1024
#     }
#     dbscan_params = {
#         'eps': 0.6,
#         'min_samples': 3
#     }
#
#     # ===================== 数据预处理 =====================
#     print("数据预处理...")
#     df = pd.read_excel(input_file, engine='openpyxl')
#     smiles_list = df['smiles'].fillna('').tolist()
#
#     # 提取Number列作为后续标签
#     number_list = df['Number'].tolist()
#     label_list = df['label'].tolist()
#     # 分子处理与去重
#     seen = set()
#     valid_mols = []
#     valid_indices = []  # 保存原始数据中的行索引
#     for idx, smi in enumerate(smiles_list):
#         mol = Chem.MolFromSmiles(smi)
#         if mol:
#             smi_hash = Chem.MolToSmiles(mol, isomericSmiles=True)
#             if smi_hash not in seen:
#                 seen.add(smi_hash)
#                 AllChem.Compute2DCoords(mol)
#                 valid_mols.append(mol)
#                 valid_indices.append(idx)  # 记录有效分子对应的原始行号
#         else:
#             print(f"无效SMILES: 第{idx + 1}行 - {smi}")
#     valid_labels = np.array(label_list)[valid_indices]
#
#     # ===================== 特征工程 =====================
#     print("生成组合指纹...")
#     fps = [get_combined_fingerprint(mol) for mol in tqdm(valid_mols, desc="生成指纹")]
#
#     # ===================== 相似度计算 =====================
#     print("计算相似度矩阵...")
#     n = len(fps)
#     sim_matrix = np.eye(n)
#
#     # 并行计算上三角部分
#     results = Parallel(n_jobs=-1)(
#         delayed(calculate_similarity)(i, j, fps)
#         for i in range(n) for j in range(i + 1, n)
#     )
#
#     # 填充矩阵
#     triu_indices = np.triu_indices(n, 1)
#     sim_matrix[triu_indices[0], triu_indices[1]] = results
#     sim_matrix = np.maximum(sim_matrix, sim_matrix.T)  # 保持对称性
#     distance_matrix = 1 - sim_matrix
#
#     # ===================== 聚类分析 =====================
#     print("\n聚类分析中...")
#     # 方法1：DBSCAN聚类
#     db = DBSCAN(eps=dbscan_params['eps'],
#                 min_samples=dbscan_params['min_samples'],
#                 metric='precomputed')
#     db_labels = db.fit_predict(distance_matrix)
#
#     # 方法2：层次聚类（智能选择最佳k）
#     silhouette_scores = []
#     calinski_scores = []
#     possible_k = range(2, min(15, len(valid_mols) - 1))
#
#     for k in tqdm(possible_k, desc="评估聚类数"):
#         model = AgglomerativeClustering(n_clusters=k,
#                                         affinity='precomputed',
#                                         linkage='average')
#         labels = model.fit_predict(distance_matrix)
#         if len(set(labels)) < 2:
#             silhouette_scores.append(-1)
#             calinski_scores.append(-1)
#             continue
#
#         # 使用PCA降维后的特征计算Calinski-Harabasz指数
#         pca = PCA(n_components=10)
#         reduced_features = pca.fit_transform([np.array(fp) for fp in fps])
#         silhouette_scores.append(silhouette_score(distance_matrix, labels, metric='precomputed'))
#         calinski_scores.append(calinski_harabasz_score(reduced_features, labels))
#
#     # 选择最佳k值
#     best_k = possible_k[np.argmax(silhouette_scores)]
#     print(f"自动选择最佳聚类数: {best_k}")
#
#     # 执行层次聚类
#     agg = AgglomerativeClustering(n_clusters=best_k,
#                                   affinity='precomputed',
#                                   linkage='average')
#     agg_labels = agg.fit_predict(distance_matrix)
#
#     # ===================== 结果评估 =====================
#     def evaluate_clustering(labels, method_name):
#         """评估聚类结果"""
#         if len(set(labels)) < 2:
#             return {'silhouette': -1, 'calinski': -1}
#
#         # 过滤噪声点（仅对DBSCAN）
#         if -1 in labels:
#             valid = np.array(labels) != -1
#             filtered_labels = labels[valid]
#             filtered_features = [np.array(fp) for i, fp in enumerate(fps) if valid[i]]
#         else:
#             filtered_labels = labels
#             filtered_features = [np.array(fp) for fp in fps]
#
#         # 降维计算Calinski
#         pca = PCA(n_components=15)
#         reduced_features = pca.fit_transform(filtered_features)
#
#         return {
#             'silhouette': silhouette_score(distance_matrix, labels, metric='precomputed'),
#             'calinski': calinski_harabasz_score(reduced_features, filtered_labels)
#         }
#
#     # 评估两种方法
#     db_eval = evaluate_clustering(db_labels, "DBSCAN")
#     agg_eval = evaluate_clustering(agg_labels, "Hierarchical")
#
#     # 选择最佳方法
#     if db_eval['silhouette'] > agg_eval['silhouette']:
#         best_labels = db_labels
#         best_method = "DBSCAN"
#     else:
#         best_labels = agg_labels
#         best_method = "Hierarchical"
#
#     # ===================== 结果保存 =====================
#     df['Cluster'] = -1
#     for idx, lab in zip(valid_indices, best_labels):
#         df.at[idx, 'Cluster'] = lab
#     df.to_excel(output_file, index=False)
#
#     # ===================== 可视化分析 =====================
#     print("生成可视化...")
#
#     # 可视化1：umap和指标分析
#     plt.figure(figsize=(15, 8))
#
#     # 手动设置颜色映射（0=蓝色，1=红色）
#     colors = np.array([plt.cm.Blues(0.6) if lbl == 0 else plt.cm.Reds(0.6) for lbl in valid_labels])
#
#     reducer = umap.UMAP(
#         n_components=2,
#         random_state=50,
#         n_neighbors=100,  # 增大局部邻域
#         min_dist=0.01,  # 增大最小距离，使布局更分散
#         metric='manhattan'  # 更换距离度量
#     )
#     projections = reducer.fit_transform([np.array(fp) for fp in fps])
#
#     ax = plt.subplot(111)
#     scatter = ax.scatter(
#         projections[:, 0], projections[:, 1],
#         c=colors,  # 直接使用预先生成的颜色数组
#         alpha=0.7, edgecolors='w', s=50
#     )
#
#     # 绘制聚类椭圆（保持不变）
#     unique_clusters = np.unique(best_labels)
#     cluster_colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_clusters)))
#     legend_elements = []
#
#     for i, cluster in enumerate(unique_clusters):
#         if cluster == -1:
#             continue
#         cluster_indices = np.where(best_labels == cluster)[0]
#         cluster_points = projections[cluster_indices]
#
#         mean = np.mean(cluster_points, axis=0)
#         cov = np.cov(cluster_points, rowvar=False)
#         eig_vals, eig_vecs = np.linalg.eig(cov)
#         order = eig_vals.argsort()[::-1]
#         eig_vals, eig_vecs = eig_vals[order], eig_vecs[:, order]
#
#         angle = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))
#         width, height = 2.4 * np.sqrt(eig_vals) * 2.0  # 2倍标准差椭圆
#
#         ellipse = Ellipse(
#             mean, width=width, height=height, angle=angle,
#             edgecolor=cluster_colors[i], facecolor='none', lw=2, alpha=0.8
#         )
#         ax.add_patch(ellipse)
#         legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {cluster}',
#                                           markerfacecolor='white', markersize=12, markeredgecolor=cluster_colors[i],
#                                           markeredgewidth=1))
#
#     # 添加红色圆点和蓝色圆点到图例
#     legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Label 0',
#                                       markerfacecolor=plt.cm.Blues(0.6), markersize=10))
#     legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Label 1',
#                                       markerfacecolor=plt.cm.Reds(0.6), markersize=10))
#
#     ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
#
#     # plt.colorbar(scatter, label='Original Label (0=Blue, 1=Red)')
#     plt.title(f'{best_method} Clustering (UMAP Visualization)')
#     plt.xlabel('UMAP Component 1')
#     plt.ylabel('UMAP Component 2')
#     plt.savefig(r"results/SAR/combing/umap_clustering1.png", dpi=300, bbox_inches='tight')
#     plt.savefig(r"results/SAR/combing/umap_clustering1.svg")
#
#     # 轮廓系数分析
#     plt.subplot(122)
#     plt.plot(possible_k, silhouette_scores, 'b-o', label='Silhouette')
#     plt.plot(possible_k, np.array(calinski_scores) / max(calinski_scores),
#              'r--^', label='Calinski (scaled)')
#     plt.axvline(x=best_k, color='k', linestyle='--')
#     plt.title('Cluster Number Optimization')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Score')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(r"results/SAR/combing/clustering_analysis1.png")
#     plt.savefig(r"results/SAR/combing/clustering_analysis1.svg")
#
#     # 可视化2：层次聚类树状图（仅当最佳方法是层次聚类时生成）
#     if best_method == "Hierarchical":
#         plt.figure(figsize=(15, 10))
#
#         # 获取有效分子对应的Number标签
#         cluster_numbers = [number_list[idx] for idx in valid_indices]  # 通过valid_indices获取对应的Number值
#
#         linked = linkage(distance_matrix, method='average')
#         # 修改dendrogram的labels参数为cluster_numbers，并添加横坐标标签
#         dendrogram(linked, orientation='top', labels=cluster_numbers, leaf_rotation=90)  # 添加leaf_rotation避免标签重叠
#         plt.xlabel("Compound Number")  # 新增横坐标标签
#         plt.ylabel("Distance")  # 新增纵坐标标签（表示层次聚类的距离度量）
#         plt.title(f"Hierarchical Cluster Dendrogram (Silhouette={agg_eval['silhouette']:.2f})")
#         plt.savefig(r"results/SAR/combing/dendrogram1.png")
#         plt.savefig(r"results/SAR/combing/dendrogram1.svg")
#
#     # 可视化3：分子结构展示（独立于聚类方法，统一处理）
#     # 统一处理所有有效聚类（排除噪声点-1）
#     unique_labels = np.unique(best_labels)
#     valid_clusters = [clust for clust in unique_labels if clust != -1]
#     n_clusters = len(valid_clusters)
#
#     if n_clusters > 0:  # 确保至少有一个有效聚类
#         fig, axes = plt.subplots(n_clusters, 1, figsize=(50, 20 * n_clusters))
#         if n_clusters == 1:
#             axes = [axes]  # 处理单子图情况
#
#         for ax_idx, cluster_id in enumerate(valid_clusters):
#             members = np.where(best_labels == cluster_id)[0]
#             member_numbers = [str(number_list[valid_indices[member]]) for member in members[:20]]
#             mols = [valid_mols[member] for member in members[:20]]  # 每类最多显示10个
#
#             if not mols:  # 跳过空聚类（理论上不会出现）
#                 continue
#
#             # 计算类内统计
#             intra_dists = distance_matrix[np.ix_(members, members)]
#             np.fill_diagonal(intra_dists, np.nan)
#             stats = {
#                 'count': len(members),
#                 'avg_sim': 1 - np.nanmean(intra_dists),
#                 'min_sim': 1 - np.nanmax(intra_dists),
#                 'max_sim': 1 - np.nanmin(intra_dists)
#             }
#
#             # 绘制分子网格图
#             img = Draw.MolsToGridImage(
#                 mols,
#                 molsPerRow=10,
#                 subImgSize=(800, 800),
#                 legends=member_numbers
#             )
#             axes[ax_idx].imshow(img)
#             axes[ax_idx].axis('off')
#             axes[ax_idx].set_title(
#                 f"Cluster {cluster_id} | 分子数: {stats['count']} | "
#                 f"平均相似度: {stats['avg_sim']:.2f} | "
#                 f"范围: {stats['min_sim']:.2f}-{stats['max_sim']:.2f}"
#             )
#
#         plt.tight_layout()
#         plt.savefig(r"results/SAR/combing/cluster_structures1.png")
#         plt.savefig(r"results/SAR/combing/cluster_structures1.svg")
#
#     # ===================== 结果报告 =====================
#     if best_method == "DBSCAN":
#         n_clusters_db = len(np.unique(best_labels)) - (1 if -1 in best_labels else 0)
#         noise_points = list(best_labels).count(-1)
#     else:
#         n_clusters_db = best_k
#         noise_points = 0
#
#     print("\n===== 聚类报告 =====")
#     print(f"有效分子数: {len(valid_mols)}")
#     print(f"最佳方法: {best_method}")
#     print(f"轮廓系数: {max(db_eval['silhouette'], agg_eval['silhouette']):.3f}")
#     print(f"Calinski-Harabasz指数: {max(db_eval['calinski'], agg_eval['calinski']):.1f}")
#
#     if best_method == "DBSCAN":
#         print(f"识别聚类数: {n_clusters_db} (包含{noise_points}个噪声点)")
#     else:
#         print(f"确定聚类数: {best_k}")
#
#
# if __name__ == "__main__":
#     main()

