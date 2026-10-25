# -*- coding: utf-8 -*-
"""
基于多策略优化的化合物聚类分析
功能：1. 组合指纹特征 2. 智能聚类算法选择 3. 多维度评估 4. 深度可视化
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
RESULTS_DIR = REPO_ROOT / "results"

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, DataStructs, rdMolDescriptors
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

def get_combined_fingerprint(mol):
    """生成组合分子指纹"""
    # Morgan指纹（局部结构）
    morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
    # 拓扑指纹（全局结构）
    topological = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=512)
    # MACCS密钥（预定义官能团）
    maccs = AllChem.GetMACCSKeysFingerprint(mol)

    # 手动合并指纹
    def _merge_fingerprints(*fps):
        """合并多个ExplicitBitVect指纹"""
        total_bits = sum(fp.GetNumBits() for fp in fps)
        combined = DataStructs.ExplicitBitVect(total_bits)
        offset = 0
        for fp in fps:
            for bit in fp.GetOnBits():
                combined.SetBit(bit + offset)
            offset += fp.GetNumBits()
        return combined

    return _merge_fingerprints(morgan, topological, maccs)


def calculate_similarity(i, j, fps):
    """并行计算相似度"""
    return DataStructs.TanimotoSimilarity(fps[i], fps[j])


def main():
    # ===================== 参数配置 =====================
    input_file = str(DATASET_DIR / "Deroceras reticulatum compounds analysis data.xlsx")
    output_file = str(RESULTS_DIR / "SAR" / "Deroceras reticulatum" / "Deroceras reticulatum analysis data1.xlsx")
    fingerprint_params = {
        'radius': 3,
        'nBits': 1024
    }
    dbscan_params = {
        'eps': 0.35,
        'min_samples': 3
    }

    # ===================== 数据预处理 =====================
    print("数据预处理...")
    df = pd.read_excel(input_file, engine='openpyxl')
    smiles_list = df['smiles'].fillna('').tolist()

    # 提取Number列作为后续标签
    number_list = df['Number'].tolist()

    # 分子处理与去重
    seen = set()
    valid_mols = []
    valid_indices = []  # 保存原始数据中的行索引
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smi_hash = Chem.MolToSmiles(mol, isomericSmiles=True)
            if smi_hash not in seen:
                seen.add(smi_hash)
                AllChem.Compute2DCoords(mol)
                valid_mols.append(mol)
                valid_indices.append(idx)  # 记录有效分子对应的原始行号
        else:
            print(f"无效SMILES: 第{idx + 1}行 - {smi}")

    # ===================== 特征工程 =====================
    print("生成组合指纹...")
    fps = [get_combined_fingerprint(mol) for mol in tqdm(valid_mols, desc="生成指纹")]

    # ===================== 相似度计算 =====================
    print("计算相似度矩阵...")
    n = len(fps)
    sim_matrix = np.eye(n)

    # 并行计算上三角部分
    results = Parallel(n_jobs=-1)(
        delayed(calculate_similarity)(i, j, fps)
        for i in range(n) for j in range(i + 1, n)
    )

    # 填充矩阵
    triu_indices = np.triu_indices(n, 1)
    sim_matrix[triu_indices[0], triu_indices[1]] = results
    sim_matrix = np.maximum(sim_matrix, sim_matrix.T)  # 保持对称性
    distance_matrix = 1 - sim_matrix

    # ===================== 聚类分析 =====================
    print("\n聚类分析中...")
    # 方法1：DBSCAN聚类
    db = DBSCAN(eps=dbscan_params['eps'],
                min_samples=dbscan_params['min_samples'],
                metric='precomputed')
    db_labels = db.fit_predict(distance_matrix)

    # 方法2：层次聚类（智能选择最佳k）
    silhouette_scores = []
    calinski_scores = []
    possible_k = range(2, min(15, len(valid_mols) - 1))

    for k in tqdm(possible_k, desc="评估聚类数"):
        model = AgglomerativeClustering(n_clusters=k,
                                        affinity='precomputed',
                                        linkage='average')
        labels = model.fit_predict(distance_matrix)
        if len(set(labels)) < 2:
            silhouette_scores.append(-1)
            calinski_scores.append(-1)
            continue

        # 使用PCA降维后的特征计算Calinski-Harabasz指数
        pca = PCA(n_components=10)
        reduced_features = pca.fit_transform([np.array(fp) for fp in fps])
        silhouette_scores.append(silhouette_score(distance_matrix, labels, metric='precomputed'))
        calinski_scores.append(calinski_harabasz_score(reduced_features, labels))

    # 选择最佳k值
    best_k = possible_k[np.argmax(silhouette_scores)]
    print(f"自动选择最佳聚类数: {best_k}")

    # 执行层次聚类
    agg = AgglomerativeClustering(n_clusters=best_k,
                                  affinity='precomputed',
                                  linkage='average')
    agg_labels = agg.fit_predict(distance_matrix)

    # ===================== 结果评估 =====================
    def evaluate_clustering(labels, method_name):
        """评估聚类结果"""
        if len(set(labels)) < 2:
            return {'silhouette': -1, 'calinski': -1}

        # 过滤噪声点（仅对DBSCAN）
        if -1 in labels:
            valid = np.array(labels) != -1
            filtered_labels = labels[valid]
            filtered_features = [np.array(fp) for i, fp in enumerate(fps) if valid[i]]
        else:
            filtered_labels = labels
            filtered_features = [np.array(fp) for fp in fps]

        # 降维计算Calinski
        pca = PCA(n_components=10)
        reduced_features = pca.fit_transform(filtered_features)

        return {
            'silhouette': silhouette_score(distance_matrix, labels, metric='precomputed'),
            'calinski': calinski_harabasz_score(reduced_features, filtered_labels)
        }

    # 评估两种方法
    db_eval = evaluate_clustering(db_labels, "DBSCAN")
    agg_eval = evaluate_clustering(agg_labels, "Hierarchical")

    # 选择最佳方法
    if db_eval['silhouette'] > agg_eval['silhouette']:
        best_labels = db_labels
        best_method = "DBSCAN"
    else:
        best_labels = agg_labels
        best_method = "Hierarchical"

    # ===================== 结果保存 =====================
    df['Cluster'] = -1
    for idx, lab in zip(valid_indices, best_labels):
        df.at[idx, 'Cluster'] = lab
    df.to_excel(output_file, index=False)

    # ===================== 可视化分析 =====================
    print("生成可视化...")

    # 可视化1：PCA和指标分析
    plt.figure(figsize=(15, 6))

    # PCA投影
    pca = PCA(n_components=2)
    projections = pca.fit_transform([np.array(fp) for fp in fps])

    plt.subplot(121)
    scatter = plt.scatter(projections[:, 0], projections[:, 1], c=best_labels,
                          cmap='viridis', alpha=0.7, edgecolors='w')
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'{best_method} Clustering (PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')

    # 轮廓系数分析
    plt.subplot(122)
    plt.plot(possible_k, silhouette_scores, 'b-o', label='Silhouette')
    plt.plot(possible_k, np.array(calinski_scores) / max(calinski_scores),
             'r--^', label='Calinski (scaled)')
    plt.axvline(x=best_k, color='k', linestyle='--')
    plt.title('Cluster Number Optimization')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "SAR" / "Deroceras reticulatum" / "clustering_analysis.png"))
    plt.savefig(str(RESULTS_DIR / "SAR" / "Deroceras reticulatum" / "clustering_analysis.svg"))

    # 可视化2：层次聚类树状图（仅当最佳方法是层次聚类时生成）
    if best_method == "Hierarchical":
        plt.figure(figsize=(15, 10))

        # 获取有效分子对应的Number标签
        cluster_numbers = [number_list[idx] for idx in valid_indices]  # 通过valid_indices获取对应的Number值

        linked = linkage(distance_matrix, method='average')
        # 修改dendrogram的labels参数为cluster_numbers，并添加横坐标标签
        dendrogram(linked, orientation='top', labels=cluster_numbers, leaf_rotation=90)  # 添加leaf_rotation避免标签重叠
        plt.xlabel("Compound Number")  # 新增横坐标标签
        plt.ylabel("Distance")  # 新增纵坐标标签（表示层次聚类的距离度量）
        plt.title(f"Hierarchical Cluster Dendrogram (Silhouette={agg_eval['silhouette']:.2f})")
        plt.savefig(str(RESULTS_DIR / "SAR" / "Deroceras reticulatum" / "dendrogram.png"))
        plt.savefig(str(RESULTS_DIR / "SAR" / "Deroceras reticulatum" / "dendrogram.svg"))

        # 可视化3：分子结构展示（处理噪声点）
    unique_labels = np.unique(best_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # 过滤噪声点
    valid_labels = [lab for lab in best_labels if lab != -1]
    valid_mols_filtered = [mol for mol, lab in zip(valid_mols, best_labels) if lab != -1]

    # 创建可视化布局
    fig, axes = plt.subplots(n_clusters, 1, figsize=(15, 5 * n_clusters))
    if n_clusters == 1:
        axes = [axes]  # 处理单子图情况

    for i, cluster_id in enumerate(unique_labels):
        if cluster_id == -1:
            continue  # 跳过噪声点

        members = np.where(best_labels == cluster_id)[0]
        # 获取每个分子在原始数据中的Number（转换为字符串确保兼容性）
        member_numbers = [str(number_list[valid_indices[member]]) for member in members[:10]]
        mols = [valid_mols[member] for member in members[:10]]  # 每类最多显示10个

        # 计算类内统计
        intra_dists = distance_matrix[np.ix_(members, members)]
        np.fill_diagonal(intra_dists, np.nan)
        stats = {
            'count': len(members),
            'avg_sim': 1 - np.nanmean(intra_dists),
            'min_sim': 1 - np.nanmax(intra_dists),
            'max_sim': 1 - np.nanmin(intra_dists)
        }

        # 绘制分子并添加Number标签（使用legends参数，需为字符串列表）
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=5,
            subImgSize=(200, 200),
            legends=member_numbers  # 关键修改：使用legends参数并确保为字符串
        )
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(
            f"Cluster {cluster_id} | molecular number: {stats['count']} | "
            f"Average similarity: {stats['avg_sim']:.2f} | "
            f"Range: {stats['min_sim']:.2f}-{stats['max_sim']:.2f}"
        )

    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "SAR" / "Deroceras reticulatum" / "cluster_structures.png"))
    plt.savefig(str(RESULTS_DIR / "SAR" / "Deroceras reticulatum" / "cluster_structures.svg"))

    # ===================== 结果报告 =====================
    print("\n===== 聚类报告 =====")
    print(f"有效分子数: {len(valid_mols)}")
    print(f"最佳方法: {best_method}")
    print(f"轮廓系数: {max(db_eval['silhouette'], agg_eval['silhouette']):.3f}")
    print(f"Calinski-Harabasz指数: {max(db_eval['calinski'], agg_eval['calinski']):.1f}")

    if best_method == "DBSCAN":
        noise_points = list(best_labels).count(-1)
        print(f"识别聚类数: {n_clusters} (包含{noise_points}个噪声点)")
    else:
        print(f"确定聚类数: {best_k}")


if __name__ == "__main__":
    main()
