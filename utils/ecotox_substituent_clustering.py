# -*- coding: utf-8 -*-
"""
基于多策略优化的化合物聚类分析 - 增强版（最终修复版）
功能：1. 组合指纹特征增强 2. 多级聚类策略 3. 动态聚类优化 4. 高级可视化
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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import seaborn as sns
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_style("whitegrid")


def safe_get_fingerprint(mol, fp_func, name, **kwargs):
    """安全获取分子指纹，包含错误处理"""
    try:
        return fp_func(mol, **kwargs)
    except Exception as e:
        smi = Chem.MolToSmiles(mol) if mol else "Invalid Molecule"
        logger.warning(f"生成 {name} 指纹失败 (分子: {smi}): {str(e)}")
        return None


def get_combined_fingerprint(mol):
    """生成稳健的组合分子指纹 - 增强错误处理"""
    if not mol:
        return None

    try:
        # Morgan指纹（局部结构）
        morgan = safe_get_fingerprint(mol, AllChem.GetMorganFingerprintAsBitVect, "Morgan", radius=3, nBits=1024)

        # 拓扑指纹（全局结构）
        atom_pair = safe_get_fingerprint(mol, rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect, "AtomPair",
                                         nBits=512)
        topological = safe_get_fingerprint(mol, rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect,
                                           "Topological", nBits=512)

        # MACCS密钥（预定义官能团）
        maccs = safe_get_fingerprint(mol, AllChem.GetMACCSKeysFingerprint, "MACCS")

        # 原子特征指纹
        feature_fp = safe_get_fingerprint(mol, AllChem.GetMorganFingerprintAsBitVect, "Feature", radius=2, nBits=512,
                                          useFeatures=True)

        # 过滤掉为None的指纹
        fps = [fp for fp in [morgan, atom_pair, topological, maccs, feature_fp] if fp is not None]

        if not fps:
            logger.warning(f"所有指纹生成失败: {Chem.MolToSmiles(mol)}")
            return None

        # 手动合并指纹
        total_bits = sum(fp.GetNumBits() for fp in fps)
        combined = DataStructs.ExplicitBitVect(total_bits)
        offset = 0
        for fp in fps:
            for bit in fp.GetOnBits():
                combined.SetBit(bit + offset)
            offset += fp.GetNumBits()

        return combined
    except Exception as e:
        smi = Chem.MolToSmiles(mol) if mol else "Invalid Molecule"
        logger.error(f"组合指纹生成失败: {smi} - {str(e)}")
        return None


def calculate_similarity(i, j, fps):
    """并行计算相似度"""
    try:
        return DataStructs.TanimotoSimilarity(fps[i], fps[j])
    except Exception as e:
        logger.warning(f"相似度计算失败 ({i}-{j}): {str(e)}")
        return 0.0  # 默认最低相似度


def multi_level_clustering(molecules, labels, min_cluster_size=10):
    """
    多级聚类策略 - 对大类进行二次聚类（带有错误处理）
    """
    final_labels = np.full(len(labels), -1, dtype=int)
    cluster_counter = 0

    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # 跳过噪声点
            noise_indices = np.where(labels == cluster_id)[0]
            final_labels[noise_indices] = -1
            continue

        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_size = len(cluster_indices)

        # 小类直接继承原标签
        if cluster_size < min_cluster_size:
            final_labels[cluster_indices] = cluster_counter
            cluster_counter += 1
            continue

        try:
            logger.info(f"对聚类 {cluster_id} (大小: {cluster_size}) 进行二次聚类...")
            # 获取大类中的分子
            sub_molecules = [molecules[i] for i in cluster_indices]

            # 为子聚类生成指纹
            sub_fps = []
            for mol in sub_molecules:
                fp = get_combined_fingerprint(mol)
                if fp:
                    sub_fps.append(fp)

            # 确保至少有2个有效的指纹
            if len(sub_fps) < 2:
                logger.warning(f"子聚类 {cluster_id} 只有 {len(sub_fps)} 个有效指纹，跳过二次聚类")
                final_labels[cluster_indices] = cluster_counter
                cluster_counter += 1
                continue

            # 计算子距离矩阵
            n_sub = len(sub_fps)
            sub_dist_matrix = np.eye(n_sub)

            # 创建所有索引对
            indices = []
            for i in range(n_sub):
                for j in range(i + 1, n_sub):
                    indices.append((i, j))

            # 并行计算相似度
            sub_results = Parallel(n_jobs=-1)(
                delayed(calculate_similarity)(i, j, sub_fps) for i, j in indices
            )

            # 填充矩阵
            for idx, (i, j) in enumerate(indices):
                sub_dist_matrix[i, j] = 1 - sub_results[idx]
                sub_dist_matrix[j, i] = 1 - sub_results[idx]

            # 确定子聚类数量
            max_sub_clusters = min(8, max(3, cluster_size // 10))
            sub_k = max(3, min(max_sub_clusters, cluster_size // 15))
            logger.info(f"为聚类 {cluster_id} 设置子聚类数: {sub_k}")

            # 执行层次子聚类
            sub_agg = AgglomerativeClustering(
                n_clusters=sub_k,
                affinity='precomputed',
                linkage='average'
            )
            sub_labels = sub_agg.fit_predict(sub_dist_matrix)

            # 映射子聚类标签
            for idx, sub_label in enumerate(sub_labels):
                original_idx = cluster_indices[idx]
                final_labels[original_idx] = cluster_counter + sub_label

            cluster_counter += sub_k
        except Exception as e:
            logger.error(f"聚类 {cluster_id} 的二次聚类失败: {str(e)}")
            # 错误处理：对当前类进行简单划分
            for idx, j in enumerate(cluster_indices):
                final_labels[j] = cluster_counter + (idx % 3)  # 简单分成3类
            cluster_counter += 3

    logger.info(f"多级聚类完成! 总聚类数: {cluster_counter}")
    return final_labels


def evaluate_clustering(distance_matrix, fps, labels, method_name=""):
    """评估聚类结果 - 增强错误处理"""
    if labels is None or len(set(labels)) < 2:
        logger.warning("聚类标签不足，无法评估")
        return {'silhouette': -1, 'calinski': -1}

    # 处理噪声点
    if -1 in labels:
        valid_indices = np.array(labels) != -1
        filtered_labels = labels[valid_indices]
        if len(filtered_labels) == 0:
            logger.warning("有效标签为空，无法评估")
            return {'silhouette': -1, 'calinski': -1}
    else:
        filtered_labels = labels

    results = {'silhouette': -1, 'calinski': -1}

    # 轮廓系数评估
    try:
        # 确保有多个聚类
        if len(set(filtered_labels)) > 1:
            results['silhouette'] = silhouette_score(
                distance_matrix, labels, metric='precomputed'
            )
        else:
            logger.warning("轮廓系数评估失败：只有1个聚类")
    except Exception as e:
        logger.error(f"轮廓系数评估失败: {str(e)}")

    # Calinski-Harabasz指数评估
    try:
        valid_features = [np.array(fp) for i, fp in enumerate(fps) if labels[i] != -1]
        if len(valid_features) > 0 and len(set(filtered_labels)) > 1:
            n_components = min(50, len(valid_features) - 1)
            if n_components > 0:
                pca = PCA(n_components=n_components)
                reduced_features = pca.fit_transform(valid_features)
                results['calinski'] = calinski_harabasz_score(
                    reduced_features, filtered_labels
                )
    except Exception as e:
        logger.error(f"Calinski-Harabasz评估失败: {str(e)}")

    return results


def plot_cluster_structure(molecules, labels, number_list, valid_indices, title_prefix=""):
    """绘制聚类分子结构 - 增强布局与信息展示（带错误处理）"""
    if not molecules or labels is None:
        logger.warning("无有效分子或标签，无法绘制结构")
        return None

    # 获取非噪声聚类标签
    non_noise_labels = np.unique(labels)
    non_noise_labels = [lab for lab in non_noise_labels if lab != -1]  # 过滤噪声标签

    if len(non_noise_labels) == 0:
        logger.warning("没有聚类结果可展示（所有点均为噪声）")
        return None

    # 统计每个聚类的分子数量
    cluster_sizes = {}
    for cluster_id in non_noise_labels:
        members = np.where(labels == cluster_id)[0]
        cluster_sizes[cluster_id] = len(members)

    # 按分子数量排序（降序）
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

    # 只选择分子数量最多的前8个聚类（或更少）
    top_n = min(8, len(sorted_clusters))
    top_clusters = [cluster_id for cluster_id, size in sorted_clusters[:top_n]]

    logger.info(f"将展示分子数量最多的 {top_n} 个聚类（共 {len(non_noise_labels)} 类）")

    # 创建可视化布局
    n_rows = (top_n + 1) // 2  # 每行最多2个聚类
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows))
    axes_flat = axes.flatten() if n_rows > 1 else [axes]

    # 遍历前top_n个聚类
    for i, cluster_id in enumerate(top_clusters):
        ax = axes_flat[i]
        members = np.where(labels == cluster_id)[0]

        # 获取有效的分子和编号
        mols_to_draw = []
        numbers_to_draw = []
        num_skipped = 0

        # 确保members是有效的索引列表
        if not members.size:
            logger.warning(f"聚类 {cluster_id} 无有效分子索引")
            continue

        for cluster_idx in members[:10]:  # 每类最多显示10个
            try:
                # 确保cluster_idx在有效分子列表的索引范围内
                if cluster_idx < len(molecules):
                    mol = molecules[cluster_idx]
                    # 检查valid_indices和number_list的范围
                    if cluster_idx < len(valid_indices):
                        original_idx = valid_indices[cluster_idx]
                        if original_idx < len(number_list):
                            if mol:
                                mols_to_draw.append(mol)
                                numbers_to_draw.append(str(number_list[original_idx]))
                        else:
                            num_skipped += 1
                    else:
                        num_skipped += 1
                else:
                    num_skipped += 1
            except Exception as e:
                logger.warning(f"绘制分子错误: {str(e)}")
                num_skipped += 1

        if num_skipped > 0:
            logger.warning(f"聚类 {cluster_id} 跳过了 {num_skipped} 个分子")

        if not mols_to_draw:
            logger.warning(f"聚类 {cluster_id} 无有效分子可展示")
            ax.axis('off')
            ax.set_title(f"聚类 {cluster_id} | 无有效分子")
            continue

        try:
            # 计算绘制分子网格的行数
            num_mols = len(mols_to_draw)
            mols_per_row = min(5, num_mols) if num_mols > 0 else 5

            # 绘制分子并添加Number标签
            img = Draw.MolsToGridImage(
                mols_to_draw,
                molsPerRow=mols_per_row,
                subImgSize=(200, 200),
                legends=numbers_to_draw,
                returnPNG=False
            )

            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"聚类 {cluster_id} | 分子数量: {len(members)}")
        except Exception as e:
            logger.error(f"绘制聚类 {cluster_id} 结构失败: {str(e)}")
            ax.axis('off')
            ax.set_title(f"聚类 {cluster_id} | 绘制失败")

    # 处理剩余的空子图
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.suptitle(f"{title_prefix}聚类分子结构 - 前{top_n}大聚类", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def extract_top_clusters_to_excel():
    """
    提取聚类结果中数量最多的前3类分子，并分别保存为单独的Excel文件
    此函数应在main()函数执行完毕后调用
    """
    # 使用与主程序相同的文件路径
    input_file = str(RESULTS_DIR / "safe" / "safe1-2.xlsx")  # 聚类结果文件
    output_prefix = str(RESULTS_DIR / "safe" / "safe1_top_cluster")

    try:
        # 读取聚类结果
        df = pd.read_excel(input_file, engine='openpyxl')

        # 筛选有效分子（Cluster >= 0）
        valid_df = df[df['Cluster'] >= 0].copy()

        if valid_df.empty:
            logger.warning("没有找到有效的聚类分子")
            return

        # 统计每个聚类的分子数量
        cluster_counts = valid_df['Cluster'].value_counts().sort_values(ascending=False)

        logger.info("各聚类分子数量统计:")
        for cluster_id, count in cluster_counts.items():
            logger.info(f"聚类 {cluster_id}: {count} 个分子")

        # 取前3个最大的聚类
        top_clusters = cluster_counts.head(3).index.tolist()

        logger.info(f"数量最多的前3个聚类: {top_clusters}")

        # 为每个顶级聚类创建单独的文件
        for i, cluster_id in enumerate(top_clusters, 1):
            # 提取当前聚类的分子
            cluster_df = valid_df[valid_df['Cluster'] == cluster_id].copy()

            # 按Number排序（可选）
            cluster_df = cluster_df.sort_values('Number')

            # 生成输出文件名
            output_file = f"{output_prefix}_{i}_cluster{cluster_id}.xlsx"

            # 保存为Excel
            cluster_df.to_excel(output_file, index=False)

            logger.info(f"聚类 {cluster_id} (第{i}大) 已保存: {output_file}")
            logger.info(f"  包含 {len(cluster_df)} 个分子")

            # 显示前几个分子的Number作为示例
            sample_numbers = cluster_df['Number'].head(5).tolist()
            logger.info(f"  示例分子编号: {sample_numbers}")

        # 同时保存一个汇总文件，包含所有前3聚类的信息
        summary_dfs = []
        for i, cluster_id in enumerate(top_clusters, 1):
            cluster_df = valid_df[valid_df['Cluster'] == cluster_id].copy()
            cluster_df.insert(0, 'Rank', i)  # 添加排名列
            summary_dfs.append(cluster_df)

        if summary_dfs:
            summary_df = pd.concat(summary_dfs, ignore_index=True)
            summary_file = f"{output_prefix}_summary_top3.xlsx"
            summary_df.to_excel(summary_file, index=False)
            logger.info(f"前3聚类汇总文件已保存: {summary_file}")

        # 生成统计报告
        stats_report = {
            '总有效分子数': len(valid_df),
            '总聚类数': len(cluster_counts),
            '前3聚类分子数': sum(cluster_counts.head(3)),
            '前3聚类占比': f"{sum(cluster_counts.head(3)) / len(valid_df) * 100:.1f}%"
        }

        logger.info("=== 提取结果统计 ===")
        for key, value in stats_report.items():
            logger.info(f"{key}: {value}")

    except Exception as e:
        logger.error(f"提取前3聚类失败: {str(e)}")



def main():
    # ===================== 参数配置 =====================
    input_file = str(RESULTS_DIR / "safe" / "safe1-1.xlsx")
    output_file = str(RESULTS_DIR / "safe" / "safe1-2.xlsx")
    output_prefix = str(RESULTS_DIR / "safe" / "safe1_optimized")

    # ===================== 数据预处理 =====================
    logger.info("数据预处理...")
    try:
        df = pd.read_excel(input_file, engine='openpyxl')
        smiles_list = df['SMILES'].fillna('').tolist()

        # 提取Number列作为后续标签
        number_list = df['Number'].tolist()

        logger.info(f"读取成功! 总记录数: {len(smiles_list)}")
    except Exception as e:
        logger.error(f"读取Excel文件失败: {str(e)}")
        return

    # 分子处理与去重
    seen = set()
    valid_mols = []
    valid_indices = []  # 保存原始数据中的行索引
    invalid_indices = []  # 无效分子索引

    for idx, smi in enumerate(smiles_list):
        if not smi or pd.isna(smi):
            invalid_indices.append(idx)
            logger.warning(f"空白SMILES: 第{idx + 1}行")
            continue

        try:
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                invalid_indices.append(idx)
                logger.warning(f"无效SMILES: 第{idx + 1}行 - {smi}")
                continue

            smi_hash = Chem.MolToSmiles(mol, isomericSmiles=True)
            if smi_hash in seen:
                logger.info(f"跳过重复分子: 第{idx + 1}行 - {smi_hash}")
                continue

            seen.add(smi_hash)
            valid_mols.append(mol)
            valid_indices.append(idx)
            logger.debug(f"有效分子: 第{idx + 1}行 - {smi_hash}")

        except Exception as e:
            invalid_indices.append(idx)
            logger.error(f"处理SMILES错误: 第{idx + 1}行 - {smi} ({str(e)})")

    logger.info(f"有效分子数: {len(valid_mols)} | 无效分子数: {len(invalid_indices)}")

    if len(valid_mols) == 0:
        logger.error("没有有效分子可用于分析! 程序终止。")
        return

    # ===================== 特征工程 =====================
    logger.info("生成稳健的组合指纹...")
    fps = []
    failed_fps = 0

    for mol in tqdm(valid_mols, desc="生成指纹"):
        fp = get_combined_fingerprint(mol)
        if fp:
            fps.append(fp)
        else:
            fps.append(None)
            failed_fps += 1

    # 移除生成指纹失败的分子
    original_count = len(valid_mols)
    valid_fps = [fp for fp in fps if fp is not None]
    valid_mols = [mol for mol, fp in zip(valid_mols, fps) if fp is not None]
    valid_indices = [idx for idx, fp in zip(valid_indices, fps) if fp is not None]
    fps = valid_fps

    removed_count = original_count - len(valid_mols)
    logger.info(f"指纹生成完成! 成功: {len(valid_mols)} | 失败: {removed_count} | 原始总数: {original_count}")

    if len(valid_mols) < 2:
        logger.error(f"有效分子太少({len(valid_mols)})，无法进行聚类分析。最小需要2个分子。")
        return

    # ===================== 相似度计算 =====================
    logger.info("计算相似度矩阵...")
    n = len(fps)
    distance_matrix = np.zeros((n, n))

    # 创建所有索引对
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            indices.append((i, j))

    total_pairs = len(indices)
    logger.info(f"需要计算 {total_pairs} 对分子相似度...")

    # 并行计算相似度 - 添加进度显示
    try:
        with tqdm(total=total_pairs, desc="计算相似度") as pbar:
            results = Parallel(n_jobs=-1, backend="threading")(
                delayed(calculate_similarity)(i, j, fps) for i, j in indices
            )
            pbar.update(total_pairs)
    except Exception as e:
        logger.error(f"并行计算失败: {str(e)}")
        logger.info("尝试单线程计算...")
        results = []
        for i, j in tqdm(indices, desc="单线程计算相似度"):
            results.append(calculate_similarity(i, j, fps))

    # 填充矩阵
    for idx, (i, j) in enumerate(indices):
        distance_matrix[i, j] = 1 - results[idx]
        distance_matrix[j, i] = 1 - results[idx]

    # 填充对角线为0
    np.fill_diagonal(distance_matrix, 0)

    logger.info("相似度矩阵计算完成!")

    # ===================== 聚类分析 =====================
    logger.info("\n开始聚类分析...")

    # 方法1：DBSCAN聚类
    try:
        db = DBSCAN(eps=0.4,  # 适当增大eps
                    min_samples=2,
                    metric='precomputed')
        db_labels = db.fit_predict(distance_matrix)
        logger.info(f"DBSCAN聚类完成! 聚类数: {len(np.unique(db_labels))}")
    except Exception as e:
        logger.error(f"DBSCAN聚类失败: {str(e)}")
        db_labels = np.zeros(len(valid_mols))  # 所有点归为0类

    # 方法2：层次聚类
    min_clusters = min(5, max(2, len(valid_mols) // 20))
    max_clusters = min(30, len(valid_mols) // 3)
    possible_k = list(range(min_clusters, max_clusters + 1))
    logger.info(f"评估聚类数范围: {min_clusters} 到 {max_clusters}")

    silhouette_scores = []
    calinski_scores = []
    combined_scores = []

    for k in possible_k:
        try:
            logger.debug(f"评估k={k}")
            model = AgglomerativeClustering(n_clusters=k,
                                            affinity='precomputed',
                                            linkage='average')
            labels = model.fit_predict(distance_matrix)

            eval_result = evaluate_clustering(distance_matrix, fps, labels)
            sil = eval_result['silhouette']
            cal = eval_result['calinski']

            silhouette_scores.append(sil)
            calinski_scores.append(cal)

            # 组合得分计算
            quality = 0.7 * (sil if sil > 0 else 0) + 0.3 * (cal / 1000 if cal > 0 else 0)
            size_factor = min(1.0, k / 20)
            combined = quality * size_factor
            combined_scores.append(combined)
        except Exception as e:
            logger.warning(f"聚类评估k={k}失败: {str(e)}")
            silhouette_scores.append(-1)
            calinski_scores.append(-1)
            combined_scores.append(-1)

    # 选择最佳k值
    try:
        # 去除无效分数
        valid_indices_scores = [i for i, score in enumerate(combined_scores) if score > 0]
        if not valid_indices_scores:
            best_idx = 0
        else:
            best_idx = valid_indices_scores[np.argmax([combined_scores[i] for i in valid_indices_scores])]

        best_k = possible_k[best_idx]
        logger.info(f"优化选择聚类数: {best_k} (轮廓系数: {silhouette_scores[best_idx]:.3f}, "
                    f"Calinski: {calinski_scores[best_idx] if calinski_scores[best_idx] > 0 else 'N/A'})")
    except Exception as e:
        logger.error(f"选择最佳k值失败: {str(e)}. 使用默认k=5")
        best_k = 5

    # 执行层次聚类
    try:
        agg = AgglomerativeClustering(n_clusters=best_k,
                                      affinity='precomputed',
                                      linkage='average')
        agg_labels = agg.fit_predict(distance_matrix)
        logger.info(f"层次聚类完成! 标签: {np.unique(agg_labels)}")
    except Exception as e:
        logger.error(f"层次聚类失败: {str(e)}. 使用DBSCAN结果")
        agg_labels = db_labels

    # ===================== 选择最佳聚类方法 =====================
    # 评估两种方法
    db_eval = evaluate_clustering(distance_matrix, fps, db_labels, "DBSCAN")
    agg_eval = evaluate_clustering(distance_matrix, fps, agg_labels, "Hierarchical")

    # 选择最佳方法
    if db_eval['silhouette'] > agg_eval['silhouette']:
        best_labels = db_labels
        best_method = "DBSCAN"
    else:
        best_labels = agg_labels
        best_method = "Hierarchical"

    logger.info(f"最佳聚类方法: {best_method}")

    # ===================== 多级聚类 =====================
    logger.info("应用多级聚类策略...")
    min_cluster_size = max(3, len(valid_mols) // 30)
    try:
        final_labels = multi_level_clustering(valid_mols, best_labels, min_cluster_size)
    except Exception as e:
        logger.error(f"多级聚类失败: {str(e)}. 使用原始聚类")
        final_labels = best_labels

    # 确保标签是整数且从0开始
    try:
        unique_labels = np.unique(final_labels)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        label_map[-1] = -1
        final_labels = np.array([label_map[l] for l in final_labels])
    except Exception as e:
        logger.error(f"重标号失败: {str(e)}")

    # 评估最终聚类结果
    final_eval = evaluate_clustering(distance_matrix, fps, final_labels, "最终聚类")
    logger.info(f"最终聚类评估: 轮廓系数 = {final_eval['silhouette']:.3f}, "
                f"Calinski-Harabasz指数 = {final_eval['calinski'] if final_eval['calinski'] > 0 else 'N/A'}")

    # ===================== 结果保存 =====================
    # 将聚类结果映射回原始数据框
    df['Cluster'] = -1  # 初始化为-1
    df['Valid'] = True  # 添加有效标记列

    # 标记有效分子
    for idx, lab in zip(valid_indices, final_labels):
        df.at[idx, 'Cluster'] = lab
        df.at[idx, 'Valid'] = True

    # 标记无效分子
    for idx in invalid_indices:
        df.at[idx, 'Cluster'] = -2  # -2表示无效分子
        df.at[idx, 'Valid'] = False

    # 保存结果
    try:
        df.to_excel(output_file, index=False)
        logger.info(f"结果保存成功: {output_file}")
    except Exception as e:
        logger.error(f"保存Excel失败: {str(e)}")

    # 保存聚类摘要
    cluster_ids = np.unique(final_labels)
    cluster_ids = cluster_ids[cluster_ids != -1]  # 排除噪声点
    cluster_summary = []

    for cluster_id in cluster_ids:
        members = np.where(final_labels == cluster_id)[0]

        sample_numbers = []
        # 安全地获取代表性的分子编号
        for member in members[:100]:  # 100个
            # 检查索引是否有效
            if member < len(valid_indices):
                original_idx = valid_indices[member]
                # 检查行号是否在有效范围内
                if original_idx < len(number_list):
                    sample_numbers.append(str(number_list[original_idx]))

        # 计算类内平均距离
        avg_distance = 0
        if len(members) > 1:
            try:
                intra_indices = np.ix_(members, members)
                intra_dists = distance_matrix[intra_indices]
                np.fill_diagonal(intra_dists, np.nan)
                avg_distance = np.nanmean(intra_dists)
            except Exception as e:
                logger.warning(f"计算聚类{cluster_id}平均距离失败: {str(e)}")
                avg_distance = -1

        # 将样本编号连接为字符串
        sample_str = ", ".join(sample_numbers)
        if len(members) > 5:
            sample_str += "..."  # 添加省略号表示还有更多分子

        cluster_summary.append({
            'Cluster ID': cluster_id,
            '分子数量': len(members),
            '平均距离': avg_distance,
            '代表性分子': sample_str
        })

    cluster_df = pd.DataFrame(cluster_summary)
    try:
        cluster_df.to_excel(f"{output_prefix}_cluster_summary.xlsx", index=False)
        logger.info(f"聚类摘要保存成功")
    except Exception as e:
        logger.error(f"保存聚类摘要失败: {str(e)}")

    # ===================== 可视化分析 =====================
    logger.info("生成高级可视化...")

    try:
        # PCA投影
        pca = PCA(n_components=2)
        projections = pca.fit_transform([np.array(fp) for fp in fps])

        # 创建图形
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        plt.suptitle("Compound clustering analysis results", fontsize=16) #化合物聚类分析结果

        # PCA投影图
        scatter1 = axs[0, 0].scatter(projections[:, 0], projections[:, 1],
                                     c=final_labels, cmap='viridis', alpha=0.7,
                                     edgecolors='w', s=40)
        plt.colorbar(scatter1, ax=axs[0, 0])
        axs[0, 0].set_title('PCA')
        axs[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
        axs[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')

        # 聚类大小分布
        cluster_sizes = [np.sum(final_labels == i) for i in np.unique(final_labels) if i != -1]

        cluster_ids = np.unique(final_labels)

        # ===== 新增：单独保存聚类分布矢量图 =====
        # 创建新图形用于单独保存
        fig_size_dist = plt.figure(figsize=(10, 7))
        ax_size = fig_size_dist.add_subplot(111)

        if cluster_sizes:
            # 生成排好序的索引
            sorted_idx = np.argsort(cluster_sizes)[::-1]
            sorted_sizes = np.array(cluster_sizes)[sorted_idx]

            # 主图内容保持不变
            bar_plot = axs[0, 1].bar(range(len(cluster_sizes)), sorted_sizes, color='teal')
            axs[0, 1].set_title('Cluster size distribution')  # 聚类大小分布
            axs[0, 1].set_xlabel('Cluster rank')  # 聚类排序
            axs[0, 1].set_ylabel('Number of molecules')  # 分子数量
            axs[0, 1].set_yscale('log')

            # 单独矢量图
            bar_dist = ax_size.bar(range(len(sorted_sizes)), sorted_sizes, color='teal', edgecolor='white')

            # 添加数值标签 (对数坐标系下更清晰)
            for i, b in enumerate(bar_dist):
                ax_size.text(b.get_x() + b.get_width() / 2, b.get_height() * 0.9,
                             f'{int(b.get_height())}', ha='center', color='white', fontsize=9)

            ax_size.set_title(f'Cluster Size Distribution (Total Clusters: {len(cluster_sizes)})', fontsize=14)
            ax_size.set_xlabel('Cluster Rank', fontsize=12)
            ax_size.set_ylabel('Number of Molecules', fontsize=12)
            ax_size.grid(True, linestyle='--', alpha=0.5)
            fig_size_dist.tight_layout()

            # 保存单独矢量图
            fig_size_dist.savefig(f"{output_prefix}_cluster_distribution.svg")
            fig_size_dist.savefig(f"{output_prefix}_cluster_distribution.pdf")
            plt.close(fig_size_dist)
            logger.info(f"已单独保存聚类分布矢量图: {output_prefix}_cluster_distribution.*")

        else:
            axs[0, 1].text(0.5, 0.5, 'No valid clusters', ha='center', va='center')  # 无有效聚类
            axs[0, 1].set_title('Cluster size distribution')  # 聚类大小分布

        # 评估指标图
        if possible_k and combined_scores:
            axs[1, 0].plot(possible_k, silhouette_scores, 'b-o', label='contour coefficient') #轮廓系数
            if any(score > 0 for score in calinski_scores):
                axs[1, 0].plot(possible_k, np.array(calinski_scores) / max(cal for cal in calinski_scores if cal > 0),
                               'r--^', label='Calinski')  # (归一化)
            if any(score > 0 for score in combined_scores):
                axs[1, 0].plot(possible_k,
                               np.array(combined_scores) / max(comb for comb in combined_scores if comb > 0),
                               'g-s', label='Overall score')  #综合得分
            axs[1, 0].axvline(x=best_k, color='k', linestyle='--', label=f'Best K={best_k}')  #
            axs[1, 0].set_title('Cluster number optimisation')  #聚类数优化
            axs[1, 0].set_xlabel('Cluster number')  #聚类数
            axs[1, 0].set_ylabel('normalised score')  #归一化得分
            axs[1, 0].legend()
        else:
            axs[1, 0].text(0.5, 0.5, '评估数据不足', ha='center', va='center')  #Insufficient assessment data
            axs[1, 0].set_title('Cluster number optimisation')  #聚类数优化

        # 留空位置用于分子结构或说明
        axs[1, 1].text(0.5, 0.5, 'See separate file for detailed molecular structure.',
                       ha='center', va='center', fontsize=12)  #分子结构详见单独文件
        axs[1, 1].set_title('Molecular structure display')  #分子结构展示

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{output_prefix}_overview.png", dpi=1200)
        plt.savefig(f"{output_prefix}_overview.svg")
        plt.savefig(f"{output_prefix}_overview.pdf")
        plt.close()
        logger.info("Overall overview diagram generated successfully")   #整体概览图生成成功
    except Exception as e:
        logger.error(f"Visualisation generation failed: {str(e)}")   #可视化生成失败

    # 分子结构展示（聚类概览图）
    try:
        structure_fig = plot_cluster_structure(
            valid_mols, final_labels, number_list, valid_indices, "Optimised"
        )   #优化后的
        if structure_fig:
            structure_fig.savefig(f"{output_prefix}_cluster_structures.png", dpi=300)
            structure_fig.savefig(f"{output_prefix}_cluster_structures.svg")
            plt.close(structure_fig)
            logger.info("Cluster structure overview diagram generated successfully")   #聚类结构概览图生成成功
    except Exception as e:
        logger.error(f"Molecular structure overview diagram generation failed: {str(e)}")  #分子结构概览图生成失败

    # 新增功能：为每个聚类单独生成结构图
    try:
        logger.info("开始生成每个聚类的详细结构图...")

        # 获取所有聚类ID（排除噪声点）
        cluster_ids = np.unique(final_labels)
        cluster_ids = [cid for cid in cluster_ids if cid != -1]

        for cluster_id in cluster_ids:
            # 获取当前聚类中的分子索引
            members = np.where(final_labels == cluster_id)[0]

            # 准备绘制的分子和编号
            mols_to_draw = []
            numbers_to_draw = []

            # 最多显示50个分子（避免图片过大）
            max_display = min(50, len(members))

            for i, idx in enumerate(members[:max_display]):
                try:
                    # 获取原始分子和编号
                    if idx < len(valid_mols):
                        mol = valid_mols[idx]
                        if mol:
                            mols_to_draw.append(mol)

                            # 获取原始行号
                            if idx < len(valid_indices):
                                orig_idx = valid_indices[idx]
                                if orig_idx < len(number_list):
                                    numbers_to_draw.append(f"{number_list[orig_idx]}")
                                else:
                                    numbers_to_draw.append(f"M{idx}")
                            else:
                                numbers_to_draw.append(f"M{idx}")
                except Exception as e:
                    logger.warning(f"Cluster{cluster_id}分子失败: {str(e)}")

            if not mols_to_draw:
                logger.warning(f"Cluster {cluster_id} 无有效分子可展示")
                continue

            # 计算每行显示的分子数
            mols_per_row = min(5, len(mols_to_draw))

            # 计算行数并调整图片高度
            n_rows = (len(mols_to_draw) + mols_per_row - 1) // mols_per_row
            fig_height = max(4, n_rows * 2.5)  # 每行2.5英寸高度

            try:
                # 创建单个聚类图
                fig = plt.figure(figsize=(15, fig_height))
                # 绘制分子网格
                img = Draw.MolsToGridImage(
                    mols_to_draw,
                    molsPerRow=mols_per_row,
                    subImgSize=(300, 300),
                    legends=numbers_to_draw,
                    returnPNG=False
                )

                # 在图像上显示聚类信息
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Clustering {cluster_id} | Number of molecules: {len(members)}", fontsize=16)
                plt.tight_layout()

                # 保存单个聚类图
                fig.savefig(f"{output_prefix}_cluster{cluster_id}_structures.png", dpi=300)
                fig.savefig(f"{output_prefix}_cluster{cluster_id}_structures.svg")
                plt.close(fig)
                logger.info(f"Clustering {cluster_id} Structure diagram saved successfully")
            except Exception as e:
                logger.error(f"Clustering {cluster_id} Structure diagram generation failed: {str(e)}")

    except Exception as e:
        logger.error(f"生成每个聚类结构图失败: {str(e)}")

    # ===================== 结果报告 =====================
    valid_clusters = [lab for lab in final_labels if lab != -1]
    n_clusters = len(np.unique(valid_clusters)) if valid_clusters else 0
    noise_count = np.sum([lab == -1 for lab in final_labels]) if final_labels is not None else 0

    logger.info("\n===== 聚类分析完成报告 =====")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"总分子数: {len(smiles_list)}")
    logger.info(f"有效分子数: {len(valid_mols)}")
    logger.info(f"最终聚类数: {n_clusters}")
    logger.info(f"噪声点数量: {noise_count}")
    logger.info(f"平均轮廓系数: {final_eval['silhouette']:.3f}")

    # 统计聚类大小
    if n_clusters > 0:
        cluster_sizes = [np.sum(final_labels == i) for i in np.unique(final_labels) if i != -1]
        avg_size = np.mean(cluster_sizes)
        min_size = np.min(cluster_sizes)
        max_size = np.max(cluster_sizes)
        logger.info(f"Cluster size: Best min={min_size}, Best max={max_size}, Average={avg_size:.1f}")

    logger.info("分析完成!")

    logger.info("开始提取数量最多的前3类分子...")
    extract_top_clusters_to_excel()

if __name__ == "__main__":
    main()










# -*- coding: utf-8 -*-
"""
聚类详细结构特征分析 - 独立运行版
功能：对数量最多的聚类进行全面的结构特征可视化
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, DataStructs, rdMolDescriptors, Descriptors
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import warnings

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_style("whitegrid")


def get_combined_fingerprint(mol):
    """生成稳健的组合分子指纹"""
    if not mol:
        return None

    try:
        # Morgan指纹
        morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
        return morgan
    except Exception as e:
        logger.warning(f"指纹生成失败: {str(e)}")
        return None


def analyze_cluster32_detailed():
    """
    对聚类进行详细的结构特征分析
    生成所有要求的可视化文件
    """
    # 文件路径配置
    cluster_file = str(RESULTS_DIR / "safe" / "safe1-2.xlsx")
    output_prefix = str(RESULTS_DIR / "safe" / "safe1-2_analysis-all")

    try:
        logger.info("=" * 60)
        logger.info("开始聚类详细结构特征分析")
        logger.info("=" * 60)

        # 读取聚类数据
        df = pd.read_excel(cluster_file, engine='openpyxl')
        logger.info(f"聚类包含 {len(df)} 个分子")

        if df.empty:
            logger.error("聚类数据为空，分析终止")
            return

        # 显示数据基本信息
        logger.info(f"数据列: {', '.join(df.columns.tolist())}")
        logger.info(f"Number列范围: {df['Number'].min()} - {df['Number'].max()}")

        # ===================== 分子结构处理 =====================
        logger.info("处理分子结构...")

        molecules = []
        valid_indices = []
        valid_numbers = []

        for idx, row in df.iterrows():
            try:
                smi = str(row['SMILES']).strip()
                if smi and smi != 'nan':
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        molecules.append(mol)
                        valid_indices.append(idx)
                        valid_numbers.append(row['Number'])
                        logger.debug(f"成功处理分子 {row['Number']}")
                    else:
                        logger.warning(f"无效SMILES: 第{idx}行 - {smi}")
                else:
                    logger.warning(f"空白SMILES: 第{idx}行")
            except Exception as e:
                logger.warning(f"分子 {idx} 处理失败: {str(e)}")

        if not molecules:
            logger.error("没有有效的分子可用于分析")
            return

        logger.info(f"成功处理 {len(molecules)} 个有效分子")

        # ===================== 1. 结构可视化 =====================
        logger.info("生成分子结构网格图...")
        generate_structure_grid(molecules, valid_numbers, output_prefix)

        # ===================== 2. 描述符分析 =====================
        logger.info("进行分子描述符分析...")
        descriptor_results = analyze_molecular_descriptors(molecules, valid_numbers, output_prefix)

        # ===================== 3. 子结构分析 =====================
        logger.info("进行子结构分析...")
        functional_group_results = analyze_functional_groups(molecules, output_prefix)

        # ===================== 4. 相似性分析 =====================
        logger.info("进行相似性分析...")
        similarity_results = analyze_similarity(molecules, output_prefix)

        # ===================== 5. 生成综合报告 =====================
        logger.info("生成综合分析报告...")
        generate_comprehensive_report(molecules, descriptor_results, functional_group_results,
                                      similarity_results, output_prefix, len(df))

        logger.info("=" * 60)
        logger.info("聚类分析完成!")
        logger.info(f"所有结果保存在: {output_prefix}_*")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"聚类分析失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def generate_structure_grid(molecules, numbers, output_prefix):
    """生成分子结构网格图"""
    try:
        n_mols = len(molecules)
        if n_mols == 0:
            return

        # 计算合适的网格布局
        mols_per_row = min(5, n_mols)
        n_rows = (n_mols + mols_per_row - 1) // mols_per_row

        # 调整图片大小
        fig_width = min(20, mols_per_row * 3)
        fig_height = max(6, n_rows * 3)

        fig = plt.figure(figsize=(fig_width, fig_height))

        # 准备标签
        legends = [f"Number: {num}" for num in numbers]

        # 绘制分子网格
        img = Draw.MolsToGridImage(
            molecules,
            molsPerRow=mols_per_row,
            subImgSize=(300, 300),
            legends=legends,
            returnPNG=False
        )

        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Cluster 32 Molecular Structures\nTotal: {n_mols} molecules',
                  fontsize=16, pad=20)
        plt.tight_layout()

        # 保存图片
        fig.savefig(f"{output_prefix}_structures-all.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{output_prefix}_structures-all.svg", bbox_inches='tight')
        plt.close(fig)
        logger.info("✓ 分子结构网格图保存成功")

    except Exception as e:
        logger.error(f"结构网格图生成失败: {str(e)}")


def analyze_molecular_descriptors(molecules, numbers, output_prefix):
    """分析分子描述符"""
    descriptors = []

    try:
        logger.info("计算分子描述符...")

        for i, mol in enumerate(tqdm(molecules, desc="计算描述符")):
            try:
                desc = {}

                # 基础物理化学描述符
                desc['Molecular_Weight'] = Descriptors.MolWt(mol)
                desc['LogP'] = Descriptors.MolLogP(mol)
                desc['TPSA'] = Descriptors.TPSA(mol)
                desc['HBD'] = Descriptors.NumHDonors(mol)  # 氢键供体
                desc['HBA'] = Descriptors.NumHAcceptors(mol)  # 氢键受体
                desc['Rotatable_Bonds'] = Descriptors.NumRotatableBonds(mol)
                desc['Aromatic_Rings'] = Descriptors.NumAromaticRings(mol)
                desc['Heavy_Atoms'] = Descriptors.HeavyAtomCount(mol)

                # 环系统
                desc['Rings'] = Descriptors.RingCount(mol)

                # 其他重要描述符
                desc['Fraction_Csp3'] = Descriptors.FractionCsp3(mol)
                desc['Num_RotatableBonds'] = Descriptors.NumRotatableBonds(mol)
                desc['Num_AromaticRings'] = Descriptors.NumAromaticRings(mol)
                desc['Num_AliphaticRings'] = Descriptors.NumAliphaticRings(mol)

                descriptors.append(desc)

            except Exception as e:
                logger.warning(f"分子 {i} 描述符计算失败: {str(e)}")
                descriptors.append({})

        # 创建描述符DataFrame
        desc_df = pd.DataFrame(descriptors)
        desc_df['Number'] = numbers

        # 重新排序列
        col_order = ['Number'] + [col for col in desc_df.columns if col != 'Number']
        desc_df = desc_df[col_order]

        # 保存描述符数据
        desc_df.to_excel(f"{output_prefix}_descriptors.xlsx", index=False)
        logger.info("✓ 分子描述符数据保存成功")

        # 生成描述符分布图
        generate_descriptor_distribution(desc_df, output_prefix)

        return desc_df

    except Exception as e:
        logger.error(f"描述符分析失败: {str(e)}")
        return pd.DataFrame()


def generate_descriptor_distribution(desc_df, output_prefix):
    """生成描述符分布图"""
    try:
        # 选择数值型列进行可视化（排除Number列）
        numeric_cols = desc_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Number' in numeric_cols:
            numeric_cols.remove('Number')

        if not numeric_cols:
            logger.warning("没有数值型描述符可用于可视化")
            return

        # 创建多子图布局
        n_plots = len(numeric_cols)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_plots == 1:
            axes = np.array([axes])
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        axes_flat = axes.flatten()

        for i, col in enumerate(numeric_cols):
            if i < len(axes_flat):
                ax = axes_flat[i]

                # 绘制直方图
                data = desc_df[col].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=min(20, len(data) // 2), alpha=0.7,
                            color='lightblue', edgecolor='black')
                    ax.set_title(f'Distribution of {col}', fontsize=12)
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')

                    # 添加统计信息
                    mean_val = data.mean()
                    std_val = data.std()
                    ax.axvline(mean_val, color='red', linestyle='--',
                               label=f'Mean: {mean_val:.2f}')
                    ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
                    ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
                    ax.legend(fontsize=8)

        # 隐藏多余的子图
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.suptitle('Molecular Descriptors Distribution', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{output_prefix}_descriptors_distribution-all.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_prefix}_descriptors_distribution-all.svg", bbox_inches='tight')
        plt.close()
        logger.info("✓ 描述符分布图保存成功")

    except Exception as e:
        logger.error(f"描述符分布图生成失败: {str(e)}")


def analyze_functional_groups(molecules, output_prefix):
    """从农药开发角度分析官能团分布"""
    try:
        # 农药相关官能团SMARTS模式 - 农药开发专用版
        functional_groups = {
            # === 核心活性基团 ===
            # 含磷官能团（有机磷农药）
            'Phosphate': 'P(=O)(O)(O)O',
            'Phosphorothioate': 'P(=O)(S)(O)',  # 硫代磷酸酯
            'Phosphonate': 'P(=O)(O)C',
            'Phosphinate': 'P(=O)(C)C',
            'Phosphine_Oxide': 'P=O',

            # 含氮官能团
            'Primary_Amine': '[NX3;H2]',
            'Secondary_Amine': '[NX3;H1]',
            'Tertiary_Amine': '[NX3]',
            'Quaternary_Ammonium': '[N+4]',  # 季铵盐类
            'Amide': 'C(=O)[NX3]',
            'Nitro': '[N+](=O)[O-]',  # 硝基化合物
            'Nitroso': 'N=O',
            'Cyanate': 'N#C',
            'Isocyanate': 'N=C=O',
            'Azide': '[N-]=[N+]=[N]',
            'Imine': 'C=N',
            'Oxime': 'C=N-O',
            'Hydrazone': 'C=N-N',
            'Enamine': 'C=C-N',

            # 杂环氮（三唑、吡啶、嘧啶等）
            'Triazole': 'c1cnn1',  # 三唑
            'Imidazole': 'c1cncn1',  # 咪唑
            'Pyridine': 'n1ccccc1',  # 吡啶
            'Pyrimidine': 'n1cncc1',  # 嘧啶
            'Pyrazole': 'c1cnnc1',  # 吡唑
            'Triazine': 'c1nc(N)nc(N)n1',  # 三嗪

            # === 卤素取代（影响脂溶性和稳定性）===
            'Fluorine': '[F]',
            'Chlorine': '[Cl]',
            'Bromine': '[Br]',
            'Iodine': '[I]',
            'CF3_Group': 'FC(F)(F)',  # 三氟甲基
            'OCF3_Group': 'OC(F)(F)F',  # 三氟甲氧基

            # === 含硫官能团 ===
            'Sulfonyl': 'S(=O)(=O)',
            'Sulfonate': 'S(=O)(=O)[OX2]',
            'Sulfonamide': 'S(=O)(=O)[NX3]',
            'Sulfoxide': '[S](=O)',
            'Sulfone': '[S](=O)(=O)',
            'Thioether': 'C-S-C',
            'Disulfide': 'SS',
            'Thiocyanate': 'S-C#N',
            'Isothiocyanate': 'N=C=S',

            # === 羧酸及其衍生物 ===
            'Carboxylic_Acid': 'C(=O)[OH]',
            'Ester': 'C(=O)[OX2]',  # 酯类（拟除虫菊酯）
            'Acid_Chloride': 'C(=O)Cl',
            'Anhydride': 'C(=O)OC(=O)',

            # === 其他重要官能团 ===
            'Alcohol': '[OX2H]',
            'Phenol': 'Oc1ccccc1',
            'Ether': '[OD2]([#6])[#6]',
            'Ketone': 'C(=O)C',
            'Aldehyde': 'C(=O)[H]',
            'Epoxide': 'C1OC1',  # 环氧化物
            'Cyano': 'C#N',  # 氰基
            'Carbamate': 'N-C(=O)O',  # 氨基甲酸酯
            'Urea': 'N-C(=O)-N',  # 脲类
            'Thiocarbamate': 'N-C(=S)O',  # 硫代氨基甲酸酯
            'Thiourea': 'N-C(=S)-N',  # 硫脲

            # === 芳香体系 ===
            'Benzene': 'c1ccccc1',
            'Naphthalene': 'c12ccccc1cccc2',
            'Biphenyl': 'c1ccccc1-c2ccccc2',
            'Phenyl': 'c1ccccc1[*]',  # 苯基

            # === 不饱和键 ===
            'Alkene': 'C=C',
            'Alkyne': 'C#C',
            'Diene': 'C=CC=C',

            # === 农药特有结构 ===
            'Chloropyridine': 'Clc1ncccc1',  # 氯吡啶
            'Chloropyrimidine': 'Clc1nccnc1',  # 氯嘧啶
            'Chlorotriazine': 'Clc1nc(N)nc(N)n1',  # 氯三嗪
            'Pyrethroid_Backbone': 'CC(C)=CC1C(=O)OC1',  # 拟除虫菊酯骨架
            'Strobilurin_Backbone': 'CC=CC(=O)Oc1ccccc1',  # 甲氧基丙烯酸酯类骨架
        }

        # 统计每个官能团的出现频率
        fg_stats = {}
        total_molecules = len(molecules)

        logger.info("从农药开发角度分析官能团分布...")

        # 添加进度条
        for fg_name, smarts in tqdm(functional_groups.items(), desc="农药官能团分析"):
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is None:
                    logger.warning(f"官能团 {fg_name} 的SMARTS模式无效: {smarts}")
                    continue

                counts = []
                for mol in molecules:
                    if mol is None:
                        counts.append(0)
                        continue
                    try:
                        matches = mol.GetSubstructMatches(pattern)
                        counts.append(len(matches))
                    except:
                        counts.append(0)

                molecules_with_fg = sum(1 for c in counts if c > 0)
                if total_molecules > 0:
                    fg_stats[fg_name] = {
                        'total_occurrences': sum(counts),
                        'frequency': molecules_with_fg / total_molecules,
                        'molecules_with': molecules_with_fg,
                        'avg_per_molecule': sum(counts) / total_molecules,
                        'coverage_percentage': (molecules_with_fg / total_molecules) * 100
                    }

            except Exception as e:
                logger.warning(f"官能团 {fg_name} 分析失败: {str(e)}")

        # 生成农药专用的官能团可视化
        generate_pesticide_functional_group_plots(fg_stats, total_molecules, output_prefix)

        # 生成农药特性分析报告
        generate_pesticide_characterization_report(fg_stats, total_molecules, output_prefix)

        return fg_stats

    except Exception as e:
        logger.error(f"农药官能团分析失败: {str(e)}")
        return {}


def generate_pesticide_functional_group_plots(fg_stats, total_molecules, output_prefix):
    """生成农药专用的官能团分析图"""
    try:
        if not fg_stats:
            logger.warning("No functional group data available for visualization")
            return

        # 按频率排序
        fg_names = list(fg_stats.keys())
        frequencies = [fg_stats[fg]['frequency'] for fg in fg_names]
        sorted_indices = np.argsort(frequencies)[::-1]
        sorted_fg_names = [fg_names[i] for i in sorted_indices]
        y_pos = np.arange(len(sorted_fg_names))

        # 只显示频率前30的官能团，避免图表过于拥挤
        display_limit = min(30, len(sorted_fg_names))
        display_fg_names = sorted_fg_names[:display_limit]
        display_y_pos = np.arange(display_limit)

        # 1. 农药官能团频率图（重点关注）
        plt.figure(figsize=(14, max(10, display_limit * 0.4)))
        display_frequencies = [fg_stats[fg]['frequency'] for fg in display_fg_names]
        bars = plt.barh(display_y_pos, display_frequencies,
                        color=['#FF6B6B' if 'Phosph' in fg else  # 磷相关-红色
                               '#4ECDC4' if 'amine' in fg.lower() else  # 胺类-青色
                               '#45B7D1' if 'azole' in fg.lower() else  # 唑类-蓝色
                               '#96CEB4' if 'Halogen' in fg else  # 卤素-绿色
                               '#FECA57' if 'Sulf' in fg else  # 硫相关-黄色
                               '#FF9FF3' for fg in display_fg_names],  # 其他-粉色
                        alpha=0.8)

        plt.yticks(display_y_pos, display_fg_names)
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Functional Groups', fontsize=12)
        plt.title('Pesticide Molecular Functional Group Distribution Analysis', fontsize=14, fontweight='bold')
        plt.xlim(0, 1)
        plt.gca().invert_yaxis()

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_pesticide_functional_groups_frequency.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 按类别分组的官能团分析
        category_stats = categorize_pesticide_functional_groups(fg_stats)
        plot_functional_group_categories(category_stats, output_prefix)

        # 3. 农药特性雷达图
        plot_pesticide_property_radar(fg_stats, output_prefix)

        logger.info(f"✓ Pesticide functional group analysis completed, professional charts generated")

    except Exception as e:
        logger.error(f"Pesticide functional group visualization failed: {str(e)}")


def categorize_pesticide_functional_groups(fg_stats):
    """将官能团按农药相关类别分组"""
    categories = {
        'Phosphorus-containing group': ['Phosphate', 'Phosphorothioate', 'Phosphonate', 'Phosphinate', 'Phosphine_Oxide'],
        'Nitrogen-containing group': ['Primary_Amine', 'Secondary_Amine', 'Tertiary_Amine', 'Amide', 'Thiocarbamate', 'Nitro', 'Nitroso', 'Cyano', 'Cyanate', 'Imine','Urea','Carbamate','Thiourea'],
        'Azole': ['Triazole', 'Imidazole', 'Pyridine', 'Pyrimidine', 'Pyrazole', 'Triazine'],
        'halogen substitution': ['Fluorine', 'Chlorine', 'Bromine', 'Iodine', 'CF3_Group', 'OCF3_Group'],
        'Sulfur-containing group': ['Sulfonyl', 'Sulfonate', 'Sulfonamide', 'Sulfoxide', 'Sulfone', 'Thioether'],
        'ester/acid': ['Carboxylic_Acid', 'Ester', 'Carbamate', 'Urea', 'Thiocarbamate', 'Thiourea'],
        'Other groups': ['Phenol', 'Ether', 'Ketone', 'Aldehyde']
    }

    category_results = {}
    for category, groups in categories.items():
        total_frequency = 0
        total_molecules_with = 0
        count = 0

        for fg in groups:
            if fg in fg_stats:
                total_frequency += fg_stats[fg]['frequency']
                total_molecules_with += fg_stats[fg]['molecules_with']
                count += 1

        if count > 0:
            category_results[category] = {
                'avg_frequency': total_frequency / count,
                'total_molecules': total_molecules_with,
                'frequency': total_molecules_with / 1108,
                'group_count': count
            }

    return category_results


def plot_functional_group_categories(category_stats, output_prefix):
    """绘制官能团类别分析图"""

    if not category_stats:
        return

    categories = list(category_stats.keys())
    frequencies = [category_stats[cat]['frequency'] for cat in categories]

    plt.figure(figsize=(10, 8))
    bars = plt.bar(categories, frequencies, color='lightsteelblue', alpha=0.8)
    plt.ylabel('frequency', fontsize=12)
    plt.xlabel('Functional Group Categories', fontsize=12)
    plt.title('Functional Group Category Distribution', fontsize=14, pad=10, y=-0.15)

    # 设置y轴范围为0-1
    plt.ylim(2.0, 0)

    # 获取当前坐标轴
    ax = plt.gca()

    # 将X轴移到上方
    ax.xaxis.set_ticks_position('top')  # 刻度放在上方
    ax.xaxis.set_label_position('top')  # 标签放在上方

    # 隐藏下方的X轴脊柱（边框线）
    ax.spines['bottom'].set_visible(False)

    # 可选：调整上方脊柱的样式，使其更明显
    ax.spines['top'].set_linewidth(1.2)

    # 旋转x轴标签（现在在上方，旋转方向可以保持不变）
    plt.xticks(rotation=45, ha='left', fontsize=10)  # ha='left' 让标签左对齐，更易读

    # 调整y轴标签位置
    ax.yaxis.set_label_coords(-0.08, 0.5)

    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_pesticide_functional_categories2.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_pesticide_functional_categories2.svg", bbox_inches='tight')
    plt.close()


def plot_pesticide_property_radar(fg_stats, output_prefix):
    """绘制特性雷达图"""
    # 定义相关特性指标
    properties = {
        'Bioactivity': ['Phosphate', 'Phosphorothioate', 'Carbamate', 'Urea', 'Triazole', 'Imidazole'],
        'Systemicity': ['Amide', 'Amine', 'Alcohol', 'Carboxylic_Acid'],
        'Stability': ['Halogen', 'CF3_Group', 'Aromatic', 'Nitro'],
        'Selectivity': ['Ester', 'Ether', 'Sulfonyl', 'Sulfonamide'],
        'Environmental_Safety': ['Ester', 'Amide', 'Alcohol', 'Phenol'],
        'Lipophilicity': ['Halogen', 'CF3_Group', 'Aromatic', 'Alkyl']
    }

    # 计算每个特性的得分
    scores = []
    labels = list(properties.keys())

    for prop, groups in properties.items():
        score = 0
        for fg in groups:
            # 模糊匹配官能团名称
            matching_fgs = [k for k in fg_stats.keys() if any(g in k for g in groups if g in k)]
            for match_fg in matching_fgs:
                score += fg_stats[match_fg]['frequency']
        scores.append(min(score, 1.0))  # 限制最大值为1

    # 绘制雷达图
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores += scores[:1]  # 闭合雷达图
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.fill(angles, scores, color='red', alpha=0.25)
    ax.plot(angles, scores, color='red', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title('Pesticide Molecular Property Radar Chart', size=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_pesticide_property_radar.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def generate_pesticide_characterization_report(fg_stats, total_molecules, output_prefix):
    """生成农药特性分析报告"""
    try:
        # 保存详细的统计数据
        fg_df = pd.DataFrame([
            {
                'Functional_Group': fg,
                'Frequency': stats['frequency'],
                'Molecules_With': stats['molecules_with'],
                'Total_Occurrences': stats['total_occurrences'],
                'Average_per_Molecule': stats['avg_per_molecule'],
                'Coverage_Percentage': stats['coverage_percentage']
            }
            for fg, stats in fg_stats.items()
        ])

        fg_df = fg_df.sort_values('Frequency', ascending=False)
        fg_df.to_excel(f"{output_prefix}_pesticide_functional_groups.xlsx", index=False)
        fg_df.to_csv(f"{output_prefix}_pesticide_functional_groups.csv", index=False)

        # 生成简要报告
        with open(f"{output_prefix}_pesticide_analysis_report.txt", "w") as f:
            f.write("=== PESTICIDE MOLECULAR CHARACTERIZATION REPORT ===\n\n")
            f.write(f"Total Molecules Analyzed: {total_molecules}\n")
            f.write(f"Functional Groups Detected: {len(fg_stats)}\n\n")

            # 重点官能团统计
            f.write("\nKEY FUNCTIONAL GROUP STATISTICS:\n")
            important_groups = ['Phosphate', 'Carbamate', 'Triazole', 'Chlorine', 'CF3_Group']
            for group in important_groups:
                if group in fg_stats:
                    freq = fg_stats[group]['frequency']
                    f.write(f"{group}: {freq:.3f} ({fg_stats[group]['molecules_with']}个分子)\n")

            f.write("\n=== 分析完成 ===\n")

        logger.info(f"✓ 农药特性分析报告已生成")

    except Exception as e:
        logger.error(f"农药分析报告生成失败: {str(e)}")


def analyze_similarity(molecules, output_prefix):
    """分析类内相似性"""
    try:
        n_mols = len(molecules)
        if n_mols <= 1:
            logger.warning("分子数量不足，无法进行相似性分析")
            return {}

        logger.info("计算类内相似度矩阵...")

        # 生成指纹
        fps = []
        valid_indices = []
        for i, mol in enumerate(tqdm(molecules, desc="生成指纹")):
            fp = get_combined_fingerprint(mol)
            if fp:
                fps.append(fp)
                valid_indices.append(i)

        if len(fps) <= 1:
            logger.warning("有效指纹数量不足，无法计算相似度")
            return {}

        # 计算相似度矩阵
        n_valid = len(fps)
        sim_matrix = np.zeros((n_valid, n_valid))

        logger.info("计算相似度...")
        for i in tqdm(range(n_valid), desc="计算相似度"):
            for j in range(i, n_valid):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        # 生成相似度热图
        generate_similarity_heatmap(sim_matrix, output_prefix, valid_indices)

        # 计算统计信息
        np.fill_diagonal(sim_matrix, np.nan)  # 忽略对角线
        similarity_stats = {
            'avg_similarity': np.nanmean(sim_matrix),
            'min_similarity': np.nanmin(sim_matrix),
            'max_similarity': np.nanmax(sim_matrix),
            'std_similarity': np.nanstd(sim_matrix)
        }

        logger.info(f"类内相似度统计: 平均={similarity_stats['avg_similarity']:.3f}, "
                    f"范围=[{similarity_stats['min_similarity']:.3f}-{similarity_stats['max_similarity']:.3f}]")

        return similarity_stats

    except Exception as e:
        logger.error(f"相似性分析失败: {str(e)}")
        return {}


def generate_similarity_heatmap(sim_matrix, output_prefix, valid_indices):
    """生成相似度热图"""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))

        # 创建热图
        im = ax.imshow(sim_matrix, cmap='viridis', aspect='auto',
                       vmin=0, vmax=1, interpolation='nearest')

        # 设置颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Tanimoto Similarity', fontsize=12)

        # 设置坐标轴
        n_mols = len(sim_matrix)
        if n_mols <= 20:  # 只在分子数量少时显示标签
            ax.set_xticks(range(n_mols))
            ax.set_yticks(range(n_mols))
            ax.set_xticklabels([f'M{i + 1}' for i in range(n_mols)], rotation=45, fontsize=8)
            ax.set_yticklabels([f'M{i + 1}' for i in range(n_mols)], fontsize=8)
        else:
            # 分子数量多时，每10个显示一个标签
            xticks = range(0, n_mols, max(1, n_mols // 10))
            yticks = range(0, n_mols, max(1, n_mols // 10))
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels([f'M{i + 1}' for i in xticks], rotation=45, fontsize=6)
            ax.set_yticklabels([f'M{i + 1}' for i in yticks], fontsize=6)

        ax.set_xlabel('Molecule Index')
        ax.set_ylabel('Molecule Index')
        ax.set_title(f'Intra-cluster Similarity Matrix - Cluster 32\n({n_mols} molecules)')

        # 添加网格
        ax.set_xticks(np.arange(-0.5, n_mols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_mols, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_similarity_heatmap-all.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_prefix}_similarity_heatmap-all.svg", bbox_inches='tight')
        plt.close()

        # 保存相似度矩阵数据
        sim_df = pd.DataFrame(sim_matrix)
        sim_df.columns = [f"Mol_{i + 1}" for i in range(len(sim_matrix))]
        sim_df.index = [f"Mol_{i + 1}" for i in range(len(sim_matrix))]
        sim_df.to_excel(f"{output_prefix}_similarity_matrix.xlsx")

        logger.info("✓ 相似性分析完成")

    except Exception as e:
        logger.error(f"相似度热图生成失败: {str(e)}")


def generate_comprehensive_report(molecules, descriptor_results, functional_group_results,
                                  similarity_results, output_prefix, original_count):
    """生成综合分析报告"""
    try:
        report_lines = [
            "=" * 70,
            "CLUSTER 32 COMPREHENSIVE STRUCTURAL ANALYSIS REPORT",
            "=" * 70,
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Original molecules in cluster: {original_count}",
            f"Valid molecules processed: {len(molecules)}",
            f"Success rate: {len(molecules) / original_count * 100:.1f}%",
            "",
            "EXECUTIVE SUMMARY:",
            "-" * 50
        ]

        # 基本信息
        if molecules:
            report_lines.append(f"• Cluster contains {len(molecules)} successfully processed molecules")

        # 描述符摘要
        if not descriptor_results.empty:
            numeric_cols = descriptor_results.select_dtypes(include=[np.number]).columns.tolist()
            if 'Number' in numeric_cols:
                numeric_cols.remove('Number')

            if numeric_cols:
                report_lines.append("\nMOLECULAR DESCRIPTORS SUMMARY:")
                report_lines.append("-" * 40)

                for col in numeric_cols[:10]:  # 只显示前10个描述符
                    if col in descriptor_results.columns:
                        stats = descriptor_results[col].describe()
                        report_lines.append(f"{col}:")
                        report_lines.append(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
                        report_lines.append(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")

        # 官能团摘要
        if functional_group_results:
            report_lines.append("\nTOP FUNCTIONAL GROUPS:")
            report_lines.append("-" * 40)

            # 按频率排序
            sorted_fg = sorted(functional_group_results.items(),
                               key=lambda x: x[1]['frequency'], reverse=True)

            for fg_name, stats in sorted_fg[:10]:  # 只显示前10个
                report_lines.append(f"{fg_name}: {stats['frequency']:.1%} "
                                    f"({stats['molecules_with']}/{len(molecules)} molecules)")

        # 相似性摘要
        if similarity_results:
            report_lines.append("\nSIMILARITY ANALYSIS:")
            report_lines.append("-" * 40)
            report_lines.append(f"Average intra-cluster similarity: {similarity_results['avg_similarity']:.3f}")
            report_lines.append(f"Similarity range: {similarity_results['min_similarity']:.3f} - "
                                f"{similarity_results['max_similarity']:.3f}")
            report_lines.append(f"Similarity standard deviation: {similarity_results['std_similarity']:.3f}")

        # 结构多样性分析
        if len(molecules) > 1:
            report_lines.append("\nSTRUCTURAL DIVERSITY:")
            report_lines.append("-" * 40)

            # 计算一些简单的多样性指标
            heavy_atoms = [mol.GetNumHeavyAtoms() for mol in molecules]
            ring_counts = [Descriptors.RingCount(mol) for mol in molecules]

            report_lines.append(f"Heavy atoms range: {min(heavy_atoms)} - {max(heavy_atoms)}")
            report_lines.append(f"Ring counts range: {min(ring_counts)} - {max(ring_counts)}")

        report_lines.extend([
            "",
            "GENERATED FILES:",
            "-" * 50,
            f"1. Structure visualization: {output_prefix}_structures.{{png,svg}}",
            f"2. Descriptor analysis: {output_prefix}_descriptors.xlsx",
            f"3. Descriptor distribution: {output_prefix}_descriptors_distribution.{{png,svg}}",
            f"4. Functional group analysis: {output_prefix}_functional_groups.xlsx",
            f"5. Functional group visualization: {output_prefix}_functional_groups.{{png,svg}}",
            f"6. Similarity matrix: {output_prefix}_similarity_matrix.xlsx",
            f"7. Similarity heatmap: {output_prefix}_similarity_heatmap.{{png,svg}}",
            f"8. This report: {output_prefix}_analysis_report.txt",
            "",
            "=" * 70,
            "ANALYSIS COMPLETE",
            "=" * 70
        ])

        # 保存报告
        report_text = "\n".join(report_lines)
        with open(f"{output_prefix}_analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write(report_text)

        # 在日志中显示摘要
        logger.info("\n" + "\n".join(report_lines[:20]))  # 只显示前20行摘要

        logger.info("✓ 综合分析报告生成完成")

    except Exception as e:
        logger.error(f"报告生成失败: {str(e)}")


if __name__ == "__main__":
    analyze_cluster32_detailed()





