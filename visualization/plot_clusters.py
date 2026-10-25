from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import os

# 降维/聚类可视化归入 fsl/dimensionality_reduction
DIM_REDUCTION_DIR = "../results/fsl/dimensionality_reduction"
os.makedirs(DIM_REDUCTION_DIR, exist_ok=True)
def supervised_tsne(X, y, perplexity=30):
    """监督t-SNE实现（通过标签调整距离矩阵）"""
    # 创建标签相似性矩阵
    y_sim = np.array([y == y_i for y_i in y], dtype=float)
    # 调整距离矩阵
    distances = pairwise_distances(X)
    adjusted_dist = distances * (1 - y_sim) + distances * 0.4 * y_sim
    # 执行t-SNE
    tsne = TSNE(n_components=2,perplexity=perplexity,metric='precomputed',random_state=50,init='random')
    return tsne.fit_transform(adjusted_dist)
def supervised_umap(X, y):
    """监督UMAP实现"""
    return umap.UMAP(n_components=2,n_neighbors=15,min_dist=0.1,target_metric='categorical',target_weight=0.5,random_state=42).fit_transform(X, y=y)

def visualize_supervised_clusters(X, labels, true_labels, method='tsne', title_prefix=""):
    """监督流形学习可视化"""
    plt.figure(figsize=(18, 6))
    # 执行监督降维
    if method == 'tsne':
        X_emb = supervised_tsne(X, true_labels)
        title_suffix = "Supervised t-SNE"
    elif method == 'umap':
        X_emb = supervised_umap(X, true_labels)
        title_suffix = "UMAP"
    else:
        raise ValueError("Unsupported method")
    # 真实标签分布
    plt.subplot(131)
    scatter = plt.scatter(X_emb[:, 0], X_emb[:, 1], c=true_labels,
                          cmap='rainbow', alpha=0.7, edgecolors='w')
    cbar = plt.colorbar(scatter)
    cbar.set_label('True Labels', fontsize=14)
    cbar.ax.tick_params(labelsize=14)  # 调整颜色条刻度字体
    plt.title(f"{title_prefix}True Labels ({title_suffix})", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # 聚类结果
    plt.subplot(132)
    scatter = plt.scatter(X_emb[:, 0], X_emb[:, 1], c=labels,
                          cmap='bwr', alpha=0.7, edgecolors='w')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster Labels', fontsize=14)
    cbar.ax.tick_params(labelsize=14)  # 调整颜色条刻度字体
    plt.title(f"{title_prefix}Clustering Result ({title_suffix})", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # 置信度可视化
    plt.subplot(133)
    sil_values = silhouette_samples(X, labels)
    # 添加 vmin=0 和 vmax=1 限制颜色条范围
    sc = plt.scatter(X_emb[:, 0], X_emb[:, 1], c=sil_values,
                     cmap='coolwarm', alpha=0.7, edgecolors='w', vmin=0.01, vmax=0.08)
    cbar = plt.colorbar(sc)
    cbar.set_label('Silhouette Coefficient', fontsize=14)
    cbar.ax.tick_params(labelsize=14)  # 调整颜色条刻度字体
    plt.title(f"{title_prefix}Cluster Confidence ({title_suffix})", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    _stem = f"{title_prefix.lower().strip().replace(' ', '_')}_{method}"
    plt.savefig(os.path.join(DIM_REDUCTION_DIR, f"{_stem}.svg"))
    plt.savefig(os.path.join(DIM_REDUCTION_DIR, f"{_stem}.png"), dpi=900)
    plt.show()