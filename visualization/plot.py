import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns# 新增seaborn导入
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import f_oneway
import matplotlib.colors as mc
import matplotlib
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import os

# fsl 支线输出目录按用途划分
FSL_DIR = "../results/fsl"
DIM_REDUCTION_DIR = os.path.join(FSL_DIR, "dimensionality_reduction")
CLUSTERING_DIR = os.path.join(FSL_DIR, "clustering")
for _d in (DIM_REDUCTION_DIR, CLUSTERING_DIR):
    os.makedirs(_d, exist_ok=True)

# 重置matplotlib配置
# matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# font_path = "/data_disk/ljw/code/wyf/TIMES.TTF"
# 注册字体到 matplotlib
# fm.fontManager.addfont(font_path)
# matplotlib.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
# font = fm.FontProperties(fname=font_path)
# # 设置全局字体为 Times New Roman
# matplotlib.rcParams['font.family'] = 'Times New Roman'
# matplotlib.rcParams['font.family'] = 'Arial'  # 使用 Arial 字体
# 查找粗体字体并设置映射
# prop = fm.FontProperties(family=matplotlib.rcParams['font.family'], weight='bold')
# matplotlib.rcParams['font.weight'] = 'bold'
# matplotlib.rcParams['font.style'] = 'normal'
# matplotlib.rcParams['font.family'] = prop.get_name()
# 辅助函数：调浅颜色
def lighten_color(color, amount=0):
        import matplotlib.colors as mc
        import colorsys
        try:
                c = mc.cnames[color]
        except:
                c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
# 辅助函数：生成在原始颜色上渐变浅的颜色
def generate_gradient_colors(original_color, num_colors):
        gradient_colors = []
        for i in range(num_colors):
                amount = i / (num_colors - 1) * 0.8# 最大调浅程度为0.8
                lightened = lighten_color(original_color, amount)
                gradient_colors.append(lightened)
        return gradient_colors
# ==================== 数据预处理模块 ====================
def process_csv_data(filepath):
    """
    处理CSV表格数据
    """
    df = pd.read_csv(filepath)
    df = df.drop(columns=["label"])
    # 筛选数值型特征
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    features = df[numerical_cols]
    
    # 处理 NaN 值：使用均值填充
    features = features.fillna(features.mean())
    # 标准化所有特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, features, df
# ==================== 可视化模块 ====================
def plot_embedding(features, cluster_labels, method_name, save_path=None, dim=2):
    plt.figure(figsize=(12, 10))
    if dim == 3:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.gca()
    if dim == 3:
        ax = plt.axes(projection='3d')
        ax.view_init(elev=5, azim=55)# 调整3D视图的仰角和方位角
    else:
        ax = plt.gca()
     # 定义聚类标签的颜色和标记（无监督场景自定义）
    colors = {0: 'red', 1: 'blue', 2: 'orange', 3: 'green', 4: 'purple', 5: 'cyan', 6: 'gray'}
    markers = {0: 'o', 1: 'o', 2: 'o', 3: 'o', 4: 'o', 5: 'o', 6: 'o'}
    unique_labels = np.unique(cluster_labels)
    for ul in unique_labels:
        idx = cluster_labels == ul
        if dim == 3:
            ax.scatter(
                    features[idx, 0], features[idx, 1], features[idx, 2],
                    c=colors[ul],
                    marker=markers[ul],
                    label=f'Cluster {ul + 1}',# 标签从1开始
                    alpha=0.7,
                    edgecolors='w',
                    linewidths=0.5)
        else:
            ax.scatter(
                    features[idx, 0], features[idx, 1],
                    c=colors[ul],
                    marker=markers[ul],
                    label=f'Cluster {ul + 1}',# 标签从1开始
                    alpha=0.7,
                    edgecolors='w',
                    linewidths=0.5)
    ax.set_title(f'{method_name} Visualization', fontsize=14, weight='bold')
    if dim == 3:
        ax.set_xlabel('Component 2', labelpad=10, fontsize=14, weight='bold')
        ax.set_xlim(8, 0)
        ax.set_ylabel('Component 1', labelpad=10, fontsize=14, weight='bold')
        ax.set_ylim(0, 8)
        ax.set_zlabel('Component 3', labelpad=2, fontsize=14, weight='bold')
        # 设置 3D 坐标轴各平面（pane）的背景为白色
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0))# X轴平面
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0))# Y轴平面
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0))# Z轴平面
    else:
        ax.set_xlabel('Component 1', fontsize=14, weight='bold')
        ax.set_ylabel('Component 2', fontsize=14, weight='bold')
    ax.grid(alpha=0.3)
    # 调整坐标轴刻度字体大小
    if dim == 3:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z', labelsize=12)
    else:
        ax.tick_params(axis='both', labelsize=14)
    # 统一图例位置到右侧
    if dim == 3:
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1, 1),
            title_fontsize=14,
            fontsize=14)
    else:
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1, 1),
            title_fontsize=14,
            fontsize=14)
    plt.subplots_adjust(left=0.08, right=0.85)# 调整左边距
    if save_path:
        plt.savefig(save_path)
        # plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.show()
# 绘制不同簇的数目横向条形图并保存
def plot_cluster_counts(cluster_labels, method_name, prefix='csv'):
    colors = {0: 'red', 1: 'blue', 2: 'orange', 3: 'green', 4: 'purple', 5: 'cyan', 6: 'gray'}
    unique_labels = np.arange(6)# 确保标签从 0 到 5
    counts = [np.sum(cluster_labels == i) for i in unique_labels]
    labels = [f'Cluster {i + 1}' for i in unique_labels]
    plt.figure(figsize=(10, 6))
    bar_width = 0.6# 减小条形图宽度
    # lightened_colors = [lighten_color(colors[i], amount=0.8) for i in unique_labels[::-1]]
    # alpha_value = 0.8  # 设置透明度
    # bars = plt.barh(labels[::-1], counts[::-1], color=lightened_colors, height=bar_width, alpha=alpha_value)

    plt.xlabel('Counts', fontsize=14, fontweight='bold')# 加粗横坐标标签
    plt.ylabel('Cluster Labels', fontsize=14, fontweight='bold')# 加粗纵坐标标签
    # plt.title(f'{method_name} Cluster Counts')
# 生成调浅后的颜色（每个标签对应颜色由深及浅）
    lightened_colors = []
    for label in unique_labels:
        base_color = colors.get(label, 'gray')# 处理可能的标签缺失
        # 调整 amount 值控制颜色调浅程度（0 为原色，1 为最浅）
        lightened = lighten_color(base_color, amount=0.4)
        lightened_colors.append(lightened)
    bars = plt.barh(
            labels[::-1],
            counts[::-1],
            color=lightened_colors[::-1],# 反转顺序匹配标签
            height=bar_width,
            alpha=0.8)
    # 在条形图顶端添加聚类的数目
    font_size = 14# 设置字体大小
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}', ha='left', va='center', fontsize=font_size)
    plt.tight_layout()
    save_path = f"{CLUSTERING_DIR}/cluster_counts.svg"
    plt.savefig(save_path)
    # plt.savefig(save_path, format='svg', dpi=800, bbox_inches='tight', pad_inches=0.1)
    print(f"{method_name} 不同簇的数目条形图已保存到 {save_path}")
    plt.show()
# 小提琴图可视化模块
def plot_violinplots(key_features_with_labels, method_name, prefix='csv'):
    """
    绘制关键特征的小提琴图（按聚类分组）
    """
    # 获取聚类标签列名
    cluster_col = [col for col in key_features_with_labels.columns if 'cluster_label' in col][0]
    # 提取聚类标签并转换为0-based索引
    cluster_labels = key_features_with_labels[cluster_col].values - 1

    # 定义与条形图一致的颜色映射
    colors = {0: 'red', 1: 'blue', 2: 'orange', 3: 'green', 4: 'purple', 5: 'cyan', 6: 'gray'}
    # 生成调浅后的颜色（与条形图相同）
    unique_labels = np.unique(cluster_labels)
    cluster_colors = [lighten_color(colors[label], amount=0.4) for label in unique_labels]
    # 转换为长格式数据
    df_melt = key_features_with_labels.melt(
        id_vars=[cluster_col],
        var_name="Feature",
        value_name="Value")
    # 创建分面小提琴图
    plt.figure(figsize=(22, 15))
    g = sns.FacetGrid(
        df_melt,
        col="Feature",
        col_wrap=5,# 每行5个子图
        sharey=False,
        height=4,
        aspect=1.2)
    # 绘制小提琴图
    g = g.map_dataframe(
        sns.violinplot,
        x=cluster_col,
        y="Value",
        palette=cluster_colors,# 使用调浅后的颜色
        saturation=0.8,
        linewidth=0.5)
    # 设置坐标轴和标题
    g.set_titles(col_template="{col_name}", size=14, weight='bold')
    g.set_axis_labels("Cluster", "Feature Value", fontsize=14)
    for ax in g.axes.flat:
        ax.tick_params(axis='both', labelsize=13)# 设置刻度字体大小为14
    # 调整布局
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.2)
    g.fig.suptitle(f'Key Features Distribution by Cluster',
                        fontsize=18, weight='bold', y=0.98)
    plt.show()
    # 保存图像
    save_path = f"{CLUSTERING_DIR}/violinplots.svg"
    plt.savefig(save_path)
    # plt.savefig(save_path, format='svg', dpi=800, bbox_inches='tight', pad_inches=0.1)
    print(f"{method_name} 关键特征小提琴图已保存到 {save_path}")

    plt.close()
# ==================== 降维分析模块 ====================
def dimension_reduction_analysis(features, original_features, df, prefix='csv'):
    """
    执行2D和3D降维分析，支持：
    - PCA (2D/3D)
    - t-SNE (2D/3D)
    - UMAP (2D/3D)
    """
    method_configs = [
        # # 2D配置
        # {'name': 'PCA', 'model': PCA, 'n_components': 2, 'args': {}},
        # {'name': 't-SNE', 'model': TSNE, 'n_components': 2,
        # 'args': {'perplexity': 8, 'n_iter': 600, 'random_state': 50}},
        {'name': 'UMAP', 'model': umap.UMAP, 'n_components': 2,
            'args': {'n_neighbors': 10, 'min_dist': 0.01, 'random_state': 65}},# 调节 UMAP 参数
        # # 3D配置
        # {'name': 'PCA 3D', 'model': PCA, 'n_components': 3, 'args': {}},
        # {'name': 't-SNE 3D', 'model': TSNE, 'n_components': 3,
        # 'args': {'perplexity': 8, 'n_iter': 600, 'random_state': 50}},
        # {'name': 'UMAP 3D', 'model': umap.UMAP, 'n_components': 3,
        # 'args': {'n_neighbors': 10, 'min_dist': 0.05, 'random_state': 70}}, # 调节 UMAP 参数
    ]
    if features.shape[1] > 50:
        pca_pre = PCA(n_components=50)
        features = pca_pre.fit_transform(features)
    for config in method_configs:
        if config['model'] == PCA:
            model = PCA(n_components=config['n_components'])
        elif config['model'] == TSNE:
            model = TSNE(n_components=config['n_components'], **config['args'])
        elif config['model'] == umap.UMAP:
            model = umap.UMAP(n_components=config['n_components'], **config['args'])
        start_time = time.time()
        embedding = model.fit_transform(features)
        elapsed = time.time() - start_time
        print(f"{config['name']} 耗时：{elapsed:.1f} s")
        # K-means 聚类分析
        kmeans = KMeans(n_clusters=6, random_state=70)
        kmeans_labels = kmeans.fit_predict(embedding)
        dim_suffix = '3d' if config['n_components'] == 3 else '2d'
        save_path = f"{DIM_REDUCTION_DIR}/{config['name'].lower().replace(' ', '_')}_viz.svg"
        plot_embedding(
                embedding,
                kmeans_labels,
                method_name=config['name'],
                save_path=save_path,
                dim=config['n_components'])
        # 绘制不同簇的数目横向条形图并保存
        plot_cluster_counts(kmeans_labels, config['name'], prefix)
        if 'UMAP' in config['name']:
            print(1)
            # 提取关键特征
            f_values = []
            num_features = features.shape[1]
            for i in range(num_features):
                feature_data = [features[kmeans_labels == label, i] for label in np.unique(kmeans_labels)]
                f_value, _ = f_oneway(*feature_data)
                f_values.append(f_value)
            # 选择F值最大的前10个特征作为关键特征
            top_features_indices = np.argsort(f_values)[-10:][::-1]
            print(f"{config['name']} 聚类后的关键特征索引: {top_features_indices}")
            # 映射到原始数据并提取保存
            key_features = original_features.iloc[:, top_features_indices]
            save_path_key_features = f"{CLUSTERING_DIR}/key_features.csv"
            key_features.to_csv(save_path_key_features, index=False)
            print(f"{config['name']} 关键特征已保存到 {save_path_key_features}")
            # 将聚类标签映射到原始数据并保存
            df_with_labels = df.copy()
            df_with_labels[f'{config["name"]}_cluster_label'] = kmeans_labels + 1
            save_path_with_labels = f"{CLUSTERING_DIR}/with_labels.csv"
            df_with_labels.to_csv(save_path_with_labels, index=False)
            print(f"{config['name']} 聚类标签已映射到原始数据并保存到 {save_path_with_labels}")
            # 添加聚类标签到关键特征
            key_features_with_labels = key_features.copy()
            key_features_with_labels[f'{config["name"]}_cluster_label'] = kmeans_labels + 1
            # 绘制小提琴图（移动到此处）
            plot_violinplots(
                    key_features_with_labels=key_features_with_labels,
                    method_name=config['name'],
                    prefix=prefix)
            # 按聚类标签保存到不同的 sheet
            save_path_multi_sheet = f"{CLUSTERING_DIR}/key_features_multi_sheet.xlsx"
            with pd.ExcelWriter(save_path_multi_sheet) as writer:
                for label in np.unique(kmeans_labels + 1):
                    subset = key_features_with_labels[key_features_with_labels[f'{config["name"]}_cluster_label'] == label]
                    subset.to_excel(writer, sheet_name=f'Cluster_{label}', index=False)
                    print(f"{config['name']} 关键特征按聚类标签保存到不同 sheet: {save_path_multi_sheet}")
# ==================== 主程序 ====================
if __name__ == "__main__":
 # 读取并预处理CSV数据
    csv_features, original_features, df = process_csv_data(r'../dataset/train/fsl-1/Pomacea canaliculata compounds analysis data-2.csv')
# 执行降维分析并保存可视化结果
    dimension_reduction_analysis(csv_features, original_features, df, prefix='csv')