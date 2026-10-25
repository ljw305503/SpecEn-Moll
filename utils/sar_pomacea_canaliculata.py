
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
RESULTS_DIR = REPO_ROOT / "results"
# #读取化合物表格、转换并保存为数值格式 CSV
# import pandas as pd
# # 读取 Excel 文件并去掉空字符串
# file_path1 = "dataset/Pomacea canaliculata compounds analysis data.xlsx"
# data1 = pd.read_excel(file_path1).replace("", pd.NA)
# # # 检测数据类型
# # print("原始数据类型:")
# print(data1.dtypes)
# # 保存为 CSV
# csv_path1 = "results/SAR/Pomacea canaliculata/Pomacea canaliculata compounds analysis data.csv"
# data1.to_csv(csv_path1, index=False)
# print("数据已成功保存为 CSV 文件")


# # 检查 SMILES 有效性并分离(检查化合物分子的SMILES是否有效，全部有效显示“All smiles are valid”，保存有效SMILES数据集和无效SMILES数据集)
# import pandas as pd
# from rdkit import Chem
# df1 = pd.read_csv("results/SAR/Pomacea canaliculata/Pomacea canaliculata compounds analysis data.csv")
# # 检查 SMILES 有效性
# valid_smiles = []
# invalid_smiles = []
# def validate_smiles(smiles):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             invalid_smiles.append(smiles)
#         return mol
#     except Exception as e:
#         invalid_smiles.append(smiles)
#         return None
# # 应用函数来检查 smiles
# df1['Mol'] = df1['smiles'].apply(validate_smiles)
# # 打印无效的 smiles 字符串
# if invalid_smiles:
#     print("Invalid smiles found:")
#     for smi in invalid_smiles:
#         print(smi)
# else:
#   print("All smiles are valid.")
# for index, row in df1.iterrows():
#     mol = Chem.MolFromSmiles(row['smiles'])
#     if mol is not None:
#         valid_smiles.append(row)
#     else:
#         invalid_smiles.append(row)
# # 保存有效的 smiles 数据
# pd.DataFrame(valid_smiles).to_csv("results/SAR/Pomacea canaliculata/Pomacea canaliculata compounds analysis data1.csv", index=False)




# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import Descriptors
# from rdkit.Chem import Lipinski

# # 读取原始数据
# df = pd.read_csv('results/SAR/Pomacea canaliculata/Pomacea canaliculata compounds analysis data1.csv')

# # 初始化计算列
# df['MW'] = None
# df['logP'] = None
# df['logD'] = None
# df['HBA'] = None
# df['HBD'] = None

# # 定义logD计算函数（默认pH=7.4）
# def calculate_logd(smiles, pH=7.4):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         return Descriptors.MolLogD(mol, pH)
#     except:
#         return None

# # 遍历计算各属性
# error_count = 0
# for idx, smiles in enumerate(df['smiles']):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
        
#         # 分子量
#         df.at[idx, 'MW'] = Descriptors.MolWt(mol)
        
#         # logP
#         df.at[idx, 'logP'] = Descriptors.MolLogP(mol)
        
#         # logD (pH=7.4)
#         df.at[idx, 'logD'] = calculate_logd(smiles)
        
#         # 氢键受体和供体
#         df.at[idx, 'HBA'] = Lipinski.NumHAcceptors(mol)
#         df.at[idx, 'HBD'] = Lipinski.NumHDonors(mol)
        
#     except Exception as e:
#         error_count += 1
#         print(f"Error processing SMILES {idx}: {str(e)}")

# print(f"成功处理 {len(df)-error_count} 条记录，{error_count} 条记录出现错误")

# # 保存结果
# output_path = 'results/SAR/Pomacea canaliculata/Pomacea canaliculata compounds analysis data1-1.csv'
# df.to_csv(output_path, index=False)
# print(f"结果已保存至：{output_path}")





# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# # 设置全局字体为 Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'

# # 读取 CSV 文件
# data = pd.read_csv('results/SAR/Pomacea canaliculata/Pomacea canaliculata compounds analysis data1-1.csv')

# # 获取所需数据
# x = data['HBA']
# y = data['MW']
# z = data['logP']

# # 获取 3 个指标的最大值和最小值
# x_min, x_max = min(x), max(x)
# y_min, y_max = min(y), 1760
# z_min, z_max = min(z), max(z)

# # 创建 3D 图形
# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection='3d')
# ax.view_init(elev=35, azim=20)

# # 绘制化合物分子对应的点
# ax.scatter(x, y, z, c='purple', marker='o', label='Compound Molecules')

# # 定义空间体的 8 个顶点
# vertices = [
#     [x_min, y_min, z_min],
#     [x_max, y_min, z_min],
#     [x_max, y_max, z_min],
#     [x_min, y_max, z_min],
#     [x_min, y_min, z_max],
#     [x_max, y_min, z_max],
#     [x_max, y_max, z_max],
#     [x_min, y_max, z_max]
# ]

# # 定义空间体的 6 个面
# faces = [
#     [vertices[0], vertices[1], vertices[2], vertices[3]],
#     [vertices[4], vertices[5], vertices[6], vertices[7]],
#     [vertices[0], vertices[1], vertices[5], vertices[4]],
#     [vertices[2], vertices[3], vertices[7], vertices[6]],
#     [vertices[0], vertices[3], vertices[7], vertices[4]],
#     [vertices[1], vertices[2], vertices[6], vertices[5]]
# ]

# # 创建 Poly3DCollection 对象并添加到 3D 图形中
# poly3d = Poly3DCollection(faces, alpha=0.2, facecolors='lightskyblue')
# ax.add_collection3d(poly3d)

# # 设置坐标轴标签和标题
# ax.set_xlabel('HBA',labelpad=20, fontsize=14, weight='bold')
# ax.set_ylabel('MW', labelpad=20, fontsize=14, weight='bold')
# ax.set_zlabel('logP', labelpad=10, fontsize=14, weight='bold')
# ax.set_title('3D Visualization of Compounds', fontsize=14, weight='bold')
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)
# ax.tick_params(axis='z', labelsize=12)
# ax.legend()

# # 固定坐标轴范围
# ax.set_xlim(40, -5)
# ax.set_ylim(0, 2000)
# ax.set_zlim(-5, 30)

# # 设置 3D 坐标轴各平面（pane）的背景为白色
# ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0))  # X轴平面
# ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0))  # Y轴平面
# ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0))  # Z轴平面

# # 显示图形
# plt.savefig('results/SAR/Pomacea canaliculata/Pc1-1.svg')
# plt.show()





# #提取分子描述符和指纹特征（使用RDKit，从合物表格中依据化合物的smiles获取分子式，InChI，InChIKey，及其相应的分子描述符和分子指纹等208个分子描述符。）
# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors
# from rdkit.Chem.rdMolDescriptors import CalcMolFormula
# from rdkit.Chem.inchi import MolToInchi, MolToInchiKey
# from rdkit import DataStructs

# # 加载数据
# input_file1 = "results/SAR/Pomacea canaliculata/Pomacea canaliculata compounds analysis data1.csv"  # 输入的化合物表格，需包含 SMILES 列
# output_file1 = "results/SAR/Pomacea canaliculata/Pomacea canaliculata compounds analysis data2.csv"  # 输出文件路径

# # 读取表格
# df1 = pd.read_csv(input_file1)

# # 确保数据中包含 'smiles' 列
# if 'smiles' not in df1.columns:
#     raise ValueError("输入表格中必须包含 'smiles' 列！")

# # 准备存储所有特征的 DataFrame
# descriptor_names = [desc[0] for desc in Descriptors._descList]
# columns = ['MolFormula', 'InChI', 'InChIKey'] + descriptor_names + [
#     'RDKFingerprint', 'TopologicalFP', 'MACCS',
#     'AtomPairsFP', 'TopologicalTorsionsFP',
#     'MorganFP_2', 'MorganFP_3', 'molprint2D'
# ]
# results = []

# # 遍历 SMILES 计算特征
# for idx, row in df1.iterrows():
#     smiles = row['smiles']
#     try:
#         # 转换 smiles 为分子对象
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             raise ValueError("无效的 smiles")
#         Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)
#         norm_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
#         # 获取分子式、InChI、InChIKey
#         mol_formula = CalcMolFormula(mol)
#         inchi = MolToInchi(mol)
#         inchi_key = MolToInchiKey(mol)

#         # 计算 208 个描述符
#         descriptor_values = [Descriptors.__dict__[name](mol) for name in descriptor_names]

#         # 计算指纹特征
#         rdk_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2).ToBitString()  # RDKFingerprint
#         topo_fp = AllChem.RDKFingerprint(mol).ToBitString()  # TopologicalFP
#         maccs_fp = MACCSkeys.GenMACCSKeys(mol).ToBitString()  # MACCS
#         atom_pairs_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol).ToBitString()  # AtomPairsFP
#         topo_torsions_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol).ToBitString()  # TopologicalTorsionsFP
#         morgan_fp_2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2).ToBitString()  # MorganFP_2
#         morgan_fp_3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3).ToBitString()  # MorganFP_3

#         # 计算 molprint2D
#         molprint2D = AllChem.GetMorganFingerprint(mol, radius=2, useFeatures=True).GetNonzeroElements()
#         molprint2D_str = ",".join(f"{k}:{v}" for k, v in molprint2D.items())  # 转换为字符串
#         # 将结果添加到列表中
#         results.append(
#             [mol_formula, inchi, inchi_key] +
#             descriptor_values +
#             [rdk_fp, topo_fp, maccs_fp, atom_pairs_fp, topo_torsions_fp, morgan_fp_2, morgan_fp_3, molprint2D_str]
#         )

#     except Exception as e:
#         print(f"SMILES {smiles} 计算失败，原因：{e}")
#         results.append([np.nan] * len(columns))

#     # 将结果保存到 DataFrame
#     features_df1 = pd.DataFrame(results, columns=columns)

# # 合并原始数据和特征
# df1 = pd.concat([df1, features_df1], axis=1)

# # 保存到 CSV 文件
# df1.to_csv(output_file1, index=False)
# print(f"特征提取完成，结果已保存到 {output_file1}")









# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns  # 新增seaborn导入
# import time
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import umap.umap_ as umap
# from sklearn.cluster import KMeans
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.stats import f_oneway
# import matplotlib.colors as mc
# import matplotlib
# import matplotlib.font_manager as fm
# from matplotlib.font_manager import FontProperties

# # 重置matplotlib配置
# matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# font: Times New Roman via rcParams
# # 注册字体到 matplotlib
# fm.fontManager.addfont(font_path)
# matplotlib.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
# # font = fm.FontProperties(fname=font_path)
# # # 设置全局字体为 Times New Roman
# # matplotlib.rcParams['font.family'] = 'Times New Roman'
# # matplotlib.rcParams['font.family'] = 'Arial'  # 使用 Arial 字体
# # 查找粗体字体并设置映射
# # prop = fm.FontProperties(family=matplotlib.rcParams['font.family'], weight='bold')
# # matplotlib.rcParams['font.weight'] = 'bold'
# # matplotlib.rcParams['font.style'] = 'normal'
# # matplotlib.rcParams['font.family'] = prop.get_name()


# # 辅助函数：调浅颜色
# def lighten_color(color, amount=0):
#     import matplotlib.colors as mc
#     import colorsys
#     try:
#         c = mc.cnames[color]
#     except:
#         c = color
#     c = colorsys.rgb_to_hls(*mc.to_rgb(c))
#     return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

# # 辅助函数：生成在原始颜色上渐变浅的颜色
# def generate_gradient_colors(original_color, num_colors):
#     gradient_colors = []
#     for i in range(num_colors):
#         amount = i / (num_colors - 1) * 0.8  # 最大调浅程度为0.8
#         lightened = lighten_color(original_color, amount)
#         gradient_colors.append(lightened)
#     return gradient_colors


# # ==================== 数据预处理模块 ====================
# def process_csv_data(filepath):
#     """
#     处理CSV表格数据
#     """
#     df = pd.read_csv(filepath)
#     df = df.drop(columns=["label"])
#     # 筛选数值型特征
#     numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     features = df[numerical_cols]

#     # 处理 NaN 值：使用均值填充
#     features = features.fillna(features.mean())

#     # 标准化所有特征
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(features)

#     return scaled_features, features, df


# # ==================== 可视化模块 ====================
# def plot_embedding(features, cluster_labels, method_name, save_path=None, dim=2):
#     plt.figure(figsize=(12, 10))
#     if dim == 3:
#         ax = plt.axes(projection='3d')
#     else:
#         ax = plt.gca()
#     if dim == 3:
#         ax = plt.axes(projection='3d')
#         ax.view_init(elev=5, azim=55)  # 调整3D视图的仰角和方位角
#     else:
#         ax = plt.gca()

#     # 定义聚类标签的颜色和标记（无监督场景自定义）
#     colors = {0: 'red', 1: 'blue', 2: 'orange', 3: 'green', 4: 'purple', 5: 'cyan', 6: 'gray'}
#     markers = {0: 'o', 1: 'o', 2: 'o', 3: 'o', 4: 'o', 5: 'o', 6: 'o'}

#     unique_labels = np.unique(cluster_labels)
#     for ul in unique_labels:
#         idx = cluster_labels == ul
#         if dim == 3:
#             ax.scatter(
#                 features[idx, 0], features[idx, 1], features[idx, 2],
#                 c=colors[ul],
#                 marker=markers[ul],
#                 label=f'Cluster {ul + 1}',  # 标签从1开始
#                 alpha=0.7,
#                 edgecolors='w',
#                 linewidths=0.5
#             )
#         else:
#             ax.scatter(
#                 features[idx, 0], features[idx, 1],
#                 c=colors[ul],
#                 marker=markers[ul],
#                 label=f'Cluster {ul + 1}',  # 标签从1开始
#                 alpha=0.7,
#                 edgecolors='w',
#                 linewidths=0.5
#             )

#     ax.set_title(f'{method_name} Visualization', fontsize=14, weight='bold')
#     if dim == 3:
#         ax.set_xlabel('Component 2', labelpad=10, fontsize=14, weight='bold')
#         ax.set_xlim(8, 0)
#         ax.set_ylabel('Component 1', labelpad=10, fontsize=14, weight='bold')
#         ax.set_ylim(0, 8)
#         ax.set_zlabel('Component 3', labelpad=2, fontsize=14, weight='bold')
#         # 设置 3D 坐标轴各平面（pane）的背景为白色
#         ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0))  # X轴平面
#         ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0))  # Y轴平面
#         ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0))  # Z轴平面

#     else:
#         ax.set_xlabel('Component 1', fontsize=14, weight='bold')
#         ax.set_ylabel('Component 2', fontsize=14, weight='bold')

#     ax.grid(alpha=0.3)

#     # 调整坐标轴刻度字体大小
#     if dim == 3:
#         ax.tick_params(axis='x', labelsize=12)
#         ax.tick_params(axis='y', labelsize=12)
#         ax.tick_params(axis='z', labelsize=12)
#     else:
#         ax.tick_params(axis='both', labelsize=14)

#     # 统一图例位置到右侧
#     if dim == 3:
#         ax.legend(
#             loc='upper left',
#             bbox_to_anchor=(1, 1),
#             title_fontsize=14,
#             fontsize=14
#         )

#     else:
#         ax.legend(
#             loc='upper left',
#             bbox_to_anchor=(1, 1),
#             title_fontsize=14,
#             fontsize=14
#         )
#     plt.subplots_adjust(left=0.08, right=0.85)  # 调整左边距

#     if save_path:
#         plt.savefig(save_path)
#         # plt.savefig(save_path, dpi=800, bbox_inches='tight')
#     plt.show()

# # 绘制不同簇的数目横向条形图并保存
# def plot_cluster_counts(cluster_labels, method_name, prefix='csv'):
#     colors = {0: 'red', 1: 'blue', 2: 'orange', 3: 'green', 4: 'purple', 5: 'cyan', 6: 'gray'}
#     unique_labels = np.arange(6)  # 确保标签从 0 到 5
#     counts = [np.sum(cluster_labels == i) for i in unique_labels]
#     labels = [f'Cluster {i + 1}' for i in unique_labels]

#     plt.figure(figsize=(10, 6))
#     bar_width = 0.6  # 减小条形图宽度
#     # lightened_colors = [lighten_color(colors[i], amount=0.8) for i in unique_labels[::-1]]
#     # alpha_value = 0.8  # 设置透明度
#     # bars = plt.barh(labels[::-1], counts[::-1], color=lightened_colors, height=bar_width, alpha=alpha_value) 
#     #    
#     plt.xlabel('Counts', fontsize=14, fontweight='bold')  # 加粗横坐标标签
#     plt.ylabel('Cluster Labels', fontsize=14, fontweight='bold')  # 加粗纵坐标标签
#     # plt.title(f'{method_name} Cluster Counts')

# # 生成调浅后的颜色（每个标签对应颜色由深及浅）
#     lightened_colors = []
#     for label in unique_labels:
#         base_color = colors.get(label, 'gray')  # 处理可能的标签缺失
#         # 调整 amount 值控制颜色调浅程度（0 为原色，1 为最浅）
#         lightened = lighten_color(base_color, amount=0.4) 
#         lightened_colors.append(lightened)

#     bars = plt.barh(
#         labels[::-1], 
#         counts[::-1], 
#         color=lightened_colors[::-1],  # 反转顺序匹配标签
#         height=bar_width, 
#         alpha=0.8
#     )

#     # 在条形图顶端添加聚类的数目
#     font_size = 14  # 设置字体大小
#     for bar in bars:
#         width = bar.get_width()
#         plt.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}', ha='left', va='center', fontsize=font_size)

#     plt.tight_layout()

#     save_path = f"results/SAR/Pomacea canaliculata/{prefix}_{method_name.lower().replace(' ', '_')}_cluster_counts.svg"
#     plt.savefig(save_path)
#     # plt.savefig(save_path, format='svg', dpi=800, bbox_inches='tight', pad_inches=0.1)
#     print(f"{method_name} 不同簇的数目条形图已保存到 {save_path}")
#     plt.show()

# # 小提琴图可视化模块
# def plot_violinplots(key_features_with_labels, method_name, prefix='csv'):
#     """
#     绘制关键特征的小提琴图（按聚类分组）
#     """
#     # 获取聚类标签列名
#     cluster_col = [col for col in key_features_with_labels.columns if 'cluster_label' in col][0]
    
#     # 提取聚类标签并转换为0-based索引
#     cluster_labels = key_features_with_labels[cluster_col].values - 1
    
#     # 定义与条形图一致的颜色映射
#     colors = {0: 'red', 1: 'blue', 2: 'orange', 3: 'green', 4: 'purple', 5: 'cyan', 6: 'gray'}
    
#     # 生成调浅后的颜色（与条形图相同）
#     unique_labels = np.unique(cluster_labels)
#     cluster_colors = [lighten_color(colors[label], amount=0.4) for label in unique_labels]

#     # 转换为长格式数据
#     df_melt = key_features_with_labels.melt(
#         id_vars=[cluster_col],
#         var_name="Feature",
#         value_name="Value"
#     )
    
#     # 创建分面小提琴图
#     plt.figure(figsize=(22, 15))
#     g = sns.FacetGrid(
#         df_melt,
#         col="Feature",
#         col_wrap=5,  # 每行5个子图
#         sharey=False,
#         height=4,
#         aspect=1.2
#     )
    
#     # 绘制小提琴图
#     g = g.map_dataframe(
#         sns.violinplot,
#         x=cluster_col,
#         y="Value",
#         palette=cluster_colors,  # 使用调浅后的颜色
#         saturation=0.8,
#         linewidth=0.5
#     )
    
#     # 设置坐标轴和标题
#     g.set_titles(col_template="{col_name}", size=14, weight='bold')
#     g.set_axis_labels("Cluster", "Feature Value", fontsize=14)

#     for ax in g.axes.flat:
#       ax.tick_params(axis='both', labelsize=13)  # 设置刻度字体大小为14

#     # 调整布局
#     plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.2)
#     g.fig.suptitle(f'Key Features Distribution by Cluster', 
#                   fontsize=18, weight='bold', y=0.98)
    
#    # 保存图像
#     save_path = f"results/SAR/Pomacea canaliculata/{prefix}_{method_name.lower().replace(' ', '_')}_violinplots.svg"
#     plt.savefig(save_path)
#     # plt.savefig(save_path, format='svg', dpi=800, bbox_inches='tight', pad_inches=0.1)
#     print(f"{method_name} 关键特征小提琴图已保存到 {save_path}")
#     plt.close()

# # ==================== 降维分析模块 ====================
# def dimension_reduction_analysis(features, original_features, df, prefix='csv'):
#     """
#     执行2D和3D降维分析，支持：
#     - PCA (2D/3D)
#     - t-SNE (2D/3D)
#     - UMAP (2D/3D)
#     """
#     method_configs = [
#         # # 2D配置
#         # {'name': 'PCA', 'model': PCA, 'n_components': 2, 'args': {}},
#         # {'name': 't-SNE', 'model': TSNE, 'n_components': 2,
#         #  'args': {'perplexity': 8, 'n_iter': 600, 'random_state': 50}},
#         {'name': 'UMAP', 'model': umap.UMAP, 'n_components': 2,
#          'args': {'n_neighbors': 10, 'min_dist': 0.01, 'random_state': 65}},  # 调节 UMAP 参数

#         # # 3D配置
#         # {'name': 'PCA 3D', 'model': PCA, 'n_components': 3, 'args': {}},
#         # {'name': 't-SNE 3D', 'model': TSNE, 'n_components': 3,
#         #  'args': {'perplexity': 8, 'n_iter': 600, 'random_state': 50}},
#         # {'name': 'UMAP 3D', 'model': umap.UMAP, 'n_components': 3,
#         #  'args': {'n_neighbors': 10, 'min_dist': 0.05, 'random_state': 70}},  # 调节 UMAP 参数
#     ]

#     if features.shape[1] > 50:
#         pca_pre = PCA(n_components=50)
#         features = pca_pre.fit_transform(features)

#     for config in method_configs:
#         if config['model'] == PCA:
#             model = PCA(n_components=config['n_components'])
#         elif config['model'] == TSNE:
#             model = TSNE(n_components=config['n_components'], **config['args'])
#         elif config['model'] == umap.UMAP:
#             model = umap.UMAP(n_components=config['n_components'], **config['args'])

#         start_time = time.time()
#         embedding = model.fit_transform(features)
#         elapsed = time.time() - start_time
#         print(f"{config['name']} 耗时：{elapsed:.1f} s")

#         # K-means 聚类分析
#         kmeans = KMeans(n_clusters=6, random_state=70)
#         kmeans_labels = kmeans.fit_predict(embedding)

#         dim_suffix = '3d' if config['n_components'] == 3 else '2d'
#         save_path = f"results/SAR/Pomacea canaliculata/{prefix}_{config['name'].lower().replace(' ', '_')}_viz.svg"
#         plot_embedding(
#             embedding,
#             kmeans_labels,
#             method_name=config['name'],
#             save_path=save_path,
#             dim=config['n_components']
#         )

#         # 绘制不同簇的数目横向条形图并保存
#         plot_cluster_counts(kmeans_labels, config['name'], prefix)

#         if 'UMAP' in config['name']:
#             # 提取关键特征
#             f_values = []
#             num_features = features.shape[1]
#             for i in range(num_features):
#                 feature_data = [features[kmeans_labels == label, i] for label in np.unique(kmeans_labels)]
#                 f_value, _ = f_oneway(*feature_data)
#                 f_values.append(f_value)

#             # 选择F值最大的前10个特征作为关键特征
#             top_features_indices = np.argsort(f_values)[-10:][::-1]
#             print(f"{config['name']} 聚类后的关键特征索引: {top_features_indices}")

#             # 映射到原始数据并提取保存
#             key_features = original_features.iloc[:, top_features_indices]
#             save_path_key_features = f"results/SAR/Pomacea canaliculata/{prefix}_{config['name'].lower().replace(' ', '_')}_key_features.csv"
#             key_features.to_csv(save_path_key_features, index=False)
#             print(f"{config['name']} 关键特征已保存到 {save_path_key_features}")

#             # 将聚类标签映射到原始数据并保存
#             df_with_labels = df.copy()
#             df_with_labels[f'{config["name"]}_cluster_label'] = kmeans_labels + 1
#             save_path_with_labels = f"results/SAR/Pomacea canaliculata/{prefix}_{config['name'].lower().replace(' ', '_')}_with_labels.csv"
#             df_with_labels.to_csv(save_path_with_labels, index=False)
#             print(f"{config['name']} 聚类标签已映射到原始数据并保存到 {save_path_with_labels}")

#             # 添加聚类标签到关键特征
#             key_features_with_labels = key_features.copy()
#             key_features_with_labels[f'{config["name"]}_cluster_label'] = kmeans_labels + 1

#             # 绘制小提琴图（移动到此处）
#             plot_violinplots(
#                 key_features_with_labels=key_features_with_labels,
#                 method_name=config['name'],
#                 prefix=prefix
#             )

#             # 按聚类标签保存到不同的 sheet
#             save_path_multi_sheet = f"results/SAR/Pomacea canaliculata/{prefix}_{config['name'].lower().replace(' ', '_')}_key_features_multi_sheet.xlsx"
#             with pd.ExcelWriter(save_path_multi_sheet) as writer:
#                 for label in np.unique(kmeans_labels + 1):
#                     subset = key_features_with_labels[key_features_with_labels[f'{config["name"]}_cluster_label'] == label]
#                     subset.to_excel(writer, sheet_name=f'Cluster_{label}', index=False)
#             print(f"{config['name']} 关键特征按聚类标签保存到不同 sheet: {save_path_multi_sheet}")


# # ==================== 主程序 ====================
# if __name__ == "__main__":
#     # 读取并预处理CSV数据
#     csv_features, original_features, df = process_csv_data(r'results/SAR/Pomacea canaliculata/Pomacea canaliculata compounds analysis data2.csv')
#     # 执行降维分析并保存可视化结果
#     dimension_reduction_analysis(csv_features, original_features, df, prefix='csv')





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
# input_file = str(DATASET_DIR / "Pomacea canaliculata compounds analysis data.xlsx")  # 输入的化合物表格，需包含 SMILES 列
# output_file_gnn = str(RESULTS_DIR / "SAR" / "Pomacea canaliculata" / "gnn" / "pc-gnn.pt")  # 输出GNN格式
# output_image_folder = str(RESULTS_DIR / "SAR" / "Pomacea canaliculata" / "gnn")  # 输出GNN图像文件夹

# # 读取表格
# df = pd.read_excel(input_file) if str(input_file).endswith(('.xlsx', '.xls')) else pd.read_csv(input_file)

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
#     current_label = int(row['label'])  # 使用临时变量存储当前标签

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

#         # 生成带标签的分子图
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















import torch
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.metrics import pairwise_distances
import seaborn as sns
from torch_geometric.data import Data
from torch.serialization import add_safe_globals
from sklearn.decomposition import PCA  # 新增PCA用于特征降维
from matplotlib.colors import ListedColormap
import matplotlib
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

add_safe_globals([Data])

# 重置matplotlib配置
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['font.family'] = 'Times New Roman'


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
            Descriptors.TPSA(mol),              # 极性表面积
            Descriptors.MolLogP(mol),            # 亲脂性
            Descriptors.NumRotatableBonds(mol),  # 可旋转键
            Descriptors.MolWt(mol),             # 分子量
            Descriptors.NumHDonors(mol),        # 氢键供体数量
            Descriptors.NumHAcceptors(mol),     # 氢键受体数量
            rdMolDescriptors.CalcNumAtoms(mol), # 原子总数
            rdMolDescriptors.CalcNumHeteroatoms(mol) # 杂原子数
        ]

        # RDK指纹特征（摩根指纹）
        fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=2048)
        fp_array = np.zeros((2048,))
        DataStructs.ConvertToNumpyArray(fp, fp_array)

        # 合并特征
        combined_feat = np.concatenate([node_feat, edge_feat, topo_feat, fp_array])
        feature_list.append(combined_feat)

    return np.array(feature_list)

def dynamic_constraint_matrix(X, y, base_alpha=0.7):
    """动态约束矩阵生成"""
    # 计算标签覆盖率
    label_coverage = np.sum(y != -1) / len(y)

    # 动态调整参数，更灵活的alpha计算
    alpha = base_alpha + 0.8 * label_coverage  # 调整系数
    k = int(np.sqrt(len(X)))                   # 自适应近邻数

    # 构建相似度矩阵
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim_matrix = X_normalized @ X_normalized.T

    # 注入动态约束
    labeled_mask = (y != -1)
    for i in range(len(y)):
        for j in range(len(y)):
            if labeled_mask[i] and labeled_mask[j]:
                if y[i] == y[j]:
                    sim_matrix[i,j] = min(1.0, sim_matrix[i,j] + alpha)
                else:
                    sim_matrix[i,j] = max(0.0, sim_matrix[i,j] - alpha)

    # 自适应稀疏化
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    return sim_matrix * nbrs.kneighbors_graph(X).toarray()

def supervised_tsne(X, y, perplexity=30):
    """监督t-SNE实现（通过标签调整距离矩阵）"""
    # 创建标签相似性矩阵
    y_sim = np.array([y == y_i for y_i in y], dtype=float)

    # 调整距离矩阵
    distances = pairwise_distances(X)
    adjusted_dist = distances * (1 - y_sim) + distances * 0.4 * y_sim

    # 执行t-SNE
    tsne = TSNE(n_components=2,
                perplexity=perplexity,
                metric='precomputed',
                random_state=50,
                init='random')
    return tsne.fit_transform(adjusted_dist)

def supervised_umap(X, y):
    """监督UMAP实现"""
    return umap.UMAP(n_components=2,
                    n_neighbors=15,
                    min_dist=0.1,
                    target_metric='categorical',
                    target_weight=0.5,
                    random_state=42).fit_transform(X, y=y)

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
    plt.savefig(str(RESULTS_DIR / "SAR" / "Pomacea canaliculata" / f"{title_prefix.lower()}_{method}.svg"))
    plt.savefig(str(RESULTS_DIR / "SAR" / "Pomacea canaliculata" / f"{title_prefix.lower()}_{method}.png"), dpi=900)
    plt.show()

def main():
    file_path = str(RESULTS_DIR / "SAR" / "Pomacea canaliculata" / "gnn" / "pc-gnn.pt")

    try:
        # 尝试使用 weights_only=True 加载
        graph_data = torch.load(file_path, weights_only=True)
    except _pickle.UnpicklingError:
        print("使用 weights_only=True 加载失败，尝试使用 weights_only=False 加载...")
        # 若失败，使用 weights_only=False 重新加载
        graph_data = torch.load(file_path, weights_only=False)
    df = pd.read_excel(DATASET_DIR / "Pomacea canaliculata compounds analysis data.xlsx")

    # 增强特征提取
    X = enhanced_feature_extractor(graph_data, df['smiles'])

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA降维，减少特征维度，去除噪声
    pca = PCA(n_components=0.95)  # 保留95%方差
    X_pca = pca.fit_transform(X_scaled)

    y = np.array([data.y.item() for data in graph_data])

    # 构建动态约束矩阵
    sim_matrix = dynamic_constraint_matrix(X_pca, y)

    # 新增相似度矩阵可视化
    plt.figure(figsize=(10, 8))
    # 获取viridis颜色映射
    viridis_cmap = plt.get_cmap('viridis')
    # 创建一个新的颜色映射，调整透明度
    new_cmap = ListedColormap(viridis_cmap(np.linspace(0, 1, 256)) * [1, 1, 1, 0.5])
    sns.heatmap(sim_matrix[:20, :20], cmap=new_cmap)
    plt.title("Similarity Matrix Subset", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(str(RESULTS_DIR / "SAR" / "Pomacea canaliculata" / "sim_matrix_heatmap.svg"))
    plt.close()  # 防止内存泄漏

    # 优化谱聚类（调整参数）
    spectral = SpectralClustering(
        n_clusters=2,
        affinity='precomputed',
        assign_labels='discretize',
        random_state=50,
        n_init=100,
        gamma=0.1  # 调整相似性矩阵的影响
    )
    cluster_labels_spectral = spectral.fit_predict(sim_matrix)

    # 尝试KMeans聚类
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels_kmeans = kmeans.fit_predict(X_pca)

    # 综合评估 - 谱聚类
    labeled_idx = (y != -1)
    if sum(labeled_idx) > 0:
        ari_spectral = adjusted_rand_score(y[labeled_idx], cluster_labels_spectral[labeled_idx])
        sil_score_spectral = silhouette_score(X_pca, cluster_labels_spectral)
        print(f"谱聚类优化后评估指标：\nARI: {ari_spectral:.3f}\nSilhouette: {sil_score_spectral:.3f}")

        # 综合评估 - KMeans
        ari_kmeans = adjusted_rand_score(y[labeled_idx], cluster_labels_kmeans[labeled_idx])
        sil_score_kmeans = silhouette_score(X_pca, cluster_labels_kmeans)
        print(f"KMeans评估指标：\nARI: {ari_kmeans:.3f}\nSilhouette: {sil_score_kmeans:.3f}")

    # 可视化 - 谱聚类
    # visualize_supervised_clusters(X_pca, cluster_labels_spectral, y, method='tsne', title_prefix="Spectral ")
    visualize_supervised_clusters(X_pca, cluster_labels_spectral, y, method='umap', title_prefix="Spectral ")

    # # 可视化 - KMeans
    # visualize_supervised_clusters(X_pca, cluster_labels_kmeans, y, method='tsne', title_prefix="KMeans-")
    # visualize_supervised_clusters(X_pca, cluster_labels_kmeans, y, method='umap', title_prefix="KMeans-")

if __name__ == "__main__":
    import _pickle
    main()
