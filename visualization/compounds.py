import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 化合物空间分布图归入 fsl/distribution
DISTRIBUTION_DIR = "../results/fsl/distribution"
os.makedirs(DISTRIBUTION_DIR, exist_ok=True)

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 读取 CSV 文件
data = pd.read_csv('../dataset/train/fsl-1/Pomacea canaliculata compounds analysis data-1.csv')
# 获取所需数据
x = data['HBA']
y = data['MW']
z = data['logP']
# 获取 3 个指标的最大值和最小值
x_min, x_max = min(x), max(x)
y_min, y_max = min(y), 1760
z_min, z_max = min(z), max(z)
# 创建 3D 图形
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=35, azim=20)
# 绘制化合物分子对应的点
ax.scatter(x, y, z, c='purple', marker='o', label='Compound Molecules')
# 定义空间体的 8 个顶点
vertices = [
[x_min, y_min, z_min],
[x_max, y_min, z_min],
[x_max, y_max, z_min],
[x_min, y_max, z_min],
[x_min, y_min, z_max],
[x_max, y_min, z_max],
[x_max, y_max, z_max],
[x_min, y_max, z_max]
]
# 定义空间体的 6 个面
faces = [
[vertices[0], vertices[1], vertices[2], vertices[3]],
[vertices[4], vertices[5], vertices[6], vertices[7]],
[vertices[0], vertices[1], vertices[5], vertices[4]],
[vertices[2], vertices[3], vertices[7], vertices[6]],
[vertices[0], vertices[3], vertices[7], vertices[4]],
[vertices[1], vertices[2], vertices[6], vertices[5]]
]
# 创建 Poly3DCollection 对象并添加到 3D 图形中
poly3d = Poly3DCollection(faces, alpha=0.2, facecolors='lightskyblue')
ax.add_collection3d(poly3d)
# 设置坐标轴标签和标题
ax.set_xlabel('HBA',labelpad=20, fontsize=14, weight='bold')
ax.set_ylabel('MW', labelpad=20, fontsize=14, weight='bold')
ax.set_zlabel('logP', labelpad=10, fontsize=14, weight='bold')
ax.set_title('3D Visualization of Compounds', fontsize=14, weight='bold')
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='z', labelsize=12)
ax.legend()
# 固定坐标轴范围
ax.set_xlim(40, -5)
ax.set_ylim(0, 2000)
ax.set_zlim(-5, 30)
# 设置 3D 坐标轴各平面（pane）的背景为白色
ax.xaxis.set_pane_color((1.0, 1.0, 1.0))# X轴平面
ax.yaxis.set_pane_color((1.0, 1.0, 1.0))# Y轴平面
ax.zaxis.set_pane_color((1.0, 1.0, 1.0))# Z轴平面
# 显示图形
plt.savefig(os.path.join(DISTRIBUTION_DIR, "compound_3d.svg"))
plt.show()