import matplotlib.pyplot as plt
import os
import numpy as np

def plot1(save_dir):
    # 5. 合并可视化
    plt.figure(1)
    plt.title("Comparison of ROC Curves", fontsize=14, weight='bold')
    plt.xticks(fontsize=13,fontweight='bold')
    plt.yticks(fontsize=13,fontweight='bold')
    plt.legend(loc="lower right",prop={'size':13,'weight':'bold'})
    plt.savefig(os.path.join(save_dir, "roc_combined.png"))
    plt.savefig(os.path.join(save_dir, "roc_combined.svg"))
    plt.show()

# # 未被调用：文件名已由 plot4 的 file_name 参数化取代（train.py:177 传入 'accuracy_comparison'）
# def plot2(save_dir,results):
#     plt.figure(figsize=(10, 6))
#
#     base_colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33F6', '#33FFF6']
#     colors = [(tuple(int(c[1:][i:i + 2], 16) / 255 for i in (0, 2, 4)) + (0.5,)) for c in
#               base_colors]  # RGBA with 60% opacity
#     bars = plt.bar(results.keys(), results.values(), color=colors, edgecolor='black', linewidth=0.8)  # 保存bar对象
#     plt.title("Model Accuracy Comparison", fontsize=18, weight='bold')
#     plt.ylabel("Accuracy", fontsize=18, weight='bold')
#     plt.xlabel("Model", fontsize=18, weight='bold')
#     plt.xticks(fontsize=15,fontweight='bold')
#     plt.yticks(fontsize=15,fontweight='bold')
#     # 设置y轴范围
#     plt.ylim(0, 1.1)  # 稍微增加y轴上限，为标签留出空间
#
#     # 在柱子上方添加数值标签
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2., height,
#                  f'{height:.2f}',  # 显示两位小数
#                  ha='center', va='bottom',  # 水平居中，垂直底部对齐
#                  fontsize=15,fontweight='bold')
#     plt.savefig(os.path.join(save_dir, "accuracy_comparison.png"))
#     plt.savefig(os.path.join(save_dir, "accuracy_comparison.svg"))
#     plt.show()

# # 未被调用：文件名已由 plot4 的 file_name 参数化取代（train.py:184 传入 'F1_Score_comparison'）
# def plot3(save_dir,results):
#     plt.figure(figsize=(10, 6))
#
#     base_colors = ['#1F77B4', '#FF7F0E', '#E377C2', '#D62728', '#17BECF']
#     colors = [(tuple(int(c[1:][i:i + 2], 16) / 255 for i in (0, 2, 4)) + (0.5,)) for c in
#               base_colors]  # RGBA with 60% opacity
#     bars = plt.bar(results.keys(), results.values(), color=colors, edgecolor='black', linewidth=0.8)  # 保存bar对象
#     plt.title("Model F1-Score Comparison", fontsize=18, weight='bold')
#     plt.ylabel("F1-Score", fontsize=18, weight='bold')
#     plt.xlabel("Model", fontsize=18, weight='bold')
#     plt.xticks(fontsize=15,fontweight='bold')
#     plt.yticks(fontsize=15,fontweight='bold')
#     # 设置y轴范围
#     plt.ylim(0, 1.1)  # 稍微增加y轴上限，为标签留出空间
#
#     # 在柱子上方添加数值标签
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2., height,
#                  f'{height:.2f}',  # 显示两位小数
#                  ha='center', va='bottom',  # 水平居中，垂直底部对齐
#                  fontsize=15,fontweight='bold')
#     plt.savefig(os.path.join(save_dir, "F1_Score_comparison.png"))
#     plt.savefig(os.path.join(save_dir, "F1_Score_comparison.svg"))
#     plt.show()


def plot4(save_dir,results,base_colors,title,y_label,x_label,file_name):
    plt.figure(figsize=(10, 6))

    # base_colors = ['#1F77B4', '#FF7F0E', '#E377C2', '#D62728', '#17BECF']
    colors = [(tuple(int(c[1:][i:i + 2], 16) / 255 for i in (0, 2, 4)) + (0.5,)) for c in
              base_colors]  # RGBA with 60% opacity
    bars = plt.bar(results.keys(), results.values(), color=colors, edgecolor='black', linewidth=0.8)  # 保存bar对象
    plt.title(title, fontsize=18, weight='bold')
    plt.ylabel(y_label, fontsize=18, weight='bold')
    plt.xlabel(x_label, fontsize=18, weight='bold')
    plt.xticks(fontsize=15,fontweight='bold')
    plt.yticks(fontsize=15,fontweight='bold')
    # 设置y轴范围
    plt.ylim(0, 1.1)  # 稍微增加y轴上限，为标签留出空间

    # 在柱子上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',  # 显示两位小数
                 ha='center', va='bottom',  # 水平居中，垂直底部对齐
                 fontsize=15,fontweight='bold')
    plt.savefig(os.path.join(save_dir, file_name+".png"))
    plt.savefig(os.path.join(save_dir, file_name+".svg"))
    plt.show()

# # 未被调用：train.py:215-216 的调用点整段被注释；功能已被 plot5 完整取代且更完善
# def plot_radar(fig, ax,color,labels,data,title):
#     # 2. 计算角度：将圆周等分给每个维度
#     num_vars = len(labels)
#     # 计算每个维度的角度，并确保图形闭合[citation:1]
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#     angles += angles[:1]  # 闭合图形
#     data_closed = np.concatenate((data, [data[0]]))  # 将数据的首尾点连接起来[citation:2]
#
#     # 绘制填充多边形和边界线
#     ax.fill(angles, data_closed,color=color, alpha=0.2)  # 填充区域
#     ax.plot(angles, data_closed, color=color,linewidth=2)  # 边界线
#
#     # 4. 设置标签和标题
#     # 设置每个维度的标签，并调整其位置使其更清晰
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
#     ax.set_yticklabels([])  # 隐藏默认的径向刻度标签[citation:1]
#     ax.set_title(title, fontsize=16, fontweight='bold', va='bottom')
#
#     # 5. 在每个数据点上添加数值标注[citation:2]
#     # for i in range(num_vars):
#     #     angle_rad = angles[i]
#     #     value = data[i]
#     #     # 将数值标签放在对应角度的外侧
#     #     ax.text(angle_rad, value + 3, str(value),
#     #             ha='center', va='center',
#     #             fontsize=11, fontweight='bold',
#     #             color='darkred',
#     #             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
#
#     # 6. 调整雷达图显示[citation:2]
#     # 将起始角度设置在顶部 (90度)
#     ax.set_theta_offset(np.pi / 2)
#     # 将角度方向设置为顺时针
#     ax.set_theta_direction(-1)
#
#     # 7. 设置径向刻度网格，便于读数
#     ax.set_ylim(0, 1)
#     ax.set_yticks([0, 0.2, 0.5, 0.8, 1])
#     ax.grid(True, linestyle='--', linewidth=2, alpha=1)

def plot5():
    # SI Table S3 nested-CV means (combined features, n=5). RF train F1 SD in Word is 0.10 (typo); use 0.01.
    data = {'RF': {'train': [1.00, 0.98, 1.00, 1.00, 1.00],
                   'test': [0.81, 0.72, 0.86, 0.91, 0.72]},
            'SVM': {'train': [0.99, 0.98, 0.98, 0.99, 0.98],
                    'test': [0.83, 0.81, 0.83, 0.87, 0.78]},
            'XGB': {'train': [0.98, 0.98, 0.99, 0.99, 0.97],
                    'test': [0.91, 0.89, 0.95, 0.96, 0.91]},
            'LR': {'train': [1.00, 0.99, 1.00, 1.00, 0.99],
                   'test': [0.86, 0.79, 0.85, 0.89, 0.81]},
            'KNN': {'train': [0.85, 0.83, 0.84, 0.87, 0.82],
                    'test': [0.80, 0.73, 0.75, 0.82, 0.69]},
            'VOTING': {'train': [0.97, 1.00, 0.98, 0.99, 0.96],
                    'test': [0.87, 0.81, 0.90, 0.93, 0.79]},
            'SpecEn-Moll': {'train': [0.99, 0.98, 1.00, 0.99, 0.98],
                    'test': [0.89, 0.84, 0.90, 0.93, 0.84]}
            }

    # 增强对比度的配色方案
    enhanced_colors = {
        'RF': {'train': '#7E1FA6', 'test': '#2C004D'},  # 红色系：亮红 vs 深红
        'SVM': {'train': '#FB8C00', 'test': '#E65100'},  # 橙色系：亮橙 vs 深橙
        'XGB': {'train': '#43A047', 'test': '#1B5E20'},  # 绿色系：亮绿 vs 深绿
        'LR': {'train': '#1E88E5', 'test': '#0D47A1'},  # 蓝色系：亮蓝 vs 深蓝
        'KNN': {'train': '#8E24AA', 'test': '#4A148C'} , # 紫色系：亮紫 vs 深紫
        'VOTING': {'train': '#FFD600', 'test': '#F57F17'} , # 黄色系：亮紫 vs 深紫
        'SpecEn-Moll': {'train': '#FF5252', 'test': '#B71C1C'}
    }
    # 基模型雷达图归 results/base_models/<MODEL>/，聚合方法归 results/ensemble/<方法>/
    result_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    _route = {m: os.path.join(result_root, "base_models", m)
              for m in data if m not in ("VOTING", "SpecEn-Moll")}
    _route["VOTING"] = os.path.join(result_root, "ensemble", "voting")
    _route["SpecEn-Moll"] = os.path.join(result_root, "ensemble", "SpecEn-Moll")
    labels = ['Acc', 'F1', 'Precision', 'Specificity', 'Recall']
    num_vars = len(labels)

    # 每个模型单独绘制一个图
    for model in data.keys():
        # 创建图形，每个模型一个单独的图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)

        # 准备数据
        train_data = np.array(data[model]['train'])
        test_data = np.array(data[model]['test'])

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        # 闭合数据
        train_data_closed = np.concatenate((train_data, [train_data[0]]))
        test_data_closed = np.concatenate((test_data, [test_data[0]]))

        # 获取颜色
        train_color = enhanced_colors[model]['train']
        test_color = enhanced_colors[model]['test']

        # 绘制填充区域（调整透明度，避免过度重叠）
        ax.fill(angles, train_data_closed, color=train_color, alpha=0.2, label='Train Set')
        ax.fill(angles, test_data_closed, color=test_color, alpha=0.2, label='Test Set')

        # 绘制边界线
        ax.plot(angles, train_data_closed, color=train_color, linewidth=2.5,
                linestyle='-', marker='o', markersize=10, markeredgecolor='white', markeredgewidth=1.5)

        ax.plot(angles, test_data_closed, color=test_color, linewidth=2.5,
                linestyle='--', marker='s', markersize=10, markeredgecolor='white', markeredgewidth=1.5)

        # 优化标签显示：避免重叠
        # 方法1：将训练集和测试集的标签合并显示在一个文本框内
        for i in range(num_vars):
            angle_rad = angles[i]
            train_value = train_data[i]
            test_value = test_data[i]

            # 计算差值，决定标签放置位置
            diff = train_value - test_value

            # 如果两者非常接近，合并显示
            if abs(diff) < 0.05:  # 差值小于0.05时合并显示
                # 在数据点附近显示合并标签
                label_text = f'T:{train_value:.3f}\nV:{test_value:.3f}'
                ax.text(angle_rad, (train_value + test_value) / 2, label_text,
                        ha='center', va='center', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor='gray', alpha=0.8))
            else:
                # 如果差值较大，分开显示但错开角度
                # 训练集标签放在稍偏左的位置
                ax.text(angle_rad - 0.08, train_value, f'{train_value:.3f}',
                        ha='center', va='center' if train_value > test_value else 'bottom',
                        fontsize=9, fontweight='bold', color=train_color,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor=train_color, alpha=0.8))

                # 测试集标签放在稍偏右的位置
                ax.text(angle_rad + 0.08, test_value, f'{test_value:.3f}',
                        ha='center', va='center' if test_value > train_value else 'top',
                        fontsize=9, fontweight='bold', color=test_color,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor=test_color, alpha=0.8))

        # 设置图形属性
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
        # 调整径向范围，给标签留出空间
        ax.set_ylim(0.5, 1.08)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=10)
        ax.grid(True, linestyle='--', linewidth=2, alpha=1)

        # 设置标题
        ax.set_title(f'{model} Model Performance\nnested cross-validation (mean, n=5)',
                     fontsize=16, fontweight='bold')

        # 添加图例（放在图形外部右侧）
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=15,prop={'weight':'bold'},
                  framealpha=0.9, edgecolor='gray')

        # 调整雷达图方向
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        save_dir = _route[model]
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "radar.png"))
        plt.savefig(os.path.join(save_dir, "radar.svg"))

        # 显示图形
        plt.show()

    print("模型平均性能对比:")
    print("=" * 50)
    for model in data:
        train_avg = np.mean(data[model]['train'])
        test_avg = np.mean(data[model]['test'])
        print(
            f"{model:4s} | 训练集平均: {train_avg:.4f} | 测试集平均: {test_avg:.4f} | 差异: {train_avg - test_avg:.4f}")



if __name__=='__main__':
    plot5()

