from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 读取 CSV 文件
df = pd.read_csv(r"D:\BS222\FSL\sjk\sj\co\all3.csv",encoding="utf-8")
# df["Activity Classification 1"] = df["Activity Classification 1"].replace({0: "00", 1: "01"})
# 目标列：Activity Classification 1
target_col = "lable"

# 选取数值列（排除非数值列）
numeric_cols = df.select_dtypes(include=['number']).columns

# 排除不需要归一化的列
exclude_cols = ['lable', 'Number']

# 进一步排除二进制列（只包含 0 和 1 的列）
binary_cols = [col for col in numeric_cols if set(df[col].dropna().unique()).issubset({0, 1})]

# 需要归一化的列 = 所有数值列 - 需要排除的列
normalize_cols = [col for col in numeric_cols if col not in exclude_cols + binary_cols]

# 初始化 MinMaxScaler
scaler = MinMaxScaler()

# 对选定的列进行归一化
df[normalize_cols] = scaler.fit_transform(df[normalize_cols])

print(df.head())  # 查看归一化后的数据


# 计算所有列与目标列的 Spearman 相关系数
correlations = {}
for col in df.columns:
    if col != target_col:  # 排除自身计算
        corr, p_value = spearmanr(df[col], df[target_col])
        correlations[col] = (corr, p_value)

# 转换为 DataFrame 方便查看
corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=["Spearman Corr", "p-value"])

# 按相关系数排序
corr_df = corr_df.sort_values(by="Spearman Corr", ascending=False)

# # 打印结果
# # print(corr_df)
# sns.pairplot(df, hue="Activity Classification 1", vars=normalize_cols, diag_kind="kde")
# plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# 仅显示最相关的前 20 个特征（去掉不需要的列）
top_n = 20
corr_df_top = corr_df.drop(index=["Activity Classification 2", "10mg"], errors="ignore").head(top_n)

fig, ax1 = plt.subplots(figsize=(12, 8))

# 颜色映射
colors = sns.color_palette("Spectral", len(corr_df_top))

# 绘制 Spearman 相关系数的条形图
ax1.barh(corr_df_top.index, corr_df_top["Spearman Corr"], color=colors, alpha=0.7)
ax1.set_xlabel("Spearman Correlation", fontsize=14)
ax1.set_ylabel("Features", fontsize=14)
ax1.set_title(f"Top {top_n} Spearman Correlations with lable", fontsize=16)
ax1.invert_yaxis()  # 让相关性最高的特征排在最上面

# 添加数值标签（相关系数）
for i, v in enumerate(corr_df_top["Spearman Corr"]):
    ax1.text(v, i, f"{v:.2f}", ha="left", va="center", fontsize=12)

# 创建第二个 y 轴用于 p-value
ax2 = ax1.twiny()
ax2.scatter(corr_df_top["p-value"], corr_df_top.index, color='black', label="p-value", marker='o', s=50)
ax2.set_xlabel("p-value (log scale)", fontsize=14)
ax2.set_xscale("log")  # 使用对数刻度，避免 p 值过大时挤在一起

# 添加图例
ax2.legend(loc="lower right")

# 保存图片
plt.savefig(r"D:\BS222\FSL\sjk\sj\co\spearman_correlation_with_pvalue.png", dpi=800, bbox_inches="tight")
plt.savefig(r"D:\BS222\FSL\sjk\sj\co\spearman_correlation_with_pvalue.svg")
# 显示图像
plt.show()

# 提取最正相关的前5个特征
positive_corr_df = corr_df[corr_df['Spearman Corr'] < 0]
top_positive_features = positive_corr_df.index[:5]

if len(top_positive_features) == 0:
    print("没有正相关特征，无法绘制")
else:
    plt.figure(figsize=(15, 6))
    for i, feature in enumerate(top_positive_features):
        plt.subplot(1, len(top_positive_features), i+1)
        # 整理数据格式
        temp_df = df[[feature, 'lable']].melt(id_vars='lable', value_vars=feature)
        temp_df.rename(columns={'value': feature}, inplace=True)
        # 绘制雨云图（小提琴图+散点+箱线图组合）
        sns.violinplot(
            data=temp_df, x='lable', y=feature, hue='lable',
            palette={0: '#7DF9FF', 1: '#F8C8DC'},  # 修改这里的键为数值类型
            split=True, inner=None
        )
        sns.swarmplot(
            data=temp_df, x='lable', y=feature, hue='lable',
            palette={0: 'blue', 1: 'red'},  # 同步修改这里
            dodge=True, size=3
        )
        sns.boxplot(
            data=temp_df, x='lable', y=feature, hue='lable',
            palette={0: '#7DF9FF', 1: '#F8C8DC'},  # 同步修改这里
            width=0.2, notch=True
        )
        plt.title(feature, fontsize=14)
        plt.xlabel('lable', fontsize=12)
        plt.ylabel(feature, fontsize=12)
        plt.legend([], frameon=False)  # 隐藏重复图例
    plt.tight_layout()
    # plt.savefig(r"D:\BS222\FSL\sjk\sj\co\raincloud_plots1.png", dpi=600, bbox_inches="tight")
    # plt.savefig(r"D:\BS222\FSL\sjk\sj\co\raincloud_plots1.svg", bbox_inches="tight")
    # plt.show()

    plt.savefig(r"D:\BS222\FSL\sjk\sj\co\raincloud_plots2.png", dpi=600, bbox_inches="tight")
    plt.savefig(r"D:\BS222\FSL\sjk\sj\co\raincloud_plots2.svg", bbox_inches="tight")
    plt.show()