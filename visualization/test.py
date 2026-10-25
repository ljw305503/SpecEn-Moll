import numpy as np
import matplotlib.pyplot as plt
import os

# 密度比分布图归入 fsl/distribution（本脚本需在 visualization/ 目录下运行）
DISTRIBUTION_DIR = "../results/fsl/distribution"
os.makedirs(DISTRIBUTION_DIR, exist_ok=True)

# density_ratios: 你已经算好的 r(x)
density_ratios = np.load(os.path.join(DISTRIBUTION_DIR, "density_ratios.npy"))
# Q1 threshold
q1_threshold = np.percentile(density_ratios, 25)

# mask
mask_keep = density_ratios >= q1_threshold

plt.figure(figsize=(6, 4))

# Before filtering
plt.hist(
 density_ratios,
 bins=50,
 alpha=0.3,
 label='Before filtering'
)

# After filtering
plt.hist(
 density_ratios[mask_keep],
 bins=50,
 alpha=0.5,
 label='After filtering'
)

# Q1 vertical line
plt.axvline(
 q1_threshold,
 color='red',
 linestyle='--',
 linewidth=2,
 label='Q1 threshold'
)

plt.xlabel('Density ratio r(x)')
plt.ylabel('Number of proxy samples')
plt.legend()
plt.tight_layout()
plt.xlim(0, 22.5)
plt.savefig(
 os.path.join(DISTRIBUTION_DIR, "density_ratio_distribution.png"),
 dpi=300,
 bbox_inches='tight'
)
plt.savefig(
 os.path.join(DISTRIBUTION_DIR, "density_ratio_distribution.pdf"),
 bbox_inches='tight'
)

plt.show()