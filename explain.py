import numpy as np
import matplotlib.pyplot as plt

# ================== 字体 ==================
plt.rcParams['font.sans-serif'] = ['Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

# ================== 数据 ==================
datasets = ["Cora","PubMed","Disease","Weibo","Reddit","Disney","Books","Enron"]

models = ["DeepWalk","Node2vec","GCN","GAT","GraphSAGE","DGI",
          "SimP-GCN","DOMINANT","ALARM","CoLA","Ours"]

roc = np.array([
[68.2,65.4,59.8,56.5,57.2,47.9,36.5,46.4],
[69.5,66.7,61.3,58.2,58.5,49.2,38.8,48.1],
[72.3,70.2,64.6,75.2,55.4,48.0,45.2,65.4],
[74.5,72.8,66.5,77.1,56.5,49.5,48.1,67.8],
[73.6,71.5,65.2,76.8,56.1,48.8,46.5,66.2],
[78.4,76.2,69.5,81.5,58.8,52.1,49.8,71.2],
[80.5,78.8,71.8,83.2,59.5,55.2,52.4,74.5],
[81.4,80.5,72.1,85.0,56.0,47.1,50.1,73.1],
[84.5,83.1,74.8,86.4,60.4,68.6,62.9,69.2],
[86.5,85.2,76.8,89.3,59.2,58.4,53.1,76.4],
[89.5,91.2,93.8,87.8,62.8,65.2,70.5,84.8]
])

ap = np.array([
[69.4,66.5,60.5,58.1,58.4,48.6,37.4,47.8],
[71.2,68.1,62.4,59.8,59.6,50.1,39.5,49.5],
[74.2,72.1,65.8,76.8,56.5,49.5,46.5,66.8],
[76.8,74.5,68.2,78.5,58.1,51.2,49.4,69.5],
[75.5,73.2,67.1,78.1,57.5,50.4,48.2,68.1],
[80.5,78.4,71.2,83.2,60.1,53.8,51.5,73.4],
[82.1,80.5,73.5,85.5,61.2,56.8,53.8,76.2],
[83.5,82.2,74.5,86.8,57.8,49.2,52.4,75.4],
[86.2,84.8,76.8,88.5,62.1,71.2,65.4,71.5],
[88.5,87.2,78.5,91.3,61.4,60.5,55.4,78.8],
[91.5,93.2,95.8,89.8,64.5,68.5,72.5,86.5]
])

# ================== 配色 ==================
colors = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#2F5D8C"
]

x = np.arange(len(datasets))
width = 0.07

# ================== 图1：ROC-AUC ==================
plt.figure(figsize=(10, 6))

for i in range(len(models)):
    plt.bar(
        x + i * width,
        roc[i],
        width,
        color=colors[i],
        edgecolor='black',
        linewidth=0.3
    )

plt.ylabel("ROC-AUC (%)")
plt.ylim(30, 100)
plt.xticks(x + width * len(models) / 2, datasets)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# 图例
handles = [plt.Rectangle((0,0),1,1,color=colors[i]) for i in range(len(models))]
plt.legend(handles, models, loc='upper center', ncol=6, frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig("roc_auc_comparison.png", dpi=300)  # ⭐ 保存
plt.show()


# ================== 图2：AP ==================
plt.figure(figsize=(10, 6))

for i in range(len(models)):
    plt.bar(
        x + i * width,
        ap[i],
        width,
        color=colors[i],
        edgecolor='black',
        linewidth=0.3
    )

plt.ylabel("AP (%)")
plt.ylim(30, 100)
plt.xticks(x + width * len(models) / 2, datasets)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# 图例
plt.legend(handles, models, loc='upper center', ncol=6, frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig("ap_comparison.png", dpi=300)  # ⭐ 保存
plt.show()