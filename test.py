import numpy as np
import matplotlib.pyplot as plt


def plot_poincare_disk(embeddings, labels, title="Hyperbolic Anomaly Detection Visualization"):
    """
    将双曲嵌入可视化在二维庞加莱圆盘中
    """
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # 1. 绘制庞加莱圆盘的边界（单位圆）
    circle = plt.Circle((0, 0), 1.0, color='navy', fill=False, linewidth=2, linestyle='--')
    ax.add_artist(circle)

    # 2. 提取不同类别的点
    # 假设 labels 中 0 为正常，1 为异常
    normal_pts = embeddings[labels == 0]
    anomaly_pts = embeddings[labels == 1]

    # 3. 绘制散点
    # 正常节点分布在边缘（蓝色）
    plt.scatter(normal_pts[:, 0], normal_pts[:, 1], c='#3498db', s=20, alpha=0.5, label='Normal Nodes',
                edgecolors='none')

    # 异常节点向中心聚集（红色）
    plt.scatter(anomaly_pts[:, 0], anomaly_pts[:, 1], c='#e74c3c', s=60, alpha=0.9, label='Anomaly Nodes', marker='*',
                edgecolors='white')

    # 4. 图形修饰
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    plt.legend(loc='upper right', fontsize=12)
    plt.title(title, fontsize=15)
    plt.grid(True, which='both', linestyle=':', alpha=0.3)

    # 隐藏坐标轴，使其更像论文插图
    plt.axis('off')

    plt.savefig('hyperbolic_visualization.png', dpi=300, bbox_inches='tight')
    print("图像已保存为 hyperbolic_visualization.png")
    plt.show()


# --- 模拟数据测试（你可以替换为你模型输出的真实数据） ---
if __name__ == "__main__":
    num_nodes = 500
    # 模拟正常节点：离中心较远 (r > 0.7)
    r_normal = 0.7 + 0.28 * np.random.rand(400)
    theta_normal = 2 * np.pi * np.random.rand(400)
    x_normal = r_normal * np.cos(theta_normal)
    y_normal = r_normal * np.sin(theta_normal)

    # 模拟异常节点：离中心较近 (r < 0.4)
    r_anomaly = 0.35 * np.random.rand(100)
    theta_anomaly = 2 * np.pi * np.random.rand(100)
    x_anomaly = r_anomaly * np.cos(theta_anomaly)
    y_anomaly = r_anomaly * np.sin(theta_anomaly)

    embeddings = np.vstack([np.column_stack([x_normal, y_normal]), np.column_stack([x_anomaly, y_anomaly])])
    labels = np.array([0] * 400 + [1] * 100)

    plot_poincare_disk(embeddings, labels)