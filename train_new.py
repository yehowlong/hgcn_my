import os

# 自动修复环境变量
os.environ['DATAPATH'] = 'data'
os.environ['LOG_DIR'] = 'logs'

import datetime
import json
import logging
import pickle
import time
import numpy as np
import torch
from sklearn.decomposition import PCA  # 确保安装了 scikit-learn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

import optimizers
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics


def visualize_dual_views(model, best_emb, data, args):
    logging.info("Generating professional manifold visualization...")
    emb = best_emb.detach().cpu().numpy()

    # 1. 降维到 2D 空间 (用于在双曲面上展示)
    if emb.shape[1] > 2:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2).fit_transform(emb)

    # 2. 计算洛伦兹坐标
    x1, x2 = emb[:, 0], emb[:, 1]
    x0 = np.sqrt(1 + x1 ** 2 + x2 ** 2)

    # 3. 模拟异常值检测 (基于双曲半径)
    # 论文逻辑：异常值往往靠近原点 (x0 较小，x1,x2 接近 0)
    # 我们将半径最小的 5% 定义为“异常值”
    dist_from_origin = x0
    threshold = np.percentile(dist_from_origin, 5)
    is_anomaly = dist_from_origin < threshold

    fig = plt.figure(figsize=(18, 9))

    # --- 左图：完整流形侧视图 ---
    ax1 = fig.add_subplot(121, projection='3d')
    # 绘制更广阔的流形碗
    r_max = 3.0
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(1, r_max, 100)
    U, V = np.meshgrid(u, v)
    X_m = np.sqrt(V ** 2 - 1) * np.cos(U)
    Y_m = np.sqrt(V ** 2 - 1) * np.sin(U)
    Z_m = V
    ax1.plot_surface(X_m, Y_m, Z_m, color='cyan', alpha=0.05, linewidth=0.1, edgecolors='gray')

    # 绘制节点：正常值(蓝色)，异常值(红色五角星)
    ax1.scatter(x1[~is_anomaly], x2[~is_anomaly], x0[~is_anomaly],
                c='blue', s=20, alpha=0.6, label='Normal')
    ax1.scatter(x1[is_anomaly], x2[is_anomaly], x0[is_anomaly],
                c='red', marker='*', s=100, label='Anomaly', edgecolors='black')

    ax1.set_title("Hyperboloid Manifold (Anomaly Highlighted)")
    ax1.view_init(elev=20, azim=45)
    ax1.legend()

    # --- 右图：俯视图 (Poincare Disk) ---
    ax2 = fig.add_subplot(122)
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=2)
    ax2.add_artist(circle)

    # 投影到圆盘
    px, py = x1 / (1 + x0), x2 / (1 + x0)
    ax2.scatter(px[~is_anomaly], py[~is_anomaly], c='blue', s=20, alpha=0.5)
    ax2.scatter(px[is_anomaly], py[is_anomaly], c='red', marker='*', s=100, edgecolors='black')

    ax2.set_xlim(-1.1, 1.1);
    ax2.set_ylim(-1.1, 1.1);
    ax2.set_aspect('equal')
    ax2.set_title("Poincare Disk Projection")
    plt.show()


def train(args):
    np.random.seed(args.seed);
    torch.manual_seed(args.seed)

    # 修复 GPU 初始化逻辑
    if int(args.cuda) >= 0 and torch.cuda.is_available():
        args.device = f'cuda:{args.cuda}'
        torch.cuda.manual_seed(args.seed)
    else:
        args.device = 'cpu'

    logging.info(f'Using device: {args.device}')

    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape

    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        Model = LPModel

    model = Model(args).to(args.device)
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)

    for x in data:
        if torch.is_tensor(data[x]): data[x] = data[x].to(args.device)

    best_val_metrics = model.init_metric_dict();
    best_test_metrics = None;
    best_emb = None;
    counter = 0
    pbar = tqdm(range(args.epochs), desc="Training Progress")

    for epoch in pbar:
        model.train();
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pbar.set_postfix({'loss': f"{train_metrics['loss'].item():.4f}", 'roc': f"{train_metrics['roc']:.2f}"})

        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                embeddings = model.encode(data['features'], data['adj_train_norm'])
                val_metrics = model.compute_metrics(embeddings, data, 'val')
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu();
                best_val_metrics = val_metrics;
                counter = 0
            else:
                counter += 1
                if counter == args.patience: break

    pbar.close()
    logging.info("Optimization Finished!")
    visualize_dual_views(model, best_emb, data, args)


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)