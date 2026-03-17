import os

os.environ['DATAPATH'] = 'data'
os.environ['LOG_DIR'] = 'logs'

import logging
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

import optimizers
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data


def visualize_dual_views(model, best_emb, data, args, final_roc):
    """
    极致对齐方案：
    1. 侧视图碗的高度 = 节点最大高度 (v_max = max(x0))
    2. 俯视图虚线圆 = 碗口投影半径 (r = sqrt(v_max-1)/sqrt(v_max+1))
    3. 正常点：Navy, 异常点：更小的红叉
    """
    logging.info("Generating paper-style aligned visualization...")
    emb = best_emb.detach().cpu().numpy()

    if emb.shape[1] > 2:
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(emb)
    else:
        emb_2d = emb

    x1, x2 = emb_2d[:, 0], emb_2d[:, 1]
    x0 = np.sqrt(1 + x1 ** 2 + x2 ** 2)

    # 径向距离得分
    radial_dist = np.arccosh(np.clip(x0, 1.0, None))
    threshold = np.percentile(radial_dist, 5)
    is_anomaly = radial_dist <= threshold

    fig = plt.figure(figsize=(20, 10), facecolor='white')

    # --- 左图：3D 侧视图 ---
    ax1 = fig.add_subplot(121, projection='3d')

    # 动态对齐：碗的高度刚好包裹住最远的点
    v_max = np.max(x0)
    v_grid = np.linspace(1, v_max, 60)
    u_grid = np.linspace(0, 2 * np.pi, 60)
    U, V = np.meshgrid(u_grid, v_grid)
    Xm, Ym, Zm = np.sqrt(V ** 2 - 1) * np.cos(U), np.sqrt(V ** 2 - 1) * np.sin(U), V

    # 绘制灰色碗
    ax1.plot_wireframe(Xm, Ym, Zm, color='lightgrey', rstride=5, cstride=5, alpha=0.3, linewidth=0.5)
    ax1.plot_surface(Xm, Ym, Zm, color='lightgrey', alpha=0.1, shade=False)

    # 绘制节点：Navy + 极小红叉
    ax1.scatter(x1[~is_anomaly], x2[~is_anomaly], x0[~is_anomaly],
                c='navy', s=15, alpha=0.6, label='Normal Node')
    ax1.scatter(x1[is_anomaly], x2[is_anomaly], x0[is_anomaly],
                c='red', marker='x', s=20, linewidths=0.5, label='Anomaly')

    # 强制比例
    ax1.set_box_aspect((1, 1, 0.8))
    xy_limit = np.sqrt(v_max ** 2 - 1)
    ax1.set_xlim(-xy_limit, xy_limit)
    ax1.set_ylim(-xy_limit, xy_limit)
    ax1.set_zlim(1, v_max)
    ax1.set_title("Hyperboloid Side View", fontsize=14, fontweight='bold')
    ax1.axis('off')

    # --- 右图：俯视图 ---
    ax2 = fig.add_subplot(122)
    # 严格数学对齐：虚线圆半径 = 碗口投影半径
    r_boundary = np.sqrt(v_max - 1) / np.sqrt(v_max + 1)

    # 绘制背景虚线圆（此时虚线圆会紧贴着最外层的点）
    circle = plt.Circle((0, 0), r_boundary, color='lightgrey', fill=False, linewidth=1.5, linestyle='--')
    ax2.add_artist(circle)

    # 数据投影
    px, py = x1 / (1 + x0), x2 / (1 + x0)
    ax2.scatter(px[~is_anomaly], py[~is_anomaly], c='navy', s=20, alpha=0.5, edgecolors='none')
    ax2.scatter(px[is_anomaly], py[is_anomaly], c='red', marker='x', s=25, linewidths=0.5)

    # 这里的显示范围强制锁定在虚线圆附近
    ax2.set_xlim(-r_boundary * 1.05, r_boundary * 1.05)
    ax2.set_ylim(-r_boundary * 1.05, r_boundary * 1.05)
    ax2.set_aspect('equal')
    ax2.set_title("Poincaré Disk Top View", fontsize=14, fontweight='bold')

    # 标注 ROC
    ax2.text(r_boundary * 0.2, -r_boundary * 0.95, f'Test ROC: {final_roc:.4f}',
             fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
    ax2.axis('off')

    plt.suptitle(f"Anomaly Detection on {args.dataset.upper()}", fontsize=18, y=0.95)
    plt.savefig("vis_anomaly_paper_aligned.png", dpi=300, bbox_inches='tight')
    plt.show()


def train(args):
    # 这里的 train 逻辑保持之前的 Early Stopping 机制...
    np.random.seed(args.seed);
    torch.manual_seed(args.seed)
    args.device = f'cuda:{args.cuda}' if int(args.cuda) >= 0 and torch.cuda.is_available() else 'cpu'

    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    Model = NCModel if args.task == 'nc' else LPModel
    if args.task != 'nc':
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])

    model = Model(args).to(args.device)
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)

    for x in data:
        if torch.is_tensor(data[x]): data[x] = data[x].to(args.device)

    best_val_metrics = model.init_metric_dict();
    best_test_metrics = None;
    best_emb = None
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
                # 仅当验证集提升时，记录测试集性能
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu();
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter >= args.patience: break

    pbar.close()
    final_roc = best_test_metrics['roc'] if best_test_metrics else 0.0
    visualize_dual_views(model, best_emb, data, args, final_roc)


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)