import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import multiprocessing as mp

os.environ['DATAPATH'] = 'data'
os.environ['LOG_DIR'] = 'logs'

import optimizers
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data

def visualize_dual_views(model, best_emb, data, args, final_metric, metric_name="Metric"):
    logging.info(f"Generating Lorentz Orthographic visualization for {args.dataset}...")
    emb = best_emb.numpy()

    if emb.shape[1] > 2:
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(emb)
    else:
        emb_2d = emb

    x1, x2 = emb_2d[:, 0], emb_2d[:, 1]
    x0 = np.sqrt(1 + x1 ** 2 + x2 ** 2)

    radial_dist = np.arccosh(np.clip(x0, 1.0, None))
    threshold = np.percentile(radial_dist, 5)
    is_anomaly = radial_dist <= threshold

    fig = plt.figure(figsize=(20, 10), facecolor='white')

    # --- 左图：Lorentz 3D ---
    ax1 = fig.add_subplot(121, projection='3d')
    v_max = np.max(x0)
    v_grid = np.linspace(1, v_max, 60)
    u_grid = np.linspace(0, 2 * np.pi, 60)
    U, V = np.meshgrid(u_grid, v_grid)
    Xm, Ym, Zm = np.sqrt(V ** 2 - 1) * np.cos(U), np.sqrt(V ** 2 - 1) * np.sin(U), V

    ax1.plot_wireframe(Xm, Ym, Zm, color='lightgrey', rstride=5, cstride=5, alpha=0.3, linewidth=0.5)
    ax1.plot_surface(Xm, Ym, Zm, color='lightgrey', alpha=0.1, shade=False)

    ax1.scatter(x1[~is_anomaly], x2[~is_anomaly], x0[~is_anomaly],
                c='navy', s=30, alpha=0.6, label='Normal Node')
    ax1.scatter(x1[is_anomaly], x2[is_anomaly], x0[is_anomaly],
                c='red', marker='x', s=40, linewidths=0.8, label='Anomaly')

    ax1.set_box_aspect((1, 1, 0.6))
    xy_limit = np.sqrt(v_max ** 2 - 1)
    ax1.set_xlim(-xy_limit, xy_limit)
    ax1.set_ylim(-xy_limit, xy_limit)
    ax1.set_zlim(1, v_max)
    ax1.set_title("Lorentz Model (3D Side View)", fontsize=14, fontweight='bold')
    ax1.view_init(elev=5, azim=0)
    ax1.legend(loc='upper right', frameon=True)
    ax1.axis('off')

    # --- 右图：诚实的学术正投影 (Orthographic) ---
    ax2 = fig.add_subplot(122)
    px, py = x1, x2
    r_boundary = xy_limit

    circle = plt.Circle((0, 0), r_boundary, color='lightgrey', fill=False, linewidth=1.5, linestyle='--')
    ax2.add_artist(circle)

    ax2.scatter(px[~is_anomaly], py[~is_anomaly], c='navy', s=25, alpha=0.5, edgecolors='none')
    ax2.scatter(px[is_anomaly], py[is_anomaly], c='red', marker='x', s=35, linewidths=0.7)

    ax2.set_xlim(-xy_limit * 1.1, xy_limit * 1.1)
    ax2.set_ylim(-xy_limit * 1.1, xy_limit * 1.1)
    ax2.set_aspect('equal')
    ax2.set_title("Top View (Orthographic Projection)", fontsize=14, fontweight='bold')

    ax2.text(xy_limit * 0.5, -xy_limit * 1.0, f'{metric_name}: {final_metric:.4f}',
             fontsize=12, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgrey'))
    ax2.axis('off')

    plt.suptitle(f"Hyperbolic Anomaly Detection ({args.dataset.upper()})", fontsize=18, y=0.98)
    plt.savefig(f"vis_lorentz_{args.dataset}.png", dpi=300, bbox_inches='tight')
    plt.close('all')


def run_single_dataset(dataset_name, args):
    original_task = args.task

    if 'nc' in dataset_name:
        args.task = 'nc'
    elif 'lp' in dataset_name:
        args.task = 'lp'
    else:
        args.task = original_task

    logging.info(f"\n{'=' * 20} Training {dataset_name} (Task: {args.task.upper()}) {'=' * 20}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data = load_data(args, os.path.join(os.environ['DATAPATH'], dataset_name))
    args.n_nodes, args.feat_dim = data['features'].shape

    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
    else:
        Model = LPModel
        # 【关键修复】：只在 LP 任务下才初始化和调用 edge 相关参数
        if 'train_edges_false' in data:
            args.nb_false_edges = len(data['train_edges_false'])
            args.nb_edges = len(data['train_edges'])
        else:
            print(f"Error: {dataset_name} lacks link prediction edges!")
            return

    model = Model(args).to(args.device)
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)

    for x in data:
        if torch.is_tensor(data[x]): data[x] = data[x].to(args.device)

    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None;
    best_emb = None;
    counter = 0

    pbar_desc = f"[{args.task.upper()}] {dataset_name.upper()}"
    pbar = tqdm(range(args.epochs), desc=pbar_desc)

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])

        # 【修复 NC 崩溃】：NC 和 LP 的 split 数据结构不同
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if args.task == 'nc':
            # NC 任务只有总分类 loss 和 acc
            pbar.set_postfix({'loss': f"{train_metrics['loss'].item():.4f}", 'acc': f"{train_metrics['acc']:.2f}"})
        else:
            # LP 任务才有结构和属性的拆分
            pbar.set_postfix({
                'L_all': f"{train_metrics['loss'].item():.4f}",
                'L_str': f"{train_metrics['loss_struct'].item():.4f}",
                'L_attr': f"{train_metrics['loss_attr'].item():.4f}",
                'roc': f"{train_metrics['roc']:.2f}"
            })

        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                val_embeddings = model.encode(data['features'], data['adj_train_norm'])
                val_metrics = model.compute_metrics(val_embeddings, data, 'val')
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(val_embeddings, data, 'test')
                best_emb = val_embeddings.detach().cpu().clone()
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter >= args.patience: break
    pbar.close()

    if best_emb is not None:
        if args.task == 'nc':
            final_metric = best_test_metrics['acc'] if best_test_metrics else 0.0
            metric_name = "Accuracy"
        else:
            final_metric = best_test_metrics['roc'] if best_test_metrics else 0.0
            metric_name = "ROC-AUC"

        visualize_dual_views(model, best_emb, data, args, final_metric, metric_name)

    args.task = original_task
    del data, model, optimizer, best_emb
    if 'embeddings' in locals(): del embeddings
    if 'val_embeddings' in locals(): del val_embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    args = parser.parse_args()
    args.device = f'cuda:{args.cuda}' if int(args.cuda) >= 0 and torch.cuda.is_available() else 'cpu'

    # 包含 pubmed 也不会再爆内存了
    target_datasets = ['disease_lp', 'disease_nc', 'airport', 'cora', 'pubmed']

    for ds in target_datasets:
        print(f"\n[{ds.upper()}] Spawning a new isolated process...")
        args.dataset = ds

        p = mp.Process(target=run_single_dataset, args=(ds, args))
        p.start()
        p.join()

        if p.exitcode != 0:
            print(f"!!! Process for {ds} ended with an error or was killed. Moving to next...")