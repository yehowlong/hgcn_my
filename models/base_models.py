"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task with Anomaly Detection Attribute Loss.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        # 【关键修复】：这里必须是 args.manifold，用于加载 FermiDirac 距离解码器
        self.decoder = model2decoder[args.manifold](self.c, args)
        # 属性解码器：将嵌入维度的特征还原回原始特征维度
        self.attr_decoder = nn.Linear(args.dim, args.feat_dim)

    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false']
            edges_true = data[f'{split}_edges']
        else:
            edges_false = data[f'{split}_edges_false']
            edges_true = data[f'{split}_edges']

        # 1. 结构损失 (Structural Loss)
        loss_struct = self.decoder.compute_loss(embeddings, edges_true, edges_false)

        # 2. 属性重建损失 (Attribute Loss)
        # 将双曲空间中的 embeddings 映射到原点处的切空间（欧式空间）
        embeddings_tg = self.manifold.logmap0(embeddings, c=self.c)
        # 通过线性层解码重构特征
        reconstructed_features = self.attr_decoder(embeddings_tg)
        # 使用 MSE 计算重构误差
        loss_attr = F.mse_loss(reconstructed_features, data['features'])

        # 3. 联合优化：权重硬编码为 1.0
        loss = loss_struct + 1.0 * loss_attr

        if split == 'train':
            metrics = {'loss': loss, 'loss_struct': loss_struct, 'loss_attr': loss_attr}
        else:
            metrics = {}

        if split == 'val':
            roc, ap = eval_utils.get_roc_score(embeddings, edges_true, edges_false, self.c, self.args)
            metrics['roc'] = roc
            metrics['ap'] = ap
        elif split == 'test':
            roc, ap = eval_utils.get_roc_score(embeddings, edges_true, edges_false, self.c, self.args)
            metrics['roc'] = roc
            metrics['ap'] = ap

        return metrics

