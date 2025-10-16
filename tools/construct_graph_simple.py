import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable

class ConstructGraph:
    def __init__(self, node_type=None, edge_type='corr', dist_type='gau', adj_norm='DAD', build_large_graph=False, random_seed=42):
        self.node_type = node_type
        self.edge_type = edge_type
        self.dist_type = dist_type
        self.adj_norm = adj_norm
        self.build_large_graph = build_large_graph
        self.random_seed = random_seed

    def construct(self, pxs, embedding, coord, labels=None, consistent_mask=None):
        """
        [最终优雅版]
        该函数现在统一处理大小图的构建，核心逻辑是：
        1. 准备一个标准化的、带自环的基础图 (base_edge)。
        2. 根据模式（大图/小图）使用这个基础图进行组装。
        """
        B, P = pxs.shape[:2]

        if consistent_mask is None:
            raise ValueError("错误：此流程强制要求提供预计算的图 (consistent_mask)。")

        # --- 1. 准备一个标准化的、带自环的基础图 (base_edge) ---
        # 这个操作只执行一次，得到的 base_edge 将被复用。
        base_edge = consistent_mask.copy()
        if self.adj_norm is not None and self.adj_norm != 'None':
            # 这个函数会给图加上自环（I）并进行归一化，保证GCN计算稳定
            base_edge = normalize_digraph(base_edge, norm_type=self.adj_norm)

        # --- 2. 根据模式，使用 base_edge 进行组装 ---
        if not self.build_large_graph:
            # --- 小图模式 ---
            # 对于批次中的每个样本，都使用这个相同的基础图。
            edges = np.stack([base_edge] * B, axis=0).astype(np.float32)
            return pxs, edges
        else:
            # --- 大图模式 ---
            assert labels is not None, "构建大图需要提供标签 (labels)。"
            
            large_node_list = []
            large_edge_list = []
            random_state = np.random.RandomState(self.random_seed)
            orig_shape = pxs.shape[2:]

            for i in range(B):  # 为批次中的每个“锚点”样本构建一个大图
                # a. 选择4个负样本 (逻辑不变)
                current_label = labels[i]
                opposite_indices = np.where(labels == 1 - current_label)[0]
                if len(opposite_indices) < 4:
                    # 如果负样本不够，允许重复采样
                    selected_neg = random_state.choice(opposite_indices, 4, replace=True)
                else:
                    selected_neg = random_state.choice(opposite_indices, 4, replace=False)
                
                # b. 拼接大图的节点特征 (逻辑不变)
                large_node = np.zeros((5 * P, *orig_shape), dtype=pxs.dtype)
                large_node[:P] = pxs[i] # 第一个块是锚点样本
                for neg_idx, global_idx in enumerate(selected_neg):
                    start, end = (neg_idx + 1) * P, (neg_idx + 2) * P
                    large_node[start:end] = pxs[global_idx]
                large_node_list.append(large_node)

                # c. 高效构建大图的邻接矩阵 (全新逻辑)
                # 创建一个 5P x 5P 的空矩阵
                large_edge = np.zeros((5 * P, 5 * P))
                # 将处理好的 base_edge 作为“积木”填充到对角线上的5个块中
                for s_idx in range(5):
                    start, end = s_idx * P, (s_idx + 1) * P
                    large_edge[start:end, start:end] = base_edge
                large_edge_list.append(large_edge)
                
            large_nodes = np.stack(large_node_list, axis=0).astype(np.float32)
            large_edges = np.stack(large_edge_list, axis=0).astype(np.float32)

            return large_nodes, large_edges
        
def normalize_digraph(A, norm_type='DAD'):  # Normalized adjacency matrix
    flag = False
    if isinstance(A, torch.Tensor) and A.layout == torch.sparse_coo:
        A = A.to_dense()
        flag = True

    if isinstance(A, torch.Tensor):
        num_node = A.shape[0]
        A = A + torch.eye(num_node).to(A.device)  # \hat{A}
        rowsum = A.sum(1)
        if norm_type == 'DA':
            r_inv = rowsum.pow(-1).flatten()
            r_inv[torch.isinf(r_inv)] = 0.
            r_inv[torch.isnan(r_inv)] = 0.
            D = torch.diag(r_inv)
            return Tensor2Sparse(D @ A) if flag else D @ A
        elif norm_type == 'DAD':
            r_inv = rowsum.pow(-1 / 2).flatten()
            r_inv[torch.isinf(r_inv)] = 0.
            r_inv[torch.isnan(r_inv)] = 0.
            D = torch.diag(r_inv)
            return Tensor2Sparse(D @ A @ D) if flag else D @ A @ D

    elif isinstance(A, np.ndarray):
        num_node = A.shape[0]
        A = A + np.eye(num_node)  # \hat{A}
        rowsum = A.sum(1)
        if norm_type == 'DA':
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_inv[np.isnan(r_inv)] = 0.
            D = np.diag(r_inv)
            return D @ A
        elif norm_type == 'DAD':
            r_inv = np.power(rowsum, -0.5).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_inv[np.isnan(r_inv)] = 0.
            D = np.diag(r_inv)
            return D @ A @ D
    else:
        raise ValueError('Unknown type of XA')


def Tensor2Sparse(Ad):
    assert isinstance(Ad, torch.Tensor) and not Ad.layout == torch.sparse_coo
    idx = torch.nonzero(Ad).T
    data = Ad[idx[0], idx[1]]
    An = torch.sparse.FloatTensor(idx, data, Ad.shape)
    return An