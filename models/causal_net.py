"""
因果图神经网络模型
从 net/networks.py 提取并优化
（动态边构建版本：基于特征实时计算个性化边矩阵）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GCN(nn.Module):
    """图卷积层"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.W.reset_parameters()
    
    def forward(self, X: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: 节点特征 [B, P, d]
            adj: 邻接矩阵 [B, P, P]
        Returns:
            输出特征 [B, P, d']
        """
        XW = self.W(X)
        AXW = torch.bmm(adj, XW)
        return AXW


class TwoStageGCN(nn.Module):
    """两阶段图卷积网络"""
    
    def __init__(self, in_dim: int, hidden: list, kernels: list):
        super(TwoStageGCN, self).__init__()
        self.hidden = hidden
        self.kernels = kernels
        
        # 第一阶段 - 两层GCN
        self.stage1_conv1 = GCN(in_features=in_dim, out_features=self.hidden[0])
        self.stage1_bn1 = nn.BatchNorm1d(self.hidden[0])
        
        self.stage1_conv2 = GCN(in_features=self.hidden[0], out_features=self.hidden[-1])
        self.stage1_bn2 = nn.BatchNorm1d(self.hidden[-1])
        
        # 第二阶段 - 维度不变
        self.stage2_conv = GCN(in_features=self.hidden[-1], out_features=self.hidden[-1])
        self.stage2_bn = nn.BatchNorm1d(self.hidden[-1])
    
    def forward(self, x, edge1, edge2=None):
        # ========== 第一阶段第一层 ==========
        edge1_norm = self._normalize_adj_batch(edge1)
        x = self.stage1_conv1(x, edge1_norm)
        x = F.relu(self.stage1_bn1(x.transpose(1, 2)).transpose(1, 2))
        
        # ========== 第一阶段第二层 ==========
        x = self.stage1_conv2(x, edge1_norm)  # 
        x = F.relu(self.stage1_bn2(x.transpose(1, 2)).transpose(1, 2))
        
        # ========== 第二阶段 ==========
        if edge2 is not None:
            edge2_norm = self._normalize_adj_batch(edge2)
            x = torch.bmm(edge2_norm, x)
        
        return x
    
    @staticmethod
    def _normalize_adj_batch(adj: torch.Tensor) -> torch.Tensor:
        """
        对称归一化邻接矩阵

        Args:
            adj: [B, P, P] 邻接矩阵
        Returns:
            [B, P, P] 归一化后的邻接矩阵
        """
        # 1. 强制清空对角线（移除任何已有的自环）
        batch_size, num_nodes = adj.shape[0], adj.shape[1]
        adj = adj.clone()  # 避免修改原始输入

        # 将对角线设为0
        diag_indices = torch.arange(num_nodes, device=adj.device)
        adj[:, diag_indices, diag_indices] = 0
        #adj[:, :, :] = 0
        # 2. 添加自环（统一权重为1）
        identity = torch.eye(num_nodes, device=adj.device).unsqueeze(0)
        adj_with_self_loops = adj + identity

        # 3. 计算度
        degree = adj_with_self_loops.sum(dim=2)  # [B, P]

        # 4. D^{-1/2}
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0  # 处理孤立节点

        # 5. 对称归一化: D^{-1/2} * A * D^{-1/2}
        adj_normalized = (degree_inv_sqrt.unsqueeze(2) * 
                         adj_with_self_loops * 
                         degree_inv_sqrt.unsqueeze(1))

        return adj_normalized



def readout(x: torch.Tensor) -> torch.Tensor:
    """
    图读出函数
    
    Args:
        x: 节点特征 [B, P, d]
    Returns:
        图特征 [B, P*d]
    """
    batch_size = x.shape[0]
    return x.reshape(batch_size, -1)


class CausalNet(nn.Module):
    """
    因果图神经网络
    
    包含特征提取器和因果推理模块
    支持动态边构建：基于输入特征实时计算个性化边矩阵
    """
    
    def __init__(
        self,
        num_class: int,
        feature_dim: int,
        hidden1: list,
        hidden2: list,
        num_patches: int,
        kernels: Optional[list] = None,
        num_neg_samples: int = 4
    ):
        super(CausalNet, self).__init__()
        
        self.num_neg_samples_default = num_neg_samples
        self.num_class = num_class
        self.feature_dim = feature_dim
        self.num_patches = num_patches  # 对应 P
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        
        # 两阶段GCN（用于因果推理）
        self.gcns2_causal = TwoStageGCN(
            in_dim=feature_dim,
            hidden=hidden2,
            kernels=kernels if kernels else [2]
        )
        
        # 因果MLP（所有因果路径共用）
        causal_feature_size = hidden2[-1] * num_patches
        self.mlp_causal = nn.Sequential(
            nn.Linear(causal_feature_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(64, num_class)
        )
    
    def compute_dynamic_edges(
        self,
        x: torch.Tensor,
        edge_prior_mask: torch.Tensor,
        is_large_graph: bool = False
    ) -> torch.Tensor:
        """
        基于输入特征动态计算个性化边矩阵
        
        Args:
            x: 节点特征
               - 小图模式: [B, P, feature_dim]
               - 大图模式: [B, (N+1)*P, feature_dim]
            edge_prior_mask: 先验候选集 [P, P]，标识哪些位置允许学习边
            is_large_graph: 是否为大图模式
            
        Returns:
            edges: 动态边矩阵（连续值相似度）
                   - 小图: [B, P, P]
                   - 大图: [B, (N+1)*P, (N+1)*P]
        
        核心思想：
            1. 先验mask只是候选集，不是最终的边
            2. 真正的边 = 实时计算的相似度 * 先验mask
            3. 每个患者有自己的边矩阵（个性化）
            4. 大图中每个子图（anchor+负样本）分别计算边
        """
        B, total_P, d = x.shape
        P = edge_prior_mask.shape[0]
        
        if is_large_graph:
            # 大图模式：分别计算每个子图的边，然后组装成块对角矩阵
            num_subgraphs = total_P // P  # N+1（1个anchor + N个负样本）
            
            all_edges = []
            for i in range(num_subgraphs):
                start = i * P
                end = (i + 1) * P
                
                # 提取第i个子图的特征 [B, P, d]
                sub_x = x[:, start:end, :]
                
                # 计算该子图内部的相似度矩阵
                sub_x_norm = F.normalize(sub_x, p=2, dim=2)  # L2归一化
                sub_similarity = torch.bmm(sub_x_norm, sub_x_norm.transpose(1, 2))  # [B, P, P]
                
                # 余弦相似度范围 [-1, 1] 映射到 [0, 1]
                sub_similarity = (sub_similarity + 1) / 2
                
                # 应用先验mask（只在允许的位置保留相似度）
                mask_expanded = edge_prior_mask.unsqueeze(0).expand(B, -1, -1)
                sub_edges = sub_similarity * mask_expanded
                
                all_edges.append(sub_edges)
            
            # 组装成块对角矩阵 [B, (N+1)*P, (N+1)*P]
            edges = self._assemble_block_diagonal(all_edges)
            
        else:
            # 小图模式：直接计算
            x_norm = F.normalize(x, p=2, dim=2)
            similarity = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [B, P, P]
            similarity = (similarity + 1) / 2
            
            # 应用先验mask
            mask_expanded = edge_prior_mask.unsqueeze(0).expand(B, -1, -1)
            edges = similarity * mask_expanded
        
        # 移除自环（对角线置0）
        num_nodes = edges.shape[1]
        identity = torch.eye(num_nodes, device=edges.device).unsqueeze(0).expand(B, -1, -1)
        edges = edges * (1 - identity)
        
        return edges
    
    def _assemble_block_diagonal(self, block_list: list) -> torch.Tensor:
        """
        将多个 [B, P, P] 的块组装成块对角矩阵 [B, (N+1)*P, (N+1)*P]
        
        结构示意（N=2的情况）：
        ┌─────────┬─────────┬─────────┐
        │ block0  │    0    │    0    │  ← anchor样本的边
        ├─────────┼─────────┼─────────┤
        │    0    │ block1  │    0    │  ← 负样本1的边
        ├─────────┼─────────┼─────────┤
        │    0    │    0    │ block2  │  ← 负样本2的边
        └─────────┴─────────┴─────────┘
        
        Args:
            block_list: 列表，每个元素是 [B, P, P] 的边矩阵
            
        Returns:
            large_edges: [B, (N+1)*P, (N+1)*P] 块对角矩阵
        """
        B = block_list[0].shape[0]
        P = block_list[0].shape[1]
        num_blocks = len(block_list)
        total_P = num_blocks * P
        
        device = block_list[0].device
        large_edges = torch.zeros(B, total_P, total_P, device=device)
        
        for idx, block in enumerate(block_list):
            start = idx * P
            end = (idx + 1) * P
            large_edges[:, start:end, start:end] = block
        
        return large_edges
    
    def prediction_whole(
        self,
        x_new: torch.Tensor,
        edge_prior_mask: torch.Tensor,
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        全图预测 - 使用动态计算的边矩阵
        
        Args:
            x_new: 节点特征
            edge_prior_mask: 先验候选集 [P, P]
            is_large_graph: 是否为大图模式
        """
        # 动态计算边矩阵
        edge = self.compute_dynamic_edges(x_new, edge_prior_mask, is_large_graph)
        
        # GCN处理
        xs = self.gcns2_causal(x_new, edge)
        graph = readout(xs)
        return self.mlp_causal(graph)
    
    def prediction_intrinsic_path(
        self,
        x_new: torch.Tensor,
        edge_prior_mask: torch.Tensor,
        masks: Tuple[torch.Tensor, torch.Tensor],
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        因果不变性预测（内在路径）
        
        Args:
            x_new: 节点特征
            edge_prior_mask: 先验候选集 [P, P]
            masks: (node_mask [P], edge_mask [P, P])
            is_large_graph: 是否为大图模式
        """
        node_mask, edge_mask = masks
        
        # 动态计算完整的边矩阵
        edge = self.compute_dynamic_edges(x_new, edge_prior_mask, is_large_graph)
        
        if is_large_graph:
            # 大图模式：只处理anchor样本
            batch_size = x_new.shape[0]
            P = self.num_patches
            x_orig = x_new[:, :P, :].clone()
            edge_orig = edge[:, :P, :P].clone()
            
            # 应用因果掩码
            x_masked = x_orig * node_mask.unsqueeze(0).unsqueeze(-1)
            inter_node_adj = edge_orig * edge_mask.unsqueeze(0)
            
            # GCN
            xs = self.gcns2_causal(x_masked, inter_node_adj)
        else:
            # 小图模式
            x_masked = x_new * node_mask.unsqueeze(0).unsqueeze(-1)
            inter_node_adj = edge * edge_mask.unsqueeze(0)
            
            P = x_new.shape[1]
            
            xs = self.gcns2_causal(x_masked, inter_node_adj)
        
        graph = readout(xs)
        return self.mlp_causal(graph)
    
    def prediction_spurious_path(
        self,
        x_new: torch.Tensor,
        edge_prior_mask: torch.Tensor,
        masks: Tuple[torch.Tensor, torch.Tensor],
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        因果变异性预测（虚假路径）
        
        Args:
            x_new: 节点特征
            edge_prior_mask: 先验候选集 [P, P]
            masks: (node_mask [P], edge_mask [P, P])
            is_large_graph: 是否为大图模式
        """
        node_mask, edge_mask = masks
        
        # 动态计算边矩阵
        edge = self.compute_dynamic_edges(x_new, edge_prior_mask, is_large_graph)
        
        # 扰动：使用非因果部分
        causal_mask = node_mask.unsqueeze(0).unsqueeze(-1)
        x_perturbed = x_new * (1 - causal_mask)
        
        causal_edge_mask = edge_mask.unsqueeze(0)
        edge_perturbed = edge * (1 - causal_edge_mask)
        
        P = x_new.shape[1]
        
        xs = self.gcns2_causal(x_perturbed, edge_perturbed)
        
        graph = readout(xs)
        return self.mlp_causal(graph)
    
    def prediction_spurious_fusion(
        self,
        x: torch.Tensor,
        edge_prior_mask: torch.Tensor,
        masks: Tuple[torch.Tensor, torch.Tensor],
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        Spurious Fusion Graph - 测试Invariance（论文Stage 3）

        使用虚假节点和边构建跨样本的融合图
        """
        node_mask, edge_mask = masks
        node_mask_spur = 1 - node_mask  # spurious节点
        # edge_mask 标识因果边，所以 spurious边 = 非因果边

        if not is_large_graph:
            raise ValueError("Spurious Fusion Graph需要大图模式")

        B = x.shape[0]
        large_P = x.shape[1]
        P = self.num_patches

        if large_P % P != 0:
            raise ValueError(f"大图节点数 ({large_P}) 不是 P ({P}) 的整数倍")
        num_subgraphs = large_P // P
        num_neg_samples = num_subgraphs - 1

        # ✅ 第1步：动态计算每个子图的完整边矩阵
        full_dynamic_edges = self.compute_dynamic_edges(x, edge_prior_mask, is_large_graph=True)
        # [B, (N+1)*P, (N+1)*P]

        # ✅ 第2步：从动态边中选择 spurious 部分
        spurious_edge_mask_large = torch.zeros(large_P, large_P, device=x.device)
        for i in range(num_subgraphs):
            start = i * P
            end = (i + 1) * P
            # 每个子图内部：选择因果边（edge_mask）
            spurious_edge_mask_large[start:end, start:end] = edge_mask

        # 应用 spurious 边 mask
        internal_edges = full_dynamic_edges * spurious_edge_mask_large.unsqueeze(0)

        # 3. Cross edges
        cross_edges = torch.zeros(B, large_P, large_P, device=x.device)

        if num_neg_samples > 0:
            edge_weight = 1.0 / num_neg_samples

            for neg_idx in range(1, num_subgraphs):
                for p in range(P):
                    anchor_idx = p
                    neg_idx_p = neg_idx * P + p

                    weight = edge_weight * node_mask_spur[p]
                    cross_edges[:, anchor_idx, neg_idx_p] = weight
                    cross_edges[:, neg_idx_p, anchor_idx] = weight

        # 4. 使用TwoStageGCN处理
        xs_all = self.gcns2_causal(x, internal_edges, cross_edges)

        # 5. 只取anchor样本的预测
        xs_anchor = xs_all[:, :P, :]
        graph = readout(xs_anchor)
        logits = self.mlp_causal(graph)

        return logits

    def prediction_intrinsic_fusion(
        self,
        x: torch.Tensor,
        edge_prior_mask: torch.Tensor,
        masks: Tuple[torch.Tensor, torch.Tensor],
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        Intrinsic Fusion Graph - 测试Sensitivity（论文Stage 3）

        使用因果节点和边构建跨样本的融合图
        """
        node_mask, edge_mask = masks

        if not is_large_graph:
            raise ValueError("Intrinsic Fusion Graph需要大图模式")

        B = x.shape[0]
        large_P = x.shape[1]
        P = self.num_patches

        if large_P % P != 0:
            raise ValueError(f"大图节点数 ({large_P}) 不是 P ({P}) 的整数倍")
        num_subgraphs = large_P // P
        num_neg_samples = num_subgraphs - 1

        # ✅ 第1步：动态计算每个子图的完整边矩阵
        full_dynamic_edges = self.compute_dynamic_edges(x, edge_prior_mask, is_large_graph=True)
        # [B, (N+1)*P, (N+1)*P]

        # ✅ 第2步：从动态边中选择 intrinsic 部分
        intrinsic_edge_mask_large = torch.zeros(large_P, large_P, device=x.device)
        for i in range(num_subgraphs):
            start = i * P
            end = (i + 1) * P
            # 每个子图内部：选择因果边（edge_mask）
            intrinsic_edge_mask_large[start:end, start:end] = edge_mask

        # 应用 intrinsic 边 mask
        internal_edges = full_dynamic_edges * intrinsic_edge_mask_large.unsqueeze(0)

        # 2. Cross edges: 跨样本intrinsic节点全连接
        cross_edges = torch.zeros(B, large_P, large_P, device=x.device)

        if num_neg_samples > 0:
            edge_weight = 1.0 / num_neg_samples

            for neg_idx in range(1, num_subgraphs):
                for p in range(P):
                    anchor_idx = p
                    neg_idx_p = neg_idx * P + p

                    weight = edge_weight * node_mask[p]
                    cross_edges[:, anchor_idx, neg_idx_p] = weight
                    cross_edges[:, neg_idx_p, anchor_idx] = weight

        # 3. 使用TwoStageGCN处理
        xs_all = self.gcns2_causal(x, internal_edges, cross_edges)

        # 4. 只取anchor样本的预测
        xs_anchor = xs_all[:, :P, :]
        graph = readout(xs_anchor)
        logits = self.mlp_causal(graph)

        return logits