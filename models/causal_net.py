"""
因果图神经网络模型
(动态边构建 + 替换式融合版本)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GCN(nn.Module):
    """图卷积层 (AXW)"""
    
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


class GCNBlock(nn.Module):
    """
    【重构】
    GCN模块，仅负责子图内部的特征学习
    (版本: 仅含Dropout，无BN或ReLU)
    """
    
    def __init__(self, in_dim: int, hidden: list, dropout_p: float = 0.2):
        """
        Args:
            in_dim: 输入特征维度
            hidden: 隐藏层维度列表，例如 [hidden1_dim, output_dim]
            dropout_p: Dropout 概率
        """
        super(GCNBlock, self).__init__()
        self.hidden = hidden
        
        # 两层GCN
        self.conv1 = GCN(in_features=in_dim, out_features=self.hidden[0])
        self.dropout1 = nn.Dropout(dropout_p)
        
        self.conv2 = GCN(in_features=self.hidden[0], out_features=self.hidden[-1])
        self.dropout2 = nn.Dropout(dropout_p)
    
    def forward(self, x: torch.Tensor, edge1: torch.Tensor, node_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            node_mask: [P] 或 None (prediction_whole不需要mask)
        """
        edge1_norm = self._normalize_adj_batch(edge1, node_mask)

        x = self.conv1(x, edge1_norm)
        x = self.dropout1(x)

        x = self.conv2(x, edge1_norm)
        x = self.dropout2(x)

        return x
    
    
    @staticmethod
    def _normalize_adj_batch(adj: torch.Tensor, node_mask: torch.Tensor = None) -> torch.Tensor:
        """
        对称归一化邻接矩阵，考虑节点mask

        Args:
            adj: [B, P, P]
            node_mask: [P] 节点mask (1=保留, 0=删除)
        """
        batch_size, num_nodes = adj.shape[0], adj.shape[1]
        adj_clone = adj.clone()

        # 【新增】如果有node_mask，屏蔽被mask掉的节点的边
        if node_mask is not None:
            valid_mask = node_mask.view(1, -1, 1) * node_mask.view(1, 1, -1)  # [1, P, P]
            adj_clone = adj_clone * valid_mask  # 无效节点的边权重置0

        # 清空对角线
        diag_indices = torch.arange(num_nodes, device=adj.device)
        adj_clone[:, diag_indices, diag_indices] = 0

        # 添加自环
        identity = torch.eye(num_nodes, device=adj.device).unsqueeze(0)
        adj_with_self_loops = adj_clone + identity

        # 计算度（只计算有效边）
        degree = adj_with_self_loops.sum(dim=2)  # [B, P]

        # D^{-1/2}
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0

        # 对称归一化
        adj_normalized = (degree_inv_sqrt.unsqueeze(2) * adj_with_self_loops * 
                          degree_inv_sqrt.unsqueeze(1))

        return adj_normalized


def readout(x: torch.Tensor) -> torch.Tensor:
    """
    图读出函数 [B, P, d] -> [B, P*d]
    """
    batch_size = x.shape[0]
    return x.reshape(batch_size, -1)


class CausalNet(nn.Module):
    """
    【重构】
    因果图神经网络
    使用 GCNBlock 进行内部特征提取
    使用 _perform_fusion_replacement (替换式融合) 进行因果测试
    """
    
    def __init__(
        self,
        num_class: int,
        feature_dim: int,
        hidden1: list, # (未使用，但保留签名)
        hidden2: list,
        num_patches: int,
        kernels: Optional[list] = None, # (未使用，但保留签名)
        num_neg_samples: int = 4
    ):
        super(CausalNet, self).__init__()
        
        self.num_neg_samples_default = num_neg_samples
        self.num_class = num_class
        self.feature_dim = feature_dim
        self.num_patches = num_patches # 对应 P
        self.hidden2 = hidden2
        
        # GCN 模块（用于子图内部特征提取）
        self.gcn_block = GCNBlock(
            in_dim=feature_dim,
            hidden=hidden2
        )
        
        # 因果MLP（所有因果路径共用）
        causal_feature_size = hidden2[-1] * num_patches
        self.mlp_causal = nn.Sequential(
            nn.Linear(causal_feature_size, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, num_class))
    
    
    def compute_dynamic_edges(
        self,
        x: torch.Tensor,
        edge_prior_mask: torch.Tensor
    ) -> torch.Tensor:
        """【简化版】动态边计算"""
        B, P, d = x.shape

        x_norm = F.normalize(x, p=2, dim=2)
        similarity = torch.bmm(x_norm, x_norm.transpose(1, 2))
        similarity = (similarity + 1) / 2

        mask_expanded = edge_prior_mask.unsqueeze(0).expand(B, -1, -1)
        edges = similarity * mask_expanded

        # 移除自环
        identity = torch.eye(P, device=edges.device).unsqueeze(0).expand(B, -1, -1)
        edges = edges * (1 - identity)

        return edges
    
    def _assemble_block_diagonal(self, block_list: list) -> torch.Tensor:
        """
        组装块对角矩阵 (未改动)
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
    
    def prediction_whole(self, x, edge_prior_mask):
        edge = self.compute_dynamic_edges(x, edge_prior_mask)
        xs = self.gcn_block(x, edge, node_mask=None)
        graph = readout(xs)
        return self.mlp_causal(graph)

    def prediction_intrinsic_path(self, x, edge_prior_mask, masks):
        node_mask, edge_mask = masks
        edge = self.compute_dynamic_edges(x, edge_prior_mask)
        x_masked = x * node_mask.unsqueeze(0).unsqueeze(-1)
        inter_node_adj = edge * edge_mask.unsqueeze(0)
        xs = self.gcn_block(x_masked, inter_node_adj, node_mask)
        graph = readout(xs)
        return self.mlp_causal(graph)

    def prediction_spurious_path(self, x, edge_prior_mask, masks):
        node_mask, edge_mask = masks
        edge = self.compute_dynamic_edges(x, edge_prior_mask)

        causal_mask = node_mask.unsqueeze(0).unsqueeze(-1)
        x_perturbed = x * (1 - causal_mask)

        causal_edge_mask = edge_mask.unsqueeze(0)
        edge_perturbed = edge * (1 - causal_edge_mask)

        spurious_node_mask = 1 - node_mask
        xs = self.gcn_block(x_perturbed, edge_perturbed, spurious_node_mask)

        graph = readout(xs)
        return self.mlp_causal(graph)

    def _perform_fusion_batch_level(
        self,
        x: torch.Tensor,  # [B, P, d]
        labels: torch.Tensor,  # [B] - 新增参数！
        edge_prior_mask: torch.Tensor,
        masks: Tuple[torch.Tensor, torch.Tensor],
        fusion_type: str
    ) -> torch.Tensor:
        """
        批次内对立类别融合

        关键改进：只使用对立类别样本作为替换源
        """
        node_mask, edge_mask = masks
        B, P, d = x.shape

        # 1. 动态边 + 内部GCN
        edges = self.compute_dynamic_edges(x, edge_prior_mask)
        intrinsic_edge = edges * edge_mask.unsqueeze(0)
        xs_stage1 = self.gcn_block(x, intrinsic_edge, node_mask)

        # 2. ✅ 关键修复：为每个样本计算对立类别的聚合特征
        replacement_features = torch.zeros_like(xs_stage1)

        for i in range(B):
            opposite_label = 1 - labels[i]
            opposite_mask = (labels == opposite_label)

            if opposite_mask.sum() > 0:
                # 只取对立类别样本的平均
                opposite_features = xs_stage1[opposite_mask]
                replacement_features[i] = opposite_features.mean(dim=0)
            else:
                # 退化方案：用除自己外的所有样本
                other_mask = torch.ones(B, dtype=torch.bool, device=x.device)
                other_mask[i] = False
                replacement_features[i] = xs_stage1[other_mask].mean(dim=0)

        # 3. 条件替换
        mask_intrinsic = node_mask.reshape(1, P, 1)
        mask_spurious = 1 - mask_intrinsic

        if fusion_type == "intrinsic":
            x_fused = (replacement_features * mask_intrinsic) + (xs_stage1 * mask_spurious)
        elif fusion_type == "spurious":
            x_fused = (xs_stage1 * mask_intrinsic) + (replacement_features * mask_spurious)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # 4. 预测
        graph = readout(x_fused)
        logits = self.mlp_causal(graph)

        return logits

    # 更新对外接口（新增 labels 参数）
    def prediction_spurious_fusion(self, x, labels, edge_prior_mask, masks):
        return self._perform_fusion_batch_level(
            x, labels, edge_prior_mask, masks, fusion_type="spurious"
        )

    def prediction_intrinsic_fusion(self, x, labels, edge_prior_mask, masks):
        return self._perform_fusion_batch_level(
            x, labels, edge_prior_mask, masks, fusion_type="intrinsic"
        )

