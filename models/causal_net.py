"""
因果图神经网络模型
从 net/networks.py 提取并优化
（最终修正版：保留节点特征 + 灵活推断负样本数 + 更新因果MLP + 修正Edge维度）
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
    
    def forward(
        self,
        x: torch.Tensor,
        edge1: torch.Tensor,
        edge2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        执行两阶段图卷积
        
        Args:
            x: 输入节点特征 [B, P, d]
            edge1: 第一阶段边矩阵 [B, P, P]
            edge2: 第二阶段边矩阵 [B, P, P]（可选）
        """
        # 第一阶段第一层
        x = self.stage1_conv1(x, edge1)
        x = F.relu(self.stage1_bn1(x.transpose(1, 2)).transpose(1, 2))
        
        # 第一阶段第二层（使用单位矩阵）
        batch_size, num_nodes = x.shape[0], x.shape[1]
        identity_matrix = torch.eye(
            num_nodes, device=x.device
        ).unsqueeze(0).repeat(batch_size, 1, 1)
        
        x = self.stage1_conv2(x, identity_matrix)
        x = F.relu(self.stage1_bn2(x.transpose(1, 2)).transpose(1, 2))
        
        # 第二阶段（如果提供了edge2）
        if edge2 is not None:
            x = torch.bmm(edge2, x)
        
        return x


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
    """
    
    def __init__(
        self,
        num_class: int,
        feature_dim: int,
        hidden1: list,
        hidden2: list,
        num_patches: int,
        kernels: Optional[list] = None,
        num_neg_samples=4
    ):
        super(CausalNet, self).__init__()
        
        self.num_neg_samples_default = num_neg_samples
        self.num_class = num_class
        self.feature_dim = feature_dim
        self.num_patches = num_patches # 对应 P
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
    
    def prediction_whole(self, x_new, edge, is_large_graph=True):
        """全图预测 - 使用相同的边矩阵"""
        
        # 【修正】处理 2D edge, 适用于 [P,P] 和 [N*P, N*P]
        if edge.dim() == 2:
            edge = edge.unsqueeze(0).repeat(x_new.shape[0], 1, 1)
        xs = self.gcns2_causal(x_new, edge)
        graph = readout(xs)
        return self.mlp_causal(graph)
    
    def prediction_intrinsic_path(
        self,
        x_new: torch.Tensor,
        edge: torch.Tensor,
        masks: Tuple[torch.Tensor, torch.Tensor],
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        因果不变性预测
        """
        node_mask, edge_mask = masks
        
        # 【修正】处理 2D edge, 适用于 [P,P] 和 [N*P, N*P]
        if edge.dim() == 2:
            edge = edge.unsqueeze(0).repeat(x_new.shape[0], 1, 1)

        if is_large_graph:
            # 大图模式
            batch_size = x_new.shape[0]
            P = self.num_patches
            x_orig = x_new[:, :P, :].clone()
            # edge 已经是 [B, N*P, N*P]，切片操作安全
            edge_orig = edge[:, :P, :P].clone()
            
            # 应用掩码
            x_masked = x_orig * node_mask.unsqueeze(0).unsqueeze(-1)
            inter_node_adj = edge_orig * edge_mask.unsqueeze(0)
            
            # 添加自环
            identity = torch.eye(P, device=inter_node_adj.device).unsqueeze(0)
            final_adj = inter_node_adj + identity
            
            # GCN
            xs = self.gcns2_causal(x_masked, final_adj)
        else:
            # 小图模式
            x_masked = x_new * node_mask.unsqueeze(0).unsqueeze(-1)
            # edge 已经是 [B, P, P]，直接使用
            inter_node_adj = edge * edge_mask.unsqueeze(0)
            
            P = x_new.shape[1]
            identity = torch.eye(P, device=edge.device).unsqueeze(0)
            final_adj = inter_node_adj + identity
            
            xs = self.gcns2_causal(x_masked, final_adj)
        
        graph = readout(xs)
        return self.mlp_causal(graph)
    
    def prediction_spurious_path(
        self,
        x_new: torch.Tensor,
        edge: torch.Tensor,
        masks: Tuple[torch.Tensor, torch.Tensor],
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        因果变异性预测
        """
        node_mask, edge_mask = masks

        # 【修正】处理 2D edge, 适用于 [P,P] 和 [N*P, N*P]
        if edge.dim() == 2:
            edge = edge.unsqueeze(0).repeat(x_new.shape[0], 1, 1)
            

        causal_mask = node_mask.unsqueeze(0).unsqueeze(-1)
        x_perturbed = x_new * (1 - causal_mask)
            
        causal_edge_mask = edge_mask.unsqueeze(0)
        # edge 已经是 [B, P, P]，直接使用
        edge_perturbed = edge * (1 - causal_edge_mask)
            
        P = x_new.shape[1]
        identity = torch.eye(P, device=edge_perturbed.device).unsqueeze(0)
        final_adj_perturbed = edge_perturbed + identity
            
        xs = self.gcns2_causal(x_perturbed, final_adj_perturbed)
        
        graph = readout(xs)
        return self.mlp_causal(graph)
    
    def prediction_spurious_fusion(
        self,
        x: torch.Tensor,
        edge: torch.Tensor,
        masks: Tuple[torch.Tensor, torch.Tensor],
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        Spurious Fusion Graph - 测试Invariance (论文 Stage 3)
        
        Args:
            x: [B, N*P, feature_dim]
            edge: [B, N*P, N*P] (此函数必须接收3D大图)
            masks: (node_mask [P], edge_mask [P, P])
        """
        node_mask, edge_mask = masks
        node_mask_spur = 1 - node_mask  # spurious节点
        edge_mask_spur = edge_mask  # spurious边
        
        if not is_large_graph:
            raise ValueError("Spurious Fusion Graph需要大图模式")
        
        if edge.dim() == 2:
             raise ValueError("Fusion GCNs 必须接收 3D [B, N*P, N*P] edge tensor")

        B = x.shape[0]
        large_P = x.shape[1]
        P = self.num_patches
        
        if large_P % P != 0:
            raise ValueError(f"大图节点数 ({large_P}) 不是 P ({P}) 的整数倍")
        num_subgraphs = large_P // P
        num_neg_samples = num_subgraphs - 1
        
        # 1. 节点特征：保留所有原始节点特征 x
        
        # 2. Internal edges: 每个样本内部的spurious连接
        internal_edges = torch.zeros_like(edge).to(edge.device)
        for i in range(B):
            for s_idx in range(num_subgraphs):
                start = s_idx * P
                end = (s_idx + 1) * P
                
                orig_edge_block = edge[i, :P, :P]
                internal_edges[i, start:end, start:end] = orig_edge_block * edge_mask_spur
        
        identity_large = torch.eye(large_P, device=edge.device).unsqueeze(0)
        internal_edges = internal_edges + identity_large
        
        # 3. Cross edges: 跨样本spurious节点全连接
        cross_edges = torch.zeros_like(edge).to(edge.device)
        
        if num_neg_samples > 0:
            edge_weight = 1.0 / num_neg_samples
            
            for neg_idx in range(1, num_subgraphs):
                for p in range(P):
                    anchor_idx = p
                    neg_idx_p = neg_idx * P + p
                    
                    weight = edge_weight * node_mask_spur[p]
                    cross_edges[:, anchor_idx, neg_idx_p] = weight
                    cross_edges[:, neg_idx_p, anchor_idx] = weight
        
        for s_idx in range(num_subgraphs):
            for p in range(P):
                diag_idx = s_idx * P + p
                cross_edges[:, diag_idx, diag_idx] = node_mask_spur[p]
        
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
        edge: torch.Tensor,
        masks: Tuple[torch.Tensor, torch.Tensor],
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        Intrinsic Fusion Graph - 测试Sensitivity (论文 Stage 3)

        Args:
            x: [B, N*P, feature_dim]
            edge: [B, N*P, N*P] (此函数必须接收3D大图)
            masks: (node_mask [P], edge_mask [P, P])
        """
        node_mask, edge_mask = masks
        
        if not is_large_graph:
            raise ValueError("Intrinsic Fusion Graph需要大图模式")
        
        if edge.dim() == 2:
             raise ValueError("Fusion GCNs 必须接收 3D [B, N*P, N*P] edge tensor")

        B = x.shape[0]
        large_P = x.shape[1]
        P = self.num_patches
        
        if large_P % P != 0:
            raise ValueError(f"大图节点数 ({large_P}) 不是 P ({P}) 的整数倍")
        num_subgraphs = large_P // P
        num_neg_samples = num_subgraphs - 1
        
        # 1. 节点特征：保留所有原始节点特征 x
        
        # 2. Internal edges: 每个样本内部的intrinsic连接
        internal_edges = torch.zeros_like(edge).to(edge.device)
        for i in range(B):
            for s_idx in range(num_subgraphs):
                start = s_idx * P
                end = (s_idx + 1) * P
                
                orig_edge_block = edge[i, :P, :P]
                internal_edges[i, start:end, start:end] = orig_edge_block * edge_mask
        
        identity_large = torch.eye(large_P, device=edge.device).unsqueeze(0)
        internal_edges = internal_edges + identity_large
        
        # 3. Cross edges: 跨样本intrinsic节点全连接
        cross_edges = torch.zeros_like(edge).to(edge.device)

        if num_neg_samples > 0:
            edge_weight = 1.0 / num_neg_samples
        
            for neg_idx in range(1, num_subgraphs):
                for p in range(P):
                    anchor_idx = p
                    neg_idx_p = neg_idx * P + p
                    
                    weight = edge_weight * node_mask[p]
                    cross_edges[:, anchor_idx, neg_idx_p] = weight
                    cross_edges[:, neg_idx_p, anchor_idx] = weight
        
        for s_idx in range(num_subgraphs):
            for p in range(P):
                diag_idx = s_idx * P + p
                cross_edges[:, diag_idx, diag_idx] = node_mask[p]
        
        # 4. 使用TwoStageGCN处理
        xs_all = self.gcns2_causal(x, internal_edges, cross_edges)
        
        # 5. 只取anchor样本的预测
        xs_anchor = xs_all[:, :P, :]
        graph = readout(xs_anchor)
        logits = self.mlp_causal(graph) 
        
        return logits