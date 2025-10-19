"""
因果图神经网络模型
从 net/networks.py 提取并优化
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
        kernels: Optional[list] = None
    ):
        super(CausalNet, self).__init__()
        
        self.num_class = num_class
        self.feature_dim = feature_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        
        # 两阶段GCN（用于因果推理）
        self.gcns2_causal = TwoStageGCN(
            in_dim=feature_dim,
            hidden=hidden2,
            kernels=kernels if kernels else [2]
        )
        
        # 因果MLP
        self.mlp_causal = nn.Sequential(
            nn.Linear(hidden2[-1] * 288, 128),  # 假设288个patch
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_class)
        )
        
        # 整体预测MLP（用于预训练）
        self.mlp_whole = nn.Sequential(
            nn.Linear(feature_dim * 288, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_class)
        )
    
    def prediction_whole(
        self,
        x: torch.Tensor,
        edge: torch.Tensor,
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        整体预测（用于预训练）
        
        Args:
            x: 节点特征 [B, P, d]
            edge: 边矩阵 [B, P, P] 或 [P, P]
            is_large_graph: 是否为大图
        """
        if is_large_graph and len(edge.shape) == 2:
            # 扩展edge到batch维度
            edge = edge.unsqueeze(0).repeat(x.shape[0], 1, 1)
        
        # 直接读出并分类
        graph = readout(x)
        return self.mlp_whole(graph)
    
    def prediction_causal_invariance(
        self,
        x_new: torch.Tensor,
        edge: torch.Tensor,
        masks: Tuple[torch.Tensor, torch.Tensor],
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        因果不变性预测
        
        Args:
            x_new: 节点特征 [B, P, d]
            edge: 边矩阵
            masks: (node_mask, edge_mask)
            is_large_graph: 是否为大图
        """
        node_mask, edge_mask = masks
        
        if is_large_graph:
            # 大图模式
            batch_size = x_new.shape[0]
            P = x_new.shape[1] // 5
            x_orig = x_new[:, :P, :].clone()
            #print(f"边矩阵形状：{edge.shape},P的大小：{P}")
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
            inter_node_adj = edge * edge_mask.unsqueeze(0)
            
            P = x_new.shape[1]
            identity = torch.eye(P, device=edge.device).unsqueeze(0)
            final_adj = inter_node_adj + identity
            
            xs = self.gcns2_causal(x_masked, final_adj)
        
        graph = readout(xs)
        return self.mlp_causal(graph)
    
    def prediction_causal_variability(
        self,
        x_new: torch.Tensor,
        edge: torch.Tensor,
        masks: Tuple[torch.Tensor, torch.Tensor],
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        因果变异性预测
        
        Args:
            x_new: 节点特征
            edge: 边矩阵
            masks: (node_mask, edge_mask)
            is_large_graph: 是否为大图
        """
        node_mask, edge_mask = masks
        
        if is_large_graph:
            batch_size = x_new.shape[0]
            P = x_new.shape[1] // 5
            x_orig = x_new[:, :P, :].clone()
            edge_orig = edge[:, :P, :P].clone()
            
            # 对非因果部分操作
            causal_mask = node_mask.unsqueeze(0).unsqueeze(-1)
            x_perturbed = x_orig * (1 - causal_mask)
            
            causal_edge_mask = edge_mask.unsqueeze(0)
            edge_perturbed = edge_orig * (1 - causal_edge_mask)
            
            # 添加自环
            identity = torch.eye(P, device=edge_perturbed.device).unsqueeze(0)
            final_adj_perturbed = edge_perturbed + identity
            
            xs = self.gcns2_causal(x_perturbed, final_adj_perturbed)
        else:
            causal_mask = node_mask.unsqueeze(0).unsqueeze(-1)
            x_perturbed = x_new * (1 - causal_mask)
            
            causal_edge_mask = edge_mask.unsqueeze(0)
            edge_perturbed = edge * (1 - causal_edge_mask)
            
            P = x_new.shape[1]
            identity = torch.eye(P, device=edge_perturbed.device).unsqueeze(0)
            final_adj_perturbed = edge_perturbed + identity
            
            xs = self.gcns2_causal(x_perturbed, final_adj_perturbed)
        
        graph = readout(xs)
        return self.mlp_causal(graph)