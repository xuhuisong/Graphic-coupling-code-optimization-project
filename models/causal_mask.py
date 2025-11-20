"""
因果掩码模型（改进版 - 目标稀疏度 + 自适应权重）

核心改进：
1. 引入目标稀疏度参数 (target_sparsity)
2. 自适应权重机制 - 动态调整正则化强度
3. 渐进式训练策略 - 预热期 + 逐步增强
4. 完善的监控机制 - 实时追踪稀疏度偏离
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Tuple, Optional


class CausalMask(nn.Module):
    """
    因果掩码学习器（改进版）
    
    Args:
        num_patches: patch数量
        edge_matrix: 边的先验候选集 [P, P]
        gumble_tau: Gumbel-Softmax 温度参数
        target_node_sparsity: 目标节点稀疏度（期望保留的因果节点比例）
        target_edge_sparsity: 目标边稀疏度（期望保留的因果边比例）
    """
    
    def __init__(
        self,
        num_patches: int,
        edge_matrix: torch.Tensor,
        gumble_tau: float = 1.0,
        target_node_sparsity: float = 0.3,
        target_edge_sparsity: float = 0.2
    ):
        super(CausalMask, self).__init__()
        
        self.P = num_patches
        self.tau = gumble_tau
        self.target_node_sparsity = target_node_sparsity
        self.target_edge_sparsity = target_edge_sparsity
        
        # 节点掩码：[P, 2] - [非因果, 因果]
        n_init = torch.cat([torch.zeros(self.P, 1), torch.ones(self.P, 1)], dim=-1)
        self.node_mask = nn.Parameter(Variable(n_init, requires_grad=True))
        
        # 边掩码：[P, P, 2] - [非因果, 因果]
        identity_matrix = torch.eye(self.P, device=edge_matrix.device)
        learnable_mask = edge_matrix * (1 - identity_matrix)
        self.register_buffer('learnable_mask', learnable_mask)
        
        e_init = torch.zeros(self.P, self.P, 2)
        for i in range(self.P):
            for j in range(self.P):
                if learnable_mask[i, j] > 0:
                    e_init[i, j, 0] = 0  # 非因果
                    e_init[i, j, 1] = 1  # 因果
                else:
                    e_init[i, j, 0] = 1
                    e_init[i, j, 1] = 0
        
        self.edge_mask = nn.Parameter(Variable(e_init, requires_grad=True))
    
    @staticmethod
    def gumbel_softmax(logits, tau: float = 1, hard: bool = True, dim: int = -1):
        """Gumbel-Softmax采样"""
        gumbel_dist = torch.distributions.gumbel.Gumbel(
            torch.tensor(0., device=logits.device, dtype=logits.dtype),
            torch.tensor(1., device=logits.device, dtype=logits.dtype)
        )
        gumbels = gumbel_dist.sample(logits.shape)
        
        gumbels = (logits + gumbels) / tau
        y_soft = gumbels.softmax(dim)
        
        if hard:
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        
        return ret[..., 1]  # 返回因果维度
    
    @staticmethod
    def hardmax(M, dim: int = -1):
        """硬argmax"""
        return M.argmax(dim).float()
    
    def calculate_sparsity(self, node_mask, edge_mask):
        """计算当前实际稀疏度"""
        node_sparsity = node_mask.mean()
        edge_sparsity = edge_mask.sum() / (self.learnable_mask.sum() + 1e-8)
        return node_sparsity, edge_sparsity
    
    def forward(self, train=True, return_probs=False):
        if train:
            node_mask_hard = self.gumbel_softmax(self.node_mask, tau=self.tau, hard=True)
            edge_mask_hard = self.gumbel_softmax(self.edge_mask, tau=self.tau, hard=True)

            # 【新增】强制对称
            edge_mask_hard = torch.triu(edge_mask_hard, diagonal=1)  # 只保留上三角
            edge_mask_hard = edge_mask_hard + edge_mask_hard.t()     # 对称复制
            edge_mask_hard = edge_mask_hard * self.learnable_mask
        else:
            node_mask_hard = self.hardmax(self.node_mask)
            edge_mask_hard = self.hardmax(self.edge_mask)

            # 【新增】强制对称
            edge_mask_hard = torch.triu(edge_mask_hard, diagonal=1)
            edge_mask_hard = edge_mask_hard + edge_mask_hard.t()
            edge_mask_hard = edge_mask_hard * self.learnable_mask

        # 对角线置0
        edge_mask_hard.fill_diagonal_(0)

        node_sparsity, edge_sparsity = self.calculate_sparsity(node_mask_hard, edge_mask_hard)

        if return_probs:
            # 概率也对称化
            node_probs = F.softmax(self.node_mask, dim=-1)[:, 1]
            edge_probs = F.softmax(self.edge_mask, dim=-1)[:, :, 1]
            edge_probs = torch.triu(edge_probs, diagonal=1) + torch.triu(edge_probs, diagonal=1).t()
            edge_probs = edge_probs * self.learnable_mask

            return ([node_mask_hard, edge_mask_hard], 
                    [node_probs, edge_probs], 
                    (node_sparsity, edge_sparsity))

        return [node_mask_hard, edge_mask_hard], (node_sparsity, edge_sparsity)
    
    def compute_sparsity_regularization(
        self, 
        lambda_reg: float = 0.01,
        lambda_edge_multiplier: float = 3.0,  # 新增参数
    ):
        """
        简洁的稀疏性正则化：直接惩罚因果节点/边的数量

        核心思想：
        - 直接最小化因果特征的比例
        - 渐进式增强：早期弱惩罚（探索），后期强惩罚（收敛）
        - 让模型在准确率和稀疏度间自然平衡

        Args:
            lambda_reg: 基础正则化系数
            epoch: 当前轮次
            max_epochs: 总轮次
            warmup_epochs: 预热期
        """

        # 计算节点和边的因果概率（软值）
        node_probs = F.softmax(self.node_mask, dim=1)[:, 1]  # [P]
        edge_probs = F.softmax(self.edge_mask, dim=-1)[:, :, 1]  # [P, P]

        # 只统计可学习边的概率
        masked_edge_probs = edge_probs * self.learnable_mask
        total_learnable_edges = self.learnable_mask.sum()

        # 计算平均稀疏度（期望有多少比例是因果的）
        node_sparsity = node_probs.mean()
        if total_learnable_edges > 0:
            edge_sparsity = masked_edge_probs.sum() / total_learnable_edges
        else:
            edge_sparsity = torch.tensor(0.0, device=node_sparsity.device)

        # 组合损失：节点 + 边
        node_loss = lambda_reg * node_sparsity
        edge_loss = lambda_reg * lambda_edge_multiplier * edge_sparsity
        total_loss = node_loss + edge_loss

        return total_loss
    
    def get_sparsity_stats(self) -> dict:
        """
        获取当前掩码的稀疏度统计信息（用于监控）
        
        Returns:
            包含详细稀疏度信息的字典
        """
        with torch.no_grad():
            # 当前软概率
            node_probs = F.softmax(self.node_mask, dim=1)[:, 1]
            edge_probs = F.softmax(self.edge_mask, dim=-1)[:, :, 1]
            masked_edge_probs = edge_probs * self.learnable_mask
            
            # 硬决策
            node_hard = self.hardmax(self.node_mask)
            edge_hard = self.hardmax(self.edge_mask) * self.learnable_mask
            
            total_learnable_edges = self.learnable_mask.sum().item()
            
            stats = {
                'node': {
                    'soft_sparsity': node_probs.mean().item(),
                    'hard_sparsity': node_hard.mean().item(),
                    'target': self.target_node_sparsity,
                    'deviation': abs(node_hard.mean().item() - self.target_node_sparsity),
                    'total_nodes': self.P,
                    'causal_nodes': int(node_hard.sum().item())
                },
                'edge': {
                    'soft_sparsity': (masked_edge_probs.sum() / (total_learnable_edges + 1e-8)).item(),
                    'hard_sparsity': (edge_hard.sum() / (total_learnable_edges + 1e-8)).item(),
                    'target': self.target_edge_sparsity,
                    'deviation': abs((edge_hard.sum() / (total_learnable_edges + 1e-8)).item() - self.target_edge_sparsity),
                    'total_learnable_edges': int(total_learnable_edges),
                    'causal_edges': int(edge_hard.sum().item())
                }
            }
            
            return stats
        
