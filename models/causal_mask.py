"""
因果掩码模型（原版正确实现）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Tuple, Optional


class CausalMask(nn.Module):
    """
    因果掩码学习器
    
    为每个节点/边学习两个logits [非因果, 因果]
    使用Gumbel-Softmax进行二选一采样
    """
    
    def __init__(
        self,
        num_patches: int,
        edge_matrix: torch.Tensor,
        gumble_tau: float = 1.0
    ):
        super(CausalMask, self).__init__()
        
        self.P = num_patches
        self.tau = gumble_tau
        
        # 节点掩码：[P, 2] - [非因果, 因果]
        n_init = torch.cat([torch.zeros(self.P, 1), torch.ones(self.P, 1)], dim=-1)
        self.node_mask = nn.Parameter(Variable(n_init, requires_grad=True))
        
        # 边掩码：[P, P, 2] - [非因果, 因果]
        # 移除对角线
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
        """计算稀疏度"""
        edge_sparsity = edge_mask.sum() / (self.learnable_mask.sum() + 1e-8)
        return (node_mask.mean() + edge_sparsity) / 2
    
    def forward(self, train=True, return_probs=False):
        """
        前向传播
        
        Returns:
            如果return_probs=False: ([node_mask, edge_mask], sparsity)
            如果return_probs=True: ([node_mask, edge_mask], [node_probs, edge_probs], sparsity)
        """
        if train:
            # 训练：Gumbel-Softmax
            node_mask_hard = self.gumbel_softmax(self.node_mask, tau=self.tau, hard=True)
            edge_mask_hard = self.gumbel_softmax(self.edge_mask, tau=self.tau, hard=True) * self.learnable_mask
        else:
            # 评估：硬argmax
            node_mask_hard = self.hardmax(self.node_mask)
            edge_mask_hard = self.hardmax(self.edge_mask) * self.learnable_mask
        
        sparsity = self.calculate_sparsity(node_mask_hard, edge_mask_hard)
        
        if return_probs:
            node_probs_soft = F.softmax(self.node_mask, dim=-1)[:, 1]
            edge_probs_soft = F.softmax(self.edge_mask, dim=-1)[:, :, 1]
            edge_probs_soft = edge_probs_soft * self.learnable_mask
            
            return [node_mask_hard, edge_mask_hard], [node_probs_soft, edge_probs_soft], sparsity
        else:
            return [node_mask_hard, edge_mask_hard], sparsity
    
    def compute_sparsity_regularization(self, lambda_reg=0.1):
        """
        稀疏性正则化损失
        
        对节点和边使用不同权重
        """
        # 节点稀疏性
        node_probs = F.softmax(self.node_mask, dim=1)
        node_causal_probs = node_probs[:, 1]
        node_sparsity_loss = torch.mean(node_causal_probs)
        
        # 边稀疏性
        edge_probs = F.softmax(self.edge_mask, dim=-1)
        edge_causal_probs = edge_probs[:, :, 1]
        
        masked_edge_probs = edge_causal_probs * self.learnable_mask
        total_learnable_edges = torch.sum(self.learnable_mask)
        
        if total_learnable_edges > 0:
            edge_sparsity_loss = torch.sum(masked_edge_probs) / total_learnable_edges
        else:
            edge_sparsity_loss = torch.tensor(0.0, device=node_sparsity_loss.device)
        
        # 不同权重
        node_sparsity_weight = 1.5
        edge_sparsity_weight = 0.1
        
        total_sparsity_loss = (node_sparsity_weight * node_sparsity_loss + 
                               edge_sparsity_weight * edge_sparsity_loss)
        
        return lambda_reg * total_sparsity_loss