"""
因果掩码模型
学习节点和边的因果掩码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CausalMask(nn.Module):
    """
    因果掩码学习器
    
    学习哪些节点和边对预测任务是因果相关的
    """
    
    def __init__(
        self,
        num_patches: int,
        edge_matrix: torch.Tensor,
        gumble_tau: float = 1.0
    ):
        super(CausalMask, self).__init__()
        
        self.P = num_patches
        self.gumble_tau = gumble_tau
        
        # 节点掩码参数
        self.node_mask_logits = nn.Parameter(torch.ones(self.P))
        
        # 边掩码参数（只学习实际存在的边）
        self.register_buffer('learnable_mask', edge_matrix)
        num_learnable_edges = int(edge_matrix.sum().item())
        self.edge_mask_logits = nn.Parameter(torch.ones(num_learnable_edges))
    
    def forward(
        self,
        train: bool = True,
        return_probs: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        前向传播
        
        Args:
            train: 是否训练模式
            return_probs: 是否返回概率
            
        Returns:
            node_mask: 节点掩码 [P]
            edge_mask: 边掩码 [P, P]
            probs: (node_probs, edge_probs) 如果return_probs=True
        """
        if train:
            # 训练时使用Gumbel-Softmax采样
            node_mask = self._gumbel_sigmoid(self.node_mask_logits, self.gumble_tau)
            
            # 边掩码
            edge_logits_sampled = self._gumbel_sigmoid(self.edge_mask_logits, self.gumble_tau)
            edge_mask = self._reconstruct_edge_mask(edge_logits_sampled)
        else:
            # 推理时使用硬阈值
            node_mask = (torch.sigmoid(self.node_mask_logits) > 0.5).float()
            edge_probs = torch.sigmoid(self.edge_mask_logits)
            edge_mask = self._reconstruct_edge_mask((edge_probs > 0.5).float())
        
        if return_probs:
            node_probs = torch.sigmoid(self.node_mask_logits)
            edge_probs = torch.sigmoid(self.edge_mask_logits)
            return node_mask, edge_mask, (node_probs, edge_probs)
        return node_mask, edge_mask, None
    
    def _gumbel_sigmoid(self, logits: torch.Tensor, tau: float) -> torch.Tensor:
        """Gumbel-Sigmoid采样"""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        y = logits + gumbel_noise
        return torch.sigmoid(y / tau)
    
    def _reconstruct_edge_mask(self, edge_values: torch.Tensor) -> torch.Tensor:
        """从压缩的边值重构完整的边掩码矩阵"""
        edge_mask = torch.zeros(self.P, self.P, device=edge_values.device)
        
        edge_indices = torch.nonzero(self.learnable_mask, as_tuple=False)
        for idx, (i, j) in enumerate(edge_indices):
            edge_mask[i, j] = edge_values[idx]
            edge_mask[j, i] = edge_values[idx]  # 对称
        
        return edge_mask
    
    def get_mask_stats(self) -> dict:
        """获取掩码统计信息"""
        with torch.no_grad():
            node_probs = torch.sigmoid(self.node_mask_logits)
            edge_probs = torch.sigmoid(self.edge_mask_logits)
            
            node_selected = (node_probs > 0.5).sum().item()
            edge_selected = (edge_probs > 0.5).sum().item()
            
            return {
                'node_selected': node_selected,
                'node_total': self.P,
                'node_ratio': node_selected / self.P,
                'edge_selected': edge_selected,
                'edge_total': len(self.edge_mask_logits),
                'edge_ratio': edge_selected / len(self.edge_mask_logits)
            }