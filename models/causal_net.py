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
    
    def forward(self, x: torch.Tensor, edge1: torch.Tensor) -> torch.Tensor:
        """
        只处理子图内部的卷积
        
        Args:
            x: 节点特征 [B, P, d]
            edge1: 子图内部边 [B, P, P]
        """
        # ========== 第一层 ==========
        edge1_norm = self._normalize_adj_batch(edge1)
        x = self.conv1(x, edge1_norm)
        # 移除了 F.relu 和 bn
        x = self.dropout1(x)
        
        # ========== 第二层 ==========
        # 注意：第二层卷积仍然使用第一层归一化后的邻接矩阵
        x = self.conv2(x, edge1_norm) 
        # 移除了 F.relu 和 bn
        x = self.dropout2(x)
        
        return x
    
    @staticmethod
    def _normalize_adj_batch(adj: torch.Tensor) -> torch.Tensor:
        """
        对称归一化邻接矩阵 (A + I)
        (这个函数保持不变)
        """
        batch_size, num_nodes = adj.shape[0], adj.shape[1]
        adj_clone = adj.clone()

        # 1. 强制清空对角线（移除任何已有的自环）
        diag_indices = torch.arange(num_nodes, device=adj.device)
        adj_clone[:, diag_indices, diag_indices] = 0
        
        # 2. 添加自环（统一权重为1）
        identity = torch.eye(num_nodes, device=adj.device).unsqueeze(0)
        adj_with_self_loops = adj_clone + identity

        # 3. 计算度
        degree = adj_with_self_loops.sum(dim=2) # [B, P]

        # 4. D^{-1/2}
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0 # 处理孤立节点

        # 5. 对称归一化: D^{-1/2} * A * D^{-1/2}
        adj_normalized = (degree_inv_sqrt.unsqueeze(2) * adj_with_self_loops * degree_inv_sqrt.unsqueeze(1))

        return adj_normalized
    
    @staticmethod
    def _normalize_adj_batch(adj: torch.Tensor) -> torch.Tensor:
        """
        对称归一化邻接矩阵 (A + I)
        """
        batch_size, num_nodes = adj.shape[0], adj.shape[1]
        adj_clone = adj.clone()

        # 1. 强制清空对角线（移除任何已有的自环）
        diag_indices = torch.arange(num_nodes, device=adj.device)
        adj_clone[:, diag_indices, diag_indices] = 0
        
        # 2. 添加自环（统一权重为1）
        identity = torch.eye(num_nodes, device=adj.device).unsqueeze(0)
        adj_with_self_loops = adj_clone + identity

        # 3. 计算度
        degree = adj_with_self_loops.sum(dim=2) # [B, P]

        # 4. D^{-1/2}
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0 # 处理孤立节点

        # 5. 对称归一化: D^{-1/2} * A * D^{-1/2}
        adj_normalized = (degree_inv_sqrt.unsqueeze(2) * adj_with_self_loops * degree_inv_sqrt.unsqueeze(1))

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
        edge_prior_mask: torch.Tensor,
        is_large_graph: bool = False
    ) -> torch.Tensor:
        """
        基于输入特征动态计算个性化边矩阵 (未改动)
        """
        B, total_P, d = x.shape
        P = edge_prior_mask.shape[0]
        
        if is_large_graph:
            num_subgraphs = total_P // P
            all_edges = []
            for i in range(num_subgraphs):
                start = i * P
                end = (i + 1) * P
                sub_x = x[:, start:end, :]
                
                sub_x_norm = F.normalize(sub_x, p=2, dim=2)
                sub_similarity = torch.bmm(sub_x_norm, sub_x_norm.transpose(1, 2))
                sub_similarity = (sub_similarity + 1) / 2
                
                mask_expanded = edge_prior_mask.unsqueeze(0).expand(B, -1, -1)
                sub_edges = sub_similarity * mask_expanded
                
                all_edges.append(sub_edges)
            
            edges = self._assemble_block_diagonal(all_edges)
            
        else:
            x_norm = F.normalize(x, p=2, dim=2)
            similarity = torch.bmm(x_norm, x_norm.transpose(1, 2))
            similarity = (similarity + 1) / 2
            
            mask_expanded = edge_prior_mask.unsqueeze(0).expand(B, -1, -1)
            edges = similarity * mask_expanded
        
        # 移除自环（对角线置0）
        num_nodes = edges.shape[1]
        identity = torch.eye(num_nodes, device=edges.device).unsqueeze(0).expand(B, -1, -1)
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
    
    def prediction_whole(
        self,
        x_new: torch.Tensor,
        edge_prior_mask: torch.Tensor,
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        全图预测 - 使用动态计算的边矩阵 (已更新)
        """
        edge = self.compute_dynamic_edges(x_new, edge_prior_mask, is_large_graph)
        
        # 【更新】使用 GCNBlock
        xs = self.gcn_block(x_new, edge)
        
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
        因果不变性预测（内在路径）(已更新)
        """
        node_mask, edge_mask = masks
        
        edge = self.compute_dynamic_edges(x_new, edge_prior_mask, is_large_graph)
        
        if is_large_graph:
            P = self.num_patches
            x_orig = x_new[:, :P, :].clone()
            edge_orig = edge[:, :P, :P].clone()
            
            x_masked = x_orig * node_mask.unsqueeze(0).unsqueeze(-1)
            inter_node_adj = edge_orig * edge_mask.unsqueeze(0)
            
            # 【更新】使用 GCNBlock
            xs = self.gcn_block(x_masked, inter_node_adj)
        else:
            x_masked = x_new * node_mask.unsqueeze(0).unsqueeze(-1)
            inter_node_adj = edge * edge_mask.unsqueeze(0)
            
            # 【更新】使用 GCNBlock
            xs = self.gcn_block(x_masked, inter_node_adj)
        
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
        因果变异性预测（虚假路径）(已更新)
        """
        node_mask, edge_mask = masks
        
        edge = self.compute_dynamic_edges(x_new, edge_prior_mask, is_large_graph)
        
        causal_mask = node_mask.unsqueeze(0).unsqueeze(-1)
        x_perturbed = x_new * (1 - causal_mask)
        
        causal_edge_mask = edge_mask.unsqueeze(0)
        edge_perturbed = edge * (1 - causal_edge_mask)
        
        # 【更新】使用 GCNBlock
        xs = self.gcn_block(x_perturbed, edge_perturbed)
        
        graph = readout(xs)
        return self.mlp_causal(graph)

    # -----------------------------------------------------------------
    # 【全新】替换式融合 (Replacement Fusion) 逻辑
    # -----------------------------------------------------------------

    def _perform_fusion_replacement(
        self,
        x: torch.Tensor,
        edge_prior_mask: torch.Tensor,
        masks: Tuple[torch.Tensor, torch.Tensor],
        fusion_type: str, # "intrinsic" 或 "spurious"
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        【新】执行替换式融合的私有辅助函数
        
        流程:
        1. 对大图中的所有子图（Anchor + Negatives）执行内部GCN（阶段一）。
        2. 聚合所有负样本（Negatives）的特征，得到 "替换源" 特征。
        3. 根据 fusion_type，有条件地用 "替换源" 替换 Anchor 的 "因果" 或 "虚假" 节点。
        4. 对被替换后的 Anchor 图进行预测。
        """
        node_mask, edge_mask = masks

        if not is_large_graph:
            raise ValueError("替换式融合 (Replacement Fusion) 需要大图模式")

        B, large_P, d = x.shape
        P = self.num_patches

        if large_P % P != 0:
            raise ValueError(f"大图节点数 ({large_P}) 不是 P ({P}) 的整数倍")
        
        num_subgraphs = large_P // P
        num_neg_samples = num_subgraphs - 1
        
        if num_neg_samples == 0:
             raise ValueError("替换式融合需要至少一个负样本")

        # ✅ 第1步：动态计算每个子图的 "因果" 边
        full_dynamic_edges = self.compute_dynamic_edges(x, edge_prior_mask, is_large_graph=True)
        
        intrinsic_edge_mask_large = torch.zeros(large_P, large_P, device=x.device)
        for i in range(num_subgraphs):
            start = i * P
            end = (i + 1) * P
            intrinsic_edge_mask_large[start:end, start:end] = edge_mask

        internal_edges = full_dynamic_edges * intrinsic_edge_mask_large.unsqueeze(0)

        # ✅ 第2步：执行阶段一（所有子图的内部卷积）
        # 使用 gcn_block，它只执行内部卷积
        xs_stage1 = self.gcn_block(x, internal_edges)
        # xs_stage1 形状 [B, (N+1)*P, d]

        # ✅ 第3步：准备替换数据
        
        # 1. 分离 Anchor 和 Negatives
        x_anchor_s1 = xs_stage1[:, :P, :]  # Anchor特征 [B, P, d]
        x_negs_s1 = xs_stage1[:, P:, :]  # 所有负样本特征 [B, N*P, d]
        
        # 2. 重塑并聚合负样本
        x_negs_s1_reshaped = x_negs_s1.reshape(B, num_neg_samples, P, d) # [B, N, P, d]
        
        # 3. 计算用于替换的"聚合信息" (取平均)
        x_replacement_info = torch.mean(x_negs_s1_reshaped, dim=1) # [B, P, d]

        # ✅ 第4步：执行“条件替换”
        
        # 准备掩码
        mask_intrinsic = node_mask.reshape(1, P, 1).to(x.device) # 因果节点 [1, P, 1]
        mask_spurious = (1 - mask_intrinsic)                     # 虚假节点 [1, P, 1]

        if fusion_type == "intrinsic":
            # 内在融合: Anchor的因果部分被替换，虚假部分保留
            x_anchor_new = (x_replacement_info * mask_intrinsic) + \
                           (x_anchor_s1 * mask_spurious)
        
        elif fusion_type == "spurious":
            # 虚假融合: Anchor的虚假部分被替换，因果部分保留
            x_anchor_new = (x_anchor_s1 * mask_intrinsic) + \
                           (x_replacement_info * mask_spurious)
        else:
            raise ValueError(f"未知的 fusion_type: {fusion_type}")
        
        # ✅ 第5步：使用被替换后的Anchor图进行预测
        graph = readout(x_anchor_new) # x_anchor_new 是 [B, P, d]
        logits = self.mlp_causal(graph)
        
        return logits


    def prediction_spurious_fusion(
        self,
        x: torch.Tensor,
        edge_prior_mask: torch.Tensor,
        masks: Tuple[torch.Tensor, torch.Tensor],
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        【重构】
        Spurious Fusion Graph (替换式) - 测试Invariance
        
        使用负样本的 *虚假节点* 特征，替换Anchor的 *虚假节点* 特征。
        """
        return self._perform_fusion_replacement(
            x,
            edge_prior_mask,
            masks,
            fusion_type="spurious",
            is_large_graph=is_large_graph
        )

    def prediction_intrinsic_fusion(
        self,
        x: torch.Tensor,
        edge_prior_mask: torch.Tensor,
        masks: Tuple[torch.Tensor, torch.Tensor],
        is_large_graph: bool = True
    ) -> torch.Tensor:
        """
        【重构】
        Intrinsic Fusion Graph (替换式) - 测试Sensitivity
        
        使用负样本的 *因果节点* 特征，替换Anchor的 *因果节点* 特征。
        """
        return self._perform_fusion_replacement(
            x,
            edge_prior_mask,
            masks,
            fusion_type="intrinsic",
            is_large_graph=is_large_graph
        )