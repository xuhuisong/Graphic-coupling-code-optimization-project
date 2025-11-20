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
            edge_prior_mask: torch.Tensor,
            masks: Tuple[torch.Tensor, torch.Tensor],
            fusion_type: str,
            labels: torch.Tensor # [B]
        ) -> torch.Tensor:
            """
            批次内对立类别融合 (Pre-GCN Version / 输入空间融合)

            核心逻辑：
            1. 在输入空间进行特征替换 (投毒)。
            2. 使用【原始特征】计算动态边权重 (锁定通道)。
            3. 使用【因果边掩码】过滤边 (结构约束)。
            4. 将融合后的特征送入 GCN，在锁定的结构上传播。
            """
            node_mask, edge_mask = masks
            B, P, d = x.shape

            # ============================================================
            # 1. 计算替换特征 (在输入 x 上计算)
            # ============================================================
            replacement_x = torch.zeros_like(x)

            for i in range(B):
                opposite_label = 1 - labels[i]
                # 找到 Batch 内所有对立类别的样本
                opposite_mask = (labels == opposite_label)

                if opposite_mask.sum() > 0:
                    # 取对立类别原始特征 x 的平均
                    opposite_features = x[opposite_mask]
                    replacement_x[i] = opposite_features.mean(dim=0)
                else:
                    # 兜底：如果没有对立类别，用除自己外的均值
                    other_mask = torch.ones(B, dtype=torch.bool, device=x.device)
                    other_mask[i] = False
                    replacement_x[i] = x[other_mask].mean(dim=0)

            # ============================================================
            # 2. 执行替换/融合 (Feature Mixing)
            # ============================================================
            # 扩展 mask 维度以匹配特征 [P, 1] -> [1, P, 1]
            mask_intrinsic = node_mask.reshape(1, P, 1)
            mask_spurious = 1 - mask_intrinsic

            if fusion_type == "intrinsic":
                # 内在融合 (Sensitivity Test)：破坏内在因果部分
                # 预期结果：预测应该翻转/错误 (Loss 变大)
                # 操作：保留虚假背景 (x * mask_spurious)，替换内在因果 (replacement * mask_intrinsic)
                x_fused = (replacement_x * mask_intrinsic) + (x * mask_spurious)

            elif fusion_type == "spurious":
                # 虚假融合 (Invariance Test)：破坏虚假背景部分
                # 预期结果：预测应该保持不变/正确 (Loss 变小)
                # 操作：保留内在因果 (x * mask_intrinsic)，替换虚假背景 (replacement * mask_spurious)
                x_fused = (x * mask_intrinsic) + (replacement_x * mask_spurious)
            else:
                raise ValueError(f"Unknown fusion_type: {fusion_type}")

            # ============================================================
            # 3. 计算动态边 & 应用掩码 (关键步骤)
            # ============================================================
            # 关键 A: 必须用【原始 x】计算相似度权重！
            # 目的：如果原图中两节点相似（有边），我们要保持这个通道畅通，
            # 这样当我们在通道一端“投毒”（替换特征）时，毒素才能流过去，导致 Loss 增加，
            # 从而倒逼模型去学习切断这条边（即让 edge_mask -> 0）。
            edges = self.compute_dynamic_edges(x, edge_prior_mask)

            # 关键 B: 必须用【因果边掩码】进行过滤！
            # 目的：只在模型认为“因果相关”的结构上传播特征。
            # 只有当 edge_mask 为 0 时，即使通道畅通，毒素也被物理切断。
            intrinsic_edge = edges * edge_mask.unsqueeze(0)

            # ============================================================
            # 4. GCN 卷积
            # ============================================================
            # 在【旧的结构(intrinsic_edge)】上跑【新的特征(x_fused)】
            # node_mask 传给 gcn_block 主要用于归一化，保持一致性
            xs_out = self.gcn_block(x_fused, intrinsic_edge, node_mask)

            # ============================================================
            # 5. 预测
            # ============================================================
            graph = readout(xs_out)
            logits = self.mlp_causal(graph)

            return logits

        # 更新对外接口（新增 labels 参数）
    def prediction_spurious_fusion(self, x, edge_prior_mask, masks, labels):
        return self._perform_fusion_batch_level(x, edge_prior_mask, masks, fusion_type="spurious", labels=labels)

        # 2. 修正 prediction_intrinsic_fusion (如果有用到，也一并检查)
    def prediction_intrinsic_fusion(self, x, edge_prior_mask, masks, labels):
        return self._perform_fusion_batch_level(x, edge_prior_mask, masks, fusion_type="intrinsic", labels=labels)
    

    def compute_prototype_divergence(
            self, 
            x: torch.Tensor, 
            masks: Tuple[torch.Tensor, torch.Tensor], 
            labels: torch.Tensor,
            mining_ratio: float = 0.5  # 动态传入的比例
        ) -> torch.Tensor:
            """
            计算因果原型的类间分离度损失 (Top-K Hard Mining 版本)
            """
            node_mask = masks[0] # [P]
            B, P, d = x.shape

            # 1. 获取加权的因果特征 (使用卷积前的原始特征)
            masked_x = x * node_mask.view(1, P, 1)

            # 2. 分离正负样本
            labels = labels.view(-1)
            idx_0 = (labels == 0).nonzero(as_tuple=True)[0]
            idx_1 = (labels == 1).nonzero(as_tuple=True)[0]

            # 异常保护
            if len(idx_0) == 0 or len(idx_1) == 0:
                return torch.tensor(0.0, device=x.device)

            # 3. 计算类原型 (Class Prototypes)
            proto_0 = masked_x[idx_0].mean(dim=0)
            proto_1 = masked_x[idx_1].mean(dim=0)

            # 4. 计算节点级的欧氏距离
            # distance[i] 越小，说明该节点越无法区分 AD/CN
            distance = torch.norm(proto_0 - proto_1, p=2, dim=-1) # [P]

            # 5. 计算逐节点损失 (Raw Loss)
            # exp(-distance)：距离越小，Loss 越大
            scale = 1.0 
            raw_loss = node_mask * torch.exp(-scale * distance) # [P]

            # 6. Top-K Hard Mining
            # 动态确定 K：只惩罚被激活节点中最差的 mining_ratio 比例
            num_active = node_mask.sum().detach().item()
            k = int(max(1, num_active * mining_ratio))
            k = min(k, P) # 边界保护

            # 选取 Loss 最大的 K 个节点 (即距离最小的困难节点)
            topk_loss, _ = torch.topk(raw_loss, k=k)

            return topk_loss.mean()
