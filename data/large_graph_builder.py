"""
优雅的大图构建器
支持灵活的负样本采样和大图组装
"""

import torch
import numpy as np
from typing import Tuple, Optional
from torch.utils.data import Dataset


class LargeGraphBuilder:
    """
    大图构建器
    
    功能：
        1. 从batch中为每个锚点样本采样N个负样本
        2. 组装成大图：原始样本 + N个负样本
        3. 构建大图的边矩阵
    
    设计原则：
        - 灵活配置负样本数量
        - 支持不同的采样策略
        - 优雅的边矩阵组装
    
    Args:
        num_neg_samples: 负样本数量（默认4）
        sampling_strategy: 采样策略 ('opposite_label', 'random', 'hard')
        random_seed: 随机种子
    """
    
    def __init__(
        self,
        num_neg_samples: int = 4,
        sampling_strategy: str = 'opposite_label',
        random_seed: int = 42
    ):
        self.num_neg_samples = num_neg_samples
        self.sampling_strategy = sampling_strategy
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
    
    def build_large_graph(
        self,
        batch_data: torch.Tensor,
        batch_labels: torch.Tensor,
        base_edge: torch.Tensor,
        all_data: Optional[np.ndarray] = None,
        all_labels: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建大图
        
        Args:
            batch_data: 当前batch的数据 [B, P, C, D, H, W]
            batch_labels: 当前batch的标签 [B]
            base_edge: 基础边矩阵 [P, P]（已归一化）
            all_data: 全部数据（用于采样）[N, P, D, H, W]
            all_labels: 全部标签（用于采样）[N]
            
        Returns:
            large_data: 大图数据 [B, (N+1)*P, C, D, H, W]
            large_edge: 大图边矩阵 [B, (N+1)*P, (N+1)*P]
        """
        B, P = batch_data.shape[0], batch_data.shape[1]
        N = self.num_neg_samples
        device = batch_data.device
        
        # 1. 采样负样本
        neg_indices_list = self._sample_negative_samples(
            batch_labels,
            all_labels if all_labels is not None else batch_labels.cpu().numpy()
        )
        
        # 2. 组装大图节点特征
        large_data = self._assemble_large_graph_features(
            batch_data,
            neg_indices_list,
            all_data if all_data is not None else batch_data.cpu().numpy()
        )
        
        # 3. 构建大图边矩阵
        large_edge = self._build_large_graph_edges(
            base_edge,
            B,
            P,
            N,
            device
        )
        
        return large_data, large_edge
    
    def _sample_negative_samples(
        self,
        batch_labels: torch.Tensor,
        all_labels: np.ndarray
    ) -> list:
        """
        为每个样本采样负样本
        
        Args:
            batch_labels: 当前batch的标签 [B]
            all_labels: 所有标签 [N]
            
        Returns:
            neg_indices_list: 每个样本的负样本索引列表
        """
        batch_labels_np = batch_labels.cpu().numpy()
        neg_indices_list = []
        
        for label in batch_labels_np:
            if self.sampling_strategy == 'opposite_label':
                # 采样对立标签的样本
                opposite_indices = np.where(all_labels == 1 - label)[0]
                
                if len(opposite_indices) < self.num_neg_samples:
                    # 不够就重复采样
                    neg_indices = self.rng.choice(
                        opposite_indices,
                        self.num_neg_samples,
                        replace=True
                    )
                else:
                    # 够就不重复采样
                    neg_indices = self.rng.choice(
                        opposite_indices,
                        self.num_neg_samples,
                        replace=False
                    )
            
            elif self.sampling_strategy == 'random':
                # 随机采样（不考虑标签）
                neg_indices = self.rng.choice(
                    len(all_labels),
                    self.num_neg_samples,
                    replace=False
                )
            
            elif self.sampling_strategy == 'hard':
                # 硬样本挖掘（需要额外实现）
                # TODO: 基于特征相似度采样最难的负样本
                raise NotImplementedError("Hard negative mining not implemented yet")
            
            neg_indices_list.append(neg_indices)
        
        return neg_indices_list
    
    def _assemble_large_graph_features(
        self,
        batch_data: torch.Tensor,
        neg_indices_list: list,
        all_data: np.ndarray
    ) -> torch.Tensor:
        """组装大图的节点特征"""
        B, P = batch_data.shape[0], batch_data.shape[1]
        N = self.num_neg_samples
        device = batch_data.device

        # batch_data shape: [B, P, 1, D, H, W]
        # 直接使用相同的shape
        data_shape = batch_data.shape[2:]  # [1, D, H, W]

        # 初始化大图数据
        large_data = torch.zeros(
            (B, (N + 1) * P, *data_shape),
            dtype=batch_data.dtype,
            device=device
        )

        # 填充数据
        for i in range(B):
            # 第一块：原始锚点样本
            large_data[i, :P] = batch_data[i]

            # 后N块：负样本
            for neg_idx, global_idx in enumerate(neg_indices_list[i]):
                start = (neg_idx + 1) * P
                end = (neg_idx + 2) * P
                # all_data shape: [N, P, D, H, W] - 没有通道维度
                neg_sample = torch.from_numpy(
                    all_data[global_idx].copy()
                ).to(device).float()

                # 添加通道维度: [P, D, H, W] -> [P, 1, D, H, W]
                neg_sample = neg_sample.unsqueeze(1)

                large_data[i, start:end] = neg_sample

        return large_data
    
    def _build_large_graph_edges(
        self,
        base_edge: torch.Tensor,
        batch_size: int,
        P: int,
        N: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        构建大图的边矩阵
        
        策略：将基础边矩阵复制到对角线的(N+1)个块中
        
        Args:
            base_edge: 基础边矩阵 [P, P]
            batch_size: batch大小
            P: 单个样本的patch数量
            N: 负样本数量
            device: 设备
            
        Returns:
            large_edge: 大图边矩阵 [B, (N+1)*P, (N+1)*P]
        """
        large_P = (N + 1) * P
        
        # 初始化大图边矩阵
        large_edge = torch.zeros(
            (batch_size, large_P, large_P),
            dtype=base_edge.dtype,
            device=device
        )
        
        # 将base_edge作为"积木"填充到对角线的(N+1)个块中
        for sample_idx in range(N + 1):
            start = sample_idx * P
            end = (sample_idx + 1) * P
            
            # 为batch中的每个样本填充
            for b in range(batch_size):
                large_edge[b, start:end, start:end] = base_edge
        
        return large_edge
    
    def build_cross_sample_edges(
        self,
        batch_size: int,
        P: int,
        node_mask: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        构建跨样本边矩阵（用于因果推理）
        
        连接原始样本的节点与负样本的对应节点
        
        Args:
            batch_size: batch大小
            P: 单个样本的patch数量
            node_mask: 节点掩码 [P]
            device: 设备
            
        Returns:
            cross_edges: 跨样本边矩阵 [B, (N+1)*P, (N+1)*P]
        """
        N = self.num_neg_samples
        large_P = (N + 1) * P
        
        # 初始化跨样本边矩阵
        cross_edges = torch.zeros(
            (batch_size, large_P, large_P),
            dtype=torch.float32,
            device=device
        )
        
        # 边权重：平均分配到N个负样本
        edge_weight = 1.0 / N
        
        # 连接原始样本与负样本的对应节点
        for neg_idx in range(1, N + 1):
            for p in range(P):
                orig_idx = p
                neg_idx_p = neg_idx * P + p
                
                # 权重取决于节点掩码（非因果节点有更强的连接）
                weight = edge_weight * (1 - node_mask[p])
                
                # 双向连接
                cross_edges[:, orig_idx, neg_idx_p] = weight
                cross_edges[:, neg_idx_p, orig_idx] = weight
        
        # 对角线：自环权重取决于节点掩码
        for sample_idx in range(N + 1):
            for p in range(P):
                diag_idx = sample_idx * P + p
                cross_edges[:, diag_idx, diag_idx] = node_mask[p]
        
        return cross_edges


class LargeGraphDataset(Dataset):
    """
    支持大图构建的数据集包装器
    
    在原始数据集基础上，动态构建大图
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        base_edge: torch.Tensor,
        num_neg_samples: int = 4,
        sampling_strategy: str = 'opposite_label'
    ):
        self.base_dataset = base_dataset
        self.base_edge = base_edge
        self.builder = LargeGraphBuilder(
            num_neg_samples=num_neg_samples,
            sampling_strategy=sampling_strategy
        )
        
        # 预加载所有数据（用于采样）
        self.all_data = base_dataset.all_patches
        self.all_labels = np.array(base_dataset.labels)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # 返回原始数据，大图在collate_fn中构建
        return self.base_dataset[idx]
    
    def collate_fn_large_graph(self, batch):
        """
        自定义collate函数，构建大图
        
        Args:
            batch: [(data, subject_id, label), ...]
            
        Returns:
            large_data: 大图数据
            large_edge: 大图边矩阵
            subject_ids: 主题ID列表
            labels: 标签tensor
        """
        data_list, subject_ids, labels = zip(*batch)
        
        # 转换为tensor
        batch_data = torch.stack(data_list)
        batch_labels = torch.tensor(labels, dtype=torch.long)
        
        # 构建大图
        large_data, large_edge = self.builder.build_large_graph(
            batch_data,
            batch_labels,
            self.base_edge,
            self.all_data,
            self.all_labels
        )
        
        return large_data, large_edge, list(subject_ids), batch_labels