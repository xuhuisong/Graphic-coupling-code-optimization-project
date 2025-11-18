"""
Graph Builder - 图边构建管理器

基于DenseNet提取的特征，自动构建图的边结构
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm

from utils.checkpoint import CheckpointManager
from core.densenet_manager import DenseNetManager
from data.dataset import PatchDataset


logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    图边构建管理器
    
    功能：
        1. 基于DenseNet特征计算patch间相似度
        2. 统计边在不同患者中的出现频率
        3. 筛选高频边，生成最终的边矩阵
        4. 智能缓存，避免重复计算
    
    设计原则：
        - 边矩阵使用全部数据构建，不区分fold
        - 边矩阵对所有fold通用，只计算一次
        - 支持多种相似度计算方法
    
    Args:
        data_dir: 数据目录
        checkpoint_manager: 缓存管理器
        densenet_manager: DenseNet管理器（用于提取特征）
        config: 图构建配置
    """
    
    def __init__(
        self,
        data_dir: str,
        checkpoint_manager: CheckpointManager,
        densenet_manager: DenseNetManager,
        config: Optional[Dict[str, Any]] = None
    ):
        self.data_dir = data_dir
        self.checkpoint_manager = checkpoint_manager
        self.densenet_manager = densenet_manager
        
        # 默认配置
        self.config = {
            'similarity_threshold': 0.7,      # patch间相似度阈值
            'frequency_threshold': 0.3,       # 边在患者中的最小出现频率
            'similarity_metric': 'cosine',    # 相似度计算方式: 'cosine', 'euclidean'
            'batch_size': 64,                 # 相似度计算批次大小
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'use_all_data': True,             # 是否使用全部数据构建边（推荐True）
        }
        
        # 更新用户配置
        if config:
            self.config.update(config)
        
        logger.info(f"GraphBuilder initialized with config: {self.config}")
    
    def get_edge_prior_mask(
        self,
        split_seed: int,
        fold_for_feature_extraction: int = 0,
        force_rebuild: bool = False
    ) -> np.ndarray:
        """
        获取边的先验候选集（learnable mask）

        返回一个二值化的mask [P, P]，标识哪些位置允许学习边
        - 1 = 该位置允许动态计算边
        - 0 = 该位置禁止有边

        注意：这不是最终的边矩阵，而是动态边构建的先验约束
        """
        # 构建缓存标识符（不包含fold信息）
        config_params = {
            'similarity_threshold': self.config['similarity_threshold'],
            'frequency_threshold': self.config['frequency_threshold'],
            'similarity_metric': self.config['similarity_metric'],
            'use_all_data': self.config['use_all_data']
        }
        
        identifier = self.checkpoint_manager.build_identifier(
            'edges',
            config_params,
            {'seed': split_seed}
        )
        
        # 检查缓存
        cache_exists = self.checkpoint_manager.check_exists('edges', identifier, extension='.npy')
        
        if cache_exists and not force_rebuild:
            logger.info(f"Loading cached edge_prior_mask: {identifier}")
            return self._load_edge_prior_mask(identifier)
        else:
            if force_rebuild:
                logger.info(f"Force rebuilding edge_prior_mask...")
            else:
                logger.info(f"No cache found, building edge_prior_mask...")
            
            return self._build_and_save(
                split_seed,
                fold_for_feature_extraction,
                identifier,
                config_params
            )
    
    def _load_edge_prior_mask(self, identifier: str) -> np.ndarray:
        """从缓存加载边矩阵"""
        edge_prior_mask = self.checkpoint_manager.load(
            'edges',
            identifier,
            extension='.npy'
        )
        
        logger.info(f"Successfully loaded edge_prior_mask: {edge_prior_mask.shape}")
        logger.info(f"Number of edges: {np.sum(edge_prior_mask)}")
        
        return edge_prior_mask
    
    def _build_and_save(
        self,
        split_seed: int,
        fold_for_feature_extraction: int,
        identifier: str,
        config_params: Dict[str, Any]
    ) -> np.ndarray:
        """构建并保存边矩阵"""

        logger.info(f"\n{'='*70}")
        logger.info(f"Starting edge_prior_mask Construction")
        logger.info(f"{'='*70}\n")

        # 1-4步保持不变...
        densenet_model = self.densenet_manager.get_pretrained_model(
            fold=fold_for_feature_extraction,
            split_seed=split_seed
        )
        all_features = self._extract_all_features(densenet_model)
        patient_edges = self._compute_patient_edges(all_features)
        edge_prior_mask = self._compute_edge_frequencies(patient_edges)

        # 5. 保存到缓存 - 修复版本
        logger.info(f"Step 5: Saving edge_prior_mask to cache...")

        # 直接使用numpy保存
        save_path = self.checkpoint_manager.get_path('edges', identifier, extension='.npy')
        np.save(str(save_path.with_suffix('')), edge_prior_mask)

        # 手动保存元数据
        from utils.checkpoint import CacheMetadata
        from datetime import datetime

        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        metadata = CacheMetadata(
            identifier=identifier,
            cache_type='edges',
            created_at=datetime.now().isoformat(),
            config_hash=self.checkpoint_manager.generate_config_hash(config_params),
            config_params=config_params,
            file_size_mb=round(file_size_mb, 2)
        )
        self.checkpoint_manager._save_metadata(metadata)

        logger.info(f"Successfully saved edge_prior_mask: {identifier} ({file_size_mb:.2f} MB)")

        logger.info(f"\n{'='*70}")
        logger.info(f"edge_prior_mask Construction Completed")
        logger.info(f"Final edge_prior_mask: {edge_prior_mask.shape}")
        logger.info(f"Number of edges: {np.sum(edge_prior_mask)}")
        logger.info(f"Edge density: {np.sum(edge_prior_mask) / (edge_prior_mask.shape[0] ** 2):.4f}")
        logger.info(f"{'='*70}\n")

        return edge_prior_mask
    
    def _extract_all_features(self, model: torch.nn.Module) -> np.ndarray:
        """
        提取所有样本的特征
        
        Args:
            model: 预训练的DenseNet模型
            
        Returns:
            特征数组 [N, P, feature_dim]
        """
        # 加载数据集
        dataset = PatchDataset(self.data_dir)
        data = dataset.all_patches  # [N, P, D, H, W]
        
        logger.info(f"Loaded dataset: {data.shape}")
        logger.info(f"  Number of samples: {data.shape[0]}")
        logger.info(f"  Patches per sample: {data.shape[1]}")
        
        # 使用DenseNetManager的特征提取功能
        features = self.densenet_manager.extract_features(
            model,
            np.array(data),  # 确保是numpy array而非memmap
            batch_size=self.config['batch_size'],
            device=self.config['device']
        )
        
        logger.info(f"Feature extraction complete: {features.shape}")
        
        return features
    
    def _compute_patient_edges(self, all_features: np.ndarray) -> np.ndarray:
        """
        计算每个患者内部的边

        [修改] 使用 TOP-K 策略替代固定阈值

        Args:
            all_features: 所有特征 [N, P, feature_dim]

        Returns:
            患者边矩阵 [N, P, P]
        """
        num_patients, num_patches, feature_dim = all_features.shape

        # 初始化结果
        patient_edges = np.zeros((num_patients, num_patches, num_patches), dtype=np.uint8)

        device = torch.device(self.config['device'])

        # 【修改】使用 TOP-K 替代阈值
        k_neighbors = self.config.get('k_neighbors', 20)  # 每个节点保留TOP-20邻居

        logger.info(f"Computing edges for {num_patients} patients...")
        logger.info(f"Using TOP-K strategy with K={k_neighbors}")

        # 逐患者计算
        for i in tqdm(range(num_patients), desc="Computing patient edges"):
            patient_features = torch.FloatTensor(all_features[i]).to(device)  # [P, feature_dim]

            # 计算相似度矩阵
            if self.config['similarity_metric'] == 'cosine':
                similarity_matrix = self._compute_cosine_similarity(patient_features)
            elif self.config['similarity_metric'] == 'euclidean':
                similarity_matrix = self._compute_euclidean_similarity(patient_features)
            else:
                raise ValueError(f"Unknown similarity metric: {self.config['similarity_metric']}")

            # 【关键修改】TOP-K 选择
            # 对每一行（每个节点），找到最相似的K个邻居
            topk_values, topk_indices = torch.topk(similarity_matrix, k=k_neighbors+1, dim=1)
            # +1 是因为自己和自己相似度最高，需要排除

            # 构建稀疏边矩阵
            edges = torch.zeros_like(similarity_matrix, dtype=torch.uint8)
            for node_idx in range(num_patches):
                neighbors = topk_indices[node_idx, 1:]
                # 【新增】同时满足TOP-K和阈值
                for neighbor in neighbors:
                    if similarity_matrix[node_idx, neighbor] >= self.config['similarity_threshold']:
                        edges[node_idx, neighbor] = 1

            # 互选
            edges = edges & edges.t()
            edges.fill_diagonal_(0)

            patient_edges[i] = edges.cpu().numpy()

        # 统计信息
        avg_edges_per_patient = np.mean([np.sum(patient_edges[i]) for i in range(num_patients)])
        logger.info(f"Average edges per patient: {avg_edges_per_patient:.1f}")
        logger.info(f"Theoretical max (K={k_neighbors}, undirected): {num_patches * k_neighbors}")

        return patient_edges
    
    def _compute_cosine_similarity(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算余弦相似度矩阵
        
        Args:
            features: [P, feature_dim]
            
        Returns:
            similarity_matrix: [P, P]，值范围[0, 1]
        """
        # L2归一化
        features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
        
        # 余弦相似度 = 归一化向量的点积
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        
        # 将[-1, 1]映射到[0, 1]
        similarity_matrix = (similarity_matrix + 1) / 2
        
        return similarity_matrix
    
    def _compute_euclidean_similarity(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算基于欧氏距离的相似度矩阵
        
        Args:
            features: [P, feature_dim]
            
        Returns:
            similarity_matrix: [P, P]，值范围[0, 1]
        """
        # 计算欧氏距离矩阵
        diff = features.unsqueeze(0) - features.unsqueeze(1)  # [P, P, feature_dim]
        distances = torch.sqrt(torch.sum(diff ** 2, dim=2))   # [P, P]
        
        # 转换为相似度（距离越小，相似度越高）
        # 使用高斯核: similarity = exp(-distance^2 / (2 * sigma^2))
        sigma = torch.std(distances)
        if sigma > 0:
            similarity_matrix = torch.exp(-distances ** 2 / (2 * sigma ** 2))
        else:
            # 如果标准差为0，说明所有特征相同
            similarity_matrix = torch.ones_like(distances)
        
        return similarity_matrix
    
    def _compute_edge_frequencies(self, patient_edges: np.ndarray) -> np.ndarray:
        """
        统计边的出现频率并筛选
        
        Args:
            patient_edges: [N, P, P]，每个患者的边矩阵
            
        Returns:
            edge_prior_mask: [P, P]，最终的边矩阵
        """
        num_patients, num_patches, _ = patient_edges.shape
        
        # 统计每条边在多少个患者中出现
        edge_counts = np.sum(patient_edges, axis=0)  # [P, P]
        
        # 计算频率
        edge_frequencies = edge_counts / num_patients  # [P, P]
        
        # 应用频率阈值
        frequency_threshold = self.config['frequency_threshold']
        edge_prior_mask = (edge_frequencies >= frequency_threshold).astype(np.uint8)
        
        # 确保对称性（无向图）
        edge_prior_mask = np.maximum(edge_prior_mask, edge_prior_mask.T)
        
        # 移除自环
        np.fill_diagonal(edge_prior_mask, 0)
        
        # 统计信息
        logger.info(f"Edge frequency statistics:")
        logger.info(f"  Frequency threshold: {frequency_threshold}")
        logger.info(f"  Max edge frequency: {edge_frequencies.max():.3f}")
        logger.info(f"  Mean edge frequency: {edge_frequencies.mean():.3f}")
        logger.info(f"  Edges above threshold: {np.sum(edge_prior_mask)}")
        
        return edge_prior_mask
    