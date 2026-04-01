"""
Feature Cache Manager
预计算并缓存所有样本的DenseNet特征，避免训练时重复计算
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

from utils.checkpoint import CheckpointManager
from data.dataset import PatchDataset

logger = logging.getLogger(__name__)


class FeatureCacheManager:
    """
    特征缓存管理器
    
    功能：
        1. 一次性提取所有样本的DenseNet特征并缓存
        2. 训练时直接加载预计算的特征，避免重复计算
        3. 支持自动检测配置变化并重新提取
    
    优势：
        - 大幅提升GPU利用率（消除特征提取瓶颈）
        - 减少训练时间（特征提取只做一次）
        - 降低显存占用（不需要保留DenseNet在GPU上）
    """
    
    def __init__(
        self,
        data_dir: str,
        checkpoint_manager: CheckpointManager,
        densenet_model: nn.Module,
        device: str = 'cuda'
    ):
        """
        Args:
            data_dir: 数据目录
            checkpoint_manager: 缓存管理器
            densenet_model: 预训练的DenseNet模型
            device: 计算设备
        """
        self.data_dir = data_dir
        self.checkpoint_manager = checkpoint_manager
        self.densenet_model = densenet_model.to(device)
        self.densenet_model.eval()
        self.device = device
        
        # 冻结模型参数
        for param in self.densenet_model.parameters():
            param.requires_grad = False
        
        logger.info(f"FeatureCacheManager initialized")
    
    def get_or_compute_features(
        self,
        fold: int,
        split_seed: int,
        batch_size: int = 64,
        force_recompute: bool = False
    ) -> np.ndarray:
        """
        获取或计算特征缓存
        
        Args:
            fold: 当前fold索引
            split_seed: 数据分割种子
            batch_size: 批处理大小
            force_recompute: 是否强制重新计算
            
        Returns:
            特征数组 [N, P, feature_dim]
        """
        # 构建缓存标识符
        config_params = {
            'model': 'densenet',
            'data_dir': self.data_dir
        }
        
        identifier = self.checkpoint_manager.build_identifier(
            'features',
            config_params,
            {'fold': fold, 'seed': split_seed}
        )
        
        # 检查缓存
        cache_exists = self.checkpoint_manager.check_exists(
            'features', identifier, extension='.npy'
        )
        
        if cache_exists and not force_recompute:
            logger.info(f"[Fold {fold}] Loading cached features: {identifier}")
            return self._load_features(identifier)
        else:
            if force_recompute:
                logger.info(f"[Fold {fold}] Force recomputing features...")
            else:
                logger.info(f"[Fold {fold}] No cache found, computing features...")
            
            return self._compute_and_save(
                fold, split_seed, identifier, config_params, batch_size
            )
    
    def _load_features(self, identifier: str) -> np.ndarray:
        """从缓存加载特征"""
        features = self.checkpoint_manager.load(
            'features',
            identifier,
            extension='.npy'
        )
        
        logger.info(f"Successfully loaded features: {features.shape}")
        return features
    
    def _compute_and_save(
        self,
        fold: int,
        split_seed: int,
        identifier: str,
        config_params: Dict[str, Any],
        batch_size: int
    ) -> np.ndarray:
        """计算并保存特征"""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Computing DenseNet Features - Fold {fold}")
        logger.info(f"{'='*70}\n")
        
        # 加载数据集（不需要数据增强）
        dataset = PatchDataset(self.data_dir, transform=None)
        all_patches = dataset.all_patches  # [N, P, D, H, W]
        
        N, P = all_patches.shape[0], all_patches.shape[1]
        
        # 获取特征维度
        if isinstance(self.densenet_model, nn.DataParallel):
            feature_dim = self.densenet_model.module.feature_dim
        else:
            feature_dim = self.densenet_model.feature_dim
        
        # 初始化特征数组
        all_features = np.zeros((N, P, feature_dim), dtype=np.float32)
        
        logger.info(f"Processing {N} samples with {P} patches each...")
        logger.info(f"Batch size: {batch_size}")
        
        # 逐样本提取特征
        with torch.no_grad():
            for i in tqdm(range(N), desc="Extracting features"):
                sample_patches = all_patches[i]  # [P, D, H, W]
                
                # 转换为tensor并应用Z-Score归一化
                patches_tensor = torch.from_numpy(sample_patches).float()
                p_mean = patches_tensor.mean()
                p_std = patches_tensor.std()
                patches_tensor = (patches_tensor - p_mean) / (p_std + 1e-6)
                
                # 添加通道维度 [P, 1, D, H, W]
                patches_tensor = patches_tensor.unsqueeze(1).to(self.device)
                
                # 分批处理
                sample_features = []
                for j in range(0, P, batch_size):
                    batch = patches_tensor[j:j+batch_size]
                    features = self.densenet_model(batch)
                    sample_features.append(features.cpu().numpy())
                
                all_features[i] = np.concatenate(sample_features, axis=0)
        
        logger.info(f"Feature extraction complete: {all_features.shape}")
        
        # 保存到缓存
        logger.info(f"Saving features to cache...")
        save_path = self.checkpoint_manager.get_path(
            'features', identifier, extension='.npy'
        )
        np.save(str(save_path.with_suffix('')), all_features)
        
        # 保存元数据
        from utils.checkpoint import CacheMetadata
        from datetime import datetime
        
        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        metadata = CacheMetadata(
            identifier=identifier,
            cache_type='features',
            created_at=datetime.now().isoformat(),
            config_hash=self.checkpoint_manager.generate_config_hash(config_params),
            config_params=config_params,
            file_size_mb=round(file_size_mb, 2)
        )
        self.checkpoint_manager._save_metadata(metadata)
        
        logger.info(f"Successfully saved features: {identifier} ({file_size_mb:.2f} MB)")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Feature Computation Completed - Fold {fold}")
        logger.info(f"{'='*70}\n")
        
        return all_features