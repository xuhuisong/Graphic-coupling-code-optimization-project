"""
DenseNet Pretraining Manager
DenseNet预训练自动化管理器
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from utils.checkpoint import CheckpointManager
from models.densenet import LightDenseNet3D, EndToEndDenseNet
from data.dataset import PatchDataset, collate_fn, get_fold_splits


logger = logging.getLogger(__name__)


class DenseNetManager:
    """
    DenseNet预训练管理器
    
    功能：
        1. 自动检测缓存的预训练模型
        2. 按需触发训练（仅在缓存不存在时）
        3. 提供统一的模型加载接口
        4. 确保与主训练流程的数据分割一致性
    
    Args:
        data_dir: 数据目录
        checkpoint_manager: 缓存管理器实例
        config: 训练配置字典
    """
    
    def __init__(
        self,
        data_dir: str,
        checkpoint_manager: CheckpointManager,
        config: Optional[Dict[str, Any]] = None
    ):
        self.data_dir = data_dir
        self.checkpoint_manager = checkpoint_manager
        
        # 默认配置
        self.config = {
            'growth_rate': 8,
            'num_init_features': 24,
            'num_epochs': 50,
            'stage1_epochs': 49,  # 阶段1轮数（无mask）
            'mask_ratio': 0.3,     # 阶段2的mask比例
            'batch_size': 16,
            'learning_rate_stage1': 0.0001,
            'learning_rate_stage2': 0.0004,
            'weight_decay_stage1': 1e-4,
            'weight_decay_stage2': 1e-5,
            'num_workers': 4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # 更新用户提供的配置
        if config:
            self.config.update(config)
        
        logger.info(f"DenseNetManager initialized with data_dir: {data_dir}")
        logger.info(f"Training config: {self.config}")
    
    def get_pretrained_model(
        self,
        fold: int,
        split_seed: int,
        force_retrain: bool = False
    ) -> LightDenseNet3D:
        """
        获取预训练的DenseNet模型
        
        自动检测缓存，如果不存在则触发训练
        
        Args:
            fold: fold索引
            split_seed: 数据分割种子
            force_retrain: 是否强制重新训练（忽略缓存）
            
        Returns:
            预训练好的LightDenseNet3D模型
        """
        # 构建缓存标识符
        config_params = {
            'growth_rate': self.config['growth_rate'],
            'num_init_features': self.config['num_init_features'],
            'num_epochs': self.config['num_epochs'],
            'stage1_epochs': self.config['stage1_epochs'],
            'mask_ratio': self.config['mask_ratio']
        }
        
        identifier = self.checkpoint_manager.build_identifier(
            'densenet',
            config_params,
            {'fold': fold, 'seed': split_seed}
        )
        
        # 检查缓存是否存在
        cache_exists = self.checkpoint_manager.check_exists('densenet', identifier)
        
        if cache_exists and not force_retrain:
            logger.info(f"[Fold {fold}] Loading cached pretrained DenseNet: {identifier}")
            return self._load_pretrained_model(identifier)
        else:
            if force_retrain:
                logger.info(f"[Fold {fold}] Force retraining DenseNet...")
            else:
                logger.info(f"[Fold {fold}] No cache found, starting training...")
            
            return self._train_and_save(fold, split_seed, identifier, config_params)
    
    def _load_pretrained_model(self, identifier: str) -> LightDenseNet3D:
        """从缓存加载预训练模型"""
        checkpoint = self.checkpoint_manager.load(
            'densenet',
            identifier,
            map_location='cpu'
        )
        
        # 创建模型并加载权重
        model = LightDenseNet3D(
            growth_rate=self.config['growth_rate'],
            num_init_features=self.config['num_init_features']
        )
        
        # checkpoint可能是完整的字典或只是state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}, "
                       f"val_acc: {checkpoint.get('val_acc', 'unknown'):.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        logger.info(f"Successfully loaded pretrained DenseNet")
        return model
    
    def _train_and_save(
        self,
        fold: int,
        split_seed: int,
        identifier: str,
        config_params: Dict[str, Any]
    ) -> LightDenseNet3D:
        """训练并保存模型"""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting DenseNet Pretraining - Fold {fold}")
        logger.info(f"{'='*70}\n")
        
        # 1. 准备数据
        train_loader, val_loader = self._prepare_dataloaders(fold, split_seed)
        
        # 2. 创建模型
        device = torch.device(self.config['device'])
        
        feature_extractor = LightDenseNet3D(
            growth_rate=self.config['growth_rate'],
            num_init_features=self.config['num_init_features']
        )
        
        # 获取patch数量
        dataset = PatchDataset(self.data_dir)
        num_patches = dataset.get_num_patches()
        
        model = EndToEndDenseNet(feature_extractor, num_patches).to(device)
        
        # 多GPU支持
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)
        
        # 3. 执行两阶段训练
        best_model_path = self._two_stage_training(
            model, train_loader, val_loader, device, fold
        )
        
        # 4. 保存最佳模型到缓存
        logger.info(f"Saving best model to cache...")
        
        # 加载最佳模型
        best_checkpoint = torch.load(best_model_path, map_location='cpu')
        
        # 提取feature_extractor的权重
        if isinstance(model, nn.DataParallel):
            feature_extractor_state = model.module.feature_extractor.state_dict()
        else:
            feature_extractor_state = model.feature_extractor.state_dict()
        
        # 保存到缓存系统
        save_data = {
            'model_state_dict': feature_extractor_state,
            'epoch': best_checkpoint['epoch'],
            'val_acc': best_checkpoint['val_acc'],
            'val_loss': best_checkpoint['val_loss']
        }
        
        self.checkpoint_manager.save(
            'densenet',
            identifier,
            save_data,
            config_params
        )
        
        # 清理临时文件
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"DenseNet Pretraining Completed - Fold {fold}")
        logger.info(f"Best Val Acc: {best_checkpoint['val_acc']:.4f}")
        logger.info(f"{'='*70}\n")
        
        # 返回纯净的feature_extractor
        clean_feature_extractor = LightDenseNet3D(
            growth_rate=self.config['growth_rate'],
            num_init_features=self.config['num_init_features']
        )
        clean_feature_extractor.load_state_dict(feature_extractor_state)
        
        return clean_feature_extractor
    
    def _prepare_dataloaders(
        self,
        fold: int,
        split_seed: int
    ) -> Tuple[DataLoader, DataLoader]:
        """准备训练和验证数据加载器"""
        
        # 加载完整数据集
        dataset = PatchDataset(self.data_dir)
        
        # 获取数据分割索引（与主训练保持一致）
        train_indices, val_indices, _ = get_fold_splits(
            self.data_dir, fold, split_seed
        )
        
        logger.info(f"Data split - Train: {len(train_indices)}, Val: {len(val_indices)}")
        
        # 创建子集
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _two_stage_training(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        fold: int
    ) -> str:
        """
        两阶段训练策略
        
        Stage 1: MLP基础训练（无mask）
        Stage 2: MLP + Mask训练（激进mask）
        
        Returns:
            最佳模型的临时保存路径
        """
        criterion = nn.CrossEntropyLoss()
        
        stage1_epochs = self.config['stage1_epochs']
        total_epochs = self.config['num_epochs']
        stage2_epochs = total_epochs - stage1_epochs
        
        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_epoch = -1
        
        # 临时保存路径
        temp_save_path = f'./temp_densenet_fold{fold}_best.pth'
        
        logger.info(f"Stage 1: Epochs 1-{stage1_epochs} (MLP基础训练, no mask)")
        logger.info(f"Stage 2: Epochs {stage1_epochs+1}-{total_epochs} (MLP+Mask训练)")
        
        optimizer = None
        scheduler = None
        
        # 获取实际的模型（处理DataParallel包装）
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        
        for epoch in range(total_epochs):
            # 动态调整训练策略
            if epoch < stage1_epochs:
                # Stage 1: 无mask训练
                if epoch == 0:
                    logger.info(f"\n{'='*50}")
                    logger.info("Entering Stage 1: MLP基础训练")
                    logger.info(f"{'='*50}\n")
                    optimizer = optim.Adam(
                        model.parameters(),
                        lr=self.config['learning_rate_stage1'],
                        weight_decay=self.config['weight_decay_stage1']
                    )
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=stage1_epochs, eta_min=1e-5
                    )
                actual_model.mask_ratio = 0.0
                stage_info = "Stage1(No Mask)"
            else:
                # Stage 2: 激进mask训练
                if epoch == stage1_epochs:
                    logger.info(f"\n{'='*50}")
                    logger.info("Entering Stage 2: MLP+Mask训练")
                    logger.info(f"{'='*50}\n")
                    optimizer = optim.Adam(
                        model.parameters(),
                        lr=self.config['learning_rate_stage2'],
                        weight_decay=self.config['weight_decay_stage2']
                    )
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=stage2_epochs, eta_min=1e-6
                    )
                actual_model.mask_ratio = self.config['mask_ratio']
                stage_info = f"Stage2(Mask={self.config['mask_ratio']:.0%})"
            
            # 训练一个epoch
            train_loss, train_acc = self._train_one_epoch(
                model, train_loader, criterion, optimizer, device, actual_model.mask_ratio
            )
            
            # 验证
            val_loss, val_acc = self._validate(
                model, val_loader, criterion, device
            )
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 保存最佳模型
            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': actual_model.feature_extractor.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, temp_save_path)
                
                logger.info(f"💎 [New Best] Epoch {epoch+1}: Val Acc={val_acc:.4f}, Val Loss={val_loss:.4f}")
            
            # 打印训练信息
            logger.info(
                f"Epoch [{epoch+1}/{total_epochs}] {stage_info} | "
                f"LR: {current_lr:.1e} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
        
        logger.info(f"\nBest model from Epoch {best_epoch+1}: "
                   f"Val Acc={best_val_acc:.4f}, Val Loss={best_val_loss:.4f}")
        
        return temp_save_path
    
    def _train_one_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        mask_ratio: float
    ) -> Tuple[float, float]:
        """训练一个epoch"""
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for patches, _, labels in train_loader:
            patches = patches.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(patches)
            loss = criterion(outputs, labels)
            
            # 如果使用mask，添加一致性损失
            if mask_ratio > 0:
                outputs_no_mask = self._forward_no_mask(model, patches)
                consistency_loss = F.kl_div(
                    F.log_softmax(outputs, dim=1),
                    F.softmax(outputs_no_mask, dim=1),
                    reduction='batchmean'
                )
                loss = loss + 0.2 * consistency_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
    ) -> Tuple[float, float]:
        """验证模型"""
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for patches, _, labels in val_loader:
                patches = patches.to(device)
                labels = labels.to(device)
                
                # 验证时不使用mask
                outputs = self._forward_no_mask(model, patches)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _forward_no_mask(self, model: nn.Module, patches: torch.Tensor) -> torch.Tensor:
        """
        无mask的前向传播（用于验证和一致性损失）
        
        处理DataParallel的特殊情况
        """
        if isinstance(model, nn.DataParallel):
            # DataParallel情况：临时切换到eval模式
            was_training = model.training
            model.eval()
            with torch.no_grad():
                result = model(patches)
            if was_training:
                model.train()
            return result
        else:
            # 普通模型：使用force_no_mask参数
            return model(patches, force_no_mask=True)
    
    def extract_features(
        self,
        model: LightDenseNet3D,
        data: np.ndarray,
        batch_size: int = 16,
        device: Optional[str] = None
    ) -> np.ndarray:
        """
        使用训练好的模型提取特征
        
        Args:
            model: 预训练的DenseNet模型
            data: 输入数据 [N, P, D, H, W]
            batch_size: 批次大小
            device: 设备（None则使用配置中的设备）
            
        Returns:
            特征数组 [N, P, feature_dim]
        """
        if device is None:
            device = self.config['device']
        
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        all_features = []
        
        logger.info(f"Extracting features from {len(data)} samples...")
        
        with torch.no_grad():
            for i in range(len(data)):
                sample_patches = data[i]  # [P, D, H, W]
                
                # 转换为tensor
                patches_tensor = torch.FloatTensor(sample_patches).unsqueeze(1).to(device)
                
                # 批量处理patch
                sample_features = []
                for j in range(0, len(patches_tensor), batch_size):
                    batch_patches = patches_tensor[j:j+batch_size]
                    features = model(batch_patches)
                    sample_features.append(features.cpu())
                
                # 合并特征
                sample_features = torch.cat(sample_features, dim=0).numpy()
                all_features.append(sample_features)
        
        features_array = np.stack(all_features, axis=0)
        logger.info(f"Feature extraction complete: {features_array.shape}")
        
        return features_array