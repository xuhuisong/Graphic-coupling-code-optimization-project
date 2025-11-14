"""
DenseNet Pretraining Manager
DenseNet预训练自动化管理器

[修改版]:
- 移除了两阶段训练 (Stage 2 Masking)。
- 替换为单阶段训练 + 早停法 (Early Stopping)。
- 训练集使用 Z-Score + 快速数据增强。
- 验证集和特征提取使用 Z-Score。
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
import math
import pickle

from utils.checkpoint import CheckpointManager
from models.densenet import LightDenseNet3D, EndToEndDenseNet
# 导入我们修改后的 PatchDataset
from data.dataset import PatchDataset, collate_fn, get_fold_splits

# [新导入] 导入 MONAI
try:
    from monai.transforms import Compose, RandFlip, RandAffine, RandGaussianNoise
except ImportError:
    logging.error("MONAI not found. Please install: pip install monai")
    exit()

logger = logging.getLogger(__name__)

# [新增] 数据增强定义
def get_train_transform():
    """
    定义一个快速、一致的数据增强管道
    它将 (P, D, H, W) 视为 (C, H, W, D) 并对 H,W,D 空间进行变换。
    """
    return Compose([
        RandFlip(spatial_axis=0, prob=0.5), 
        RandAffine(
            prob=0.5,
            rotate_range=(math.pi/32, math.pi/32, math.pi/32),
            translate_range=(3, 3, 3),
            scale_range=(0.05, 0.05, 0.05),
            padding_mode='border'
        ),
        RandGaussianNoise(prob=0.5, std=0.05)
    ])


class DenseNetManager:
    """
    DenseNet预训练管理器
    (功能描述保持不变)
    """
    
    def __init__(
        self,
        data_dir: str,
        checkpoint_manager: CheckpointManager,
        config: Optional[Dict[str, Any]] = None
    ):
        self.data_dir = data_dir
        self.checkpoint_manager = checkpoint_manager
        
        # [修改点 1] 简化配置
        self.config = {
            'growth_rate': 8,
            'num_init_features': 24,
            'num_epochs': 70,      # 总 Epoch 上限
            'patience': 40,         # 早停的耐心值
            'batch_size': 24,
            'learning_rate': 0.0005,  # 单一学习率
            'weight_decay': 1e-4,     # 单一权重衰减
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
        """
        # [修改点 2] 更新缓存标识符
        config_params = {
            'growth_rate': self.config['growth_rate'],
            'num_init_features': self.config['num_init_features'],
            'num_epochs': self.config['num_epochs'],
            'patience': self.config['patience'],
            'learning_rate': self.config['learning_rate']
            # 移除了 stage2/mask 参数
        }
        
        identifier = self.checkpoint_manager.build_identifier(
            'densenet',
            config_params,
            {'fold': fold, 'seed': split_seed}
        )
        
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
        # (此函数逻辑保持不变)
        checkpoint = self.checkpoint_manager.load(
            'densenet',
            identifier,
            map_location='cpu'
        )
        
        model = LightDenseNet3D(
            growth_rate=self.config['growth_rate'],
            num_init_features=self.config['num_init_features']
        )
        
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
        
        try:
            num_patches = val_loader.dataset.dataset.get_num_patches()
        except Exception:
            temp_dataset = PatchDataset(self.data_dir, transform=None)
            num_patches = temp_dataset.get_num_patches()
            del temp_dataset

        
        model = EndToEndDenseNet(feature_extractor, num_patches).to(device)
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)
        
        # 3. [修改点 3] 执行单阶段训练
        best_model_path = self._train_model_with_early_stopping(
            model,
            train_loader,
            val_loader,
            device,
            fold,
            num_epochs=self.config['num_epochs'],
            patience=self.config['patience']
        )
        
        # 4. 保存最佳模型到缓存
        logger.info(f"Saving best model to cache...")
        best_checkpoint = torch.load(best_model_path, map_location='cpu')
        
        # (保持不变：保存 extractor 的 state_dict)
        feature_extractor_state = best_checkpoint['model_state_dict']
        
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
        
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"DenseNet Pretraining Completed - Fold {fold}")
        logger.info(f"Best Val Acc: {best_checkpoint['val_acc']:.4f}")
        logger.info(f"{'='*70}\n")
        
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
        """
        准备训练和验证数据加载器
        [修改点 4]：为训练集和验证集创建不同的 Dataset 实例
        """
        
        # 1. 为训练集创建带增强的 Dataset
        train_transform = get_train_transform()
        train_dataset = PatchDataset(self.data_dir, transform=train_transform)
        
        # 2. 为验证集创建不带增强的 Dataset
        eval_dataset = PatchDataset(self.data_dir, transform=None)
        
        # 3. 获取数据分割索引
        train_indices, val_indices, _ = get_fold_splits(
            self.data_dir, fold, split_seed
        )
        
        logger.info(f"Data split - Train: {len(train_indices)}, Val: {len(val_indices)}")
        
        # 4. 从各自的数据集中创建子集
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(eval_dataset, val_indices)
        
        # 5. 创建数据加载器
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
    
    # [修改点 5] 新的训练函数 (替换 _two_stage_training)
    def _train_model_with_early_stopping(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        fold: int,
        num_epochs: int,
        patience: int
    ) -> str:
        """
        单阶段训练策略 + 早停法
        
        Returns:
            最佳模型的临时保存路径
        """
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_epoch = -1
        patience_counter = 0
        
        temp_save_path = f'./temp_densenet_fold{fold}_best.pth'
        
        logger.info(f"Starting Single-Stage Training (Epochs: {num_epochs}, Patience: {patience})")
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        actual_model.mask_ratio = 0.0 # 确保 mask 始终关闭
        
        for epoch in range(num_epochs):
            
            # 训练
            train_loss, train_acc = self._train_one_epoch(
                model, train_loader, criterion, optimizer, device, mask_ratio=0.0
            )
            
            # 验证
            val_loss, val_acc = self._validate(
                model, val_loader, criterion, device
            )
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 早停逻辑
            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0 # 重置耐心
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': actual_model.feature_extractor.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, temp_save_path)
                
                logger.info(f"💎 [New Best] Epoch {epoch+1}: Val Acc={val_acc:.4f}, Val Loss={val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"  (No improvement, patience: {patience_counter}/{patience})")

            # 打印训练信息
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] | "
                f"LR: {current_lr:.1e} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            if patience_counter >= patience:
                logger.info(f"\n🔥 [Triggering Early Stopping] Validation accuracy did not improve for {patience} epochs.")
                break # 停止训练
        
        logger.info(f"\nBest model from Epoch {best_epoch+1}: "
                   f"Val Acc={best_val_acc:.4f}, Val Loss={best_val_loss:.4f}")
        
        return temp_save_path
    
    # [修改点 6] 简化 _train_one_epoch
    def _train_one_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        mask_ratio: float # (此参数保留以匹配函数签名，但始终为 0)
    ) -> Tuple[float, float]:
        """训练一个epoch (已移除 mask 逻辑)"""
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for patches, _, labels in train_loader:
            patches = patches.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(patches) # force_no_mask 默认为 False
            loss = criterion(outputs, labels)
            
            # (移除了 mask 和一致性损失)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    # [修改点 7] 简化 _validate
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
    ) -> Tuple[float, float]:
        """验证模型 (已移除 _forward_no_mask)"""
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for patches, _, labels in val_loader:
                patches = patches.to(device)
                labels = labels.to(device)
                
                # model.eval() 会自动处理 (mask_ratio=0)
                outputs = model(patches) 
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy

    def extract_features(
            self,
            model: LightDenseNet3D,
            data: np.ndarray,
            batch_size: int = 16,
            device: Optional[str] = None
        ) -> np.ndarray:
            """
            使用训练好的模型提取特征
            [修改]：增加了 Z-Score 归一化以匹配训练
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

                    # [核心修改] 转换为 tensor 并应用 Z-Score
                    patches_tensor = torch.from_numpy(sample_patches).float() # (P, D, H, W)

                    p_mean = patches_tensor.mean()
                    p_std = patches_tensor.std()
                    patches_tensor = (patches_tensor - p_mean) / (p_std + 1e-6)

                    # 添加通道维度并移到 GPU
                    patches_tensor = patches_tensor.unsqueeze(1).to(device) # (P, 1, D, H, W)

                    sample_features = []
                    for j in range(0, len(patches_tensor), batch_size):
                        batch_patches = patches_tensor[j:j+batch_size]
                        features = model(batch_patches)
                        sample_features.append(features.cpu())

                    sample_features = torch.cat(sample_features, dim=0).numpy()
                    all_features.append(sample_features)

            features_array = np.stack(all_features, axis=0)
            logger.info(f"Feature extraction complete: {features_array.shape}")

            return features_array