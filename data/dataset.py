"""
Dataset Classes for Patch-based Medical Image Processing
用于基于patch的医学影像数据加载

[新增功能]:
- 支持预计算特征模式（Feature Cache Mode）
- 保留原始patch模式用于特征提取
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional


class PatchDataset(Dataset):
    """
    Patch数据集
    
    [支持两种工作模式]:
    1. 原始模式 (precomputed_features=None):
       - 返回原始patches [P, 1, D, H, W]
       - 用于DenseNet特征提取阶段
       - 应用Z-Score归一化和数据增强
    
    2. 特征模式 (precomputed_features!=None):
       - 直接返回预计算的特征 [P, feature_dim]
       - 用于主训练阶段（跳过特征提取）
       - 大幅提升GPU利用率
    
    Args:
        data_dir: 数据目录路径
        transform: MONAI数据增强变换（仅在原始模式使用）
        precomputed_features: 预计算特征数组 [N, P, feature_dim] (可选)
    """
    
    def __init__(
        self, 
        data_dir: str, 
        transform=None,
        precomputed_features: Optional[np.ndarray] = None
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.precomputed_features = precomputed_features
        
        # 加载数据
        data_path = os.path.join(data_dir, 'data.npy')
        label_path = os.path.join(data_dir, 'label.pkl')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        # [原始模式] 加载原始patches（只在非特征模式时需要）
        if precomputed_features is None:
            self.all_patches = np.load(data_path, mmap_mode='r')
        else:
            self.all_patches = None  # 节省内存
        
        # 加载标签和subject_ids
        with open(label_path, 'rb') as f:
            loaded_data = pickle.load(f)
            self.labels = loaded_data[0]
            self.subject_ids = loaded_data[1]
            
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)
        
        # 验证数据一致性
        if precomputed_features is not None:
            assert len(precomputed_features) == len(self.labels), \
                f"Feature-label mismatch: {len(precomputed_features)} vs {len(self.labels)}"
            self.num_samples = len(self.labels)
            self.num_patches = precomputed_features.shape[1]
            self.feature_dim = precomputed_features.shape[2]
        else:
            assert len(self.all_patches) == len(self.labels), \
                f"Data-label mismatch: {len(self.all_patches)} vs {len(self.labels)}"
            self.num_samples = len(self.labels)
            self.num_patches = self.all_patches.shape[1]
            self.patch_shape = self.all_patches.shape[2:]
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        """
        获取单个样本
        
        [特征模式] Returns:
            features: 特征tensor [P, feature_dim]
            subject_id: 患者ID
            label: 类别标签
        
        [原始模式] Returns:
            patches: patch tensor [P, 1, D, H, W]
            subject_id: 患者ID
            label: 类别标签
        """
        label = int(self.labels[idx])
        subject_id = self.subject_ids[idx]
        
        # ============================================================
        # [特征模式] 直接返回预计算的特征
        # ============================================================
        if self.precomputed_features is not None:
            features = torch.from_numpy(
                self.precomputed_features[idx]
            ).float()  # [P, feature_dim]
            
            return features, subject_id, label
        
        # ============================================================
        # [原始模式] 返回原始patches（用于特征提取）
        # ============================================================
        # 1. 加载数据 (P, D, H, W)
        subject_patches = np.array(self.all_patches[idx])
        
        # 2. 转为 Tensor (P, D, H, W)
        patches_tensor = torch.from_numpy(subject_patches).float()
        
        # 3. Z-Score 归一化 (样本级别)
        p_mean = patches_tensor.mean()
        p_std = patches_tensor.std()
        patches_tensor = (patches_tensor - p_mean) / (p_std + 1e-6)
        
        # 4. 应用快速、一致的数据增强
        # MONAI 将 (P, D, H, W) 视为 (C, H, W, D)
        if self.transform:
            patches_tensor = self.transform(patches_tensor)
            
        # 5. 增加通道维度，以匹配模型输入
        patches_tensor = patches_tensor.unsqueeze(1)  # Shape: (P, 1, D, H, W)
        
        return patches_tensor, subject_id, label
        
    def get_num_patches(self) -> int:
        return self.num_patches
    
    def get_patch_shape(self) -> Tuple[int, ...]:
        """获取patch形状（仅在原始模式可用）"""
        if hasattr(self, 'patch_shape'):
            return self.patch_shape
        else:
            raise AttributeError("Patch shape not available in feature mode")


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """
    自定义batch整理函数
    
    [特征模式]:
        batch_data: [B, P, feature_dim]
    
    [原始模式]:
        batch_patches: [B, P, 1, D, H, W]
    
    Args:
        batch: 数据列表
        
    Returns:
        batch_data: [B, P, ...] (特征或patches)
        patient_ids: [B]
        batch_labels: [B]
    """
    data, patient_ids, labels = zip(*batch)
    
    batch_data = torch.stack(data)
    batch_labels = torch.tensor(labels, dtype=torch.long)
    
    return batch_data, list(patient_ids), batch_labels


def get_fold_splits(
    data_dir: str,
    fold: int,
    split_seed: int,
    num_folds: int = 5,
    val_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    获取指定fold的数据分割索引
    
    这个函数确保与主训练流程使用完全相同的分割方式
    
    Args:
        data_dir: 数据目录
        fold: fold索引 (0-4)
        split_seed: 随机种子
        num_folds: 总fold数
        val_ratio: 验证集比例
        
    Returns:
        train_indices: 训练集索引
        val_indices: 验证集索引
        test_indices: 测试集索引
    """
    from sklearn.model_selection import StratifiedKFold
    
    # 加载标签
    label_path = os.path.join(data_dir, 'label.pkl')
    with open(label_path, 'rb') as f:
        labels, _ = pickle.load(f)
    
    # 使用StratifiedKFold进行分割
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=split_seed)
    splits = list(skf.split(labels, labels))
    
    # 获取当前fold的训练/测试分割
    train_val_indices, test_indices = splits[fold]
    
    # 从训练集中分出验证集
    # 注意：必须使用相同的随机种子和shuffle逻辑
    train_indices = train_val_indices.copy()
    np.random.seed(split_seed)  # 关键：使用相同的种子
    np.random.shuffle(train_indices)
    
    val_size = int(len(train_indices) * val_ratio)
    val_indices = train_indices[:val_size]
    train_indices = train_indices[val_size:]
    
    return train_indices, val_indices, test_indices