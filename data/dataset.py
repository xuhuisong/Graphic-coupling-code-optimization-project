"""
Dataset Classes for Patch-based Medical Image Processing
用于基于patch的医学影像数据加载
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List


class PatchDataset(Dataset):
    """
    Patch数据集
    
    从预处理好的 data.npy 和 label.pkl 文件加载patch数据
    
    Args:
        data_dir: 数据目录路径
        
    Attributes:
        all_patches: 所有样本的patch数据 [N, P, D, H, W]
        labels: 样本标签 [N]
        subject_ids: 样本ID [N]
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
        # 加载数据
        data_path = os.path.join(data_dir, 'data.npy')
        label_path = os.path.join(data_dir, 'label.pkl')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        # 使用内存映射模式加载大文件，节省内存
        self.all_patches = np.load(data_path, mmap_mode='r')
        
        with open(label_path, 'rb') as f:
            self.labels, self.subject_ids = pickle.load(f)
        
        # 验证数据一致性
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
        
        Args:
            idx: 样本索引
            
        Returns:
            patches: patch tensor [P, 1, D, H, W]
            subject_id: 样本ID
            label: 标签
        """
        # 从内存映射中加载数据并立即转换为副本
        subject_patches = np.array(self.all_patches[idx])
        
        # 转换为tensor并添加通道维度
        patches_tensor = torch.from_numpy(subject_patches).float().unsqueeze(1)
        
        label = int(self.labels[idx])
        subject_id = self.subject_ids[idx]
        
        return patches_tensor, subject_id, label
    
    def get_num_patches(self) -> int:
        """获取每个样本的patch数量"""
        return self.num_patches
    
    def get_patch_shape(self) -> Tuple[int, ...]:
        """获取单个patch的形状"""
        return self.patch_shape


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """
    自定义batch整理函数
    
    Args:
        batch: 数据列表
        
    Returns:
        batch_patches: [B, P, 1, D, H, W]
        patient_ids: [B]
        batch_labels: [B]
    """
    patches, patient_ids, labels = zip(*batch)
    
    batch_patches = torch.stack(patches)
    batch_labels = torch.tensor(labels, dtype=torch.long)
    
    return batch_patches, list(patient_ids), batch_labels


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