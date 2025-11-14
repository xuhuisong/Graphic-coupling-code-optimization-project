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
    
    [修改版]:
    1. __init__ 接受一个可选的 transform。
    2. __getitem__ 应用 Z-Score 归一化。
    3. __getitem__ 以 (C, H, W, D) 格式应用 transform，以实现快速、一致的增强。
    4. __getitem__ 最终返回 (P, 1, D, H, W) 以匹配模型的输入。
    """
    
    def __init__(self, data_dir: str, transform = None): # <--- 修改点 1
        self.data_dir = data_dir
        self.transform = transform # <--- 修改点 2
        
        # 加载数据
        data_path = os.path.join(data_dir, 'data.npy')
        label_path = os.path.join(data_dir, 'label.pkl')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        self.all_patches = np.load(data_path, mmap_mode='r')
        
        with open(label_path, 'rb') as f:
            loaded_data = pickle.load(f)
            self.labels = loaded_data[0]
            self.subject_ids = loaded_data[1]
            
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)
            
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
        
        Returns:
            patches: patch tensor [P, 1, D, H, W]
            ...
        """
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
        patches_tensor = patches_tensor.unsqueeze(1) # Shape: (P, 1, D, H, W)
        
        label = int(self.labels[idx])
        subject_id = self.subject_ids[idx]
        
        return patches_tensor, subject_id, label
        
    def get_num_patches(self) -> int:
        return self.num_patches
    
    def get_patch_shape(self) -> Tuple[int, ...]:
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