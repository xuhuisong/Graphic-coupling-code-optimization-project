"""
LightDenseNet3D Model Definition
轻量级3D DenseNet特征提取器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LightDenseNet3D(nn.Module):
    """
    轻量级3D DenseNet特征提取器
    
    用于从3D医学影像patch中提取特征向量。
    设计精简，适合作为图神经网络的前置特征提取器。
    
    Args:
        growth_rate: DenseNet的增长率（每层增加的通道数）
        num_init_features: 初始卷积层的输出通道数
        
    Attributes:
        feature_dim: 输出特征向量的维度
    """
    
    def __init__(self, growth_rate: int = 8, num_init_features: int = 24):
        super(LightDenseNet3D, self).__init__()
        
        # 初始特征提取层
        self.features = nn.Sequential(
            nn.Conv3d(1, num_init_features, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        num_features = num_init_features
        
        # 第一个Dense Block (3层)
        self.dense1 = self._make_dense_block(num_features, growth_rate, 3)
        num_features += 3 * growth_rate
        
        # 第一个Transition Layer
        self.trans1 = self._make_transition(num_features, num_features // 2)
        num_features = num_features // 2
        
        # 第二个Dense Block (4层)
        self.dense2 = self._make_dense_block(num_features, growth_rate, 4)
        num_features += 4 * growth_rate
        
        # 第二个Transition Layer
        self.trans2 = self._make_transition(num_features, num_features // 2)
        num_features = num_features // 2
        
        # 最终归一化
        self.norm = nn.BatchNorm3d(num_features)
        self.relu = nn.ReLU(inplace=True)
        
        # 记录特征维度（供外部使用）
        self.feature_dim = num_features
        
        # 权重初始化
        self._initialize_weights()
    
    def _make_dense_layer(self, in_channels: int, growth_rate: int) -> nn.Sequential:
        """
        创建单个Dense Layer
        
        使用1x1卷积降维 + 3x3卷积提取特征的瓶颈结构
        """
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, 4 * growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv3d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
    
    def _make_dense_block(self, in_channels: int, growth_rate: int, num_layers: int) -> nn.ModuleList:
        """
        创建Dense Block（多个Dense Layer的组合）
        
        每层的输入是前面所有层输出的拼接
        """
        layers = []
        for i in range(num_layers):
            layers.append(self._make_dense_layer(in_channels + i * growth_rate, growth_rate))
        return nn.ModuleList(layers)
    
    def _make_transition(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        创建Transition Layer（用于降维和下采样）
        """
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )
    
    def _initialize_weights(self) -> None:
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入tensor，shape为 [B, 1, D, H, W]
            
        Returns:
            特征向量，shape为 [B, feature_dim]
        """
        # 初始特征提取
        features = self.features(x)
        
        # Dense Block 1
        for layer in self.dense1:
            new_features = layer(features)
            features = torch.cat([features, new_features], 1)
        features = self.trans1(features)
        
        # Dense Block 2
        for layer in self.dense2:
            new_features = layer(features)
            features = torch.cat([features, new_features], 1)
        features = self.trans2(features)
        
        # 最终归一化
        features = self.norm(features)
        features = self.relu(features)
        
        # 全局平均池化 + 展平
        out = F.adaptive_avg_pool3d(features, (1, 1, 1))
        out = torch.flatten(out, 1)
        
        return out


class EndToEndDenseNet(nn.Module):
    """
    端到端的DenseNet训练模型
    
    包含特征提取器 + 分类器，用于预训练阶段
    支持可选的Patch Masking策略来提高特征鲁棒性
    
    Args:
        feature_extractor: LightDenseNet3D实例
        num_patches: 每个样本的patch数量
        num_classes: 分类类别数
    """
    
    def __init__(
        self, 
        feature_extractor: LightDenseNet3D, 
        num_patches: int, 
        num_classes: int = 2
    ):
        super(EndToEndDenseNet, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.mask_ratio = 0.0  # 训练时随机mask掉的patch比例
        
        feature_dim = feature_extractor.feature_dim
        
        # Patch特征变换层（可选，用于增强表达能力）
        self.patch_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.Dropout(0.1)
        )
        
        # 分类器
        classifier_input_dim = feature_dim * num_patches
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def generate_random_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        生成随机mask（训练时使用）
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            mask tensor，shape为 [B, num_patches]
        """
        if not self.training or self.mask_ratio == 0:
            return torch.ones(batch_size, self.num_patches, dtype=torch.bool, device=device)
        
        num_masked = int(self.num_patches * self.mask_ratio)
        mask = torch.ones(batch_size, self.num_patches, device=device)
        
        for i in range(batch_size):
            masked_indices = torch.randperm(self.num_patches, device=device)[:num_masked]
            mask[i, masked_indices] = 0
        
        return mask.bool()
    
    def forward(self, x: torch.Tensor, force_no_mask: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入tensor，shape为 [B, num_patches, 1, D, H, W]
            force_no_mask: 是否强制不使用mask（验证时使用）
            
        Returns:
            分类logits，shape为 [B, num_classes]
        """
        batch_size, num_patches, _, D, H, W = x.shape
        
        # 1. 提取每个patch的特征
        x_reshaped = x.view(-1, 1, D, H, W)  # [B*P, 1, D, H, W]
        patch_features = self.feature_extractor(x_reshaped)  # [B*P, feature_dim]
        patch_features = patch_features.view(batch_size, num_patches, -1)  # [B, P, feature_dim]
        
        # 2. 可选的MLP变换（当前注释掉，保持简洁）
        # patch_features_flat = patch_features.view(-1, patch_features.size(-1))
        # transformed_flat = self.patch_transform(patch_features_flat)
        # patch_features = transformed_flat.view(batch_size, num_patches, -1)
        
        # 3. 应用随机mask（仅训练时）
        if self.training and not force_no_mask and self.mask_ratio > 0:
            mask = self.generate_random_mask(batch_size, x.device)
            patch_features = patch_features * mask.unsqueeze(-1)
        
        # 4. 聚合所有patch特征
        patient_features = patch_features.reshape(batch_size, -1)
        
        # 5. 分类
        logits = self.classifier(patient_features)
        
        return logits