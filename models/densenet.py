"""
DenseNet Feature Extractor and End-to-End Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LightDenseNet3D(nn.Module):
    # (保持你提供的原始架构不变, growth_rate=16, 2x trans)
    def __init__(self, growth_rate=16, num_init_features=24):
        super(LightDenseNet3D, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, num_init_features, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        num_features = num_init_features
        
        self.dense1 = self._make_dense_block(num_features, growth_rate, 3)
        num_features += 3 * growth_rate
        self.trans1 = self._make_transition(num_features, num_features // 2)
        num_features = num_features // 2
        
        self.dense2 = self._make_dense_block(num_features, growth_rate, 4)
        num_features += 4 * growth_rate
        self.trans2 = self._make_transition(num_features, num_features // 2)
        num_features = num_features // 2
        
        self.norm = nn.BatchNorm3d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.feature_dim = num_features
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def _make_dense_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, 4 * growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv3d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
    
    def _make_dense_block(self, in_channels, growth_rate, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(self._make_dense_layer(in_channels + i * growth_rate, growth_rate))
        return nn.ModuleList(layers)
    
    def _make_transition(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        features = self.features(x)
        for layer in self.dense1:
            new_features = layer(features)
            features = torch.cat([features, new_features], 1)
        features = self.trans1(features)
        for layer in self.dense2:
            new_features = layer(features)
            features = torch.cat([features, new_features], 1)
        features = self.trans2(features)
        features = self.norm(features)
        features = self.relu(features)
        out = F.adaptive_avg_pool3d(features, (1, 1, 1))
        out = torch.flatten(out, 1)
        return out

# ========================== 3. 端到端模型 (已修改) ==========================
class EndToEndDenseNet(nn.Module):
    def __init__(self, feature_extractor, num_patches, num_classes=2):
        super(EndToEndDenseNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.mask_ratio = 0.0
        
        feature_dim = feature_extractor.feature_dim
        
        self.patch_transform_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), 
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim), 
            nn.Dropout(0.1)
        )
        
        classifier_input_dim = feature_dim * num_patches
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128), 
            nn.ReLU(inplace=True), 
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # [核心修改 1]
        # x 进来的形状是 (B, 288, 24, 24, 24)
        # 我们需要添加 C=1 的维度
        batch_size, num_patches, _, D, H, W = x.shape
        
        # 1. 提取Patch特征
        x_reshaped = x.view(-1, 1, D, H, W) # (B*288, 1, 24, 24, 24)
        patch_features = self.feature_extractor(x_reshaped)
        patch_features = patch_features.view(batch_size, num_patches, -1)
        
        # 2. MLP变换
        patch_features_flat = patch_features.view(-1, patch_features.size(-1))
        transformed_flat = self.patch_transform_mlp(patch_features_flat)
        patch_features = transformed_flat.view(batch_size, num_patches, -1)
        
        # 4. 分类 (保持 "reshape" 拼接)
        patient_features = patch_features.reshape(batch_size, -1)
        logits = self.classifier(patient_features)
        
        return logits