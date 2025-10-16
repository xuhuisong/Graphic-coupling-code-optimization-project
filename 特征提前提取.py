#!/usr/bin/env python3
"""
按折提取特征的完整解决方案
为每个fold使用对应的预训练DenseNet提取特征，避免数据泄露
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split

class LightDenseNet3D(nn.Module):
    """DenseNet特征提取器 - 与预训练保持一致"""
    def __init__(self, growth_rate=8, num_init_features=24):
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
        out = torch.nn.functional.adaptive_avg_pool3d(features, (1, 1, 1))
        out = torch.flatten(out, 1)
        return out

def extract_features_for_fold(data_path, output_base_path, fold, split_seed, model_path, device='cuda:0', batch_size=16):
    """
    为指定fold提取特征
    
    Args:
        data_path: 原始数据路径
        output_base_path: 输出基础路径
        fold: 折索引 (0-4)
        split_seed: 分割种子
        model_path: 对应fold的预训练模型路径
        device: 设备
        batch_size: 批次大小
    """
    print(f"\n{'='*60}")
    print(f"开始为 Fold {fold} 提取特征")
    print(f"模型: {model_path}")
    print(f"{'='*60}")
    
    # 加载原始数据
    print("加载原始数据...")
    data = np.load(os.path.join(data_path, 'data.npy'))
    with open(os.path.join(data_path, 'label.pkl'), 'rb') as f:
        labels, subject_ids = pickle.load(f)
    
    print(f"数据形状: {data.shape}")
    print(f"标签数量: {len(labels)}")
    
    # 创建与主训练相同的数据分割
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
    splits = list(skf.split(data, labels))
    train_val_indices, test_indices = splits[fold]
    
    # 进一步分割训练集和验证集 (与主训练保持一致)
    train_val_labels = labels[train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=0.2, stratify=train_val_labels, random_state=42
    )
    
    print(f"Fold {fold} 数据分割:")
    print(f"  训练集: {len(train_indices)} 样本")
    print(f"  验证集: {len(val_indices)} 样本") 
    print(f"  测试集: {len(test_indices)} 样本")
    
    # 加载对应fold的预训练模型
    print(f"加载预训练模型: {model_path}")
    model = LightDenseNet3D(growth_rate=8, num_init_features=24)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"模型加载完成，特征维度: {model.feature_dim}")
    
    # 为每个样本提取特征
    print("开始特征提取...")
    all_features = []
    
    with torch.no_grad():
        for sample_idx in tqdm(range(len(data)), desc=f"Fold {fold} 特征提取"):
            sample_patches = data[sample_idx]  # [288, 24, 24, 24]
            
            # 转换为tensor
            patches_tensor = torch.FloatTensor(sample_patches).unsqueeze(1).to(device)  # [288, 1, 24, 24, 24]
            
            # 批量处理patch (避免显存不足)
            sample_features = []
            for i in range(0, len(patches_tensor), batch_size):
                end_idx = min(i + batch_size, len(patches_tensor))
                batch_patches = patches_tensor[i:end_idx]
                features = model(batch_patches)
                sample_features.append(features.cpu())
            
            # 合并特征 [288, 28]
            sample_features = torch.cat(sample_features, dim=0).numpy()
            all_features.append(sample_features)
    
    # 转换为numpy数组
    features_array = np.stack(all_features, axis=0)  # [N, 288, 28]
    
    print(f"特征提取完成: {features_array.shape}")
    print(f"内存占用: {features_array.nbytes / (1024**2):.1f} MB (原始: {data.nbytes / (1024**3):.1f} GB)")
    
    # 创建输出目录
    output_path = os.path.join(output_base_path, f'fold_{fold}')
    os.makedirs(output_path, exist_ok=True)
    
    # 保存特征数据
    feature_file = os.path.join(output_path, 'data.npy')
    label_file = os.path.join(output_path, 'label.pkl')
    
    np.save(feature_file, features_array)
    with open(label_file, 'wb') as f:
        pickle.dump((labels, subject_ids), f)
    
    # 复制其他必要文件
    for filename in ['coordinates.npy', 'group_correlation_edges.npy']:
        src = os.path.join(data_path, filename)
        dst = os.path.join(output_path, filename)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)
    
    print(f"Fold {fold} 特征数据已保存到: {output_path}")
    
    # 清理显存
    del model
    torch.cuda.empty_cache()
    
    return output_path

def extract_all_folds(data_path, output_base_path, pretrained_base_path, split_seed=3333, device='cuda:0'):
    """
    为所有fold提取特征
    
    Args:
        data_path: 原始数据路径
        output_base_path: 输出基础路径  
        pretrained_base_path: 预训练模型基础路径
        split_seed: 分割种子
        device: 设备
    """
    print(f"开始为所有fold提取特征...")
    print(f"原始数据: {data_path}")
    print(f"输出路径: {output_base_path}")
    print(f"预训练模型: {pretrained_base_path}")
    print(f"分割种子: {split_seed}")
    
    # 创建输出目录
    os.makedirs(output_base_path, exist_ok=True)
    
    extracted_paths = {}
    
    for fold in range(5):
        # 构建预训练模型路径
        model_path = os.path.join(pretrained_base_path, f'fold_{fold}_feature_extractor_288patch.pth')
        
        if not os.path.exists(model_path):
            print(f"警告: 找不到 Fold {fold} 的预训练模型: {model_path}")
            continue
        
        # 提取特征
        output_path = extract_features_for_fold(
            data_path=data_path,
            output_base_path=output_base_path,
            fold=fold,
            split_seed=split_seed,
            model_path=model_path,
            device=device
        )
        
        extracted_paths[fold] = output_path
    
    # 保存路径映射
    mapping_file = os.path.join(output_base_path, f'fold_paths_seed{split_seed}.pkl')
    with open(mapping_file, 'wb') as f:
        pickle.dump(extracted_paths, f)
    
    print(f"\n{'='*80}")
    print("所有fold特征提取完成!")
    print(f"路径映射已保存: {mapping_file}")
    
    for fold, path in extracted_paths.items():
        # 计算大小
        data_file = os.path.join(path, 'data.npy')
        if os.path.exists(data_file):
            size_mb = os.path.getsize(data_file) / (1024**2)
            print(f"  Fold {fold}: {path} ({size_mb:.1f} MB)")
    
    print(f"{'='*80}")
    
    return extracted_paths

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='按fold提取特征')
    parser.add_argument('--data_path', default='MCIvsCN.ex001_2_p288/MCIvsCN.ex001_2_p288_pw24_all', 
                        help='原始数据路径')
    parser.add_argument('--output_path', default='MCIvsCN.ex001_2_p288/MCIvsCN.ex001_2_p288_pw24_all_features_by_fold',
                        help='输出基础路径')
    parser.add_argument('--pretrained_path', default='pretrained_densenet_models',
                        help='预训练模型基础路径')
    parser.add_argument('--split_seed', type=int, default=1449,
                        help='分割种子')
    parser.add_argument('--fold', type=int, help='只处理指定fold (可选)')
    parser.add_argument('--device', default='cuda:0', help='设备')
    
    args = parser.parse_args()
    
    if args.fold is not None:
        # 只处理指定fold
        model_path = os.path.join(args.pretrained_path, f'fold_{args.fold}_feature_extractor_288patch.pth')
        extract_features_for_fold(
            data_path=args.data_path,
            output_base_path=args.output_path, 
            fold=args.fold,
            split_seed=args.split_seed,
            model_path=model_path,
            device=args.device
        )
    else:
        # 处理所有fold
        extract_all_folds(
            data_path=args.data_path,
            output_base_path=args.output_path,
            pretrained_base_path=args.pretrained_path,
            split_seed=args.split_seed,
            device=args.device
        )

if __name__ == "__main__":
    main()