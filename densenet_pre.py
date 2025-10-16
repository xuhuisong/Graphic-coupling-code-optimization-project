"""
独立DenseNet预训练器
解决数据泄露问题：为每一折独立预训练DenseNet特征提取器
使用方法：
1. 先运行此脚本预训练所有折的DenseNet
2. 在主训练脚本中加载对应折的预训练模型
"""

import os
import math 
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
warnings.filterwarnings("ignore")

class LightDenseNet3D(nn.Module):
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

class EndToEndModel(nn.Module):
    def __init__(self, feature_extractor, num_patches, num_classes=2):
        super(EndToEndModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.mask_ratio = 0.0
        
        feature_dim = feature_extractor.feature_dim
        
        # Patch特征变换层
        self.patch_transform_mlp = nn.Sequential(
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
    
    def generate_random_mask(self, batch_size, device):
        if not self.training or self.mask_ratio == 0:
            return torch.ones(batch_size, self.num_patches, dtype=torch.bool, device=device)
        
        num_masked = int(self.num_patches * self.mask_ratio)
        mask = torch.ones(batch_size, self.num_patches, device=device)
        for i in range(batch_size):
            masked_indices = torch.randperm(self.num_patches, device=device)[:num_masked]
            mask[i, masked_indices] = 0
        return mask.bool()
    
    def forward(self, x, force_no_mask=False):
        batch_size, num_patches, _, D, H, W = x.shape
        
        # 1. 提取Patch特征
        x_reshaped = x.view(-1, 1, D, H, W)
        patch_features = self.feature_extractor(x_reshaped)
        patch_features = patch_features.view(batch_size, num_patches, -1)
        
        # 2. MLP变换
        #patch_features_flat = patch_features.view(-1, patch_features.size(-1))
        #transformed_flat = self.patch_transform_mlp(patch_features_flat)
        #patch_features = transformed_flat.view(batch_size, num_patches, -1)
        
        # 3. 应用Mask
        if self.training and not force_no_mask and self.mask_ratio > 0:
            mask = self.generate_random_mask(batch_size, x.device)
            patch_features = patch_features * mask.unsqueeze(-1)
        
        # 4. 分类
        patient_features = patch_features.reshape(batch_size, -1)
        logits = self.classifier(patient_features)
        
        return logits

class PreprocessedPatchDataset(Dataset):
    """从预处理好的 data.npy 和 label.pkl 文件加载数据的Dataset"""
    
    def __init__(self, data_dir):
        print(f"从 '{data_dir}' 加载预处理数据...")
        
        data_path = os.path.join(data_dir, 'data.npy')
        label_path = os.path.join(data_dir, 'label.pkl')

        try:
            self.all_patches = np.load(data_path)
            with open(label_path, 'rb') as f:
                self.labels, self.subject_ids = pickle.load(f)
        except FileNotFoundError as e:
            print(f"错误：找不到所需文件: {e}")
            raise e
            
        print("数据加载成功!")
        print(f"  - Patches shape: {self.all_patches.shape}")
        print(f"  - Labels shape: {self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        subject_patches = self.all_patches[idx]
        patches_tensor = torch.from_numpy(subject_patches).float().unsqueeze(1)
        label = self.labels[idx]
        subject_id = self.subject_ids[idx]
        return patches_tensor, subject_id, label

    def get_num_patches(self):
        return self.all_patches.shape[1]

def collate_fn(batch):
    patches, patient_ids, labels = zip(*batch)
    batch_patches = torch.stack(patches)
    batch_labels = torch.tensor(labels, dtype=torch.long)
    return batch_patches, patient_ids, batch_labels

def forward_with_no_mask(model, patches):
    """
    专门用于无mask推理的辅助函数，避免DataParallel的关键字参数问题
    """
    if isinstance(model, nn.DataParallel):
        # 临时关闭训练模式来避免mask
        was_training = model.training
        model.eval()
        with torch.no_grad():
            result = model(patches)
        if was_training:
            model.train()
        return result
    else:
        # 非DataParallel情况下可以直接使用关键字参数
        return model(patches, force_no_mask=True)

def pretrain_feature_extractor(model, train_loader, val_loader, device, fold_idx, num_epochs):
    """两阶段预训练DenseNet特征提取器"""
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc, best_val_loss, best_epoch = 0.0, float('inf'), -1
    checkpoint_path = f'pretrained_densenet_models/fold_{fold_idx}_feature_extractor_288patch.pth'
    
    # 两阶段配置
    stage1_epochs = 49
    stage2_epochs = num_epochs - 49
    AGGRESSIVE_MASK_RATIO = 0.3
    
    print(f"\n{'='*60}\n两阶段预训练 - Fold {fold_idx}\n{'='*60}")
    print(f"阶段1: Epoch 1-{stage1_epochs} (MLP基础训练)")
    print(f"阶段2: Epoch {stage1_epochs+1}-{num_epochs} (MLP+Mask训练)")
    print(f"{'='*60}\n")
    
    optimizer, scheduler = None, None

    for epoch in range(num_epochs):
        # 动态调整训练策略
        if epoch < stage1_epochs:
            if epoch == 0:
                print(f"\n{'='*50}\n进入阶段1: MLP基础训练\n{'='*50}")
                optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs, eta_min=1e-5)
            actual_model.mask_ratio = 0.0
            stage_info = "阶段1 (MLP基础训练)"
        else:
            if epoch == stage1_epochs:
                print(f"\n{'='*50}\n进入阶段2: MLP激进Mask训练\n{'='*50}")
                optimizer = optim.Adam(model.parameters(), lr=0.0004, weight_decay=1e-5)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_epochs, eta_min=1e-6)
            actual_model.mask_ratio = AGGRESSIVE_MASK_RATIO
            stage_info = f"阶段2 (MLP+mask={AGGRESSIVE_MASK_RATIO:.0%})"

        # 训练
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for patches, _, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            outputs = model(patches)
            loss = criterion(outputs, labels)
            
            if actual_model.mask_ratio > 0:
                # 使用修复后的无mask推理函数
                outputs_no_mask = forward_with_no_mask(model, patches)
                consistency_loss = F.kl_div(F.log_softmax(outputs, dim=1), F.softmax(outputs_no_mask, dim=1), reduction='batchmean')
                loss = loss + 0.2 * consistency_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        scheduler.step()
        
        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for patches, _, labels in val_loader:
                patches, labels = patches.to(device), labels.to(device)
                # 使用修复后的无mask推理函数
                outputs = forward_with_no_mask(model, patches)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # 保存最佳模型
        if val_acc > best_val_acc or (val_acc == best_val_acc and avg_val_loss < best_val_loss):
            best_val_acc, best_val_loss, best_epoch = val_acc, avg_val_loss, epoch
            torch.save({
                'epoch': epoch, 
                'model_state_dict': actual_model.feature_extractor.state_dict(),
                'val_acc': val_acc, 
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"💎 [保存最佳DenseNet] Epoch {epoch+1}: Val Acc: {val_acc:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{num_epochs}] {stage_info}: LR: {current_lr:.1e}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    print(f"\n{'='*60}\nFold {fold_idx} 预训练完成\n{'='*60}")
    print(f"最佳DenseNet保存在 Epoch {best_epoch+1}")
    print(f"最佳验证指标 -> Acc: {best_val_acc:.4f}, Loss: {best_val_loss:.4f}")
    print(f"模型参数已保存至: {checkpoint_path}")
    
    return checkpoint_path

class DenseNetPretrainer:
    """DenseNet预训练器主类"""
    
    def __init__(self, data_dir, split_seed, save_dir='pretrained_densenet_models'):
        """
        初始化预训练器
        
        Args:
            data_dir: 数据目录
            split_seed: 数据划分种子 (与主训练保持一致)
            save_dir: 模型保存目录
        """
        self.data_dir = data_dir
        self.split_seed = split_seed
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"DenseNet预训练器初始化完成")
        print(f"数据目录: {data_dir}")
        print(f"分割种子: {split_seed}")
        print(f"保存目录: {save_dir}")
    
    def pretrain_all_folds(self, num_folds=5, num_epochs=50):
        """
        为所有折预训练DenseNet
        
        Args:
            num_folds: 折数
            num_epochs: 训练轮数
        
        Returns:
            dict: 各折模型路径字典
        """
        torch.manual_seed(42)
        np.random.seed(42)
        
        BATCH_SIZE = 16
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 加载数据集
        dataset = PreprocessedPatchDataset(data_dir=self.data_dir)
        num_patches = dataset.get_num_patches()
        labels = dataset.labels
        
        # 使用与主训练相同的种子进行分割
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=self.split_seed)
        
        fold_model_paths = {}
        
        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(labels, labels)):
            print(f"\n{'='*70}\nFOLD {fold_idx}/{num_folds}\n{'='*70}")

            # 进一步分割训练集和验证集 - 完全与第二个代码等价
            train_indices = train_val_idx.copy()  # 复制数组避免修改原数组
            np.random.seed(self.split_seed)  # 修改2：使用相同的种子
            np.random.shuffle(train_indices)
            val_size = int(len(train_indices) * 0.2)
            val_idx = train_indices[:val_size]
            train_idx = train_indices[val_size:]
            print(f"{fold_idx}:{val_idx}")
            # 创建数据加载器
            train_loader = DataLoader(
                Subset(dataset, train_idx), 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                collate_fn=collate_fn, 
                num_workers=4
            )
            val_loader = DataLoader(
                Subset(dataset, val_idx), 
                batch_size=BATCH_SIZE, 
                shuffle=False, 
                collate_fn=collate_fn, 
                num_workers=4
            )
            
            # 创建模型
            feature_extractor = LightDenseNet3D(growth_rate=8, num_init_features=24)
            model = EndToEndModel(feature_extractor, num_patches).to(device)
            
            # 只在有多个GPU且数据量足够大时使用DataParallel
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                print(f"检测到 {torch.cuda.device_count()} 个GPU，使用DataParallel")
                model = nn.DataParallel(model)
            
            # 预训练当前折
            model_path = pretrain_feature_extractor(
                model, train_loader, val_loader, device, fold_idx, num_epochs
            )
            
            fold_model_paths[fold_idx] = model_path
            
            # 清理显存
            del model, feature_extractor
            torch.cuda.empty_cache()
        
        print(f"\n{'='*70}")
        print("所有折的DenseNet预训练完成!")
        print(f"{'='*70}")
        
        # 保存模型路径映射
        paths_file = os.path.join(self.save_dir, f'model_paths_seed{self.split_seed}.pkl')
        with open(paths_file, 'wb') as f:
            pickle.dump(fold_model_paths, f)
        
        print(f"模型路径映射已保存至: {paths_file}")
        
        for fold_idx, path in fold_model_paths.items():
            print(f"  Fold {fold_idx}: {path}")
        
        return fold_model_paths

def main():
    """独立运行的主函数"""
    # 配置参数 - 请根据您的设置修改
    DATA_DIR = 'MCIvsCN.ex001_2_p288/MCIvsCN.ex001_2_p288_pw24_all'  # 修改为您的数据目录
    SPLIT_SEED = 1449  # 与主训练脚本保持一致！
    SAVE_DIR = f'pretrained_densenet_split_seed{SPLIT_SEED}'
    NUM_EPOCHS = 50
    
    print(f"开始DenseNet预训练...")
    print(f"数据目录: {DATA_DIR}")
    print(f"分割种子: {SPLIT_SEED}")
    print(f"保存目录: {SAVE_DIR}")
    print(f"训练轮数: {NUM_EPOCHS}")
    
    # 创建预训练器
    pretrainer = DenseNetPretrainer(
        data_dir=DATA_DIR,
        split_seed=SPLIT_SEED,
        save_dir=SAVE_DIR
    )
    
    # 执行预训练
    model_paths = pretrainer.pretrain_all_folds(
        num_folds=5,
        num_epochs=NUM_EPOCHS
    )
    
    print(f"\n{'='*70}")
    print("DenseNet预训练全部完成!")
    print(f"所有模型已保存到: {SAVE_DIR}")
    print("现在可以在主训练脚本中使用这些预训练模型了。")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()