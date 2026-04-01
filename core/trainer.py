"""
Causal Graph Neural Network Trainer (简化优雅版 - 无大图逻辑)
端到端训练流程：预训练 + 两阶段因果学习

核心改进：
1. 移除复杂的大图构建逻辑
2. 使用批次内聚合作为融合替换源
3. Stage 1 和 Stage 2 统一数据流
4. 简洁的稀疏度控制策略
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Any, Optional
import numpy as np

from utils.checkpoint import CheckpointManager
from models.causal_net import CausalNet
from models.causal_mask import CausalMask
from utils.metrics import compute_binary_metrics
from utils.mask_monitor import MaskMonitor
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
logger = logging.getLogger(__name__)


class CausalTrainer:
    """
    因果图神经网络训练器（简化版）
    
    训练流程：
    1. 预训练阶段 (40 epochs)：整体图预测，建立基础特征表示
    2. 阶段1 (40 epochs)：学习因果掩码（内在子图 + 虚假子图）
    3. 阶段2 (60 epochs)：因果测试与鲁棒性增强（批次内融合）
    
    核心组件：
    - DenseNet：冻结的特征提取器
    - CausalMask：可学习的因果掩码（节点 + 边）
    - CausalNet：图神经网络分类器
    
    关键简化：
    - 移除大图构建器，使用批次内均值作为融合替换源
    - Stage 1 和 Stage 2 使用统一的数据结构 [B, P, d]
    """
    
    def __init__(
            self,
            config: Dict[str, Any],
            fold: int,
            densenet_model: nn.Module,
            edge_prior_mask: np.ndarray,
            checkpoint_manager: CheckpointManager,
            work_dir: str,
            device: str = 'cuda',
            rank: int = 0,
            coordinates: Optional[np.ndarray] = None,
            use_feature_cache: bool = True
        ):
        self.config = config
        self.fold = fold
        self.densenet_model = densenet_model.to(device)
        self.edge_prior_mask = torch.FloatTensor(edge_prior_mask).to(device)
        self.checkpoint_manager = checkpoint_manager
        self.work_dir = work_dir
        self.device = device
        self.rank = rank
        self.coordinates = coordinates
        self.use_feature_cache = use_feature_cache

        # 冻结DenseNet特征提取器
        for param in self.densenet_model.parameters():
            param.requires_grad = False
        self.densenet_model.eval()

        if use_feature_cache:
            logger.info("✅ Feature cache enabled - DenseNet frozen")
            self.feature_extract_batch_size = 32
        else:
            logger.warning("⚠️ Feature cache disabled - GPU utilization may be low!")
            # 多GPU处理
            if torch.cuda.device_count() > 1:
                num_gpus = torch.cuda.device_count()
                self.densenet_model = nn.DataParallel(self.densenet_model)
                self.feature_extract_batch_size = 64 * num_gpus
            else:
                self.feature_extract_batch_size = 32

        # 初始化占位符
        self.model = None
        self.mask = None
        self.optimizer = None
        self.optimizer_mask = None
        self.optimizer_pretrain = None
        self.lr_scheduler = None
        self.lr_scheduler_mask = None
        self.scheduler_pretrain = None

        # 状态追踪变量
        self.global_step = 0
        self.pretrain_best_val_acc = 0.0
        self.pretrain_best_model_state = None
        self.pretrain_best_epoch = -1
        self.best_val_acc = 0.0
        self.best_test_acc = 0.0
        self.best_epoch = -1
        self.best_model_state = None
        self.epoch_results = {}
        self.current_mask_sums = {}

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.lambda_l1 = config['train']['loss_weights']['lambda_l1']

        # Mask 监控器
        self.mask_monitor = MaskMonitor(
            save_dir=str(Path(work_dir).parent),
            fold=fold
        )

        logger.info(f"✅ Trainer initialized for Fold {fold}")


    def _build_models(self):
        """构建因果图神经网络和掩码模型"""
        if isinstance(self.densenet_model, nn.DataParallel):
            feature_dim = self.densenet_model.module.feature_dim
        else:
            feature_dim = self.densenet_model.feature_dim

        # 主GNN模型
        self.model = CausalNet(
            num_class=2,
            feature_dim=feature_dim,
            hidden1=self.config['model']['args']['hidden1'],
            hidden2=self.config['model']['args']['hidden2'],
            kernels=self.config['model']['args'].get('kernels', [2]),
            num_patches=self.edge_prior_mask.shape[0],
            # 【新增】传入坐标和参数
            coordinates=self.coordinates,
            spatial_sigma=48.0 # 可以根据 patch size (24) * 2 设置
        ).to(self.device)

        # 因果掩码模型
        self.mask = CausalMask(
            num_patches=self.edge_prior_mask.shape[0], 
            edge_matrix=self.edge_prior_mask,    
            gumble_tau=self.config['misc']['gumble_tau']
        ).to(self.device)

        # 多GPU并行
        if torch.cuda.device_count() > 1:
            logger.info(f"🎮 Wrapping CausalNet and Mask with DataParallel across {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
            self.mask = nn.DataParallel(self.mask)

        logger.info("✅ Models built")
    
    def _setup_optimizers(self):
            """配置优化器和学习率调度器 (双阶段余弦退火版)"""
            # 1. 预训练阶段 (保持不变)
            self.optimizer_pretrain = optim.Adam(
                self.model.parameters(),
                lr=self.config['densenet']['pretrain']['learning_rate'],
                weight_decay=self.config['densenet']['pretrain']['weight_decay']
            )
            self.scheduler_pretrain = CosineAnnealingLR(
                self.optimizer_pretrain,
                T_max=self.config['train']['pre_epoch'],
                eta_min=1e-6
            )

            # 2. 主训练阶段优化器 (保持不变)
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['train']['base_lr'],
                weight_decay=self.config['train']['weight_decay_adamw']
            )

            self.optimizer_mask = optim.SGD(
                self.mask.parameters(),
                lr=self.config['train']['base_lr_mask'],
                momentum=0.9,
                nesterov=True,
                weight_decay=self.config['train']['weight_decay']
            )

            # ============================================================
            # 【修改】初始化 Stage 1 的余弦调度器
            # T_max = Stage 1 的总 epoch 数
            # ============================================================
            stage1_duration = self.config['train']['stage_transition_epoch'] - self.config['train']['pre_epoch']

            # 确保 duration > 0，防止配置错误
            stage1_duration = max(1, stage1_duration)

            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=stage1_duration,
                eta_min=1e-6
            )
            self.lr_scheduler_mask = CosineAnnealingLR(
                self.optimizer_mask,
                T_max=stage1_duration,
                eta_min=1e-6
            )

            logger.info(f"✅ Optimizers configured with CosineAnnealingLR (Stage 1 duration: {stage1_duration})")
    
    # ============================================================================
    # 主训练入口
    # ============================================================================
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        完整训练流程
        
        Returns:
            最终评估结果字典
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 Training Start - Fold {self.fold}")
        logger.info(f"{'='*80}\n")
        
        # 初始化模型和优化器
        self._build_models()
        self._setup_optimizers()
        
        # 阶段1: 预训练 (整体图预测)
        pre_epochs = self.config['train']['pre_epoch']
        if pre_epochs > 0:
            logger.info("\n" + "="*80)
            logger.info("📚 Phase 1: Pre-training (Whole Graph Prediction)")
            logger.info("="*80)
            self._pretrain_phase(train_loader, val_loader, test_loader, pre_epochs)
        
        # 阶段2+3: 主训练 (因果掩码学习 + 鲁棒性增强)
        logger.info("\n" + "="*80)
        logger.info("🎯 Phase 2+3: Causal Learning (Mask + GNN Co-training)")
        logger.info("="*80)
        self._main_training(train_loader, val_loader, test_loader)
        
        # 最终评估
        final_results = self._final_evaluation(test_loader)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Training Completed - Fold {self.fold}")
        logger.info(f"   Best Val Acc:  {self.best_val_acc:.4f}")
        logger.info(f"   Best Test Acc: {self.best_test_acc:.4f}")
        logger.info(f"{'='*80}\n")
        
        return final_results
    
    # ============================================================================
    # 预训练阶段 (整体图预测)
    # ============================================================================
    
    def _pretrain_phase(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int
    ):
        """预训练阶段：学习整体图的预测能力"""
        for epoch in range(num_epochs):
            if self.rank == 0:
                self.epoch_results[epoch] = {}

            # 训练
            self._train_pretrain_epoch(epoch, train_loader)

            # 评估
            if self.rank == 0:
                with torch.no_grad():
                    self._eval_pretrain_epoch(epoch, val_loader, 'val')
                    self._eval_pretrain_epoch(epoch, test_loader, 'test')

                self._print_pretrain_summary(epoch)

            # 学习率调度
            self.scheduler_pretrain.step()

        # 加载最佳模型
        if self.rank == 0 and self.pretrain_best_model_state:
            logger.info(f"✅ Loading best pretrain model (Epoch {self.pretrain_best_epoch+1})")
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(self.pretrain_best_model_state)
            else:
                self.model.load_state_dict(self.pretrain_best_model_state)
    
    def _train_pretrain_epoch(self, epoch: int, train_loader: DataLoader):
        """预训练的单个epoch"""
        self.model.train()
        
        losses = []
        accuracies = []
        
        for data, _, label in train_loader:
            self.global_step += 1
            
            data = data.to(self.device)
            label = label.to(self.device)
            
            # 提取特征
            x_features = self._extract_features(data)
            
            # 整体预测
            outputs = self.model.module.prediction_whole(x_features, self.edge_prior_mask) \
                if isinstance(self.model, nn.DataParallel) else \
                self.model.prediction_whole(x_features, self.edge_prior_mask)
            
            # 计算损失
            loss = self.criterion(outputs, label).mean()
            l1_loss = self._compute_l1_regularization()
            loss_total = loss + l1_loss
            
            # 反向传播
            self.optimizer_pretrain.zero_grad()
            loss_total.backward()
            self.optimizer_pretrain.step()
            
            # 记录
            if self.rank == 0:
                losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                acc = (predicted == label).float().mean().item()
                accuracies.append(acc)
        
        # 保存结果
        if self.rank == 0:
            self.epoch_results[epoch]['train'] = {
                'loss_all': np.mean(losses),
                'acc_Intrinsic': np.mean(accuracies)
            }
    
    def _eval_pretrain_epoch(self, epoch: int, data_loader: DataLoader, phase: str):
        """预训练的评估"""
        self.model.eval()

        all_outputs = []
        all_labels = []

        for data, _, label in data_loader:
            data = data.to(self.device)
            label = label.to(self.device)

            x_features = self._extract_features(data)
            outputs = self.model.module.prediction_whole(x_features, self.edge_prior_mask) \
                if isinstance(self.model, nn.DataParallel) else \
                self.model.prediction_whole(x_features, self.edge_prior_mask)

            all_outputs.append(outputs)
            all_labels.append(label)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = compute_binary_metrics(all_outputs, all_labels)
        self.epoch_results[epoch][phase] = metrics

        # 保存最佳模型
        if phase == 'val' and metrics['accuracy'] > self.pretrain_best_val_acc:
            self.pretrain_best_val_acc = metrics['accuracy']
            self.pretrain_best_epoch = epoch

            if isinstance(self.model, nn.DataParallel):
                self.pretrain_best_model_state = self.model.module.state_dict()
            else:
                self.pretrain_best_model_state = self.model.state_dict()

            logger.info(f"💎 New Best Pretrain - Val Acc: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
    
    # ============================================================================
    # 主训练阶段 (因果掩码学习)
    # ============================================================================
    
    def _main_training(self, train_loader, val_loader, test_loader):
            """
            主训练阶段：阶段1（因果分离） + 阶段2（鲁棒性增强）

            包含策略：
            1. Stage 1: 探索因果结构 (Mask LR 从高到低)
            2. Transition: 
               - 加载 Stage 1 最佳模型权重
               - 重置学习率为初始值的一半 (Warm Restart)
               - 重置调度器为新的余弦退火
            3. Stage 2: 在强约束下微调 (Mask LR 重新开始衰减)
            """
            start_epoch = self.config['train']['pre_epoch']
            num_epochs = self.config['train']['num_epoch']
            stage_transition = self.config['train']['stage_transition_epoch']

            # 阶段1的最佳模型追踪
            stage1_best_val_acc = 0.0
            stage1_best_model_state = None
            stage1_best_epoch = -1

            for epoch in range(start_epoch, num_epochs):
                is_stage1 = epoch < stage_transition

                # ============================================================
                # 🔄 阶段切换逻辑：重置学习率与调度器 (Warm Restart)
                # ============================================================
                if epoch == stage_transition:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"🔄 Entering Stage 2 (Epoch {epoch+1}): Performing Warm Restart")

                    # 1. 计算新的基础学习率 (初始的一半)
                    base_lr_gnn = self.config['train']['base_lr'] * 0.5
                    base_lr_mask = self.config['train']['base_lr_mask'] * 0.5

                    # 2. 手动更新优化器的当前学习率
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = base_lr_gnn
                    for param_group in self.optimizer_mask.param_groups:
                        param_group['lr'] = base_lr_mask

                    logger.info(f"   ⬇️ Reset GNN LR:  {self.config['train']['base_lr']} -> {base_lr_gnn}")
                    logger.info(f"   ⬇️ Reset Mask LR: {self.config['train']['base_lr_mask']} -> {base_lr_mask}")

                    # 3. 为 Stage 2 重建余弦退火调度器
                    # 这样调度器会以新的 LR 为起点，进行新一轮的衰减
                    stage2_duration = num_epochs - stage_transition
                    self.lr_scheduler = CosineAnnealingLR(
                        self.optimizer, 
                        T_max=stage2_duration, 
                        eta_min=1e-6
                    )
                    self.lr_scheduler_mask = CosineAnnealingLR(
                        self.optimizer_mask, 
                        T_max=stage2_duration, 
                        eta_min=1e-6
                    )
                    logger.info(f"   🔄 Schedulers re-initialized for {stage2_duration} epochs")
                    logger.info(f"{'='*60}\n")
                # ============================================================

                if self.rank == 0:
                    self.epoch_results[epoch] = {}

                # 1. 训练
                self._train_main_epoch(epoch, train_loader, is_stage1)

                # 2. 评估
                if self.rank == 0:
                    with torch.no_grad():
                        self._eval_main_epoch(epoch, val_loader, 'val')
                        self._eval_main_epoch(epoch, test_loader, 'test')

                    # [Stage 1] 保存最佳模型
                    if is_stage1:
                        val_acc = self.epoch_results[epoch]['val']['accuracy']
                        if val_acc > stage1_best_val_acc:
                            stage1_best_val_acc = val_acc
                            stage1_best_epoch = epoch

                            # 保存模型状态字典
                            if isinstance(self.model, nn.DataParallel):
                                stage1_best_model_state = self.model.module.state_dict()
                            else:
                                stage1_best_model_state = self.model.state_dict()

                            # 注意：我们不保存 mask 的状态，让它自然演化，
                            # 或者如果希望 mask 也回滚，也可以在这里保存 mask state

                            logger.info(f"💎 New Best Stage1 - Val Acc: {val_acc:.4f}")

                    # 打印日志
                    self._print_main_summary(epoch, is_stage1)

                    # 3. 学习率调度 (统一使用 Step，不再依赖 val_loss)
                    # 因为我们已经强制使用了 CosineAnnealingLR
                    self.lr_scheduler.step()
                    self.lr_scheduler_mask.step()

                # [Stage 1 结束] 加载最佳模型作为 Stage 2 起点
                # 注意：必须在 epoch 结束且尚未进入 Stage 2 循环前完成
                if epoch == stage_transition - 1 and self.rank == 0 and stage1_best_model_state:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"✅ Stage 1 Completed! Loading best GNN weights for Stage 2")
                    logger.info(f"   Best Stage1 Epoch: {stage1_best_epoch + 1}")
                    logger.info(f"   Best Stage1 Val Acc: {stage1_best_val_acc:.4f}")
                    logger.info(f"{'='*80}\n")

                    # 加载 GNN 权重
                    if isinstance(self.model, nn.DataParallel):
                        self.model.module.load_state_dict(stage1_best_model_state)
                    else:
                        self.model.load_state_dict(stage1_best_model_state)

                    # 重置 Stage 2 的追踪指标
                    self.best_val_acc = 0.0
                    self.best_test_acc = 0.0
                    self.best_epoch = -1   
    
    def _train_main_epoch(self, epoch: int, train_loader: DataLoader, is_stage1: bool):
            """
            主训练的单个epoch（Mask和GNN交替训练）

            优化点：
            - 注册 prototype_div 指标
            - 统一数据流，移除大图逻辑
            - 修正调用参数顺序
            """
            self.model.train()
            self.mask.train()

            # 损失和准确率累积器
            # 【修改】这里加入了 'prototype_div' 以便记录日志
            losses_mask = {
                'all': [], 
                'Intrinsic': [], 
                'Spurious': [], 
                'spurious_fusion': [],
                'intrinsic_fusion': [], 
                'sparsity_reg': [],
                'prototype_div': []   # <--- 新增指标注册
            }

            losses_gnn = {
                'all': [], 'Intrinsic': [], 'spurious_fusion': [], 'l1_reg': []
            }
            accs_mask = {}
            accs_gnn = {}

            for batch_idx, (data, _, label) in enumerate(train_loader):
                self.global_step += 1
                label = label.to(self.device)

                # ✅ 统一处理：无论Stage 1还是Stage 2，都直接使用原始数据提取特征
                # (不再需要 build_large_graph)
                data = data.to(self.device)
                x_features = self._extract_features(data)

                # ========== 步骤1: 训练Mask（固定GNN） ==========
                for param in self.model.parameters():
                    param.requires_grad = False

                mask_module = self.mask.module if isinstance(self.mask, nn.DataParallel) else self.mask
                masks, sparsity = mask_module(train=True)

                if is_stage1:
                    result_mask = self._compute_stage1_mask_loss(
                        x_features, masks, label, epoch, self.edge_prior_mask
                    )
                else:
                    # Stage 2: 使用动态 Ratio + 原型分离
                    result_mask = self._compute_stage2_mask_loss(
                        x_features, masks, label, epoch, self.edge_prior_mask
                    )

                self.optimizer_mask.zero_grad()
                if 'all' in result_mask['loss'] and isinstance(result_mask['loss']['all'], torch.Tensor):
                    result_mask['loss']['all'].backward()
                self.optimizer_mask.step()

                # 记录Mask损失
                for k in losses_mask.keys():
                    # 使用 .get(k, 0) 防止某个阶段没有该 loss 报错
                    val = result_mask['loss'].get(k, 0)
                    if isinstance(val, torch.Tensor):
                        losses_mask[k].append(val.item())
                    else:
                        losses_mask[k].append(float(val))

                for k, v in result_mask['preds'].items():
                    if k not in accs_mask:
                        accs_mask[k] = []
                    accs_mask[k].append(self._compute_accuracy(v, label))

                # ========== 步骤2: 训练GNN（固定Mask） ==========
                for param in self.model.parameters():
                    param.requires_grad = True

                masks, sparsity = mask_module(train=False)
                masks = [m.detach() for m in masks]

                if is_stage1:
                    result_gnn = self._compute_stage1_gnn_loss(
                        x_features, masks, label, self.edge_prior_mask
                    )
                else:
                    result_gnn = self._compute_stage2_gnn_loss(
                        x_features, masks, label, self.edge_prior_mask
                    )

                self.optimizer.zero_grad()
                if 'all' in result_gnn['loss'] and isinstance(result_gnn['loss']['all'], torch.Tensor):
                    result_gnn['loss']['all'].backward()
                self.optimizer.step()

                # 记录GNN损失
                for k in losses_gnn.keys():
                    val = result_gnn['loss'].get(k, 0)
                    if isinstance(val, torch.Tensor):
                        losses_gnn[k].append(val.item())
                    else:
                        losses_gnn[k].append(float(val))

                for k, v in result_gnn['preds'].items():
                    if k not in accs_gnn:
                        accs_gnn[k] = []
                    accs_gnn[k].append(self._compute_accuracy(v, label))

                # 更新当前掩码统计
                if self.rank == 0:
                    self.current_mask_sums = {
                        'node': masks[0].sum().item(),
                        'edge': masks[1].sum().item()
                    }

            # Epoch结束，保存结果
            if self.rank == 0:
                train_res = {'mask': {}, 'gnn': {}}

                for k in losses_mask.keys():
                    train_res['mask'][k] = float(np.mean(losses_mask[k])) if len(losses_mask[k]) > 0 else 0.0
                for k in losses_gnn.keys():
                    train_res['gnn'][k] = float(np.mean(losses_gnn[k])) if len(losses_gnn[k]) > 0 else 0.0
                for k, v in accs_mask.items():
                    train_res['mask'][f'acc_{k}'] = float(np.mean(v)) if len(v) > 0 else 0.0
                for k, v in accs_gnn.items():
                    train_res['gnn'][f'acc_{k}'] = float(np.mean(v)) if len(v) > 0 else 0.0

                self.epoch_results[epoch]['train'] = train_res
    
    def _eval_main_epoch(self, epoch: int, data_loader: DataLoader, phase: str):
        """主训练阶段的评估"""
        self.model.eval()
        self.mask.eval()

        all_outputs = []
        all_labels = []

        for data, _, label in data_loader:
            data = data.to(self.device)
            label = label.to(self.device)

            x_features = self._extract_features(data)
            mask_module = self.mask.module if isinstance(self.mask, nn.DataParallel) else self.mask
            masks, probs, sparsity = mask_module(train=False, return_probs=True)

            model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            outputs = model_module.prediction_intrinsic_path(
                x_features, self.edge_prior_mask, masks
            )

            all_outputs.append(outputs)
            all_labels.append(label)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_binary_metrics(all_outputs, all_labels)
        self.epoch_results[epoch][phase] = metrics

        # 保存 Mask Logits
        if self.rank == 0 and phase == 'val':
            mask_module = self.mask.module if isinstance(self.mask, nn.DataParallel) else self.mask

            # 准备额外信息
            additional_info = {
                'val_acc': metrics['accuracy'],
                'val_auc': metrics['auc'],
                'val_f1': metrics['f1'],
                'val_sensitivity': metrics['sensitivity'],
                'val_specificity': metrics['specificity'],
                'val_precision': metrics['precision'],
            }

            # 如果有测试集结果，也加入
            if 'test' in self.epoch_results.get(epoch, {}):
                test_metrics = self.epoch_results[epoch]['test']
                additional_info.update({
                    'test_acc': test_metrics['accuracy'],
                    'test_auc': test_metrics['auc'],
                    'test_f1': test_metrics['f1'],
                    'test_sensitivity': test_metrics['sensitivity'],
                    'test_specificity': test_metrics['specificity'],
                    'test_precision': test_metrics['precision'],
                })

            # 添加训练损失信息（如果有）
            if 'train' in self.epoch_results.get(epoch, {}):
                train_res = self.epoch_results[epoch]['train']
                mask_res = train_res.get('mask', {})
                gnn_res = train_res.get('gnn', {})

                additional_info.update({
                    'train_mask_loss': mask_res.get('all', 0),
                    'train_gnn_loss': gnn_res.get('all', 0),
                    'train_sparsity_loss': mask_res.get('sparsity_reg', 0),
                })

            # 保存当前epoch的masks
            self.mask_monitor.save_epoch_masks(
                epoch=epoch,
                mask_module=mask_module,
                additional_info=additional_info
            )

        # 仅在阶段2更新最终评估的最佳模型
        stage_transition = self.config['train']['stage_transition_epoch']
        if phase == 'val' and epoch >= stage_transition and metrics['accuracy'] > self.best_val_acc:
            self.best_val_acc = metrics['accuracy']
            self.best_epoch = epoch

            if 'test' in self.epoch_results[epoch]:
                self.best_test_acc = self.epoch_results[epoch]['test']['accuracy']
                logger.info(
                    f"💎 New Best Stage2 - Val Acc: {metrics['accuracy']:.4f}, "
                    f"Val AUC: {metrics['auc']:.4f}, Test Acc: {self.best_test_acc:.4f}"
                )
            else:
                logger.info(
                    f"💎 New Best Stage2 Val - Acc: {metrics['accuracy']:.4f}, "
                    f"AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}"
                )
    
    # ============================================================================
    # 损失计算函数（移除所有 is_large_graph 参数）
    # ============================================================================
    
    def _compute_stage1_mask_loss(self, x, masks, label, epoch, edge_prior_mask):
        """
        阶段1 Mask损失：内在子图 + 虚假子图
        
        目标：
        - 内在子图：准确预测标签
        - 虚假子图：最大化熵（无判别力）
        - 稀疏性：直接惩罚因果特征数量
        """
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        # 内在子图预测
        y_pred = model.prediction_intrinsic_path(x, edge_prior_mask, masks)
        loss_pred = self.criterion(y_pred, label).mean()
        
        # 虚假子图（熵损失）
        y_spu = model.prediction_spurious_path(x, edge_prior_mask, masks)
        loss_spu = self._entropy_loss(y_spu)
        
        # 稀疏性正则（简洁版）
        mask_module = self.mask.module if isinstance(self.mask, nn.DataParallel) else self.mask
        reg_loss = mask_module.compute_sparsity_regularization(
            lambda_reg=self.config['train']['loss_weights']['lambda_sparsity'],
            lambda_edge_multiplier=self.config['train']['loss_weights'].get('lambda_edge_multiplier', 3.0)
        )
        
        # 组合损失
        loss_weights = self.config['train']['loss_weights']
        loss_all = loss_weights['L_pred'] * loss_pred + loss_weights['L_spu'] * loss_spu + reg_loss
        
        return {
            'loss': {
                'all': loss_all,
                'Intrinsic': loss_pred,
                'Spurious': loss_spu,
                'sparsity_reg': reg_loss
            },
            'preds': {
                'Intrinsic': y_pred,
                'Spurious': y_spu
            }
        }
    
    def _compute_stage2_mask_loss(self, x, masks, label, epoch, edge_prior_mask):
            """阶段2 Mask损失 (Final Version - Added Sensitivity Loss)"""
            model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

            # 1. 基础预测 (L_pred)
            y_pred = model.prediction_intrinsic_path(x, edge_prior_mask, masks)
            loss_pred = self.criterion(y_pred, label).mean()

            # 2. 虚假融合 / 不变性测试 (L_inv)
            # 逻辑：保留内在因果，替换虚假背景 -> 预测应【不变】 (Label)
            y_inv = model.prediction_spurious_fusion(x, edge_prior_mask, masks, labels=label)
            loss_inv = self.criterion(y_inv, label).mean()

            # 3. 【新增】内在融合 / 敏感性测试 (L_sen) 
            # 逻辑：保留虚假背景，替换内在因果 -> 预测应【翻转】 (1 - Label)
            # -------------------------------------------------------------------------
            y_sen = model.prediction_intrinsic_fusion(x, edge_prior_mask, masks, labels=label)
            # 注意：目标是 (1 - label)，迫使模型意识到内在因果部分被篡改了
            loss_sen = self.criterion(y_sen, 1 - label).mean()
            # -------------------------------------------------------------------------

            # 4. 动态计算 Mining Ratio (课程学习)
            start_ratio = 0.4
            end_ratio = 0.1
            current_stage_epoch = epoch - self.config['train']['pre_epoch']
            max_stage_epochs = self.config['train']['num_epoch'] - self.config['train']['pre_epoch']
            
            if max_stage_epochs > 0:
                progress = max(0, min(1, current_stage_epoch / max_stage_epochs))
            else:
                progress = 1.0
            current_ratio = start_ratio - (start_ratio - end_ratio) * progress

            # 5. 原型分离损失
            loss_proto = model.compute_prototype_divergence(
                x, masks, label, 
                mining_ratio=current_ratio
            )

            # 6. 稀疏性正则
            mask_module = self.mask.module if isinstance(self.mask, nn.DataParallel) else self.mask
            reg_loss = mask_module.compute_sparsity_regularization(
                lambda_reg=self.config['train']['loss_weights']['lambda_sparsity'],
                lambda_edge_multiplier=self.config['train']['loss_weights'].get('lambda_edge_multiplier', 3.0)
            )

            # 7. 组合所有损失
            loss_weights = self.config['train']['loss_weights']
            lambda_proto = loss_weights.get('lambda_proto', 1.0)

            loss_all = (
                loss_weights['L_inv'] * loss_inv + 
                loss_weights.get('L_sen', 0.5) * loss_sen +  # 【新增】加入 L_sen
                loss_weights['L_pred'] * loss_pred + 
                lambda_proto * loss_proto + 
                reg_loss
            )

            return {
                'loss': {
                    'all': loss_all,
                    'spurious_fusion': loss_inv,
                    'intrinsic_fusion': loss_sen,  # 【新增】记录日志
                    'prototype_div': loss_proto,
                    'Intrinsic': loss_pred,
                    'sparsity_reg': reg_loss
                },
                'preds': {
                    'spurious_fusion': y_inv,
                    'intrinsic_fusion': y_sen,     # 【新增】记录预测值用于计算准确率
                    'Intrinsic': y_pred
                }
            }

    def _compute_stage1_gnn_loss(self, x, masks, label, edge_prior_mask):
        """阶段1 GNN损失：内在子图预测"""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        y_pred = model.prediction_intrinsic_path(x, edge_prior_mask, masks)
        loss_pred = self.criterion(y_pred, label).mean()
        #l1_loss = self._compute_l1_regularization()
        l1_loss = 0
        loss_all = loss_pred + l1_loss
        
        return {
            'loss': {
                'all': loss_all,
                'Intrinsic': loss_pred,
                'l1_reg': l1_loss
            },
            'preds': {
                'Intrinsic': y_pred
            }
        }
    
    def _compute_stage2_gnn_loss(self, x, masks, label, edge_prior_mask):
        """阶段2 GNN损失：内在子图 + 虚假融合"""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        y_pred = model.prediction_intrinsic_path(x, edge_prior_mask, masks)
        loss_pred = self.criterion(y_pred, label).mean()
        
        y_inv = model.prediction_spurious_fusion(x, edge_prior_mask, masks, labels=label)
        loss_inv = self.criterion(y_inv, label).mean()
        
        #l1_loss = self._compute_l1_regularization()
        l1_loss = 0
        loss_all = loss_pred + loss_inv + l1_loss
        
        return {
            'loss': {
                'all': loss_all,
                'Intrinsic': loss_pred,
                'spurious_fusion': loss_inv,
                'l1_reg': l1_loss
            },
            'preds': {
                'Intrinsic': y_pred,
                'spurious_fusion': y_inv
            }
        }
    
    # ============================================================================
    # 辅助函数
    # ============================================================================
    def _extract_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        特征提取函数

        [特征缓存模式] (self.use_feature_cache=True):
            - data 已经是预计算的特征 [B, P, feature_dim]
            - 直接移到GPU并返回，无需计算
            - 大幅提升GPU利用率（避免实时特征提取的CPU瓶颈）

        [原始模式] (self.use_feature_cache=False):
            - data 是原始patches [B, P, 1, D, H, W]
            - 使用冻结的DenseNet逐批提取特征
            - GPU利用率较低（特征提取成为瓶颈）

        Args:
            data: 输入数据
                - 特征模式: [B, P, feature_dim]
                - 原始模式: [B, P, 1, D, H, W]

        Returns:
            特征张量 [B, P, feature_dim]
        """
        # ============================================================
        # [特征缓存模式] 直接使用预计算特征
        # ============================================================
        if self.use_feature_cache:
            # data 已经是特征 [B, P, feature_dim]
            # 只需要移到正确的设备
            return data.to(self.device)

        # ============================================================
        # [原始模式] 实时提取特征（性能较低，不推荐）
        # ============================================================
        B, P = data.shape[0], data.shape[1]

        # Reshape: [B, P, 1, D, H, W] -> [B*P, 1, D, H, W]
        data_reshaped = data.view(-1, 1, data.shape[3], data.shape[4], data.shape[5])
        total_patches = data_reshaped.shape[0]

        batch_size = getattr(self, 'feature_extract_batch_size', 32)

        all_features = []
        with torch.no_grad():
            for i in range(0, total_patches, batch_size):
                batch = data_reshaped[i:i+batch_size]
                features_batch = self.densenet_model(batch)
                all_features.append(features_batch)

        features = torch.cat(all_features, dim=0)
        features = features.view(B, P, -1)

        return features
    
    def _compute_l1_regularization(self) -> torch.Tensor:
        """计算GNN参数的L1正则化"""
        l1_reg = torch.tensor(0., device=self.device)
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        for name, param in model.named_parameters():
            if 'mlp_causal.0.weight' in name:
                l1_reg += torch.sum(torch.abs(param))
        
        return self.lambda_l1 * l1_reg
    
    def _entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        熵损失（用于虚假子图）
        
        目标：最大化预测熵，使虚假子图无判别力
        """
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        return -entropy.mean()  # 负号：最大化熵
    
    def _compute_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """计算分类准确率"""
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean().item()
        return acc
    
    # ============================================================================
    # 日志打印
    # ============================================================================
    
    def _print_pretrain_summary(self, epoch: int):
        """打印预训练阶段的训练总结"""
        res = self.epoch_results[epoch]
        train_res = res.get('train', {})
        val_res = res.get('val', {})
        test_res = res.get('test', {})

        logger.info("="*80)
        logger.info(f"📚 Epoch {epoch+1}/{self.config['train']['pre_epoch']} [Pre-training]")
        logger.info("-"*80)
        logger.info(f"Train - Loss: {train_res.get('loss_all', 0):.4f}, "
                   f"Acc: {train_res.get('acc_Intrinsic', 0):.4f}")

        logger.info(f"Val   - Acc: {val_res.get('accuracy', 0):.4f}, "
                   f"AUC: {val_res.get('auc', 0):.4f}, "
                   f"F1: {val_res.get('f1', 0):.4f}")
        logger.info(f"        Sens: {val_res.get('sensitivity', 0):.4f}, "
                   f"Spec: {val_res.get('specificity', 0):.4f}")

        logger.info(f"Test  - Acc: {test_res.get('accuracy', 0):.4f}, "
                   f"AUC: {test_res.get('auc', 0):.4f}, "
                   f"F1: {test_res.get('f1', 0):.4f}")
        logger.info(f"        Sens: {test_res.get('sensitivity', 0):.4f}, "
                   f"Spec: {test_res.get('specificity', 0):.4f}")

        logger.info("="*80 + "\n")
    
    def _print_main_summary(self, epoch: int, is_stage1: bool):
            """打印主训练阶段的训练总结"""
            res = self.epoch_results[epoch]
            train_res = res.get('train', {})
            val_res = res.get('val', {})
            test_res = res.get('test', {})

            mask_res = train_res.get('mask', {})
            gnn_res = train_res.get('gnn', {})

            stage_name = "Stage 1 (Causal Separation)" if is_stage1 else "Stage 2 (Robustness Enhancement)"

            logger.info("="*80)
            logger.info(f"🎯 Epoch {epoch+1}/{self.config['train']['num_epoch']} [{stage_name}]")
            logger.info("-"*80)

            # 官方评估指标
            logger.info(f"📊 Validation Metrics:")
            logger.info(f"   Acc: {val_res.get('accuracy', 0):.4f}, "
                       f"AUC: {val_res.get('auc', 0):.4f}, "
                       f"F1: {val_res.get('f1', 0):.4f}")
            logger.info(f"   Sens: {val_res.get('sensitivity', 0):.4f}, "
                       f"Spec: {val_res.get('specificity', 0):.4f}, "
                       f"Prec: {val_res.get('precision', 0):.4f}")

            logger.info(f"\n📊 Test Metrics:")
            logger.info(f"   Acc: {test_res.get('accuracy', 0):.4f}, "
                       f"AUC: {test_res.get('auc', 0):.4f}, "
                       f"F1: {test_res.get('f1', 0):.4f}")
            logger.info(f"   Sens: {test_res.get('sensitivity', 0):.4f}, "
                       f"Spec: {test_res.get('specificity', 0):.4f}, "
                       f"Prec: {test_res.get('precision', 0):.4f}")

            # Mask训练详情
            logger.info(f"\n🎭 Mask Training:")
            logger.info(f"   Total Loss: {mask_res.get('all', 0):.4f}")
            if is_stage1:
                logger.info(f"     ├─ Intrinsic:  {mask_res.get('Intrinsic', 0):.4f} "
                           f"(Acc: {mask_res.get('acc_Intrinsic', 0):.2%})")
                logger.info(f"     ├─ Spurious:   {mask_res.get('Spurious', 0):.4f} "
                           f"(Acc: {mask_res.get('acc_Spurious', 0):.2%})")
                logger.info(f"     └─ Sparsity:   {mask_res.get('sparsity_reg', 0):.4f}")
            else:
                logger.info(f"     ├─ Intrinsic:       {mask_res.get('Intrinsic', 0):.4f} "
                           f"(Acc: {mask_res.get('acc_Intrinsic', 0):.2%})")
                logger.info(f"     ├─ Spurious Fusion: {mask_res.get('spurious_fusion', 0):.4f} "
                           f"(Acc: {mask_res.get('acc_spurious_fusion', 0):.2%})")
                logger.info(f"     ├─ Intrinsic Fusion (Sens): {mask_res.get('intrinsic_fusion', 0):.4f} "
                           f"(Acc: {mask_res.get('acc_intrinsic_fusion', 0):.2%})")
                # 【修改】新增 Prototype Div 日志打印
                logger.info(f"     ├─ Prototype Div:   {mask_res.get('prototype_div', 0):.4f}")

                logger.info(f"     └─ Sparsity:        {mask_res.get('sparsity_reg', 0):.4f}")

            # GNN训练详情
            logger.info(f"\n🧠 GNN Training:")
            logger.info(f"   Total Loss: {gnn_res.get('all', 0):.4f}")
            logger.info(f"     ├─ Intrinsic: {gnn_res.get('Intrinsic', 0):.4f} "
                       f"(Acc: {gnn_res.get('acc_Intrinsic', 0):.2%})")
            if not is_stage1:
                logger.info(f"     ├─ Spurious Fusion: {gnn_res.get('spurious_fusion', 0):.4f} "
                           f"(Acc: {gnn_res.get('acc_spurious_fusion', 0):.2%})")
            logger.info(f"     └─ L1 Reg:     {gnn_res.get('l1_reg', 0):.4f}")

            # 学习率
            lr_gnn = self.optimizer.param_groups[0]['lr']
            lr_mask = self.optimizer_mask.param_groups[0]['lr']
            logger.info(f"\n⚙️  Learning Rates:")
            logger.info(f"   GNN:  {lr_gnn:.6f}")
            logger.info(f"   Mask: {lr_mask:.6f}")

            # 掩码稀疏度统计
            if self.current_mask_sums:
                mask_module = self.mask.module if isinstance(self.mask, nn.DataParallel) else self.mask
                total_nodes = mask_module.P
                # 注意：如果使用了僵尸边处理策略，这里的分母计算可能需要微调，但作为日志展示通常足够
                total_edges = int(mask_module.learnable_mask.sum().item())

                node_sum = int(self.current_mask_sums.get('node', 0))
                edge_sum = int(self.current_mask_sums.get('edge', 0))

                node_pct = node_sum / total_nodes * 100 if total_nodes > 0 else 0
                edge_pct = edge_sum / total_edges * 100 if total_edges > 0 else 0

                logger.info(f"\n🎭 Mask Sparsity:")
                logger.info(f"   Nodes: {node_sum}/{total_nodes} ({node_pct:.1f}%)")
                logger.info(f"   Edges: {edge_sum}/{total_edges} ({edge_pct:.1f}%)")

            logger.info("="*80 + "\n")
    
    def _final_evaluation(self, test_loader: DataLoader) -> Dict[str, float]:
        """最终评估并返回结果"""
        logger.info("\n" + "="*80)
        logger.info("🏁 Final Evaluation")
        logger.info("="*80)

        # 确保test_acc被正确记录
        if self.best_test_acc == 0.0 and self.best_epoch >= 0:
            if 'test' in self.epoch_results.get(self.best_epoch, {}):
                self.best_test_acc = self.epoch_results[self.best_epoch]['test']['accuracy']
                logger.info(f"ℹ️  Retrieved test acc from epoch {self.best_epoch + 1}")

        results = {
            'fold': self.fold,
            'best_epoch': self.best_epoch,
            'val_acc': self.best_val_acc,
            'test_acc': self.best_test_acc
        }

        # 生成 Mask 分析报告
        if self.rank == 0:
            logger.info("\n" + "-"*80)
            logger.info("📊 Generating Mask Analysis Reports...")
            logger.info("-"*80)

            try:
                self.mask_monitor.save_summary()
                self.mask_monitor.generate_analysis_report()

                logger.info("✅ Mask analysis completed")
                logger.info(f"   Summary saved: {self.mask_monitor.fold_dir / 'masks_summary.csv'}")
                logger.info(f"   Report saved: {self.mask_monitor.fold_dir / 'mask_analysis_report.txt'}")
                logger.info(f"   Detailed masks: {self.mask_monitor.mask_dir}/")

            except Exception as e:
                logger.warning(f"⚠️  Failed to generate mask analysis: {str(e)}")

        logger.info("\n" + "-"*80)
        logger.info(f"Best Epoch:    {self.best_epoch + 1}")
        logger.info(f"Best Val Acc:  {self.best_val_acc:.4f}")
        logger.info(f"Best Test Acc: {self.best_test_acc:.4f}")
        logger.info("="*80)

        return results