"""
优雅的因果图神经网络训练器
从 main_causal.py 精炼提取，结构清晰，逻辑优雅
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score

from utils.checkpoint import CheckpointManager
from models.causal_net import CausalNet
from models.causal_mask import CausalMask
from data.large_graph_builder import LargeGraphBuilder
from utils.metrics import compute_binary_metrics, print_metrics

logger = logging.getLogger(__name__)


class CausalTrainer:
    """
    优雅的因果图神经网络训练器
    
    三阶段训练流程：
    1. 预训练：训练整体预测 (40 epochs)
    2. 阶段1：Mask+GNN联合训练（内在子图+虚假子图） (40 epochs)
    3. 阶段2：Mask+GNN联合训练（虚假子图干扰性+内在子图干扰） (60 epochs)
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
        rank: int = 0
    ):
        self.config = config
        self.fold = fold
        self.densenet_model = densenet_model.to(device)
        self.edge_prior_mask = torch.FloatTensor(edge_prior_mask).to(device)
        self.checkpoint_manager = checkpoint_manager
        self.work_dir = work_dir
        self.device = device
        self.rank = rank
        
        # 冻结DenseNet
        for param in self.densenet_model.parameters():
            param.requires_grad = False
        self.densenet_model.eval()
        
        # 模型
        self.model = None
        self.mask = None
        
        # 优化器
        self.optimizer = None
        self.optimizer_mask = None
        self.lr_scheduler = None
        self.lr_scheduler_mask = None
        
        # 训练状态
        self.global_step = 0

        # 预训练阶段状态
        self.pretrain_best_val_acc = 0.0
        self.pretrain_best_model_state = None
        self.pretrain_best_epoch = -1

        # 主训练阶段状态（用于最终评估）
        self.best_val_acc = 0.0
        self.best_test_acc = 0.0
        self.best_epoch = -1
        self.best_model_state = None

        self.epoch_results = {}
        self.current_mask_sums = {}
        self.large_graph_builder = LargeGraphBuilder(
            num_neg_samples=config['large_graph']['num_neg_samples'],
            sampling_strategy=config['large_graph']['sampling_strategy'],
            random_seed=config['seed']
        )
        self.all_data = None
        self.all_labels = None        
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.lambda_l1 = config['train']['loss_weights']['lambda_l1']
        
        logger.info(f"✅ Trainer initialized for Fold {fold}")
    
    def _build_models(self):
        """构建模型"""
        # 主GNN模型
        self.model = CausalNet(
            num_class=2,
            feature_dim=self.densenet_model.feature_dim,
            hidden1=self.config['model']['args']['hidden1'],
            hidden2=self.config['model']['args']['hidden2'],
            kernels=self.config['model']['args'].get('kernels', [2]),
            num_patches=self.edge_prior_mask.shape[0],
            num_neg_samples=self.config['large_graph']['num_neg_samples']
        ).to(self.device)
        
        # 因果掩码模型
        self.mask = CausalMask(
            num_patches=self.edge_prior_mask.shape[0], 
            edge_matrix=self.edge_prior_mask,    
            gumble_tau=self.config['misc']['gumble_tau']
        ).to(self.device)
        
        # DataParallel
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.mask = nn.DataParallel(self.mask)
        
        logger.info("✅ Models built")
    
    def _setup_optimizers(self):
        """设置优化器"""
        # 预训练阶段：使用与DenseNet相同的Adam配置
        self.optimizer_pretrain = optim.Adam(
            self.model.parameters(),
            lr=self.config['densenet']['pretrain']['learning_rate'],
            weight_decay=self.config['densenet']['pretrain']['weight_decay']
        )
        self.scheduler_pretrain = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_pretrain,
            T_max=self.config['train']['pre_epoch'],
            eta_min=1e-5
        )

        # 主训练阶段：使用SGD
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config['train']['base_lr'],
            momentum=0.9,
            nesterov=True,
            weight_decay=self.config['train']['weight_decay']
        )
        self.optimizer_mask = optim.SGD(
            self.mask.parameters(),
            lr=self.config['train']['base_lr_mask'],
            momentum=0.9,
            nesterov=True,
            weight_decay=self.config['train']['weight_decay']
        )

        # 主训练学习率调度器
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            verbose=(self.rank == 0),
            patience=self.config['train']['stepsize'],
            factor=self.config['train']['gamma']
        )
        self.lr_scheduler_mask = ReduceLROnPlateau(
            self.optimizer_mask,
            verbose=(self.rank == 0),
            patience=self.config['train']['stepsize'],
            factor=self.config['train']['gamma']
        )

        logger.info("✅ Optimizers configured")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """完整训练流程"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 Training Start - Fold {self.fold}")
        logger.info(f"{'='*80}\n")
        
        # 预加载数据用于大图构建
        if self.all_data is None:
            logger.info("预加载数据用于大图构建...")
            dataset = train_loader.dataset
            if hasattr(dataset, 'dataset'):
                dataset = dataset.dataset
            self.all_data = np.array(dataset.all_patches)
            self.all_labels = np.array(dataset.labels)
            logger.info(f"✅ 数据预加载完成: {self.all_data.shape}")        
        
        # 初始化
        self._build_models()
        self._setup_optimizers()
        
        # 阶段1: 预训练 (40 epochs)
        pre_epochs = self.config['train']['pre_epoch']
        if pre_epochs > 0:
            logger.info("\n" + "="*80)
            logger.info("📚 Phase 1: Pre-training (Whole Prediction)")
            logger.info("="*80)
            self._pretrain_phase(train_loader, val_loader, test_loader, pre_epochs)
        
        # 阶段2+3: 主训练 (100 epochs)
        logger.info("\n" + "="*80)
        logger.info("🎯 Phase 2+3: Main Training (Mask + GNN)")
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
    
    #==================== 预训练阶段 ====================
    
    def _pretrain_phase(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int
    ):
        """预训练阶段"""
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

            # 更新学习率 - 添加这行
            self.scheduler_pretrain.step()

        # 加载最佳模型
        if self.rank == 0 and self.best_model_state:
            logger.info(f"✅ Loading best pretrain model (Epoch {self.best_epoch+1})")
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(self.best_model_state)
            else:
                self.model.load_state_dict(self.best_model_state)
    
    def _train_pretrain_epoch(self, epoch: int, train_loader: DataLoader):
        """预训练的一个epoch"""
        self.model.train()
        
        losses = []
        accuracies = []
        
        for data, _, label in train_loader:
            self.global_step += 1
            
            # 转换数据
            data = data.to(self.device)
            label = label.to(self.device)
            
            # 提取特征
            x_features = self._extract_features(data)
            
            # 整体预测
            outputs = self.model.module.prediction_whole(x_features, self.edge_prior_mask) \
                if isinstance(self.model, nn.DataParallel) else \
                self.model.prediction_whole(x_features, self.edge_prior_mask)
            
            loss = self.criterion(outputs, label).mean()
            l1_loss = self._compute_l1_regularization()
            loss_total = loss + l1_loss
            
            # 反向传播
            self.optimizer_pretrain.zero_grad()  # 改这里
            loss_total.backward()
            self.optimizer_pretrain.step()  # 改这里
            
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

        # ✅ 使用预训练专属的最佳模型追踪
        if phase == 'val' and metrics['accuracy'] > self.pretrain_best_val_acc:
            self.pretrain_best_val_acc = metrics['accuracy']
            self.pretrain_best_epoch = epoch

            # 保存预训练最佳模型
            if isinstance(self.model, nn.DataParallel):
                self.pretrain_best_model_state = self.model.module.state_dict()
            else:
                self.pretrain_best_model_state = self.model.state_dict()

            logger.info(f"💎 New Best Pretrain Val Acc: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
    
    #==================== 主训练阶段 ====================
    def _main_training(self, train_loader, val_loader, test_loader):
        """主训练阶段（阶段1+阶段2）"""
        start_epoch = self.config['train']['pre_epoch']
        num_epochs = self.config['train']['num_epoch']
        stage_transition = self.config['train']['stage_transition_epoch']

        # ✅ 阶段1的最佳模型追踪
        stage1_best_val_acc = 0.0
        stage1_best_model_state = None
        stage1_best_epoch = -1

        for epoch in range(start_epoch, num_epochs):
            is_stage1 = epoch < stage_transition

            if self.rank == 0:
                self.epoch_results[epoch] = {}

            # 训练
            self._train_main_epoch(epoch, train_loader, is_stage1)

            # 评估
            if self.rank == 0:
                with torch.no_grad():
                    self._eval_main_epoch(epoch, val_loader, 'val')
                    self._eval_main_epoch(epoch, test_loader, 'test')

                # ✅ 阶段1：保存阶段1最佳模型
                if is_stage1:
                    val_acc = self.epoch_results[epoch]['val']['accuracy']
                    if val_acc > stage1_best_val_acc:
                        stage1_best_val_acc = val_acc
                        stage1_best_epoch = epoch
                        if isinstance(self.model, nn.DataParallel):
                            stage1_best_model_state = self.model.module.state_dict()
                        else:
                            stage1_best_model_state = self.model.state_dict()

                        logger.info(f"💎 New Best Stage1 Val Acc: {val_acc:.4f}")

                self._print_main_summary(epoch, is_stage1)

                # 更新学习率
                if self.config['train']['scheduler'] == 'auto':
                    val_loss = self.epoch_results[epoch]['val'].get('gnn', {}).get('loss_all', 0)
                    self.lr_scheduler.step(val_loss)
                    self.lr_scheduler_mask.step(val_loss)
                else:
                    self.lr_scheduler.step()
                    self.lr_scheduler_mask.step()

            # ✅ 阶段1结束
            if epoch == stage_transition - 1 and self.rank == 0 and stage1_best_model_state:
                logger.info(f"\n{'='*80}")
                logger.info(f"✅ Stage 1 Completed! Loading best Stage1 model as Stage2 starting point")
                logger.info(f"   Best Stage1 Epoch: {stage1_best_epoch + 1}")
                logger.info(f"   Best Stage1 Val Acc: {stage1_best_val_acc:.4f}")
                logger.info(f"{'='*80}\n")

                # 重置阶段2的追踪器
                self.best_val_acc = 0.0
                self.best_test_acc = 0.0
                self.best_epoch = -1

    def _train_main_epoch(self, epoch: int, train_loader: DataLoader, is_stage1: bool):
            """主训练的一个epoch (V3 - 修复了 'int' object has no attribute 'item' 错误)"""

            # ========== 【新】安全获取损失值的辅助函数 ==========
            def _safe_get_item(loss_dict: dict, key: str) -> float:
                """
                安全地从损失字典中获取值并转换为浮点数。
                无论值是 Tensor, int, 还是 float, 都能处理。
                """
                # 设置一个默认张量，以防key不存在
                default_tensor = torch.tensor(0.0, device=self.device)
                val = loss_dict.get(key, default_tensor)

                if isinstance(val, torch.Tensor):
                    return val.item() # 如果是张量，调用 .item()
                elif isinstance(val, (int, float)):
                    return float(val) # 如果是 int/float (比如 0)，直接转换
                else:
                    logger.warning(f"无法识别的损失类型 {type(val)} for key {key}. 返回 0.0")
                    return 0.0
            # =======================================================

            self.model.train()
            self.mask.train()

            losses_mask = {
                'all': [], 'Intrinsic': [], 'Spurious': [], 'spurious_fusion': [],
                'intrinsic_fusion': [], 'sparsity_reg': []
            }
            losses_gnn = {
                'all': [], 'Intrinsic': [], 'spurious_fusion': [], 'l1_reg': []
            }
            accs_mask = {}
            accs_gnn = {}

            lambda_reg = 0.1 * (1 + epoch / self.config['train']['num_epoch'])
            stage_transition_epoch = self.config['train']['stage_transition_epoch']

            for batch_idx, (data, _, label) in enumerate(train_loader):

                is_first_batch_of_stage2 = (not is_stage1) and \
                                           (epoch == stage_transition_epoch) and \
                                           (batch_idx == 0)

                if is_first_batch_of_stage2:
                    logger.info(f"\n{'!'*80}")
                    logger.info(f"DEBUG [Epoch {epoch+1}, Batch {batch_idx}]: === 正在进入 Stage 2 的第一个批次 ===")
                    logger.info(f"{'!'*80}\n")

                self.global_step += 1
                label = label.to(self.device)

                # ============ 按阶段构建数据和特征 ============
                if is_stage1:
                    data = data.to(self.device)
                    x_features = self._extract_features(data) # [B, P, feature_dim]
                else:
                    large_data, large_edge = self.large_graph_builder.build_large_graph(
                        batch_data=data,
                        batch_labels=label,
                        base_edge=self.edge_prior_mask.cpu(),
                        all_data=self.all_data,
                        all_labels=self.all_labels
                    )
                    large_data = large_data.to(self.device)
                    x_features = self._extract_features(large_data) # [B, (N+1)*P, feature_dim]

                    # ======== 排查点 1: 检查 Stage 2 的输入特征 (x_features) ========
                    if is_first_batch_of_stage2:
                        logger.info(f"DEBUG [E{epoch+1}, B{batch_idx}]: Stage 2 - x_features shape: {x_features.shape}")
                        if torch.isnan(x_features).any():
                            logger.error(f"FATAL [E{epoch+1}, B{batch_idx}]: x_features 中检测到 NaN!")
                            raise ValueError("NaN found in features after _extract_features")
                        if torch.isinf(x_features).any():
                            logger.error(f"FATAL [E{epoch+1}, B{batch_idx}]: x_features 中检测到 Inf!")
                            raise ValueError("Inf found in features after _extract_features")
                        logger.info(f"DEBUG [E{epoch+1}, B{batch_idx}]: x_features (排查点 1) 检查通过 (无 NaN/Inf).")


                #========== 1. Mask训练 ==========
                for param in self.model.parameters():
                    param.requires_grad = False

                mask_module = self.mask.module if isinstance(self.mask, nn.DataParallel) else self.mask
                masks, sparsity = mask_module(train=True)

                if is_stage1:
                    result_mask = self._compute_stage1_mask_loss(
                        x_features, masks, label, lambda_reg, self.edge_prior_mask, is_large_graph=False
                    )
                else:
                    result_mask = self._compute_stage2_mask_loss(
                        x_features, masks, label, lambda_reg, self.edge_prior_mask, is_large_graph=True
                    )

                    # ======== 排查点 2: 检查 Stage 2 的 Mask 损失值 (已修复) ========
                    if is_first_batch_of_stage2:
                        # 【修复】: 使用 _safe_get_item 辅助函数
                        l_all = _safe_get_item(result_mask['loss'], 'all')
                        l_inv = _safe_get_item(result_mask['loss'], 'spurious_fusion')
                        l_sen = _safe_get_item(result_mask['loss'], 'intrinsic_fusion') # <-- 现在这里是安全的
                        l_int = _safe_get_item(result_mask['loss'], 'Intrinsic')
                        l_spa = _safe_get_item(result_mask['loss'], 'sparsity_reg')

                        logger.info(f"DEBUG [E{epoch+1}, B{batch_idx}]: Stage 2 - Mask 损失 (Backward 之前):")
                        logger.info(f"  ├─ Total Loss: {l_all:.4e}")
                        logger.info(f"  ├─ L_Intrinsic: {l_int:.4e}")
                        logger.info(f"  ├─ L_spurious_fusion (L_inv): {l_inv:.4e}")
                        logger.info(f"  ├─ L_intrinsic_fusion (L_sen): {l_sen:.4e}") # <-- 重点观察对象 (应为 0.0)
                        logger.info(f"  └─ L_Sparsity: {l_spa:.4e}")

                        if not np.isfinite(l_all):
                            logger.error(f"FATAL [E{epoch+1}, B{batch_idx}]: Mask 损失值为 NaN/Inf！")
                            raise ValueError("Mask loss exploded before backward")
                        logger.info(f"DEBUG [E{epoch+1}, B{batch_idx}]: Mask 损失 (排查点 2) 检查通过 (有限值).")


                self.optimizer_mask.zero_grad()
                # 只有当 'all' 键存在且是张量时才进行反向传播
                if 'all' in result_mask['loss'] and isinstance(result_mask['loss']['all'], torch.Tensor):
                    result_mask['loss']['all'].backward()
                else:
                    logger.warning(f"DEBUG [E{epoch+1}, B{batch_idx}]: 'all' 损失不是张量，跳过 Mask backward。")

                # ======== 排查点 3: 检查 Stage 2 的 Mask 梯度 ========
                if is_first_batch_of_stage2:
                    max_grad_norm = 0.0
                    nan_grad = False
                    inf_grad = False
                    for param in self.mask.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any(): nan_grad = True
                            if torch.isinf(param.grad).any(): inf_grad = True
                            current_norm = param.grad.data.norm(2).item()
                            if current_norm > max_grad_norm:
                                max_grad_norm = current_norm

                    logger.info(f"DEBUG [E{epoch+1}, B{batch_idx}]: Stage 2 - Mask 梯度 (Backward 之后):")
                    logger.info(f"  ├─ 最大梯度范数 (Max Grad Norm): {max_grad_norm:.4e}")

                    if nan_grad or inf_grad:
                        logger.error(f"FATAL [E{epoch+1}, B{batch_idx}]: Mask 梯度中检测到 NaN/Inf！")
                        raise ValueError("Mask gradients exploded (NaN/Inf)")
                    logger.info(f"DEBUG [E{epoch+1}, B{batch_idx}]: Mask 梯度 (排查点 3) 检查通过 (无 NaN/Inf).")

                self.optimizer_mask.step()

                # (记录 Mask 损失)
                for k in losses_mask.keys():
                    losses_mask[k].append(_safe_get_item(result_mask['loss'], k)) # 【修复】使用 safe_get
                for k, v in result_mask['preds'].items():
                    if k not in accs_mask: accs_mask[k] = []
                    accs_mask[k].append(self._compute_accuracy(v, label))


                #========== 2. GNN训练 ==========
                for param in self.model.parameters():
                    param.requires_grad = True

                masks, sparsity = mask_module(train=False) # 使用更新后的 Mask
                masks = [m.detach() for m in masks]

                if is_stage1:
                    result_gnn = self._compute_stage1_gnn_loss(
                        x_features, masks, label, self.edge_prior_mask, is_large_graph=False
                    )
                else:
                    result_gnn = self._compute_stage2_gnn_loss(
                        x_features, masks, label, self.edge_prior_mask, is_large_graph=True
                    )

                    # ======== 排查点 4: 检查 Stage 2 的 GNN 损失和L1 (已修复) ========
                    if is_first_batch_of_stage2:
                        # 【修复】: 使用 _safe_get_item 辅助函数
                        l1_reg_val = _safe_get_item(result_gnn['loss'], 'l1_reg')

                        logger.info(f"DEBUG [E{epoch+1}, B{batch_idx}]: Stage 2 - GNN L1 Reg (Backward 之前): {l1_reg_val:.4e}")

                        if not np.isfinite(l1_reg_val):
                             logger.error(f"FATAL [E{epoch+1}, B{batch_idx}]: GNN L1 Reg 值为 NaN/Inf！...")
                             raise ValueError("GNN weights (L1 Reg) exploded")

                        # 【修复】: 检查 GNN 的总损失
                        l_all_gnn = _safe_get_item(result_gnn['loss'], 'all')
                        logger.info(f"DEBUG [E{epoch+1}, B{batch_idx}]: Stage 2 - GNN Total Loss (Backward 之前): {l_all_gnn:.4e}")

                        if not np.isfinite(l_all_gnn):
                            logger.error(f"FATAL [E{epoch+1}, B{batch_idx}]: GNN 损失值为 NaN/Inf！(Mask 损坏导致)")
                            raise ValueError("GNN loss exploded (likely due to corrupted mask)")

                        logger.info(f"DEBUG [E{epoch+1}, B{batch_idx}]: GNN 损失 (排查点 4) 检查通过。")


                # 只有当 'all' 键存在且是张量时才进行反向传播
                if 'all' in result_gnn['loss'] and isinstance(result_gnn['loss']['all'], torch.Tensor):
                    result_gnn['loss']['all'].backward()
                else:
                    logger.warning(f"DEBUG [E{epoch+1}, B{batch_idx}]: 'all' 损失不是张量，跳过 GNN backward。")

                # ======== 排查点 5: 检查 Stage 2 的 GNN 梯度 ========
                if is_first_batch_of_stage2:
                    max_grad_norm_gnn = 0.0
                    nan_grad_gnn = False
                    inf_grad_gnn = False
                    for param in self.model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any(): nan_grad_gnn = True
                            if torch.isinf(param.grad).any(): inf_grad_gnn = True
                            current_norm = param.grad.data.norm(2).item()
                            if current_norm > max_grad_norm_gnn:
                                max_grad_norm_gnn = current_norm

                    logger.info(f"DEBUG [E{epoch+1}, B{batch_idx}]: Stage 2 - GNN 梯度 (Backward 之后):")
                    logger.info(f"  ├─ 最大梯度范数 (Max Grad Norm): {max_grad_norm_gnn:.4e}")

                    if nan_grad_gnn or inf_grad_gnn:
                        logger.error(f"FATAL [E{epoch+1}, B{batch_idx}]: GNN 梯度中检测到 NaN/Inf！")
                        raise ValueError("GNN gradients exploded (NaN/Inf)")
                    logger.info(f"DEBUG [E{epoch+1}, B{batch_idx}]: GNN 梯度 (排查点 5) 检查通过 (无 NaN/Inf).")

                self.optimizer.step()

                # (记录 GNN 损失)
                for k in losses_gnn.keys():
                    losses_gnn[k].append(_safe_get_item(result_gnn['loss'], k)) # 【修复】使用 safe_get
                for k, v in result_gnn['preds'].items():
                    if k not in accs_gnn: accs_gnn[k] = []
                    accs_gnn[k].append(self._compute_accuracy(v, label))

                if self.rank == 0:
                    self.current_mask_sums = {
                        'node': masks[0].sum().item(),
                        'edge': masks[1].sum().item()
                    }

            # ============ Epoch 结束，保存结果 ============
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
        """主训练评估"""
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
            outputs = model_module.prediction_intrinsic_path(x_features, self.edge_prior_mask, masks, is_large_graph=False)

            all_outputs.append(outputs)
            all_labels.append(label)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_binary_metrics(all_outputs, all_labels)
        self.epoch_results[epoch][phase] = metrics

        # ✅ 仅在阶段2更新最终评估的最佳模型
        stage_transition = self.config['train']['stage_transition_epoch']
        if phase == 'val' and epoch >= stage_transition and metrics['accuracy'] > self.best_val_acc:
            self.best_val_acc = metrics['accuracy']
            self.best_epoch = epoch

            if 'test' in self.epoch_results[epoch]:
                self.best_test_acc = self.epoch_results[epoch]['test']['accuracy']
                logger.info(
                    f"💎 New Best Stage2 - Val Acc: {metrics['accuracy']:.4f}, "
                    f"Val AUC: {metrics['auc']:.4f}, "
                    f"Test Acc: {self.best_test_acc:.4f}"
                )
            else:
                logger.info(
                    f"💎 New Best Stage2 Val - Acc: {metrics['accuracy']:.4f}, "
                    f"AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}"
                )
                
    
    #==================== 损失计算 ====================
    
    def _compute_stage1_mask_loss(self, x, masks, label, lambda_reg, edge_prior_mask, is_large_graph):
        """阶段1 Mask损失：内在子图 + 虚假子图"""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        # 内在子图
        y_pred = model.prediction_intrinsic_path(x, edge_prior_mask, masks, is_large_graph)
        loss_pred = self.criterion(y_pred, label).mean()
        
        # 虚假子图（熵损失）
        y_spu = model.prediction_spurious_path(x, edge_prior_mask, masks, is_large_graph)
        loss_spu = self._entropy_loss(y_spu)
        
        # 稀疏性正则
        mask_module = self.mask.module if isinstance(self.mask, nn.DataParallel) else self.mask
        reg_loss = mask_module.compute_sparsity_regularization(lambda_reg=lambda_reg)
        
        # 总损失
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
    
    def _compute_stage2_mask_loss(self, x, masks, label, lambda_reg, edge_prior_mask, is_large_graph):
        """阶段2 Mask损失：虚假子图干扰 + 内在子图干扰 + 内在子图"""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        # 内在子图
        y_pred = model.prediction_intrinsic_path(x, edge_prior_mask, masks, is_large_graph)
        loss_pred = self.criterion(y_pred, label).mean()
        
        # 虚假子图干扰性
        y_inv = model.prediction_spurious_fusion(x, edge_prior_mask, masks, is_large_graph)   # 使用相同方法
        loss_inv = self.criterion(y_inv, label).mean()
        
        # 内在子图干扰
        y_sen = model.prediction_intrinsic_fusion(x, edge_prior_mask, masks, is_large_graph)
        loss_sen = self.criterion(y_sen, 1 - label).mean()
        # 稀疏性正则
        mask_module = self.mask.module if isinstance(self.mask, nn.DataParallel) else self.mask
        reg_loss = mask_module.compute_sparsity_regularization(lambda_reg=lambda_reg)
        
        # 总损失
        loss_weights = self.config['train']['loss_weights']
        loss_all = (loss_weights['L_inv'] * loss_inv + 
                   loss_weights['L_sen'] * loss_sen + 
                   loss_weights['L_pred'] * loss_pred + 
                   reg_loss)
        
        return {
            'loss': {
                'all': loss_all,
                'spurious_fusion': loss_inv,
                'intrinsic_fusion': loss_sen,
                'Intrinsic': loss_pred,
                'sparsity_reg': reg_loss
            },
            'preds': {
                'spurious_fusion': y_inv,
                'intrinsic_fusion': y_sen,
                'Intrinsic': y_pred
            }
        }
    def _compute_stage1_gnn_loss(self, x, masks, label, edge_prior_mask, is_large_graph):
        """阶段1 GNN损失：内在子图"""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        y_pred = model.prediction_intrinsic_path(x, edge_prior_mask, masks, is_large_graph)
        loss_pred = self.criterion(y_pred, label).mean()
        l1_loss = self._compute_l1_regularization()
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
    
    def _compute_stage2_gnn_loss(self, x, masks, label, edge_prior_mask, is_large_graph):
        """阶段2 GNN损失：内在子图 + 虚假子图干扰"""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        y_pred = model.prediction_intrinsic_path(x, edge_prior_mask, masks, is_large_graph)
        loss_pred = self.criterion(y_pred, label).mean()
        
        y_inv = model.prediction_spurious_fusion(x, edge_prior_mask, masks, is_large_graph) 
        loss_inv = self.criterion(y_inv, label).mean()
        
        l1_loss = self._compute_l1_regularization()
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
    
    #==================== 辅助函数 ====================
    
    def _extract_features(self, data: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """使用DenseNet批量提取特征（避免OOM）"""
        B = data.shape[0]
        total_P = data.shape[1]

        data_reshaped = data.view(-1, 1, data.shape[3], data.shape[4], data.shape[5])
        total_patches = data_reshaped.shape[0]

        # 批量提取，避免显存爆炸
        all_features = []
        with torch.no_grad():
            for i in range(0, total_patches, batch_size):
                batch = data_reshaped[i:i+batch_size]
                features_batch = self.densenet_model(batch)
                all_features.append(features_batch.cpu())  # 立即移到CPU
                del features_batch
                torch.cuda.empty_cache()

        # 在CPU上拼接，再移回GPU
        features = torch.cat(all_features, dim=0).to(self.device)
        features = features.view(B, total_P, -1)
        return features
    
    def _compute_l1_regularization(self) -> torch.Tensor:
        """计算L1正则化"""
        l1_reg = torch.tensor(0., device=self.device)
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        for name, param in model.named_parameters():
            if 'mlp_causal.0.weight' in name:
                l1_reg += torch.sum(torch.abs(param))
        
        return self.lambda_l1 * l1_reg
    
    def _entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """熵损失（用于虚假子图）"""
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        return -entropy.mean()  # 负号使其最大化熵
    
    def _compute_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """计算准确率"""
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean().item()
        return acc
    
    #==================== 打印 ====================
    
    def _print_pretrain_summary(self, epoch: int):
        """打印预训练总结"""
        res = self.epoch_results[epoch]
        train_res = res.get('train', {})
        val_res = res.get('val', {})
        test_res = res.get('test', {})

        logger.info("="*80)
        logger.info(f"📚 Epoch {epoch+1}/{self.config['train']['pre_epoch']} [Pre-train]")
        logger.info("-"*80)
        logger.info(f"Train - Loss: {train_res.get('loss_all', 0):.4f}, "
                   f"Acc: {train_res.get('acc_Intrinsic', 0):.4f}")

        # ✅ 增强验证集日志
        logger.info(f"Val   - Acc: {val_res.get('accuracy', 0):.4f}, "
                   f"AUC: {val_res.get('auc', 0):.4f}, "
                   f"F1: {val_res.get('f1', 0):.4f}")
        logger.info(f"        Sens: {val_res.get('sensitivity', 0):.4f}, "
                   f"Spec: {val_res.get('specificity', 0):.4f}")

        # ✅ 增强测试集日志
        logger.info(f"Test  - Acc: {test_res.get('accuracy', 0):.4f}, "
                   f"AUC: {test_res.get('auc', 0):.4f}, "
                   f"F1: {test_res.get('f1', 0):.4f}")
        logger.info(f"        Sens: {test_res.get('sensitivity', 0):.4f}, "
                   f"Spec: {test_res.get('specificity', 0):.4f}")

        logger.info("="*80 + "\n")
    

    def _print_main_summary(self, epoch: int, is_stage1: bool):
        """打印主训练总结"""
        res = self.epoch_results[epoch]
        train_res = res.get('train', {})
        val_res = res.get('val', {})
        test_res = res.get('test', {})

        mask_res = train_res.get('mask', {})
        gnn_res = train_res.get('gnn', {})

        stage_name = "Stage 1 (Intrinsic+Spurious)" if is_stage1 else "Stage 2 (spurious_fusion+intrinsic_fusion)"

        logger.info("="*80)
        logger.info(f"🎯 Epoch {epoch+1}/{self.config['train']['num_epoch']} [{stage_name}]")
        logger.info("-"*80)

        # ✅ 增强官方评估结果
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

        # ✅ 修复 Mask 训练详情（键名匹配）
        logger.info(f"\n🎭 Mask Training:")
        logger.info(f"   Total Loss: {mask_res.get('all', 0):.4f}")  # ✅ 改为 'all'
        if is_stage1:
            logger.info(f"     ├─ Intrinsic:  {mask_res.get('Intrinsic', 0):.4f} "  # ✅ 改为 'Intrinsic'
                       f"(Acc: {mask_res.get('acc_Intrinsic', 0):.2%})")
            logger.info(f"     ├─ Spurious: {mask_res.get('Spurious', 0):.4f} "  # ✅ 改为 'Spurious'
                       f"(Acc: {mask_res.get('acc_Spurious', 0):.2%})")
            logger.info(f"     └─ Sparsity:    {mask_res.get('sparsity_reg', 0):.4f}")  # ✅ 改为 'sparsity_reg'
        else:
            logger.info(f"     ├─ Intrinsic:     {mask_res.get('Intrinsic', 0):.4f} "  # ✅
                       f"(Acc: {mask_res.get('acc_Intrinsic', 0):.2%})")
            logger.info(f"     ├─ spurious_fusion:         {mask_res.get('spurious_fusion', 0):.4f} "  # ✅ 改为 'spurious_fusion'
                       f"(Acc: {mask_res.get('acc_causal', 0):.2%})")
            logger.info(f"     ├─ intrinsic_fusion: {mask_res.get('intrinsic_fusion', 0):.4f} "  # ✅ 改为 'intrinsic_fusion'
                       f"(Acc: {mask_res.get('acc_intrinsic_fusion', 0):.2%})")
            logger.info(f"     └─ Sparsity:       {mask_res.get('sparsity_reg', 0):.4f}")  # ✅

        # ✅ 修复 GNN 训练详情（键名匹配）
        logger.info(f"\n🧠 GNN Training:")
        logger.info(f"   Total Loss: {gnn_res.get('all', 0):.4f}")  # ✅ 改为 'all'
        logger.info(f"     ├─ Intrinsic: {gnn_res.get('Intrinsic', 0):.4f} "  # ✅ 改为 'Intrinsic'
                   f"(Acc: {gnn_res.get('acc_Intrinsic', 0):.2%})")
        if not is_stage1:
            logger.info(f"     ├─ spurious_fusion:     {gnn_res.get('spurious_fusion', 0):.4f} "  # ✅ 改为 'spurious_fusion'
                       f"(Acc: {gnn_res.get('acc_causal', 0):.2%})")
        logger.info(f"     └─ L1 Reg:     {gnn_res.get('l1_reg', 0):.4f}")  # ✅ 改为 'l1_reg'

        # 训练参数
        lr_gnn = self.optimizer.param_groups[0]['lr']
        lr_mask = self.optimizer_mask.param_groups[0]['lr']
        logger.info(f"\n⚙️  Learning Rates:")
        logger.info(f"   GNN:  {lr_gnn:.6f}")
        logger.info(f"   Mask: {lr_mask:.6f}")

        # 掩码统计
        if self.current_mask_sums:
            mask_module = self.mask.module if isinstance(self.mask, nn.DataParallel) else self.mask
            total_nodes = mask_module.P
            total_edges = int(mask_module.learnable_mask.sum().item())

            node_sum = int(self.current_mask_sums.get('node', 0))
            edge_sum = int(self.current_mask_sums.get('edge', 0))

            node_pct = node_sum / total_nodes * 100 if total_nodes > 0 else 0
            edge_pct = edge_sum / total_edges * 100 if total_edges > 0 else 0

            logger.info(f"\n🎭 Mask Statistics:")
            logger.info(f"   Nodes: {node_sum}/{total_nodes} ({node_pct:.1f}%)")
            logger.info(f"   Edges: {edge_sum}/{total_edges} ({edge_pct:.1f}%)")

        logger.info("="*80 + "\n")
    
    def _final_evaluation(self, test_loader: DataLoader) -> Dict[str, float]:
        """最终评估"""
        logger.info("\n" + "="*80)
        logger.info("🏁 Final Evaluation")
        logger.info("="*80)

        # ✅ 如果 best_test_acc 仍然是 0，说明没有正确记录
        # 这时应该从 epoch_results 中获取最佳 epoch 对应的测试集准确率
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

        logger.info(f"Best Epoch:    {self.best_epoch + 1}")
        logger.info(f"Best Val Acc:  {self.best_val_acc:.4f}")
        logger.info(f"Best Test Acc: {self.best_test_acc:.4f}")
        logger.info("="*80)

        return results