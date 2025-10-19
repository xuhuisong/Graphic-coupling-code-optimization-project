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
    2. 阶段1：Mask+GNN联合训练（不变性+变异性） (40 epochs)
    3. 阶段2：Mask+GNN联合训练（因果性+反事实） (60 epochs)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        fold: int,
        densenet_model: nn.Module,
        edge_matrix: np.ndarray,
        checkpoint_manager: CheckpointManager,
        work_dir: str,
        device: str = 'cuda',
        rank: int = 0
    ):
        self.config = config
        self.fold = fold
        self.densenet_model = densenet_model.to(device)
        self.edge_matrix = torch.FloatTensor(edge_matrix).to(device)
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
            num_patches=self.edge_matrix.shape[0],
            num_neg_samples=self.config['large_graph']['num_neg_samples']
        ).to(self.device)
        
        # 因果掩码模型
        self.mask = CausalMask(
            num_patches=self.edge_matrix.shape[0],
            edge_matrix=self.edge_matrix,
            gumble_tau=self.config['misc']['gumble_tau']
        ).to(self.device)
        
        # DataParallel
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.mask = nn.DataParallel(self.mask)
        
        logger.info("✅ Models built")
    
    def _setup_optimizers(self):
        """设置优化器"""
        if self.config['train']['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['train']['base_lr'],
                momentum=0.9,
                nesterov=self.config['train'].get('nesterov', True),
                weight_decay=self.config['train']['weight_decay']
            )
            self.optimizer_mask = optim.SGD(
                self.mask.parameters(),
                lr= self.config['train']['base_lr_mask'],
                momentum=0.9,
                nesterov=self.config['train'].get('nesterov', True),
                weight_decay=self.config['train']['weight_decay']
            )
        
        # 学习率调度器
        if self.config['train']['scheduler'] == 'auto':
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
        elif self.config['train']['scheduler'] == 'step':
            self.lr_scheduler = StepLR(
                self.optimizer,
                step_size=self.config['train']['stepsize'],
                gamma=self.config['train']['gamma']
            )
            self.lr_scheduler_mask = StepLR(
                self.optimizer_mask,
                step_size=self.config['train']['stepsize'],
                gamma=self.config['train']['gamma']
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
            outputs = self.model.module.prediction_whole(x_features, self.edge_matrix) \
                if isinstance(self.model, nn.DataParallel) else \
                self.model.prediction_whole(x_features, self.edge_matrix)
            
            loss = self.criterion(outputs, label).mean()
            l1_loss = self._compute_l1_regularization()
            loss_total = loss + l1_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            
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
                'acc_invariance': np.mean(accuracies)
            }
    
    def _eval_pretrain_epoch(self, epoch: int, data_loader: DataLoader, phase: str):
        self.model.eval()

        all_outputs = []
        all_labels = []

        for data, _, label in data_loader:
            data = data.to(self.device)
            label = label.to(self.device)

            x_features = self._extract_features(data)
            outputs = self.model.module.prediction_whole(x_features, self.edge_matrix) \
                if isinstance(self.model, nn.DataParallel) else \
                self.model.prediction_whole(x_features, self.edge_matrix)

            all_outputs.append(outputs)
            all_labels.append(label)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # ✅ 替换为统一指标计算
        metrics = compute_binary_metrics(all_outputs, all_labels)

        # 保存结果
        self.epoch_results[epoch][phase] = metrics

        # 更新最佳模型
        if phase == 'val' and metrics['accuracy'] > self.best_val_acc:
            self.best_val_acc = metrics['accuracy']
            self.best_epoch = epoch
            if isinstance(self.model, nn.DataParallel):
                self.best_model_state = self.model.module.state_dict()
            else:
                self.best_model_state = self.model.state_dict()
            
            logger.info(f"💎 New Best Val Acc: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
    
    #==================== 主训练阶段 ====================
    
    def _main_training(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader
    ):
        """主训练阶段（阶段1+阶段2）"""
        start_epoch = self.config['train']['pre_epoch']
        num_epochs = self.config['train']['num_epoch']
        stage_transition = self.config['train']['stage_transition_epoch']
        
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
                
                self._print_main_summary(epoch, is_stage1)
                
                # 更新学习率
                if self.config['train']['scheduler'] == 'auto':
                    val_loss = self.epoch_results[epoch]['val'].get('gnn', {}).get('loss_all', 0)
                    self.lr_scheduler.step(val_loss)
                    self.lr_scheduler_mask.step(val_loss)
                else:
                    self.lr_scheduler.step()
                    self.lr_scheduler_mask.step()
    
    def _train_main_epoch(self, epoch: int, train_loader: DataLoader, is_stage1: bool):
        """主训练的一个epoch"""
        self.model.train()
        self.mask.train()

        # ✅ 修复1：损失记录器始终初始化
        losses_mask = {
            'all': [], 
            'invariance': [], 
            'variability': [], 
            'causal': [], 
            'counterfactual': [], 
            'sparsity_reg': []
        }
        losses_gnn = {
            'all': [], 
            'invariance': [], 
            'causal': [], 
            'l1_reg': []
        }
        accs_mask = {}
        accs_gnn = {}

        # 计算正则化强度
        lambda_reg = 0.05 * (1 + epoch / self.config['train']['num_epoch'])

        for data, _, label in train_loader:
            self.global_step += 1

            label = label.to(self.device)

            # 构建大图
            large_data, large_edge = self.large_graph_builder.build_large_graph(
                batch_data=data,
                batch_labels=label,
                base_edge=self.edge_matrix.cpu(),
                all_data=self.all_data,
                all_labels=self.all_labels
            )
            large_data = large_data.to(self.device)
            large_edge = large_edge.to(self.device)

            # 提取大图特征
            x_features = self._extract_features(large_data)

            #========== 1. Mask训练 ==========
            for param in self.model.parameters():
                param.requires_grad = False

            mask_module = self.mask.module if isinstance(self.mask, nn.DataParallel) else self.mask
            masks, sparsity = mask_module(train=True)

            if is_stage1:
                result_mask = self._compute_stage1_mask_loss(x_features, masks, label, lambda_reg, large_edge)
            else:
                result_mask = self._compute_stage2_mask_loss(x_features, masks, label, lambda_reg, large_edge)

            self.optimizer_mask.zero_grad()
            result_mask['loss']['all'].backward()
            self.optimizer_mask.step()

            # ✅ 修复2：立即记录损失（不管 rank）
            losses_mask['all'].append(result_mask['loss']['all'].item())
            if 'invariance' in result_mask['loss']:
                losses_mask['invariance'].append(result_mask['loss']['invariance'].item())
            if 'variability' in result_mask['loss']:
                losses_mask['variability'].append(result_mask['loss']['variability'].item())
            if 'causal' in result_mask['loss']:
                losses_mask['causal'].append(result_mask['loss']['causal'].item())
            if 'counterfactual' in result_mask['loss']:
                losses_mask['counterfactual'].append(result_mask['loss']['counterfactual'].item())
            if 'sparsity_reg' in result_mask['loss']:
                losses_mask['sparsity_reg'].append(result_mask['loss']['sparsity_reg'].item())

            # 记录准确率
            for pred_type, pred_output in result_mask['preds'].items():
                if pred_type not in accs_mask:
                    accs_mask[pred_type] = []
                accs_mask[pred_type].append(
                    self._compute_accuracy(pred_output, label)
                )

            #========== 2. GNN训练 ==========
            for param in self.model.parameters():
                param.requires_grad = True

            masks, sparsity = mask_module(train=False)
            masks = [m.detach() for m in masks]

            if is_stage1:
                result_gnn = self._compute_stage1_gnn_loss(x_features, masks, label, large_edge)
            else:
                result_gnn = self._compute_stage2_gnn_loss(x_features, masks, label, large_edge)

            self.optimizer.zero_grad()
            result_gnn['loss']['all'].backward()
            self.optimizer.step()

            # ✅ 修复3：立即记录损失（不管 rank）
            losses_gnn['all'].append(result_gnn['loss']['all'].item())
            if 'invariance' in result_gnn['loss']:
                losses_gnn['invariance'].append(result_gnn['loss']['invariance'].item())
            if 'causal' in result_gnn['loss']:
                losses_gnn['causal'].append(result_gnn['loss']['causal'].item())
            if 'l1_reg' in result_gnn['loss']:
                losses_gnn['l1_reg'].append(result_gnn['loss']['l1_reg'].item())

            # 记录准确率
            for pred_type, pred_output in result_gnn['preds'].items():
                if pred_type not in accs_gnn:
                    accs_gnn[pred_type] = []
                accs_gnn[pred_type].append(
                    self._compute_accuracy(pred_output, label)
                )

            # 记录掩码统计（仅主进程需要，用于日志）
            if self.rank == 0:
                self.current_mask_sums = {
                    'node': masks[0].sum().item(),
                    'edge': masks[1].sum().item()
                }

        # ✅ 修复4：保存epoch结果（确保列表非空）
        if self.rank == 0:
            train_res = {
                'mask': {},
                'gnn': {}
            }

            # 保存损失均值
            for k in losses_mask.keys():
                if len(losses_mask[k]) > 0:
                    train_res['mask'][k] = float(np.mean(losses_mask[k]))
                else:
                    train_res['mask'][k] = 0.0

            for k in losses_gnn.keys():
                if len(losses_gnn[k]) > 0:
                    train_res['gnn'][k] = float(np.mean(losses_gnn[k]))
                else:
                    train_res['gnn'][k] = 0.0

            # 保存准确率均值
            for k, v in accs_mask.items():
                if len(v) > 0:
                    train_res['mask'][f'acc_{k}'] = float(np.mean(v))
                else:
                    train_res['mask'][f'acc_{k}'] = 0.0

            for k, v in accs_gnn.items():
                if len(v) > 0:
                    train_res['gnn'][f'acc_{k}'] = float(np.mean(v))
                else:
                    train_res['gnn'][f'acc_{k}'] = 0.0

            self.epoch_results[epoch]['train'] = train_res
    
    def _eval_main_epoch(self, epoch: int, data_loader: DataLoader, phase: str):
        """主训练评估"""
        self.model.eval()
        self.mask.eval()

        all_outputs = []
        all_labels = []

        for data, _, label in data_loader:
            batch_data = data
            data = data.to(self.device)
            label = label.to(self.device)

            x_features = self._extract_features(data)

            mask_module = self.mask.module if isinstance(self.mask, nn.DataParallel) else self.mask
            masks, probs, sparsity = mask_module(train=False, return_probs=True)

            model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            # 评估时使用小图
            B, P = batch_data.shape[0], batch_data.shape[1]
            small_edge = self.edge_matrix.unsqueeze(0).repeat(B, 1, 1)
            outputs = model_module.prediction_causal_invariance(x_features, small_edge, masks, is_large_graph=False)

            all_outputs.append(outputs)
            all_labels.append(label)

        # 计算指标
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = compute_binary_metrics(all_outputs, all_labels)

        self.epoch_results[epoch][phase] = metrics

        # ✅ 修复：正确更新 best_val_acc 和 best_test_acc
        if phase == 'val' and metrics['accuracy'] > self.best_val_acc:
            self.best_val_acc = metrics['accuracy']
            self.best_epoch = epoch

            # ✅ 如果当前 epoch 已经评估过测试集，记录其准确率
            if 'test' in self.epoch_results[epoch]:
                self.best_test_acc = self.epoch_results[epoch]['test']['accuracy']
                logger.info(
                    f"💎 New Best - Val Acc: {metrics['accuracy']:.4f}, "
                    f"Val AUC: {metrics['auc']:.4f}, "
                    f"Test Acc: {self.best_test_acc:.4f}"
                )
            else:
                logger.info(
                    f"💎 New Best Val - Acc: {metrics['accuracy']:.4f}, "
                    f"AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}"
                )
    
    #==================== 损失计算 ====================
    
    def _compute_stage1_mask_loss(self, x, masks, label, lambda_reg, edge):
        """阶段1 Mask损失：不变性 + 变异性"""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        # 不变性
        yci = model.prediction_causal_invariance(x, edge, masks, True)
        loss_ci = self.criterion(yci, label).mean()
        
        # 变异性（熵损失）
        ycv = model.prediction_causal_variability(x, edge, masks, True)
        loss_cv = self._entropy_loss(ycv)
        
        # 稀疏性正则
        mask_module = self.mask.module if isinstance(self.mask, nn.DataParallel) else self.mask
        reg_loss = mask_module.compute_sparsity_regularization(lambda_reg=lambda_reg)
        
        # 总损失
        loss_weights = self.config['train']['loss_weights']
        loss_all = loss_weights['LCI'] * loss_ci + loss_weights['LCV'] * loss_cv + reg_loss
        
        return {
            'loss': {
                'all': loss_all,
                'invariance': loss_ci,
                'variability': loss_cv,
                'sparsity_reg': reg_loss
            },
            'preds': {
                'invariance': yci,
                'variability': ycv
            }
        }
    
    def _compute_stage2_mask_loss(self, x, masks, label, lambda_reg, edge):
        """阶段2 Mask损失：因果 + 反事实 + 不变性"""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        # 不变性
        yci = model.prediction_causal_invariance(x, edge, masks, True)
        loss_ci = self.criterion(yci, label).mean()
        
        # 因果性
        yc = model.prediction_spurious_fusion(x, edge, masks, True)   # 使用相同方法
        loss_c = self.criterion(yc, label).mean()
        
        # 反事实
        yo = model.prediction_intrinsic_fusion(x, edge, masks, True)
        loss_o = self.criterion(yo, 1 - label).mean()
        
        # 稀疏性正则
        mask_module = self.mask.module if isinstance(self.mask, nn.DataParallel) else self.mask
        reg_loss = mask_module.compute_sparsity_regularization(lambda_reg=lambda_reg)
        
        # 总损失
        loss_weights = self.config['train']['loss_weights']
        loss_all = (loss_weights['LC'] * loss_c + 
                   loss_weights['LO'] * loss_o + 
                   loss_weights['LCI'] * loss_ci + 
                   reg_loss)
        
        return {
            'loss': {
                'all': loss_all,
                'causal': loss_c,
                'counterfactual': loss_o,
                'invariance': loss_ci,
                'sparsity_reg': reg_loss
            },
            'preds': {
                'causal': yc,
                'counterfactual': yo,
                'invariance': yci
            }
        }
    def _compute_stage1_gnn_loss(self, x, masks, label, edge):
        """阶段1 GNN损失：不变性"""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        yci = model.prediction_causal_invariance(x, edge, masks, True)
        loss_ci = self.criterion(yci, label).mean()
        l1_loss = self._compute_l1_regularization()
        loss_all = loss_ci + l1_loss
        
        return {
            'loss': {
                'all': loss_all,
                'invariance': loss_ci,
                'l1_reg': l1_loss
            },
            'preds': {
                'invariance': yci
            }
        }
    
    def _compute_stage2_gnn_loss(self, x, masks, label, edge):
        """阶段2 GNN损失：不变性 + 因果"""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        yci = model.prediction_causal_invariance(x, edge, masks, True)
        loss_ci = self.criterion(yci, label).mean()
        
        yc = model.prediction_spurious_fusion(x, edge, masks, True) 
        loss_c = self.criterion(yc, label).mean()
        
        l1_loss = self._compute_l1_regularization()
        loss_all = loss_ci + loss_c + l1_loss
        
        return {
            'loss': {
                'all': loss_all,
                'invariance': loss_ci,
                'causal': loss_c,
                'l1_reg': l1_loss
            },
            'preds': {
                'invariance': yci,
                'causal': yc
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
        """熵损失（用于变异性）"""
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
                   f"Acc: {train_res.get('acc_invariance', 0):.4f}")

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

        stage_name = "Stage 1 (Invariance+Variability)" if is_stage1 else "Stage 2 (Causal+Counterfactual)"

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
            logger.info(f"     ├─ Invariance:  {mask_res.get('invariance', 0):.4f} "  # ✅ 改为 'invariance'
                       f"(Acc: {mask_res.get('acc_invariance', 0):.2%})")
            logger.info(f"     ├─ Variability: {mask_res.get('variability', 0):.4f} "  # ✅ 改为 'variability'
                       f"(Acc: {mask_res.get('acc_variability', 0):.2%})")
            logger.info(f"     └─ Sparsity:    {mask_res.get('sparsity_reg', 0):.4f}")  # ✅ 改为 'sparsity_reg'
        else:
            logger.info(f"     ├─ Invariance:     {mask_res.get('invariance', 0):.4f} "  # ✅
                       f"(Acc: {mask_res.get('acc_invariance', 0):.2%})")
            logger.info(f"     ├─ Causal:         {mask_res.get('causal', 0):.4f} "  # ✅ 改为 'causal'
                       f"(Acc: {mask_res.get('acc_causal', 0):.2%})")
            logger.info(f"     ├─ Counterfactual: {mask_res.get('counterfactual', 0):.4f} "  # ✅ 改为 'counterfactual'
                       f"(Acc: {mask_res.get('acc_counterfactual', 0):.2%})")
            logger.info(f"     └─ Sparsity:       {mask_res.get('sparsity_reg', 0):.4f}")  # ✅

        # ✅ 修复 GNN 训练详情（键名匹配）
        logger.info(f"\n🧠 GNN Training:")
        logger.info(f"   Total Loss: {gnn_res.get('all', 0):.4f}")  # ✅ 改为 'all'
        logger.info(f"     ├─ Invariance: {gnn_res.get('invariance', 0):.4f} "  # ✅ 改为 'invariance'
                   f"(Acc: {gnn_res.get('acc_invariance', 0):.2%})")
        if not is_stage1:
            logger.info(f"     ├─ Causal:     {gnn_res.get('causal', 0):.4f} "  # ✅ 改为 'causal'
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