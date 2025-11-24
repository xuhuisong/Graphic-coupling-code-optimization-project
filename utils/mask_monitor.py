"""
Mask Logits 监控工具
用于保存和分析训练过程中的因果掩码演化
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MaskMonitor:
    """
    因果掩码监控器
    
    功能：
    1. 每个epoch保存 node/edge logits 到 CSV
    2. 生成稀疏度统计摘要
    3. 可选：生成可视化图表
    
    文件结构：
    experiments/exp_name/fold_0/
    ├── masks/
    │   ├── epoch_060_node_logits.csv
    │   ├── epoch_060_edge_logits.csv
    │   ├── epoch_061_node_logits.csv
    │   ├── epoch_061_edge_logits.csv
    │   └── ...
    └── masks_summary.csv
    """
    
    def __init__(self, save_dir: str, fold: int):
        """
        Args:
            save_dir: 实验根目录（如 experiments/exp_name）
            fold: 当前fold索引
        """
        self.fold = fold
        self.fold_dir = Path(save_dir) / f"fold_{fold}"
        self.mask_dir = self.fold_dir / "masks"
        
        # 创建目录
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        
        # 摘要数据
        self.summary_data = []
        
        logger.info(f"MaskMonitor initialized: {self.mask_dir}")
    
    def save_epoch_masks(
        self, 
        epoch: int, 
        mask_module,
        additional_info: Optional[Dict] = None
    ):
        """
        保存当前epoch的mask logits
        
        Args:
            epoch: 当前epoch
            mask_module: CausalMask 模块实例
            additional_info: 额外信息（如loss, accuracy等）
        """
        with torch.no_grad():
            # 1. 获取原始 logits
            node_logits = mask_module.node_mask.cpu().numpy()  # [P, 2]
            edge_logits = mask_module.edge_mask.cpu().numpy()  # [P, P, 2]
            learnable_mask = mask_module.learnable_mask.cpu().numpy()  # [P, P]
            
            # 2. 计算概率（softmax）
            node_probs = F.softmax(mask_module.node_mask, dim=1).cpu().numpy()  # [P, 2]
            edge_probs = F.softmax(mask_module.edge_mask, dim=-1).cpu().numpy()  # [P, P, 2]
            
            # 3. 保存 Node Logits
            self._save_node_logits(epoch, node_logits, node_probs)
            
            # 4. 保存 Edge Logits（只保存可学习的边）
            self._save_edge_logits(epoch, edge_logits, edge_probs, learnable_mask)
            
            # 5. 更新摘要
            self._update_summary(epoch, node_logits, edge_logits, learnable_mask, additional_info)
        
        logger.debug(f"Epoch {epoch}: Masks saved to {self.mask_dir}")
    
    def _save_node_logits(self, epoch: int, logits: np.ndarray, probs: np.ndarray):
        """保存节点logits到CSV"""
        P = logits.shape[0]
        
        # 构建DataFrame
        df = pd.DataFrame({
            'node_id': range(P),
            'logit_non_causal': logits[:, 0],
            'logit_causal': logits[:, 1],
            'logit_diff': logits[:, 1] - logits[:, 0],  # 正值表示倾向因果
            'prob_non_causal': probs[:, 0],
            'prob_causal': probs[:, 1],
            'decision': (logits[:, 1] > logits[:, 0]).astype(int)  # 硬决策
        })
        
        # 保存
        filepath = self.mask_dir / f"epoch_{epoch:03d}_node_logits.csv"
        df.to_csv(filepath, index=False, float_format='%.6f')
    
    def _save_edge_logits(
        self, 
        epoch: int, 
        logits: np.ndarray, 
        probs: np.ndarray,
        learnable_mask: np.ndarray
    ):
        """保存边logits到CSV（只保存可学习的边）"""
        P = logits.shape[0]
        
        # 找出所有可学习的边
        learnable_edges = np.where(learnable_mask > 0)
        num_edges = len(learnable_edges[0])
        
        # 构建数据
        data = {
            'source_node': learnable_edges[0],
            'target_node': learnable_edges[1],
            'logit_non_causal': logits[learnable_edges[0], learnable_edges[1], 0],
            'logit_causal': logits[learnable_edges[0], learnable_edges[1], 1],
            'logit_diff': (logits[learnable_edges[0], learnable_edges[1], 1] - 
                          logits[learnable_edges[0], learnable_edges[1], 0]),
            'prob_non_causal': probs[learnable_edges[0], learnable_edges[1], 0],
            'prob_causal': probs[learnable_edges[0], learnable_edges[1], 1],
            'decision': (logits[learnable_edges[0], learnable_edges[1], 1] > 
                        logits[learnable_edges[0], learnable_edges[1], 0]).astype(int)
        }
        
        df = pd.DataFrame(data)
        
        # 保存
        filepath = self.mask_dir / f"epoch_{epoch:03d}_edge_logits.csv"
        df.to_csv(filepath, index=False, float_format='%.6f')
        
        logger.debug(f"Saved {num_edges} learnable edges")
    
    def _update_summary(
        self, 
        epoch: int, 
        node_logits: np.ndarray,
        edge_logits: np.ndarray,
        learnable_mask: np.ndarray,
        additional_info: Optional[Dict]
    ):
        """更新训练摘要"""
        # 节点统计
        node_decisions = (node_logits[:, 1] > node_logits[:, 0]).astype(int)
        num_causal_nodes = node_decisions.sum()
        node_sparsity = num_causal_nodes / len(node_logits)
        
        # 边统计
        learnable_edges = np.where(learnable_mask > 0)
        edge_decisions = (edge_logits[learnable_edges[0], learnable_edges[1], 1] > 
                         edge_logits[learnable_edges[0], learnable_edges[1], 0]).astype(int)
        num_causal_edges = edge_decisions.sum()
        total_learnable_edges = len(learnable_edges[0])
        edge_sparsity = num_causal_edges / total_learnable_edges if total_learnable_edges > 0 else 0
        
        # Logits 分布统计
        node_logit_diffs = node_logits[:, 1] - node_logits[:, 0]
        edge_logit_diffs = (edge_logits[learnable_edges[0], learnable_edges[1], 1] - 
                           edge_logits[learnable_edges[0], learnable_edges[1], 0])
        
        # 构建摘要行
        summary = {
            'epoch': epoch,
            'num_causal_nodes': num_causal_nodes,
            'node_sparsity': node_sparsity,
            'num_causal_edges': num_causal_edges,
            'edge_sparsity': edge_sparsity,
            'node_logit_diff_mean': node_logit_diffs.mean(),
            'node_logit_diff_std': node_logit_diffs.std(),
            'node_logit_diff_min': node_logit_diffs.min(),
            'node_logit_diff_max': node_logit_diffs.max(),
            'edge_logit_diff_mean': edge_logit_diffs.mean(),
            'edge_logit_diff_std': edge_logit_diffs.std(),
            'edge_logit_diff_min': edge_logit_diffs.min(),
            'edge_logit_diff_max': edge_logit_diffs.max(),
        }
        
        # 添加额外信息
        if additional_info:
            summary.update(additional_info)
        
        self.summary_data.append(summary)
    
    def save_summary(self):
        """保存训练摘要到CSV"""
        if not self.summary_data:
            logger.warning("No summary data to save")
            return
        
        df = pd.DataFrame(self.summary_data)
        filepath = self.fold_dir / "masks_summary.csv"
        df.to_csv(filepath, index=False, float_format='%.6f')
        
        logger.info(f"✅ Mask summary saved: {filepath}")
        logger.info(f"   Total epochs: {len(self.summary_data)}")
    
    def generate_analysis_report(self):
        """生成简要分析报告"""
        if not self.summary_data:
            return
        
        df = pd.DataFrame(self.summary_data)
        
        report = f"""
{'='*60}
Mask Evolution Analysis Report - Fold {self.fold}
{'='*60}

Training Overview:
  Total Epochs: {len(df)}
  Epoch Range: {df['epoch'].min()} - {df['epoch'].max()}

Node Sparsity Evolution:
  Initial: {df['node_sparsity'].iloc[0]:.2%}
  Final:   {df['node_sparsity'].iloc[-1]:.2%}
  Min:     {df['node_sparsity'].min():.2%} (Epoch {df.loc[df['node_sparsity'].idxmin(), 'epoch']})
  Max:     {df['node_sparsity'].max():.2%} (Epoch {df.loc[df['node_sparsity'].idxmax(), 'epoch']})

Edge Sparsity Evolution:
  Initial: {df['edge_sparsity'].iloc[0]:.2%}
  Final:   {df['edge_sparsity'].iloc[-1]:.2%}
  Min:     {df['edge_sparsity'].min():.2%} (Epoch {df.loc[df['edge_sparsity'].idxmin(), 'epoch']})
  Max:     {df['edge_sparsity'].max():.2%} (Epoch {df.loc[df['edge_sparsity'].idxmax(), 'epoch']})

Logit Diff Statistics (Final Epoch):
  Node Mean: {df['node_logit_diff_mean'].iloc[-1]:.4f}
  Node Std:  {df['node_logit_diff_std'].iloc[-1]:.4f}
  Edge Mean: {df['edge_logit_diff_mean'].iloc[-1]:.4f}
  Edge Std:  {df['edge_logit_diff_std'].iloc[-1]:.4f}

{'='*60}
"""
        
        # 保存报告
        report_path = self.fold_dir / "mask_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(report)
        logger.info(f"✅ Analysis report saved: {report_path}")