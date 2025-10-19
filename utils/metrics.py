"""
统一评估指标模块
用于二分类任务的全面评估
"""

import numpy as np
import torch
from typing import Dict, Union, Tuple
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)


class BinaryClassificationMetrics:
    """
    二分类任务评估指标计算器
    
    功能：
        - 一次性计算所有常用指标
        - 支持 torch.Tensor 和 numpy.ndarray
        - 自动处理设备转换
        - 返回格式化的字典
    
    Example:
        >>> metrics = BinaryClassificationMetrics()
        >>> outputs = torch.tensor([[0.2, 0.8], [0.9, 0.1]])
        >>> labels = torch.tensor([1, 0])
        >>> results = metrics.compute(outputs, labels)
        >>> print(results['accuracy'])
        1.0
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: 分类阈值（默认0.5）
        """
        self.threshold = threshold
    
    @staticmethod
    def _to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """将 Tensor 转换为 numpy 数组"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def compute(
        self,
        outputs: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        return_confusion_matrix: bool = False
    ) -> Dict[str, float]:
        """
        计算所有二分类指标
        
        Args:
            outputs: 模型输出 logits，shape [N, 2] 或概率 [N]
            labels: 真实标签，shape [N]
            return_confusion_matrix: 是否返回混淆矩阵
            
        Returns:
            包含所有指标的字典：
            {
                'accuracy': 准确率,
                'balanced_accuracy': 平衡准确率,
                'auc': ROC AUC,
                'sensitivity': 敏感度/召回率,
                'specificity': 特异度,
                'precision': 精确度,
                'f1': F1分数,
                'confusion_matrix': 混淆矩阵 (可选)
            }
        """
        # 转换为 numpy
        labels_np = self._to_numpy(labels)
        
        # 处理 outputs 格式
        outputs_np = self._to_numpy(outputs)
        
        if len(outputs_np.shape) == 2:
            # 如果是 logits [N, 2]，转为概率
            probs = self._softmax(outputs_np)[:, 1]
            preds = (probs >= self.threshold).astype(int)
        else:
            # 如果已经是概率 [N]
            probs = outputs_np
            preds = (probs >= self.threshold).astype(int)
        
        # 计算基础指标
        acc = accuracy_score(labels_np, preds)
        balanced_acc = balanced_accuracy_score(labels_np, preds)
        
        # AUC (需要概率)
        try:
            auc = roc_auc_score(labels_np, probs)
        except ValueError:
            # 如果只有一个类别，AUC无法计算
            auc = 0.0
        
        # 混淆矩阵相关指标
        tn, fp, fn, tp = confusion_matrix(labels_np, preds).ravel()
        
        sensitivity = recall_score(labels_np, preds, zero_division=0)  # TP / (TP + FN)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TN / (TN + FP)
        precision = precision_score(labels_np, preds, zero_division=0)  # TP / (TP + FP)
        f1 = f1_score(labels_np, preds, zero_division=0)
        
        results = {
            'accuracy': float(acc),
            'balanced_accuracy': float(balanced_acc),
            'auc': float(auc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1': float(f1),
        }
        
        if return_confusion_matrix:
            results['confusion_matrix'] = {
                'TP': int(tp),
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn)
            }
        
        return results
    
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """手动计算 softmax（避免导入torch）"""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def format_results(self, results: Dict[str, float]) -> str:
        """
        格式化输出结果
        
        Args:
            results: compute() 返回的指标字典
            
        Returns:
            格式化的字符串
        """
        lines = [
            f"Accuracy:          {results['accuracy']:.4f}",
            f"Balanced Acc:      {results['balanced_accuracy']:.4f}",
            f"AUC:               {results['auc']:.4f}",
            f"Sensitivity:       {results['sensitivity']:.4f}",
            f"Specificity:       {results['specificity']:.4f}",
            f"Precision:         {results['precision']:.4f}",
            f"F1-Score:          {results['f1']:.4f}"
        ]
        
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            lines.append("\nConfusion Matrix:")
            lines.append(f"  TP: {cm['TP']}, TN: {cm['TN']}")
            lines.append(f"  FP: {cm['FP']}, FN: {cm['FN']}")
        
        return "\n".join(lines)


# 便捷函数
def compute_binary_metrics(
    outputs: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    快速计算二分类指标的便捷函数
    
    Args:
        outputs: 模型输出
        labels: 真实标签
        threshold: 分类阈值
        
    Returns:
        指标字典
    """
    calculator = BinaryClassificationMetrics(threshold=threshold)
    return calculator.compute(outputs, labels)


def print_metrics(
    outputs: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    phase: str = "Eval"
) -> Dict[str, float]:
    """
    计算并打印指标
    
    Args:
        outputs: 模型输出
        labels: 真实标签
        phase: 阶段名称（如 'Train', 'Val', 'Test'）
        
    Returns:
        指标字典
    """
    calculator = BinaryClassificationMetrics()
    results = calculator.compute(outputs, labels, return_confusion_matrix=True)
    
    print(f"\n{'='*50}")
    print(f"{phase} Metrics:")
    print('='*50)
    print(calculator.format_results(results))
    print('='*50)
    
    return results