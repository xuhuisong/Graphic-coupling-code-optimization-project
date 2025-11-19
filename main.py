"""
统一训练入口
优雅的端到端训练流程
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.checkpoint import CheckpointManager
from utils.logging_config import setup_logging
from core.densenet_manager import DenseNetManager
from core.graph_builder import GraphBuilder
from core.trainer import CausalTrainer
from data.dataset import PatchDataset, get_fold_splits
from torch.utils.data import DataLoader, Subset
from data.dataset import collate_fn


logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment(config: dict) -> Path:
    """设置实验目录"""
    exp_dir = Path(f"./experiments/{config['exp_name']}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'results').mkdir(exist_ok=True)
    
    return exp_dir


def run_training_for_fold(
    config: dict,
    fold: int,
    densenet_model,
    edge_prior_mask,
    checkpoint_manager: CheckpointManager,
    exp_dir: Path
) -> dict:
    """
    运行单个 fold 的训练
    
    Args:
        config: 配置字典
        fold: fold 索引
        densenet_model: 预训练的 DenseNet 模型
        edge_matrix: 图边矩阵
        checkpoint_manager: 缓存管理器
        exp_dir: 实验目录
        
    Returns:
        训练结果字典
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"📂 Fold {fold}/{config['data']['num_folds']-1}")
    logger.info(f"{'='*80}\n")
    
    # 1. 准备数据
    data_dir = os.path.join(config['data']['data_dir'], config['data']['data_name'])
    dataset = PatchDataset(data_dir)
    
    # 获取数据分割
    train_indices, val_indices, test_indices = get_fold_splits(
        data_dir,
        fold,
        config['split_seed'],
        num_folds=config['data']['num_folds']
    )
    
    logger.info(f"📊 Data split:")
    logger.info(f"   Train: {len(train_indices)} samples")
    logger.info(f"   Val:   {len(val_indices)} samples")
    logger.info(f"   Test:  {len(test_indices)} samples")
    
    # 创建数据加载器
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config['train']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['densenet']['pretrain']['num_workers'],  # 🔧 使用配置的值
        pin_memory=True,       # ✅ 保持
        persistent_workers=True,  # 🔧 新增：避免每epoch重启worker
        prefetch_factor=4      # 🔧 新增：预加载4个batch
    )

    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=config['train']['test_batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['densenet']['pretrain']['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=config['train']['test_batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['densenet']['pretrain']['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    # 2. 创建训练器
    logger.info("\n" + "-"*80)
    logger.info("🔧 Initializing trainer...")
    logger.info("-"*80)
    
    work_dir = exp_dir / f"fold_{fold}"
    work_dir.mkdir(exist_ok=True)
    
    trainer = CausalTrainer(
        config=config,
        fold=fold,
        densenet_model=densenet_model,
        edge_prior_mask=edge_prior_mask,
        checkpoint_manager=checkpoint_manager,
        work_dir=str(work_dir),
        device='cuda',
        rank=0
    )
    
    logger.info("✅ Trainer initialized")
    
    # 3. 开始训练
    logger.info("\n" + "="*80)
    logger.info("🚀 Starting training...")
    logger.info("="*80)
    
    results = trainer.train(train_loader, val_loader, test_loader)
    
    logger.info(f"\n✅ Fold {fold} training completed!")
    logger.info(f"   Val Acc:  {results['val_acc']:.4f}")
    logger.info(f"   Test Acc: {results['test_acc']:.4f}")
    
    return results


def main():
    """主函数"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='优雅的因果图神经网络训练系统')
    parser.add_argument(
        '--config',
        type=str,
        default='config/train_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--folds',
        type=int,
        nargs='+',
        default=None,
        help='要训练的 fold 列表（如：--folds 0 1 2）'
    )
    args = parser.parse_args()
    
    # 1. 加载配置
    print("="*80)
    print("📋 加载配置文件...")
    print("="*80)
    
    config = load_config(args.config)
    print(f"配置文件: {args.config}")
    print(f"实验名称: {config['exp_name']}")
    
    # 2. 设置实验目录和日志
    exp_dir = setup_experiment(config)
    log_file = exp_dir / 'logs' / f"train_{config['exp_name']}.log"
    setup_logging(log_file=str(log_file), level='INFO')
    
    logger.info("="*80)
    logger.info("🎯 优雅的因果图神经网络训练系统")
    logger.info("="*80)
    logger.info(f"实验目录: {exp_dir}")
    logger.info(f"日志文件: {log_file}")
    
    # 3. 初始化管理器
    logger.info("\n" + "="*80)
    logger.info("🔧 初始化管理器...")
    logger.info("="*80)
    
    checkpoint_manager = CheckpointManager(
        base_cache_dir=config['cache']['base_dir'],
        auto_clean=config['cache'].get('auto_clean', False)
    )
    logger.info(f"✅ CheckpointManager 初始化完成")
    
    densenet_manager = DenseNetManager(
        data_dir=os.path.join(config['data']['data_dir'], config['data']['data_name']),
        checkpoint_manager=checkpoint_manager,
        config=config['densenet']
    )
    logger.info(f"✅ DenseNetManager 初始化完成")
    
    graph_builder = GraphBuilder(
        data_dir=os.path.join(config['data']['data_dir'], config['data']['data_name']),
        checkpoint_manager=checkpoint_manager,
        densenet_manager=densenet_manager,
        config=config['graph']['build']
    )
    logger.info(f"✅ GraphBuilder 初始化完成")
    
    # 4. 构建全局边矩阵（所有 fold 共享）
    logger.info("\n" + "="*80)
    logger.info("🕸️  构建/加载边先验候选集...")
    logger.info("="*80)
    
    edge_prior_mask = graph_builder.get_edge_prior_mask(
        split_seed=config['split_seed'],
        fold_for_feature_extraction=0
    )
    logger.info(f"✅ 边先验候选集准备完成:")
    logger.info(f"   形状: {edge_prior_mask.shape}")
    logger.info(f"   可学习边位置数: {edge_prior_mask.sum()}")
    logger.info(f"   密度: {edge_prior_mask.sum() / (edge_prior_mask.shape[0] ** 2):.4f}")
    
    # 5. 执行每个 fold 的训练
    folds_to_train = args.folds if args.folds else config['train']['folds']
    all_results = []
    
    logger.info("\n" + "="*80)
    logger.info(f"🚂 开始训练 {len(folds_to_train)} 个 fold...")
    logger.info("="*80)
    
    for fold in folds_to_train:
        # 获取该 fold 的预训练 DenseNet
        logger.info(f"\n📦 获取 Fold {fold} 的预训练 DenseNet...")
        densenet_model = densenet_manager.get_pretrained_model(
            fold=fold,
            split_seed=config['split_seed']
        )
        logger.info(f"✅ DenseNet 模型准备完成")
        
        # 执行训练
        fold_results = run_training_for_fold(
            config=config,
            fold=fold,
            densenet_model=densenet_model,
            edge_prior_mask=edge_prior_mask,
            checkpoint_manager=checkpoint_manager,
            exp_dir=exp_dir
        )
        
        all_results.append(fold_results)
    
    # 6. 汇总结果
    logger.info("\n" + "="*80)
    logger.info("📊 训练完成！汇总结果：")
    logger.info("="*80)
    
    import numpy as np
    
    val_accs = [r['val_acc'] for r in all_results]
    test_accs = [r['test_acc'] for r in all_results]
    
    logger.info(f"\n📈 验证集准确率:")
    for i, acc in enumerate(val_accs):
        logger.info(f"   Fold {folds_to_train[i]}: {acc:.4f}")
    logger.info(f"   平均: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    
    logger.info(f"\n📈 测试集准确率:")
    for i, acc in enumerate(test_accs):
        logger.info(f"   Fold {folds_to_train[i]}: {acc:.4f}")
    logger.info(f"   平均: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    
    # 7. 保存结果
    import json
    results_file = exp_dir / 'results' / 'final_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'fold_results': all_results,
            'summary': {
                'val_acc_mean': float(np.mean(val_accs)),
                'val_acc_std': float(np.std(val_accs)),
                'test_acc_mean': float(np.mean(test_accs)),
                'test_acc_std': float(np.std(test_accs))
            }
        }, f, indent=2)
    
    logger.info(f"\n✅ 结果已保存到: {results_file}")
    logger.info("\n" + "="*80)
    logger.info("🎉 所有任务完成！优雅地结束！")
    logger.info("="*80)

if __name__ == "__main__":
    main()