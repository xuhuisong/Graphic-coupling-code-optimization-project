from __future__ import print_function
import os
import time
import yaml
import pickle
import pandas as pd # 导入pandas用于后续结果处理
from shutil import copytree, ignore_patterns
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

# [DDP修改] 1. 导入分布式训练所需的新模块
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from tools.utils import seed_torch, str2bool, str2list, get_graph_name, import_class, coef_list, Prior
from settle_results import SettleResults
import warnings
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")
TF_ENABLE_ONEDNN_OPTS = 0
# 使用warn_only选项，允许非确定性操作但会发出警告
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='PRISM Framework for Alzheimer\'s Disease Detection')
    
    # 基本配置
    parser.add_argument('--exp_name', default='', type=str, help='实验名称')
    parser.add_argument('--save_dir', default='./results', type=str, help='结果保存文件夹')
    parser.add_argument('--data_dir', default='./', type=str, help='数据目录')
    parser.add_argument('--config', default='./train_causal.yaml', type=str, help='配置文件路径')
    
    # 随机种子和数据分割
    parser.add_argument('--seed', default=1, type=int, help='随机种子')
    parser.add_argument('--split_seed', default=1, type=int, help='交叉验证分割随机种子')
    parser.add_argument('--fold', default=0, type=int, help='交叉验证折数(0-4)')
    
    # 数据参数
    parser.add_argument('--data_name', default='arising_2_0n0', type=str, help='数据集名称')
    parser.add_argument('--patch_size', default=5, type=int, help='脑部补丁大小')
    
    # 训练参数
    parser.add_argument('--pre_epoch', type=int, default=30, help='预训练阶段的轮数')
    parser.add_argument('--num_epoch', type=int, default=150, help='总训练轮数')
    parser.add_argument('--stage_transition_epoch', type=int, default=100, help='阶段转换轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='总训练批次大小 (会被分配到各个GPU)')
    parser.add_argument('--test_batch_size', type=int, default=32, help='测试批次大小')
    parser.add_argument('--inc_mode', type=str, default='lin', help='增长模式: lin(线性), log(对数), power(幂)')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='SGD', help='优化器类型')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='初始学习率')
    parser.add_argument('--base_lr_mask', type=float, default=16, help='掩码学习率')
    parser.add_argument('--scheduler', type=str, default='auto', help='学习率调度器')
    parser.add_argument('--stepsize', type=int, default=10, help='调度器步长')
    parser.add_argument('--gamma', type=float, default=0.5, help='学习率衰减率')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='是否使用Nesterov动量')
    
    # 损失权重 - 更新为PRISM框架的术语
    parser.add_argument('--LC', type=float, default=1.0, help='内在子图损失权重')
    parser.add_argument('--LO', type=float, default=1, help='反事实损失权重')
    parser.add_argument('--LCI', type=float, default=1, help='不变性损失权重')
    parser.add_argument('--LCV', type=float, default=1, help='变异性损失权重')
    parser.add_argument('--LG', type=float, default=0.002, help='引导损失权重')
    parser.add_argument('--lambda_l1', type=float, default=0.0001, help='L1正则化强度')
    
    # 模型参数
    parser.add_argument('--model', default='net.networks', help='模型类')
    parser.add_argument('--model_args', default=dict(), help='模型参数')
    parser.add_argument('--pretrained_path', default=None, type=str, help='预训练模型路径 (手动模式)')
    parser.add_argument('--freeze_extractor', type=str2bool, default=True, help='是否冻结特征提取器')
    
    # [新增] DenseNet预训练相关参数
    parser.add_argument('--auto_pretrain_densenet', type=str2bool, default=True, 
                        help='是否自动使用对应折的预训练DenseNet (推荐)')
    parser.add_argument('--densenet_epochs', type=int, default=50, 
                        help='DenseNet预训练轮数 (仅在需要重新预训练时使用)')
    parser.add_argument('--densenet_save_dir', default='pretrained_densenet', type=str, 
                        help='DenseNet预训练模型保存基础目录')
    
    # 图构建参数
    parser.add_argument('--graph_args', default=dict(), help='图构建参数')
    parser.add_argument('--feeder', default='tools.feeder.FeederGraph', help='数据加载器')
    
    # 可视化和调试
    parser.add_argument('--save_score', type=str2bool, default=True, help='是否保存分类得分')
    parser.add_argument('--log_interval', type=int, default=100, help='打印消息的间隔(#迭代)')
    parser.add_argument('--save_interval', type=int, default=5, help='保存模型的间隔(#迭代)')
    parser.add_argument('--eval_interval', type=int, default=5, help='评估模型的间隔(#迭代)')
    parser.add_argument('--print_log', type=str2bool, default=True, help='是否打印日志')
    
    # 其他参数
    parser.add_argument('--num_worker', type=int, default=0, help='数据加载器的工作进程数')
    parser.add_argument('--train_feeder_args', default=dict(), help='训练数据加载器参数')
    parser.add_argument('--test_feeder_args', default=dict(), help='测试数据加载器参数')
    parser.add_argument('--device', type=int, default=0, nargs='+', 
                        help='用于训练或测试的GPU索引 (DDP模式下此参数会被忽略)')
    parser.add_argument('--start_epoch', type=int, default=0, help='从哪个轮次开始训练')
    parser.add_argument('--gumble_tau', type=float, default=1, help='Gumbel-Softmax的温度参数')
    
    return parser

class Processor:
    """
    因果图神经网络训练处理器
    """
    # [修正] __init__ 方法的修改
    def __init__(self, rank, world_size, arg):
        """初始化处理器，设置配置、加载数据和模型"""
        self.rank = rank
        self.world_size = world_size
        self.arg = arg
        
        # [修正] 让所有进程都调用save_arg()来设置必要的属性
        self.save_arg()
        
        # 只有主进程写完文件后，其他进程再继续
        dist.barrier()
        
        self.lambda_l1 = getattr(arg, 'lambda_l1', 0.0001)
        self.load_precomputed_graph()
        self.load_data()

        if self.rank == 0:
            self.train_writer = SummaryWriter(os.path.join(self.work_dir, 'train'), 'train')
            self.val_writer = SummaryWriter(os.path.join(self.work_dir, 'val'), 'val')
            self.test_writer = SummaryWriter(os.path.join(self.work_dir, 'test'), 'test')

        self.global_step = 0
        self.best_acc = 0
        self.best_val_acc = 0
        self.best_model_state = None
        self.best_mask_state = None
        self.best_masks = None
        self.best_epoch = -1
        self.epoch_mask_sums = {}
        self.best_epoch_pretrain = -1
        self.best_epoch_stage1 = -1
        self.best_epoch_stage2 = -1
        self.load_model()
        self.load_optimizer()

        # [修正] save_arg 方法的修改
        
    def get_fold_feature_path(self, base_data_name, fold, split_seed):
        """
        获取指定fold的特征数据路径

        Args:
            base_data_name: 原始数据名称 (如: ADvsCN.ex001_2_p288_pw24_all)
            fold: 折索引
            split_seed: 分割种子

        Returns:
            tuple: (fold_path, feature_base_path)
        """
        # 构建fold特征路径
        feature_base_path = f"{base_data_name}_features_by_fold"
        fold_path = os.path.join(feature_base_path, f"fold_{fold}")

        return fold_path, feature_base_path  # 返回两个值
        
    def load_data(self):
        """加载训练、验证和测试数据，应用一致性边掩码"""
        self.print_log('正在加载数据集(包含独立验证集)...')
        Feeder = import_class(self.arg.feeder)
        val_ratio = 0.2

        train_set = Feeder(
            fold=self.arg.fold, split_seed=self.arg.split_seed, out_dir=self.data_path, 
            mode='train', graph_arg=self.arg.graph_args, **self.arg.train_feeder_args,
            build_large_graph=True, consistent_mask=self.consistent_edges_mask, val_ratio=val_ratio
        )
        val_set = Feeder(
            fold=self.arg.fold, split_seed=self.arg.split_seed, out_dir=self.data_path, 
            mode='val', graph_arg=self.arg.graph_args, **self.arg.test_feeder_args,
            build_large_graph=False, consistent_mask=self.consistent_edges_mask, val_ratio=val_ratio
        )
        test_set = Feeder(
            fold=self.arg.fold, split_seed=self.arg.split_seed, out_dir=self.data_path, 
            mode='test', graph_arg=self.arg.graph_args, **self.arg.test_feeder_args,
            build_large_graph=False, consistent_mask=self.consistent_edges_mask, val_ratio=val_ratio
         )

        self.DiffNode = Prior(train_set, device=self.rank)
        prior_sparsity = self.DiffNode.sum() / self.DiffNode.shape[0]
        self.print_log(f'先验知识稀疏度: 节点 {self.DiffNode.sum()}/{self.DiffNode.shape[0]}={prior_sparsity:.4f}')

        self.data_loader = dict()

        train_sampler = DistributedSampler(train_set, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        per_gpu_batch_size = self.arg.batch_size // self.world_size
        self.print_log(f"总批次大小: {self.arg.batch_size}, GPU数量: {self.world_size}, 每GPU批次大小: {per_gpu_batch_size}", log_type="info")

        self.data_loader['train'] = DataLoader(
            dataset=train_set, batch_size=per_gpu_batch_size, sampler=train_sampler,
            num_workers=self.arg.num_worker, drop_last=True, pin_memory=True
        )
        self.data_loader['val'] = DataLoader(
            dataset=val_set, batch_size=self.arg.test_batch_size, num_workers=self.arg.num_worker,
            shuffle=False, drop_last=False
        )
        self.data_loader['test'] = DataLoader(
            dataset=test_set, batch_size=self.arg.test_batch_size, num_workers=self.arg.num_worker,
            shuffle=False, drop_last=False
        )
        self.print_log(f"数据加载完成: 训练集 {len(train_set)} 样本, 验证集 {len(val_set)} 样本, 测试集 {len(test_set)} 样本")  
      
        
    def save_arg(self):
        """保存配置参数并创建工作目录"""
        # 这部分是变量设置，所有进程都需要执行
        self.num_class = int(self.arg.data_name.split('_')[1])
        self.reg = coef_list(
            init=0.01, final=1, pre_epoch=self.arg.pre_epoch,
            inc_epoch=self.arg.num_epoch, num_epoch=self.arg.num_epoch, kind=self.arg.inc_mode
        )
        self.arg.graph_args = str2list(self.arg.graph_args, flag='simple')
        self.arg.model_args = str2list(self.arg.model_args, flag='deep')

        # [新增] 智能选择数据路径
        original_data_name = self.arg.data_name

        # 检查是否有对应fold的特征数据
        fold_path, feature_base_path = self.get_fold_feature_path(
            original_data_name,
            self.arg.fold,
            self.arg.split_seed
        )

        # 检查fold特征是否存在
        full_fold_path = os.path.join(self.arg.data_dir, fold_path)
        fold_data_file = os.path.join(full_fold_path, 'data.npy')

        if os.path.exists(fold_data_file):
            if self.rank == 0:
                print(f"[INFO] 发现fold {self.arg.fold}的预提取特征")
                print(f"[INFO] 特征路径: {full_fold_path}")

                # 检查特征文件大小
                feature_size = os.path.getsize(fold_data_file) / (1024**2)  # MB
                print(f"[INFO] 特征文件大小: {feature_size:.1f} MB")

            self.data_path = full_fold_path

            # 关闭DenseNet预训练，因为已经使用提取的特征
            original_auto_pretrain = getattr(self.arg, 'auto_pretrain_densenet', True)
            self.arg.auto_pretrain_densenet = False

            if self.rank == 0 and original_auto_pretrain:
                print(f"[INFO] 已禁用DenseNet预训练 (使用预提取特征)")
        else:
            if self.rank == 0:
                print(f"[INFO] 未找到fold {self.arg.fold}的特征数据: {full_fold_path}")
                print(f"[INFO] 使用原始数据: {original_data_name}")

            self.data_path = os.path.join(self.arg.data_dir, original_data_name)

        # 继续原有逻辑
        self.graph_name = get_graph_name(**self.arg.graph_args)

        netw = 'C{}k{}G{}'.format(
            '.'.join([str(s) for s in self.arg.model_args['hidden1']]),
            '.'.join([str(s) for s in self.arg.model_args['kernels']]),
            '.'.join([str(s) for s in self.arg.model_args['hidden2']]),
        )
        losses = 'lr{:g}m{:g}_{:d}.{:d}{}_C{:g}O{:g}CI{:g}S0'.format(
            self.arg.base_lr, self.arg.base_lr_mask, self.arg.pre_epoch, 
            self.arg.num_epoch, self.arg.inc_mode, self.arg.LC, self.arg.LO, self.arg.LCI
        )

        self.arg.exp_name = '__'.join([losses, self.arg.exp_name])
        self.model_name = 't{}__{}'.format(time.strftime('%Y%m%d%H%M%S'), self.arg.exp_name)

        self.work_dir = os.path.join(
            self.arg.save_dir, 
            f'fold{self.arg.fold}'
        )
        self.output_device = self.rank

        # [修正] 这部分是文件IO操作，只有主进程(rank=0)执行
        if self.rank == 0:
            os.makedirs(self.work_dir, exist_ok=True)
            os.makedirs(os.path.join(self.work_dir, 'epoch'), exist_ok=True)
            os.makedirs(os.path.join(self.work_dir, 'best_models'), exist_ok=True)

            self.print_log(',\t'.join([self.graph_name, netw, self.arg.exp_name]))

            arg_dict = vars(self.arg)
            with open(os.path.join(self.work_dir, 'config.yaml'), 'w') as f:
                yaml.dump(arg_dict, f)

            # [新增] 记录数据路径信息
            data_info = {
                'original_data_name': original_data_name,
                'actual_data_path': self.data_path,
                'using_extracted_features': os.path.exists(fold_data_file),
                'fold': self.arg.fold,
                'split_seed': self.arg.split_seed
            }

            with open(os.path.join(self.work_dir, 'data_info.yaml'), 'w') as f:
                yaml.dump(data_info, f)

            if os.path.exists(fold_data_file):
                print(f"[INFO] 本次训练将使用预提取特征，内存占用大幅减少")
            else:
                print(f"[INFO] 本次训练将使用原始数据，请确保有足够内存")
            
    # 将此函数添加到您的 Processor 类中
    def load_precomputed_graph(self):
        """
        加载预先计算好的一致性边/图邻接矩阵。
        这是一个简洁、高效的最终方案。
        """
        # 1. 构建预计算文件的完整路径
        #    文件名需要与您在Jupyter中生成的文件名一致。
        graph_file_name = 'group_correlation_edges.npy'
        print(f"正在加载预计算的图: ")
        graph_path = os.path.join(self.data_path, graph_file_name)
        
        # 2. 检查文件是否存在，如果不存在则报错
        if not os.path.exists(graph_path):
            error_msg = (
                f"错误: 找不到预计算的图文件 '{graph_path}'。\n"
                f"请确保您已经成功运行了离线的图构建脚本，"
                f"并将生成的 '{graph_file_name}' 文件放置在正确的数据目录中: '{self.data_path}'"
            )
            self.print_log(error_msg, log_type="error")
            raise FileNotFoundError(error_msg)

        # 3. 加载文件并直接赋值给类属性 self.consistent_edges_mask
        #    这个属性将在后续的 load_data 方法中被 Feeder 使用
        self.consistent_edges_mask = np.load(graph_path)
        # 4. 打印日志以确认加载成功
        num_edges = np.sum(self.consistent_edges_mask)
        self.print_log(
            f"预计算图加载成功。 形状: {self.consistent_edges_mask.shape}, "
            f"包含 {int(num_edges)} 条边。",
            log_type="info"
        )


    def load_model(self):
        """加载损失函数和模型"""
        self.loss = nn.CrossEntropyLoss(reduction="mean").to(self.rank)
        self.losses = nn.CrossEntropyLoss(reduction="none").to(self.rank)
        
        base_mask_model = import_class(self.arg.model).CausalMask(
            patch_num=self.data_loader['train'].dataset.P, 
            channel=self.arg.model_args['hidden1'][-1],
            consistent_edges=self.consistent_edges_mask, tau=self.arg.gumble_tau
        ).to(self.rank)

        # [修改] 获取预训练路径 - 优先使用当前折的预训练模型
        pretrained_path = self._get_pretrained_path()
        freeze_extractor = getattr(self.arg, 'freeze_extractor', True)
        if 'freeze_extractor' in self.arg.model_args:
            freeze_extractor = self.arg.model_args['freeze_extractor']

        model_kwargs = {
            'num_class': self.num_class, **self.arg.model_args,
            'pretrained_path': pretrained_path, 'freeze_extractor': freeze_extractor
        }
        
        base_causal_model = import_class(self.arg.model).CausalNet(**model_kwargs).to(self.rank)
        
        self.mask = DDP(base_mask_model, device_ids=[self.rank], find_unused_parameters=True)
        self.model = DDP(base_causal_model, device_ids=[self.rank], find_unused_parameters=True)
    
    def _get_pretrained_path(self):
        """获取当前折对应的预训练DenseNet路径"""
        if hasattr(self.arg, 'auto_pretrain_densenet') and self.arg.auto_pretrain_densenet:
            # 自动预训练模式 - 查找或触发预训练
            densenet_save_dir = getattr(self.arg, 'densenet_save_dir', 'pretrained_densenet')
            
            # 构建当前设置对应的保存目录
            specific_save_dir = f"{densenet_save_dir}_split_seed{self.arg.split_seed}"
            
            # 检查是否已有预训练模型
            model_paths_file = os.path.join(specific_save_dir, f'model_paths_seed{self.arg.split_seed}.pkl')
            fold_model_path = os.path.join(specific_save_dir, f'fold_{self.arg.fold}_feature_extractor_288patch.pth')
            
            if os.path.exists(fold_model_path):
                self.print_log(f"找到预训练DenseNet: {fold_model_path}")
                return fold_model_path
            elif os.path.exists(model_paths_file):
                # 从映射文件加载
                with open(model_paths_file, 'rb') as f:
                    model_paths = pickle.load(f)
                if self.arg.fold in model_paths:
                    path = model_paths[self.arg.fold]
                    self.print_log(f"从映射文件找到预训练DenseNet: {path}")
                    return path
            
            # 如果没有找到，则需要先运行预训练
            if self.rank == 0:
                self.print_log("未找到预训练DenseNet模型！", log_type="error")
                self.print_log("请先运行 densenet_pretrainer_standalone.py 进行预训练", log_type="error")
                self.print_log(f"或确保以下路径存在: {fold_model_path}", log_type="error")
            raise FileNotFoundError(f"预训练DenseNet模型不存在: {fold_model_path}")
        
        else:
            # 手动指定模式
            pretrained_path = getattr(self.arg, 'pretrained_path', None)
            if pretrained_path is None and 'pretrained_path' in self.arg.model_args:
                pretrained_path = self.arg.model_args['pretrained_path']
            return pretrained_path
        
        

    def load_optimizer(self):
        """初始化优化器和学习率调度器"""
        if self.arg.optimizer == 'SGD':
            self.optimizer_mask = optim.SGD(
                self.mask.parameters(), lr=self.arg.base_lr_mask, 
                weight_decay=self.arg.weight_decay, momentum=0.9, nesterov=self.arg.nesterov
            )
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.arg.base_lr, 
                weight_decay=self.arg.weight_decay, momentum=0.9, nesterov=self.arg.nesterov
            )
        elif self.arg.optimizer == 'Adam':
            self.optimizer_mask = optim.Adam(
                self.mask.parameters(), lr=self.arg.base_lr_mask, weight_decay=self.arg.weight_decay
            )
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器类型: {self.arg.optimizer}")

        # ReduceLROnPlateau的verbose只在主进程上显示
        verbose_flag = self.rank == 0
        if self.arg.scheduler == 'auto':
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer, verbose=verbose_flag, patience=self.arg.stepsize, factor=self.arg.gamma
            )
            self.lr_scheduler_mask = ReduceLROnPlateau(
                self.optimizer_mask, verbose=verbose_flag, patience=self.arg.stepsize, factor=self.arg.gamma
            )
        elif self.arg.scheduler == 'step':
            self.lr_scheduler = StepLR(self.optimizer, step_size=self.arg.stepsize, gamma=self.arg.gamma)
            self.lr_scheduler_mask = StepLR(self.optimizer_mask, step_size=self.arg.stepsize, gamma=self.arg.gamma)
        else:
            raise ValueError(f"不支持的学习率调度器类型: {self.arg.scheduler}")

    def start(self):
        """开始训练流程，包括预训练和主训练阶段"""
        if self.rank == 0: self.epoch_results = {}
        
        # ====================== 预训练阶段 ======================
        for epoch in range(self.arg.start_epoch, self.arg.pre_epoch):
            self.data_loader['train'].sampler.set_epoch(epoch)
            if self.rank == 0: self.epoch_results[epoch] = {'train': {}, 'val': {}, 'test': {}}
            self.train_emb(epoch)
            if self.rank == 0:
                with torch.no_grad():
                    self.eval_emb(epoch, save_score=self.arg.save_score, loader_name=['val'])
                    self.eval_emb(epoch, save_score=self.arg.save_score, loader_name=['test'])
                self.print_epoch_summary(epoch, is_pre_train=True)
        
        dist.barrier()
        
        if self.rank == 0:
            # [修改] 记录本阶段的最佳轮次
            self.best_epoch_pretrain = self.best_epoch
            self.print_log("预训练阶段结束，同步模型状态...", log_type="info")
            self.save_best_model(stage='pretrain')
            if self.best_model_state: self.model.module.load_state_dict(self.best_model_state)
            
            with torch.no_grad():
                test_acc = 0.0
                if self.best_epoch != -1:
                    # 使用 self.best_epoch_pretrain 来获取正确的测试结果
                    test_acc = self.epoch_results[self.best_epoch_pretrain]['test']['acc_official']
                self.print_log(f"预训练阶段完成: 最佳模型轮次 {self.best_epoch_pretrain+1}, 验证准确率 {self.best_val_acc:.4f}, 测试准确率 {test_acc:.4f}", log_type="result")
            torch.save(self.model.module.state_dict(), os.path.join(self.work_dir, 'tmp_pretrain_model.pt'))

        dist.barrier()
        self.model.module.load_state_dict(torch.load(os.path.join(self.work_dir, 'tmp_pretrain_model.pt'), map_location=f'cuda:{self.rank}'))
        
        # [修改] 重置计数器，为下一阶段做准备
        if self.rank == 0: self.best_val_acc = 0; self.best_epoch = -1
        
        # ====================== 第一阶段训练 ======================
        for epoch in range(self.arg.pre_epoch, self.arg.stage_transition_epoch):
            self.data_loader['train'].sampler.set_epoch(epoch)
            if self.rank == 0: self.epoch_results[epoch] = {'train': {}, 'val': {}, 'test': {}}
            self.train(epoch)
            if self.rank == 0:
                with torch.no_grad():
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['val'])
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
                self.print_epoch_summary(epoch, is_pre_train=False)

        dist.barrier()
        
        if self.rank == 0:
            # [修改] 记录本阶段的最佳轮次
            self.best_epoch_stage1 = self.best_epoch
            self.print_log("第一阶段结束，同步模型状态...", log_type="info")
            self.save_best_model(stage='stage1')
            if self.best_model_state: self.model.module.load_state_dict(self.best_model_state)
            if self.best_mask_state: self.mask.module.load_state_dict(self.best_mask_state)

            with torch.no_grad():
                test_acc = 0.0
                if self.best_epoch != -1:
                    test_acc = self.epoch_results[self.best_epoch_stage1]['test']['acc_official']
                self.print_log(f"第一阶段完成: 最佳模型轮次 {self.best_epoch_stage1+1}, 验证准确率 {self.best_val_acc:.4f}, 测试准确率 {test_acc:.4f}", log_type="result")
            torch.save(self.model.module.state_dict(), os.path.join(self.work_dir, 'tmp_stage1_model.pt'))
            torch.save(self.mask.module.state_dict(), os.path.join(self.work_dir, 'tmp_stage1_mask.pt'))

        dist.barrier()
        self.model.module.load_state_dict(torch.load(os.path.join(self.work_dir, 'tmp_stage1_model.pt'), map_location=f'cuda:{self.rank}'))
        self.mask.module.load_state_dict(torch.load(os.path.join(self.work_dir, 'tmp_stage1_mask.pt'), map_location=f'cuda:{self.rank}'))

        # [修改] 重置计数器，为下一阶段做准备
        if self.rank == 0: self.best_val_acc = 0; self.best_epoch = -1
        
        # ====================== 第二阶段训练 ======================
        for epoch in range(self.arg.stage_transition_epoch, self.arg.num_epoch):
            self.data_loader['train'].sampler.set_epoch(epoch)
            if self.rank == 0: self.epoch_results[epoch] = {'train': {}, 'val': {}, 'test': {}}
            self.train(epoch)
            if self.rank == 0:
                with torch.no_grad():
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['val'])
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
                self.print_epoch_summary(epoch, is_pre_train=False)
        
        dist.barrier()

        # ====================== 最终评估和清理 ======================
        if self.rank == 0:
            # [修改] 记录本阶段的最佳轮次
            self.best_epoch_stage2 = self.best_epoch
            self.print_log("第二阶段结束，进行最终评估...", log_type="info")
            self.save_best_model(stage='stage2')
            if self.best_model_state: self.model.module.load_state_dict(self.best_model_state)
            if self.best_mask_state: self.mask.module.load_state_dict(self.best_mask_state)

            with torch.no_grad():
                test_acc = 0.0
                if self.best_epoch != -1:
                    test_acc = self.epoch_results[self.best_epoch_stage2]['test']['acc_official']
                self.best_acc = test_acc
            
            # [修改] 在最终评估前，调用最终总结报告
            self._print_final_summary()
            
            # 结果处理和混淆矩阵
            try:
                ss = SettleResults(self.data_loader['test'].dataset.out_dir, self.work_dir, self.arg.exp_name)
                best_score_file = os.path.join(self.work_dir, 'best_models', 'best_stage2_test_score.pkl')
                if os.path.exists(best_score_file):
                    with open(best_score_file, 'rb') as f: test_scores = pickle.load(f)
                    with open(os.path.join(self.data_loader['test'].dataset.out_dir, 'label.pkl'), 'rb') as f: label_all, sub_name_all = pickle.load(f)
                    label_dict = dict(zip(sub_name_all, label_all))
                    score_df = pd.DataFrame(test_scores)
                    label_df = pd.DataFrame(label_dict, index=['true_label'])
                    scores = pd.concat([score_df, label_df], axis=0).dropna(axis=1)
                    from tools.utils import get_CM, plot_confusion_matrix
                    cm = get_CM(scores.iloc[-1, :], scores.iloc[:-1, :])
                    classes = ['0', 'n0'] if self.data_loader['test'].dataset.num_class == 2 else [str(kk) for kk in range(self.data_loader['test'].dataset.num_class)]
                    plot_confusion_matrix(
                        self.data_loader['test'].dataset.data_name + '\n' + self.arg.exp_name,
                        os.path.join(self.work_dir, 'CM.png'), cm=cm, classes=classes
                    )
            except Exception as e:
                self.print_log(f"警告: 最终结果处理失败: {str(e)}", log_type="info")

            # 清理临时文件
            if os.path.exists(os.path.join(self.work_dir, 'tmp_pretrain_model.pt')): os.remove(os.path.join(self.work_dir, 'tmp_pretrain_model.pt'))
            if os.path.exists(os.path.join(self.work_dir, 'tmp_stage1_model.pt')): os.remove(os.path.join(self.work_dir, 'tmp_stage1_model.pt'))
            if os.path.exists(os.path.join(self.work_dir, 'tmp_stage1_mask.pt')): os.remove(os.path.join(self.work_dir, 'tmp_stage1_mask.pt'))
            
            self.print_log(f'训练完成: {self.work_dir}', log_type="info")

    def print_epoch_summary(self, epoch, is_pre_train=False):
        """
        [最终完美版]
        打印内容详尽、格式优美的轮次总结。
        - 清晰区分 预训练/第一阶段/第二阶段。
        - 详细列出 验证/测试集 的官方评估指标。
        - 分别展示 掩码(Mask)和GNN 的训练详情，包含所有子损失和对应的准确率。
        - 汇总展示 学习率、正则化强度 和 掩码统计等关键参数。
        """
        # 确保只有主进程(rank 0)执行打印
        if self.rank != 0: 
            return

        is_first_stage = epoch < self.arg.stage_transition_epoch
        stage_name = "预训练" if is_pre_train else ("第一阶段" if is_first_stage else "第二阶段")
        results = self.epoch_results[epoch]

        # 打印总标题
        self.print_log("="*120, print_time=False)
        self.print_log(f"轮次 {epoch + 1}/{self.arg.num_epoch} ({stage_name}) 总结", log_type="summary")
        self.print_log("-"*120, print_time=False)

        # 场景一：预训练阶段的打印
        if is_pre_train:
            header = f"{'指标':<15} | {'训练':<25} | {'验证':<25} | {'测试':<25}"
            self.print_log(header, print_time=False)
            self.print_log("-"*120, print_time=False)

            # 准备数据
            train_res = results.get('train', {})
            val_res = results.get('val', {})
            test_res = results.get('test', {})
            
            # 官方准确率
            val_acc_str = f"\033[1m\033[33m{val_res.get('acc_official', 0.0):.4f}※\033[0m" if val_res.get('acc_official', 0.0) == self.best_val_acc and self.best_val_acc > 0 else f"{val_res.get('acc_official', 0.0):.4f}"
            acc_row = f"{'官方准确率':<14} | {train_res.get('acc_invariance', 0.0):<25.4f} | {val_acc_str:<35} | {test_res.get('acc_official', 0.0):<25.4f}"
            self.print_log(acc_row, print_time=False)
            
            # AUC
            auc_row = f"{'AUC':<14} | {train_res.get('auc', 0.0):<25.4f} | {val_res.get('auc', 0.0):<25.4f} | {test_res.get('auc', 0.0):<25.4f}"
            self.print_log(auc_row, print_time=False)
            
            # 总损失
            loss_row = f"{'总损失':<14} | \033[31m{train_res.get('loss_all', 0.0):.4f}\033[0m{'':<19} | \033[31m{val_res.get('loss_all', 0.0):.4f}\033[0m{'':<19} | \033[31m{test_res.get('loss_all', 0.0):.4f}\033[0m{'':<19}"
            self.print_log(loss_row, print_time=False)

            # 学习率
            lr_gnn = self.optimizer.param_groups[0]['lr']
            lr_row = f"{'学习率':<14} | GNN: {lr_gnn:<20.6f} | {'-':<25} | {'-':<25}"
            self.print_log(lr_row, print_time=False)
            
        # 场景二：主训练阶段的打印
        else:
            # 1. 官方评估结果
            self.print_log("【官方评估结果】", print_time=False)
            val_res = results.get('val', {})
            test_res = results.get('test', {})
            val_acc_str = f"\033[1m\033[33m{val_res.get('acc_official', 0.0):.4f}※\033[0m" if val_res.get('acc_official', 0.0) == self.best_val_acc and self.best_val_acc > 0 else f"{val_res.get('acc_official', 0.0):.4f}"
            official_row_val = f"  验证集: 准确率={val_acc_str} | AUC={val_res.get('auc', 0.0):.4f}"
            official_row_test = f"  测试集: 准确率={test_res.get('acc_official', 0.0):.4f} | AUC={test_res.get('auc', 0.0):.4f}"
            self.print_log(official_row_val, print_time=False)
            self.print_log(official_row_test, print_time=False)
            self.print_log("-"*120, print_time=False)

            # 准备训练数据
            train_res = results.get('train', {})
            mask_res = train_res.get('mask', {})
            gnn_res = train_res.get('gnn', {})

            # 2. 掩码训练详情
            self.print_log("【掩码训练详情】", print_time=False)
            self.print_log(f"  总损失: \033[31m{mask_res.get('loss_all', 0):.4f}\033[0m", print_time=False)
            
            # 根据不同阶段打印不同子损失
            if is_first_stage:
                self.print_log(f"    ├─ 不变性损失 (Lci): {mask_res.get('loss_invariance', 0):.4f} (Acc: \033[32m{mask_res.get('acc_invariance', 0):.2%}\033[0m)")
                self.print_log(f"    ├─ 变异性损失 (Lcv): {mask_res.get('loss_variability', 0):.4f} (Acc: \033[32m{mask_res.get('acc_variability', 0):.2%}\033[0m)")
                self.print_log(f"    ├─ 引导损失 (Lg):    {mask_res.get('loss_guide', 0):.4f}")
                self.print_log(f"    └─ 稀疏性损失:        {mask_res.get('loss_sparsity_reg', 0):.4f}")
            else: # 第二阶段
                self.print_log(f"    ├─ 不变性损失 (Lci): {mask_res.get('loss_invariance', 0):.4f} (Acc: \033[32m{mask_res.get('acc_invariance', 0):.2%}\033[0m)")
                self.print_log(f"    ├─ 因果损失 (Lc):    {mask_res.get('loss_causal', 0):.4f} (Acc: \033[32m{mask_res.get('acc_causal', 0):.2%}\033[0m)")
                self.print_log(f"    ├─ 反事实损失 (Lo):  {mask_res.get('loss_counterfactual', 0):.4f} (Acc: \033[32m{mask_res.get('acc_counterfactual', 0):.2%}\033[0m)")
                self.print_log(f"    └─ 稀疏性损失:        {mask_res.get('loss_sparsity_reg', 0):.4f}")

            # 3. GNN训练详情
            self.print_log("【GNN训练详情】", print_time=False)
            self.print_log(f"  总损失: \033[31m{gnn_res.get('loss_all', 0):.4f}\033[0m", print_time=False)
            self.print_log(f"    ├─ L1正则化损失:      {gnn_res.get('loss_l1_reg', 0):.4f}")
            if is_first_stage:
                self.print_log(f"    └─ 不变性损失 (Lci): {gnn_res.get('loss_invariance', 0):.4f} (Acc: \033[32m{gnn_res.get('acc_invariance', 0):.2%}\033[0m)")
            else: # 第二阶段
                self.print_log(f"    ├─ 不变性损失 (Lci): {gnn_res.get('loss_invariance', 0):.4f} (Acc: \033[32m{gnn_res.get('acc_invariance', 0):.2%}\033[0m)")
                self.print_log(f"    └─ 因果损失 (Lc):    {gnn_res.get('loss_causal', 0):.4f} (Acc: \033[32m{gnn_res.get('acc_causal', 0):.2%}\033[0m)")

            # 4. 训练参数与状态
            self.print_log("【训练参数与状态】", print_time=False)
            lr_gnn = self.optimizer.param_groups[0]['lr']
            lr_mask = self.optimizer_mask.param_groups[0]['lr']
            lambda_reg = 0.05 * (1 + epoch / self.arg.num_epoch)
            self.print_log(f"  学习率: GNN={lr_gnn:.6f} | Mask={lr_mask:.6f} || 稀疏正则强度 λ_reg={lambda_reg:.4f}", print_time=False)

            if hasattr(self, 'current_mask_sums'):
                node_sum = self.current_mask_sums.get('node', 0)
                edge_sum = self.current_mask_sums.get('edge', 0)
                total_nodes = self.mask.module.P
                total_edges = torch.sum(self.mask.module.learnable_mask).item()
                node_percentage = node_sum / total_nodes * 100 if total_nodes > 0 else 0
                edge_percentage = edge_sum / total_edges * 100 if total_edges > 0 else 0
                mask_row = f"  因果掩码和: 节点={int(node_sum)}/{total_nodes} ({node_percentage:.1f}%) | 边={int(edge_sum)}/{int(total_edges)} ({edge_percentage:.1f}%)"
                self.print_log(mask_row, print_time=False)

        self.print_log("="*120, print_time=False)

    
    def train_emb(self, epoch, save_model=False):
        self.model.train()
        self.print_log(f'预训练轮次: {epoch + 1}')
        loader = self.data_loader['train']
        
        if self.rank == 0:
            self.train_writer.add_scalar(self.model_name + '/epoch', epoch, self.global_step)
            self.record_time()

        loss_value, all_accuracies, l1_losses = [], [], []

        for batch_idx, (data, edges, label, index) in enumerate(loader):
            self.global_step += 1
            x_node, edge, label = self.converse2tensor(data, edges, label)
            #x_new = self.model.emb(x_node)
            x_new = x_node
            yw = self.model.module.prediction_whole(x_new, edge, is_large_graph=True)
            lossW = self.losses(yw, label)
            l1_loss = self.compute_l1_regularization(self.model.module)
            loss_all = lossW.mean() + l1_loss
            
            self.optimizer.zero_grad()
            loss_all.backward()
            self.optimizer.step()

            if self.rank == 0:
                loss_value.append(lossW.mean().item())
                l1_losses.append(l1_loss.item())
                _, predict_label = torch.max(yw.data, 1)
                acc = torch.mean((predict_label == label.data).float())
                all_accuracies.append(acc.item())
                self.train_writer.add_scalar(self.model_name + '/acc', acc.item(), self.global_step)
                self.train_writer.add_scalar(self.model_name + '/loss_w', lossW.mean().item(), self.global_step)
                self.train_writer.add_scalar(self.model_name + '/l1_loss', l1_loss.item(), self.global_step)
                self.train_writer.add_scalar(self.model_name + '/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

        if self.rank == 0:
            self.print_log(f'\t预训练平均损失: {np.mean(loss_value):.4f}, L1损失: {np.mean(l1_losses):.4f}')
            train_results = {'loss_all': np.mean(loss_value), 'loss_l1': np.mean(l1_losses), 'acc_invariance': np.mean(all_accuracies)}
            self.epoch_results[epoch]['train'] = train_results
            if save_model: torch.save(self.model.module.state_dict(), os.path.join(self.work_dir, 'epoch', f'epoch-{epoch + 1}.pt'))

    def eval_emb(self, epoch, save_score=False, loader_name=['val']):
        """改进的预训练评估函数，包含详细指标"""
        if self.rank != 0: return
        self.model.eval()
        self.print_log(f'预训练评估轮次: {epoch + 1}')

        for ln in loader_name:
            loss_value, score_dict, all_labels, all_probs = [], {}, [], []

            for batch_idx, (data, edges, label, index) in enumerate(self.data_loader[ln]):
                x_node, edge, label = self.converse2tensor(data, edges, label)

                yw = self.model.module.prediction_whole(x_node, edge, is_large_graph=False)
                lossW = self.losses(yw, label)

                probs = F.softmax(yw, dim=1)[:, 1].detach().cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(label.cpu().numpy())
                loss_value.extend(lossW.data.cpu().numpy())

                sub_list = self.data_loader[ln].dataset.sample_name[index] if len(index) > 1 else [self.data_loader[ln].dataset.sample_name[index]]
                score_dict.update(dict(zip(sub_list, yw.data.cpu().numpy())))

            loss = np.mean(loss_value)
            # 使用原有的准确率计算方法，保持与原代码一致
            accuracy = self.data_loader[ln].dataset.top_k(np.array(list(score_dict.values())), 1)

            # 计算额外的详细指标
            from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
            try:
                auc_score = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0

                # 获取预测标签（基于概率阈值0.5）
                all_predictions = (np.array(all_probs) > 0.5).astype(int)

                # 计算分类指标
                average_method = 'binary' if len(np.unique(all_labels)) == 2 else 'macro'
                precision = precision_score(all_labels, all_predictions, average=average_method, zero_division=0)
                recall = recall_score(all_labels, all_predictions, average=average_method, zero_division=0)
                f1 = f1_score(all_labels, all_predictions, average=average_method, zero_division=0)

                # 计算特异性（仅二分类）
                if len(np.unique(all_labels)) == 2:
                    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                else:
                    specificity = 0.0
                    sensitivity = recall

            except Exception as e:
                auc_score = precision = recall = f1 = specificity = sensitivity = 0.0

            self.print_log(f'\t{ln}数据集 - ACC: {accuracy:.4f}, AUC: {auc_score:.4f}, '
                          f'P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}')

            # 更新调度器和最佳模型
            if ln == 'val':
                if self.arg.scheduler == 'auto': self.lr_scheduler.step(loss)
                else: self.lr_scheduler.step()
                if accuracy > self.best_val_acc:
                    self.best_val_acc = accuracy
                    self.best_model_state = self.model.module.state_dict().copy()
                    self.best_epoch = epoch
                    self.print_log(f"发现新的最佳预训练模型 (验证集准确率: {accuracy:.4f})")

            # 保存完整结果
            eval_results = {
                'loss_all': loss,
                'acc_official': accuracy,
                'auc': auc_score,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity,
                'sensitivity': sensitivity
            }
            self.epoch_results[epoch][ln] = eval_results

            if save_score:
                with open(os.path.join(self.work_dir, 'epoch', f'epoch{epoch + 1}_{ln}_score.pkl'), 'wb') as f:
                    pickle.dump(score_dict, f)


    def train(self, epoch, save_model=False):
        # 设置模型为训练模式
        self.mask.train()
        self.model.train()

        # [最终方案] 分阶段设置不同的稀疏性惩罚强度
        is_first_stage = epoch < self.arg.stage_transition_epoch
        if is_first_stage:
            # 第一阶段：使用一个较小且平滑增长的惩罚，鼓励探索
            lambda_reg = 0.05 * (1 + epoch / self.arg.num_epoch)
        else:

            lambda_reg = 1.0 

        # 准备训练环境
        loader = self.data_loader['train']
        if self.rank == 0:
            self.train_writer.add_scalar(self.model_name + '/epoch', epoch, self.global_step)
            self.train_writer.add_scalar(self.model_name + '/lambda_reg', lambda_reg, self.global_step)
            self.record_time()

        # 初始化损失和准确率记录器 (仅在主进程上有用)
        losses_mask = {'all': [], 'causal': [], 'counterfactual': [], 'invariance': [], 'variability': [], 'guide': [], 'sparsity_reg': []}
        losses_gnn = {'all': [], 'causal': [], 'invariance': [], 'l1_reg': []}
        accuracies_mask, accuracies_gnn = {}, {}

        # 遍历批次数据
        for batch_idx, (data, edges, label, index) in enumerate(loader):
            self.global_step += 1
            x_node, edge, label = self.converse2tensor(data, edges, label)

            # ==================== 掩码训练阶段 ====================
            for param in self.model.parameters(): param.requires_grad = False
            masks, sparsity = self.mask.module(train=True) # 使用 .module 访问原始模型
            #x_new = self.model.emb(x_node)
            x_new = x_node
            if is_first_stage:
                result_mask = self._compute_first_stage_mask_loss(x_new, edge, masks, label, epoch, lambda_reg)
                if self.rank == 0:
                    accuracies_mask.setdefault('invariance', []).append(self._calculate_accuracy(result_mask['predictions']['invariance'], label))
                    accuracies_mask.setdefault('variability', []).append(self._calculate_accuracy(result_mask['predictions']['variability'], label))
            else:
                result_mask = self._compute_second_stage_mask_loss(x_new, edge, masks, label, lambda_reg)
                if self.rank == 0:
                    accuracies_mask.setdefault('invariance', []).append(self._calculate_accuracy(result_mask['predictions']['invariance'], label))
                    accuracies_mask.setdefault('causal', []).append(self._calculate_accuracy(result_mask['predictions']['causal'], label))
                    accuracies_mask.setdefault('counterfactual', []).append(self._calculate_accuracy(result_mask['predictions']['counterfactual'], label))

            self.optimizer_mask.zero_grad()
            result_mask['loss']['all'].backward()
            self.optimizer_mask.step()
            if self.rank == 0: self._record_mask_losses(losses_mask, result_mask)

            # ==================== GNN训练阶段 ====================
            for param in self.model.parameters(): param.requires_grad = True
            # 获取更新后的掩码(不计算梯度)
            masks, _ = self.mask.module(train=False)
            masks = [mm.detach() for mm in masks]
            #x_new = self.model.emb(x_node)
            x_new = x_node
            if is_first_stage:
                result_gnn = self._compute_first_stage_gnn_loss(x_new, edge, masks, label)
                if self.rank == 0: accuracies_gnn.setdefault('invariance', []).append(self._calculate_accuracy(result_gnn['predictions']['invariance'], label))
            else:
                result_gnn = self._compute_second_stage_gnn_loss(x_new, edge, masks, label)
                if self.rank == 0:
                    accuracies_gnn.setdefault('invariance', []).append(self._calculate_accuracy(result_gnn['predictions']['invariance'], label))
                    accuracies_gnn.setdefault('causal', []).append(self._calculate_accuracy(result_gnn['predictions']['causal'], label))

            self.optimizer.zero_grad()
            result_gnn['loss']['all'].backward()
            self.optimizer.step()

            # [恢复] 在主进程上记录GNN损失和当前的掩码和
            if self.rank == 0: 
                self._record_gnn_losses(losses_gnn, result_gnn)
                # 这句是关键，为print_epoch_summary准备数据
                self.current_mask_sums = {'node': masks[0].sum().item(), 'edge': masks[1].sum().item()}

        # ==================== Epoch结束后的处理 ====================
        if self.rank == 0:
            # 将一个epoch内所有batch的平均损失和准确率存入结果字典
            train_results = {
                'mask': {k: np.mean(v) for k, v in losses_mask.items() if v},
                'gnn': {k: np.mean(v) for k, v in losses_gnn.items() if v}
            }
            for k, v in accuracies_mask.items(): train_results['mask'][f'acc_{k}'] = np.mean(v)
            for k, v in accuracies_gnn.items(): train_results['gnn'][f'acc_{k}'] = np.mean(v)
            self.epoch_results[epoch]['train'] = train_results

            # 如果需要，保存模型
            if save_model:
                state_dict = self.mask.module.state_dict()
                save_path = os.path.join(self.work_dir, 'epoch', f'save{epoch + 1}_mask.pt')
                torch.save(state_dict, save_path)

    # 替换您 Processor 类中的整个 eval 函数
    def eval(self, epoch, save_score=False, loader_name=['val']):
        """改进的主训练评估函数，包含详细指标"""
        if self.rank != 0: return
        self.mask.eval()
        self.model.eval()

        stage_name = "stage1" if epoch < self.arg.stage_transition_epoch else "stage2"

        for ln in loader_name:
            loss_values, all_labels, all_probs = [], [], []
            invariance_scores = {}

            for batch_idx, (data, edges, label, index) in enumerate(self.data_loader[ln]):
                x_node, edge, label = self.converse2tensor(data, edges, label)

                masks, probs, _ = self.mask.module(train=False, return_probs=True)
                current_masks = [m.clone() for m in masks]
                current_probs = [p.clone() for p in probs]

                yci = self.model.module.prediction_causal_invariance(x_node, edge, masks, is_large_graph=False)

                loss_ci = self.losses(yci, label)
                loss_values.append(loss_ci.mean().item())

                probs_softmax = F.softmax(yci, dim=1)[:, 1].detach().cpu().numpy()
                all_probs.extend(probs_softmax)
                all_labels.extend(label.cpu().numpy())

                sub_list = self.data_loader[ln].dataset.sample_name[index] if len(index) > 1 else [self.data_loader[ln].dataset.sample_name[index]]
                invariance_scores.update(dict(zip(sub_list, yci.data.cpu().numpy())))

            avg_loss = np.mean(loss_values) if loss_values else 0
            # 使用原有的准确率计算方法，保持与原代码一致
            official_accuracy = self.data_loader[ln].dataset.top_k(np.array(list(invariance_scores.values())), 1)

            # 计算额外的详细指标
            from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
            try:
                auc_score = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0

                # 获取预测标签（基于概率阈值0.5）
                all_predictions = (np.array(all_probs) > 0.5).astype(int)

                # 计算分类指标
                average_method = 'binary' if len(np.unique(all_labels)) == 2 else 'macro'
                precision = precision_score(all_labels, all_predictions, average=average_method, zero_division=0)
                recall = recall_score(all_labels, all_predictions, average=average_method, zero_division=0)
                f1 = f1_score(all_labels, all_predictions, average=average_method, zero_division=0)

                # 计算特异性（仅二分类）
                if len(np.unique(all_labels)) == 2:
                    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                else:
                    specificity = 0.0
                    sensitivity = recall

            except Exception as e:
                auc_score = precision = recall = f1 = specificity = sensitivity = 0.0

            # 更新调度器和最佳模型
            if ln == 'val':
                if self.arg.scheduler == 'auto':
                    self.lr_scheduler_mask.step(avg_loss)
                    self.lr_scheduler.step(avg_loss)
                else:
                    self.lr_scheduler_mask.step()
                    self.lr_scheduler.step()

                if official_accuracy > self.best_val_acc:
                    self.best_val_acc = official_accuracy
                    self.best_model_state = self.model.module.state_dict().copy()
                    self.best_mask_state = self.mask.module.state_dict().copy()
                    self.best_epoch = epoch
                    self.print_log(f"发现新的最佳模型 (验证集准确率: {official_accuracy:.4f})")
                    self._save_causal_artifacts(current_masks, current_probs, stage_name)

            # 保存完整结果
            eval_results = {
                'loss_all': avg_loss,
                'acc_official': official_accuracy,
                'auc': auc_score,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity,
                'sensitivity': sensitivity
            }
            self.epoch_results[epoch][ln] = eval_results

            if save_score:
                with open(os.path.join(self.work_dir, 'epoch', f'epoch{epoch + 1}_{ln}_score.pkl'), 'wb') as f:
                    pickle.dump(invariance_scores, f)

                    
                
    def _compute_detailed_metrics(self, all_labels, all_probs, all_predictions):
        """计算详细的分类指标"""
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
        import numpy as np

        metrics = {}

        try:
            # 基本指标
            metrics['accuracy'] = np.mean(all_predictions == all_labels)
            metrics['auc'] = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0

            # 分类指标（支持二分类和多分类）
            average_method = 'binary' if len(np.unique(all_labels)) == 2 else 'macro'
            metrics['precision'] = precision_score(all_labels, all_predictions, average=average_method, zero_division=0)
            metrics['recall'] = recall_score(all_labels, all_predictions, average=average_method, zero_division=0)
            metrics['f1'] = f1_score(all_labels, all_predictions, average=average_method, zero_division=0)

            # 混淆矩阵和特异性（仅二分类）
            if len(np.unique(all_labels)) == 2:
                tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 等同于recall
            else:
                metrics['specificity'] = 0.0
                metrics['sensitivity'] = metrics['recall']

        except Exception as e:
            # 如果计算失败，返回默认值
            metrics = {k: 0.0 for k in ['accuracy', 'auc', 'precision', 'recall', 'f1', 'specificity', 'sensitivity']}

        return metrics

    def compute_l1_regularization(self, model_module):
        l1_reg = 0
        for name, param in model_module.named_parameters():
            if 'mlp_causal.0.weight' in name:
                l1_reg += torch.sum(torch.abs(param))
        return self.lambda_l1 * l1_reg
    
    def _compute_first_stage_mask_loss(self, x_new, edge, masks, label, epoch, lambda_reg=0.05):
        yci = self.model.module.prediction_causal_invariance(x_new, edge, masks, is_large_graph=True)
        loss_ci = self.losses(yci, label)
        ycv = self.model.module.prediction_causal_variability(x_new, edge, masks, is_large_graph=True)
        loss_cv = self.entropy_loss(ycv)
        reg_loss = self.mask.module.compute_sparsity_regularization(lambda_reg=lambda_reg)
        loss_all = self.arg.LCI * loss_ci.mean() + self.arg.LCV * loss_cv.mean()  + reg_loss
        return {'loss': {'all': loss_all, 'invariance': loss_ci.mean(), 'variability': loss_cv.mean(),  'sparsity_reg': reg_loss}, 
                'predictions': {'invariance': yci, 'variability': ycv}}

    def _compute_second_stage_mask_loss(self, x_new, edge, masks, label, lambda_reg=0.1):
        yci = self.model.module.prediction_causal_invariance(x_new, edge, masks, is_large_graph=True)
        loss_ci = self.losses(yci, label)
        yc = self.model.module.prediction_causal(x_new, edge, masks, is_large_graph=True)
        loss_c = self.losses(yc, label)
        yo = self.model.module.prediction_counterfactual(x_new, edge, masks, is_large_graph=True)
        loss_o = self.loss(yo, 1 - label)
        reg_loss = self.mask.module.compute_sparsity_regularization(lambda_reg=lambda_reg)
        loss_all = (self.arg.LC * loss_c.mean() + self.arg.LO * loss_o.mean() + self.arg.LCI * loss_ci.mean() + reg_loss)
        return {'loss': {'all': loss_all, 'causal': self.arg.LC * loss_c.mean(), 'counterfactual': self.arg.LO * loss_o.mean(), 'invariance': self.arg.LCI * loss_ci.mean(), 'sparsity_reg': reg_loss}, 
                'predictions': {'causal': yc, 'counterfactual': yo, 'invariance': yci}}

    def _compute_first_stage_gnn_loss(self, x_new, edge, masks, label):
        yci = self.model.module.prediction_causal_invariance(x_new, edge, masks, is_large_graph=True)
        loss_ci = self.losses(yci, label)
        l1_loss = self.compute_l1_regularization(self.model.module)
        loss_all = loss_ci.mean() + l1_loss
        return {'loss': {'all': loss_all, 'invariance': loss_ci.mean(), 'l1_reg': l1_loss}, 'predictions': {'invariance': yci}}

    def _compute_second_stage_gnn_loss(self, x_new, edge, masks, label):
        yc = self.model.module.prediction_causal(x_new, edge, masks, is_large_graph=True)
        loss_c = self.losses(yc, label)
        yci = self.model.module.prediction_causal_invariance(x_new, edge, masks, is_large_graph=True)
        loss_ci = self.losses(yci, label)
        loss_all = self.arg.LCI * loss_ci.mean() + self.arg.LC * loss_c.mean()
        return {'loss': {'all': loss_all, 'causal': self.arg.LC * loss_c.mean(), 'invariance': loss_ci.mean()}, 
                'predictions': {'causal': yc, 'invariance': yci}}

    def _record_mask_losses(self, losses_container, current_losses):
        for key, value in current_losses['loss'].items():
            if key in losses_container and isinstance(value, torch.Tensor):
                losses_container[key].append(value.item())

    def _record_gnn_losses(self, losses_container, current_losses):
        for key, value in current_losses['loss'].items():
            if key in losses_container and isinstance(value, torch.Tensor):
                losses_container[key].append(value.item())

    def _calculate_accuracy(self, predictions, labels):
        _, predict_label = torch.max(predictions.data, 1)
        acc = torch.mean((predict_label == labels.data).float())
        return acc.item()

    def entropy_loss(self, logits):
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=1)
        return -entropy

    def converse2tensor(self, data, edges, label):
        data = torch.FloatTensor(data.float()).to(self.rank, non_blocking=True)
        label = torch.LongTensor(label.long()).to(self.rank, non_blocking=True)
        all_edge = torch.FloatTensor(edges.float()).to(self.rank, non_blocking=True)
        return data, all_edge, label

    def guide(self, M):
        aa = torch.norm(self.DiffNode.squeeze(1) - M, p=2)
        return aa

    def print_log(self, text_str, print_time=True, log_type=None):
        if self.rank != 0: return
        # 省略颜色代码以简化
        prefix = "" if not log_type else f"[{log_type.upper()}] "
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            text_str = f"[{localtime}] {prefix}{text_str}"
        else:
            text_str = prefix + text_str
        print(text_str)
        if self.arg.print_log:
            with open(os.path.join(self.work_dir, 'log.txt'), 'a') as f:
                print(text_str.encode('utf-8').decode('utf-8'), file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time
        
    def save_best_model(self, stage='pretrain', epoch=None):
        if self.rank != 0: return
        save_dir = os.path.join(self.work_dir, 'best_models')

        if epoch is None:
            model_state = self.best_model_state
            mask_state = self.best_mask_state
        else:
            model_state = self.model.module.state_dict()
            mask_state = self.mask.module.state_dict() if hasattr(self, 'mask') else None

        if model_state: torch.save(model_state, os.path.join(save_dir, f'best_{stage}_model.pt'))
        if mask_state: torch.save(mask_state, os.path.join(save_dir, f'best_{stage}_mask.pt'))

        # 省略了保存logits等逻辑，因为它们依赖于best_masks等变量，这些变量在eval中被设置
        self.print_log(f"已保存{stage}阶段最佳模型(轮次 {self.best_epoch+1})", log_type="info")

    def load_best_model(self, stage='pretrain'):
        if self.rank != 0: return
        save_dir = os.path.join(self.work_dir, 'best_models')
        model_path = os.path.join(save_dir, f'best_{stage}_model.pt')
        mask_path = os.path.join(save_dir, f'best_{stage}_mask.pt')

        if os.path.exists(model_path):
            self.model.module.load_state_dict(torch.load(model_path, map_location=f'cuda:{self.rank}'))
            self.print_log(f"已加载{stage}阶段最佳模型", log_type="info")
        if stage != 'pretrain' and os.path.exists(mask_path):
            self.mask.module.load_state_dict(torch.load(mask_path, map_location=f'cuda:{self.rank}'))
            self.print_log(f"已加载{stage}阶段最佳掩码", log_type="info")
            
    def _save_causal_artifacts(self, masks, probs, stage_name):
        """
        将CausalMask识别出的因果节点/边及其对应的因果概率值保存下来。
        """
        if self.rank != 0:
            return

        node_mask_hard, edge_mask_hard = masks
        node_probs, edge_probs = probs

        # --- 保存二元的（0/1）结果 ---
        causal_node_indices = np.where(node_mask_hard.cpu().numpy() == 1)[0]
        node_path = os.path.join(self.work_dir, 'best_models', f'best_{stage_name}_causal_nodes.csv')
        np.savetxt(node_path, causal_node_indices, fmt='%d', header='causal_node_index', comments='')
        
        edge_path = os.path.join(self.work_dir, 'best_models', f'best_{stage_name}_causal_edges.csv')
        pd.DataFrame(edge_mask_hard.cpu().numpy()).to_csv(edge_path, header=False, index=False)
        self.print_log(f"已将 {stage_name} 阶段最佳二元因果图保存。", log_type="info")
        
        # --- 保存连续的、可排序的因果概率值 ---
        node_scores_df = pd.DataFrame({
            'node_index': np.arange(len(node_probs)),
            'causal_probability': node_probs.cpu().numpy()
        })
        node_scores_df = node_scores_df.sort_values(by='causal_probability', ascending=False)
        node_scores_path = os.path.join(self.work_dir, 'best_models', f'best_{stage_name}_node_scores.csv')
        node_scores_df.to_csv(node_scores_path, index=False, float_format='%.6f')

        edge_probs_np = edge_probs.cpu().numpy()
        source_nodes, target_nodes = np.where(self.mask.module.learnable_mask.cpu().numpy() > 0)
        edge_scores_df = pd.DataFrame({
            'source_node': source_nodes,
            'target_node': target_nodes,
            'causal_probability': edge_probs_np[source_nodes, target_nodes]
        })
        edge_scores_df = edge_scores_df.sort_values(by='causal_probability', ascending=False)
        edge_scores_path = os.path.join(self.work_dir, 'best_models', f'best_{stage_name}_edge_scores.csv')
        edge_scores_df.to_csv(edge_scores_path, index=False, float_format='%.6f')

        self.print_log(f"已将 {stage_name} 阶段的因果概率值保存至CSV文件，可供排序和筛选。", log_type="info")

    def _print_final_summary(self):
        """打印并保存最终性能总结"""
        if self.rank != 0:
            return

        self.print_log("="*140, print_time=True)
        self.print_log("训练流程完成 - 各阶段性能总结", log_type="result")
        self.print_log("="*140, print_time=False)

        # 准备数据结构
        summary_data = []
        metric_names = ['ACC', 'AUC', 'Precision', 'Recall', 'F1', 'Specificity', 'Sensitivity']

        # 打印表头
        header = f"{'阶段':<20} | {'模型类型':<10} | {'轮次':<6} | {'数据集':<6} |"
        for metric in metric_names:
            header += f" {metric:<10} |"
        self.print_log(header, print_time=False)
        self.print_log("-"*140, print_time=False)

        # 定义阶段信息
        stages_info = [
            ('预训练', 'pretrain', self.best_epoch_pretrain),
            ('第一阶段', 'stage1', self.best_epoch_stage1),
            ('第二阶段', 'stage2', self.best_epoch_stage2)
        ]

        for stage_name, stage_key, best_epoch in stages_info:
            if best_epoch == -1:
                continue

            # 最佳模型性能
            for dataset in ['val', 'test']:
                if best_epoch in self.epoch_results and dataset in self.epoch_results[best_epoch]:
                    res = self.epoch_results[best_epoch][dataset]

                    # 构建输出行
                    model_type = "最佳" if dataset == 'val' else "最佳"
                    line = f"{stage_name:<20} | {model_type:<10} | {best_epoch + 1:<6} | {dataset.upper():<6} |"

                    metrics_values = [
                        res.get('acc_official', 0.0),
                        res.get('auc', 0.0),
                        res.get('precision', 0.0),
                        res.get('recall', 0.0),
                        res.get('f1', 0.0),
                        res.get('specificity', 0.0),
                        res.get('sensitivity', 0.0)
                    ]

                    for value in metrics_values:
                        if dataset == 'test' and stage_key == 'stage2':
                            line += f" \033[1m\033[33m{value:.4f}\033[0m{'':<2} |"
                        else:
                            line += f" {value:<10.4f} |"

                    self.print_log(line, print_time=False)

                    # 添加到汇总数据
                    summary_data.append({
                        'Stage': stage_name,
                        'Model_Type': '最佳模型',
                        'Epoch': best_epoch + 1,
                        'Dataset': dataset.upper(),
                        'Accuracy': metrics_values[0],
                        'AUC': metrics_values[1],
                        'Precision': metrics_values[2],
                        'Recall': metrics_values[3],
                        'F1': metrics_values[4],
                        'Specificity': metrics_values[5],
                        'Sensitivity': metrics_values[6]
                    })

            # 最后一轮模型性能
            stage_ranges = {
                'pretrain': (0, self.arg.pre_epoch),
                'stage1': (self.arg.pre_epoch, self.arg.stage_transition_epoch),
                'stage2': (self.arg.stage_transition_epoch, self.arg.num_epoch)
            }

            if stage_key in stage_ranges:
                last_epoch = stage_ranges[stage_key][1] - 1
                # 确保last_epoch存在于结果中，且打印最后一轮结果（无论是否与最佳相同）
                if last_epoch >= 0 and last_epoch in self.epoch_results:
                    for dataset in ['val', 'test']:
                        if dataset in self.epoch_results[last_epoch]:
                            res = self.epoch_results[last_epoch][dataset]

                            # 如果最后一轮与最佳相同，标注为"最后※"，否则为"最后"
                            model_type = "最后※" if last_epoch == best_epoch else "最后"
                            line = f"{'':<20} | {model_type:<10} | {last_epoch + 1:<6} | {dataset.upper():<6} |"

                            metrics_values = [
                                res.get('acc_official', 0.0),
                                res.get('auc', 0.0),
                                res.get('precision', 0.0),
                                res.get('recall', 0.0),
                                res.get('f1', 0.0),
                                res.get('specificity', 0.0),
                                res.get('sensitivity', 0.0)
                            ]

                            for value in metrics_values:
                                # 如果是最后一轮且与最佳相同，用灰色标注
                                if last_epoch == best_epoch:
                                    line += f" \033[90m{value:.4f}\033[0m{'':<6} |"
                                else:
                                    line += f" {value:<10.4f} |"

                            self.print_log(line, print_time=False)

                            # 添加到汇总数据
                            summary_data.append({
                                'Stage': stage_name,
                                'Model_Type': '最后一轮',
                                'Epoch': last_epoch + 1,
                                'Dataset': dataset.upper(),
                                'Accuracy': metrics_values[0],
                                'AUC': metrics_values[1],
                                'Precision': metrics_values[2],
                                'Recall': metrics_values[3],
                                'F1': metrics_values[4],
                                'Specificity': metrics_values[5],
                                'Sensitivity': metrics_values[6]
                            })

            self.print_log("-"*140, print_time=False)

        # 保存为CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_path = os.path.join(self.work_dir, 'performance_summary.csv')
            summary_df.to_csv(csv_path, index=False, float_format='%.6f')
            self.print_log(f"性能总结已保存至: {csv_path}", log_type="info")

        self.print_log("="*140, print_time=False)
        
        
# DDP主工作函数
def main_worker(rank, world_size, arg):
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # 保持我们之前的网络修复
    os.environ['MASTER_PORT'] = '12355' 
    
    # --- 【调试修改】: 添加第一行打印日志 ---
    # 为了区分不同进程的输出，我们加入 rank 信息
    print(f"[Rank {rank}] 准备初始化进程组 (Attempting to init process group)...")
    # ------------------------------------

    # 绝大概率程序会卡在下面这一行
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    
    # --- 【调试修改】: 添加第二行打印日志 ---
    # 如果您能看到这条消息，说明初始化成功了
    print(f"[Rank {rank}] 成功：进程组初始化完成！(Process group initialized successfully!)")
    # ------------------------------------

    torch.cuda.set_device(rank)
    seed_torch(arg.seed + rank)
    processor = Processor(rank, world_size, arg)
    processor.start()
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()

    # DDP启动逻辑
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"检测到 {world_size} 个可用的GPU。")
    else:
        print("未检测到GPU，DDP训练无法进行。"); exit()

    if world_size < 2:
        print("GPU数量少于2，将以单卡模式运行。")
        main_worker(0, 1, arg)
    else:
        mp.spawn(main_worker, args=(world_size, arg), nprocs=world_size, join=True)