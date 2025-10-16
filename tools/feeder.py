from __future__ import print_function
import numpy as np
import os
import pickle
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from .construct_graph_simple import ConstructGraph

class FeederGraph:
    def __init__(self, fold, split_seed, out_dir, mode, graph_arg, debug=False, save=False, 
                 build_large_graph=False, random_seed=42, consistent_mask=None, skip_edge_computation=False,
                 val_ratio=0.2):  # 验证集比例参数

        self.fold_num = 5
        self.split_seed = split_seed
        self.fold = fold
        self.out_dir = out_dir
        self.mode = mode
        self.graph_arg = graph_arg
        self.debug = debug
        self.save = save
        self.build_large_graph = build_large_graph
        self.random_seed = random_seed
        self.consistent_mask = consistent_mask
        self.skip_edge_computation = skip_edge_computation
        self.val_ratio = val_ratio

        pp = Prepare(data_path=self.out_dir, fold_num=5)
        # 传递验证集比例参数
        self.pxs, self.label, self.sample_name = pp.train_test_split(
            seed=split_seed, fold=fold, mode=mode, val_ratio=val_ratio)
        self.N, self.P = self.pxs.shape[:2]
        self.node_prior_use = self.pxs
        self.coordinates = np.load(os.path.join(self.out_dir, 'coordinates.npy'))
        coord = self.coordinates[None].repeat(self.N, axis=0)  # [P,3]->[B,P,3]
        
        # 如果skip_edge_computation为True，只加载数据不计算边
        if skip_edge_computation:
            self.node = self.pxs
            self.edge = None  # 不计算边
            return
        # 构建图
        graph = ConstructGraph(**self.graph_arg, build_large_graph=self.build_large_graph, random_seed=self.random_seed)
        
        # 传入一致性边掩码
        if self.build_large_graph:
            self.node, self.edge = graph.construct(self.pxs, None, coord, self.label, consistent_mask=self.consistent_mask)
        else:
            self.node, self.edge = graph.construct(self.pxs, None, coord, consistent_mask=self.consistent_mask)
        
        #self.node = self.pxs
        
        # Debug模式处理
        if self.debug:
            self.node = self.node[0:10]
            self.edge = self.edge[0:10]
            self.label = self.label[0:10]
            self.sample_name = self.sample_name[0:10]

        weights_dict = dict(zip(np.unique(self.label), [1. / sum(self.label == l) for l in np.unique(self.label)]))
        self.samples_weights = [weights_dict[l] for l in self.label]
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = self.node[index]
        edge = self.edge[index]
        label = self.label[index]
        return data, edge, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]

        return sum(hit_top_k) * 1.0 / len(hit_top_k)

class Prepare:
    def __init__(self, data_path, fold_num=5):
        self.data_path = data_path
        self.fold_num = fold_num
        with open(os.path.join(data_path, 'label.pkl'), 'rb') as f:
            self.label_all, self.sub_name_all = pickle.load(f)
        
        # 加载原始数据
        #self.data_all = np.load(os.path.join(data_path, 'data.npy'))
        self.data_all = np.load(os.path.join(data_path, 'data.npy'), mmap_mode='r')
        self.coordinates = np.load(os.path.join(data_path, 'coordinates.npy'))
        
        # 执行Z-score标准化
        #self.data_all = self.z_score_normalize(self.data_all)
        # 保存标准化统计信息，方便后续使用
        #stats_file = os.path.join(data_path, 'normalization_stats.npz')
        #np.savez(stats_file, mean=self.mean, std=self.std)
        #print(f"Z-score标准化已应用。数据均值: {self.mean:.6f}, 标准差: {self.std:.6f}")
        #print(f"标准化统计信息已保存到: {stats_file}")

    def z_score_normalize(self, data):
        """对输入数据进行Z-score标准化
        
        Parameters:
        -----------
        data : numpy.ndarray
            Shape [N, P, W, H, D] 的输入数据
            N: 样本数, P: 每个样本的patch数, W,H,D: 每个patch的尺寸
            
        Returns:
        --------
        normalized_data : numpy.ndarray
            标准化后的数据，与输入形状相同
        """
        # 获取数据形状
        original_shape = data.shape
        
        # 将数据重塑为1D形式进行标准化
        # 将所有样本、所有patch和所有空间维度展平为一个大向量
        flattened_data = data.reshape(-1)
        
        # 计算全局均值和标准差
        self.mean = np.mean(flattened_data)
        self.std = np.std(flattened_data)
        
        # 确保标准差不为零，避免除零错误
        if self.std == 0:
            self.std = 1e-6
        
        # 应用标准化
        normalized_data = (data - self.mean) / self.std
        
        return normalized_data

    def train_test_split(self, seed, fold, mode, val_ratio=0.2):
        skf = StratifiedKFold(n_splits=self.fold_num, shuffle=True, random_state=seed)
        split = skf.split(self.label_all, self.label_all)
        train_idx_list, test_idx_list = [], []
        save_idx_dir = os.path.join(self.data_path, 'test_idx_' + str(seed))
        os.makedirs(save_idx_dir, exist_ok=True)

        for i, (train, test) in enumerate(split):
            train_idx_list.append(train)
            test_idx_list.append(test)
            if os.path.isfile(os.path.join(save_idx_dir, '%d.txt' % i)):
                test_save = np.loadtxt(os.path.join(save_idx_dir, '%d.txt' % i)).astype(int)
                assert (test_save == test).all()
            else:
                np.savetxt(os.path.join(save_idx_dir, '%d.txt' % i), test, fmt="%d")

        if mode == 'train':
            # 获取当前折的训练索引
            train_indices = train_idx_list[fold]

            # 从训练索引中排除验证集部分
            np.random.seed(seed)
            np.random.shuffle(train_indices)
            val_size = int(len(train_indices) * val_ratio)
            train_indices = train_indices[val_size:]

            data, label, sub = sub_data(self.data_all, self.label_all, self.sub_name_all, [train_indices], 0)

        elif mode == 'val':
            # 获取当前折的训练索引并提取验证集部分
            train_indices = train_idx_list[fold]
            np.random.seed(seed)
            np.random.shuffle(train_indices)
            val_size = int(len(train_indices) * val_ratio)
            val_indices = train_indices[:val_size]
            print(val_indices)
            data, label, sub = sub_data(self.data_all, self.label_all, self.sub_name_all, [val_indices], 0)

        elif mode == 'test':
            data, label, sub = sub_data(self.data_all, self.label_all, self.sub_name_all, test_idx_list, fold)

        else:
            raise ValueError(f"不支持的模式: {mode}")

        return data, label, sub

def sub_data(data, label, sub_name, idx_list, fold):
    data = data[idx_list[fold], ...]
    label = label[idx_list[fold]]
    sub_name = sub_name[idx_list[fold]]
    return data, label, sub_name