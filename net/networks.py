import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd

class CausalMask(nn.Module):
    def __init__(self, patch_num, channel, tau, consistent_edges=None):
        super(CausalMask, self).__init__()
        self.P = patch_num
        self.channel = channel
        self.tau = tau
         
        key_indices =[7, 13, 16, 19, 20, 25, 26, 31, 32, 50, 55, 56, 60, 61, 62, 63, 64, 65, 67, 68, 70, 75, 77, 78, 79, 86, 87, 96, 98, 99, 107, 109, 111, 112, 114, 116, 117, 118, 119, 121, 124, 126, 127, 128, 131, 145, 152, 153, 155, 157, 160, 162, 163, 165, 166, 167, 169, 170, 171, 172, 173, 174, 176, 177, 178, 179, 183, 192, 195, 200, 201, 202, 204, 206, 208, 210, 211, 212, 213, 217, 219, 220, 221, 222, 223, 225, 226, 229, 231, 235, 252, 254, 259, 264, 266, 278]


        #key_indices =[]
        # 创建两个全为0的张量
        dim1 = torch.ones(self.P, 1)
        dim2 = torch.zeros(self.P, 1)

        dim1[key_indices]= 0

        # 第二维度：关键位置为1
        dim2[key_indices] = 1.0

        # 合并两个维度
        n = torch.cat([dim1, dim2], dim=-1)
        
        self.node_mask = nn.Parameter(Variable(n, requires_grad=True))

        # [最终方案] 边掩码初始化 - 使用一致性边信息，并永久排除对角线
        if consistent_edges is not None:
            consistent_edges = torch.FloatTensor(consistent_edges)

            # 1. 初始的可学习边掩码，基于一致性边
            learnable_mask = (consistent_edges > 0).float()

            # 2. 创建一个单位矩阵，用于移除对角线
            identity_matrix = torch.eye(self.P, device=learnable_mask.device)

            # 3. 从可学习掩码中强制移除对角线 (核心修改)
            self.learnable_edges_mask = learnable_mask * (1 - identity_matrix)
            self.register_buffer('learnable_mask', self.learnable_edges_mask)

            # 4. 初始化边掩码的logits，确保对角线和非一致性边都不可学习
            e_init = torch.zeros(self.P, self.P, 2)
            for i in range(self.P):
                for j in range(self.P):
                    if self.learnable_edges_mask[i, j] > 0:
                        # 可学习的边(非对角线)，logits倾向于1 (因果)
                        e_init[i, j, 0] = 0
                        e_init[i, j, 1] = 1
                    else:
                        # 不可学习的边(包括所有对角线)，logits倾向于0 (非因果)
                        e_init[i, j, 0] = 1
                        e_init[i, j, 1] = 0
            
            self.edge_mask = nn.Parameter(Variable(e_init, requires_grad=True))
            print(f"Initialized edge mask with {torch.sum(self.learnable_edges_mask).item()} learnable edges (diagonal excluded)")
        else:
            # 默认情况下的初始化也应排除对角线
            e = torch.cat([torch.zeros(self.P, self.P, 1), torch.ones(self.P, self.P, 1)], dim=-1)
            self.edge_mask = nn.Parameter(Variable(e, requires_grad=True))
            
            identity_matrix = torch.eye(self.P)
            self.learnable_edges_mask = torch.ones(self.P, self.P) * (1 - identity_matrix)
            self.register_buffer('learnable_mask', self.learnable_edges_mask)
            
    @staticmethod
    def gumbel_softmax(logits, tau: float = 1, hard: bool = True, dim: int = -1):
        gumbel_dist = torch.distributions.gumbel.Gumbel(
            torch.tensor(0., device=logits.device, dtype=logits.dtype),
            torch.tensor(1., device=logits.device, dtype=logits.dtype))
        gumbels = gumbel_dist.sample(logits.shape)

        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft

        return ret[..., 1]
    
    @staticmethod
    def hardmax(M, dim: int = -1):
        return M.argmax(dim).float()
    
    def calculate_sparsity(self, node_mask, edge_mask):
        """一个独立的稀疏度计算函数"""
        # 计算稀疏度时，确保分母不为0
        edge_sparsity = edge_mask.sum() / (self.learnable_mask.sum() + 1e-8)
        return (node_mask.mean() + edge_sparsity) / 2

    def forward(self, train=True, return_probs=False):
        """
        [最终优化]
        - 统一返回格式，始终返回硬掩码和稀疏度。
        - 在评估模式下，可选择性地额外返回连续的因果概率。
        """
        if train:
            # 训练模式：使用Gumbel-Softmax生成0/1掩码
            node_mask_hard = self.gumbel_softmax(self.node_mask, tau=self.tau, hard=True)
            edge_mask_hard = self.gumbel_softmax(self.edge_mask, tau=self.tau, hard=True) * self.learnable_mask
        else: # 评估或保存模式
            # 硬掩码（0/1），用于模型推理
            node_mask_hard = self.hardmax(self.node_mask)
            edge_mask_hard = self.hardmax(self.edge_mask) * self.learnable_mask

        # [核心修改] 无论何种模式，都计算稀疏度
        sparsity = self.calculate_sparsity(node_mask_hard, edge_mask_hard)

        # 如果需要，额外计算概率值
        if return_probs:
            node_probs_soft = F.softmax(self.node_mask, dim=-1)[:, 1]
            edge_probs_soft = F.softmax(self.edge_mask, dim=-1)[:, :, 1]
            edge_probs_soft = edge_probs_soft * self.learnable_mask
            
            return [node_mask_hard, edge_mask_hard], [node_probs_soft, edge_probs_soft], sparsity
        else:
            # 默认只返回硬掩码和稀疏度
            return [node_mask_hard, edge_mask_hard], sparsity
        
    def compute_sparsity_regularization(self, lambda_reg=0.1):
        """
        [最终优化版] 计算稀疏性正则化损失
        引入了独立的权重来控制节点和边的稀疏度。
        """
        # --- 1. 计算节点稀疏性损失 (逻辑不变) ---
        node_probs = F.softmax(self.node_mask, dim=1)
        node_causal_probs = node_probs[:, 1]  # 因果维度的概率
        node_sparsity_loss = torch.mean(node_causal_probs)
        
        # --- 2. 计算边稀疏性损失 (逻辑不变) ---
        edge_probs = F.softmax(self.edge_mask, dim=-1)
        edge_causal_probs = edge_probs[:, :, 1]  # 因果维度的概率
        
        masked_edge_probs = edge_causal_probs * self.learnable_mask
        total_learnable_edges = torch.sum(self.learnable_mask)
        
        if total_learnable_edges > 0:
            edge_sparsity_loss = torch.sum(masked_edge_probs) / total_learnable_edges
        else:
            edge_sparsity_loss = torch.tensor(0.0, device=node_sparsity_loss.device)
        
        # --- 3. [核心修改] 为节点和边的稀疏性损失赋予不同权重 ---
        
        # 对节点稀疏性保持一个较高的惩罚
        node_sparsity_weight = 1.5
        
        # 对边的稀疏性使用一个小得多的惩罚，鼓励模型保留更多的边
        # 这个值是您可以精调的关键超参数，以获得您想要的边数量
        edge_sparsity_weight = 0.1 # 建议从 0.1 或 0.05 开始尝试
        
        # 加权计算总的稀疏性损失
        total_sparsity_loss = (node_sparsity_weight * node_sparsity_loss + 
                               edge_sparsity_weight * edge_sparsity_loss)
        
        # 乘以全局的稀疏性惩罚系数 lambda_reg
        return lambda_reg * total_sparsity_loss
    
class LightDenseNet3D(nn.Module):
    """
    作为特征提取器的3D CNN骨干网络。
    """
    def __init__(self, growth_rate=16, num_init_features=24):
        super(LightDenseNet3D, self).__init__()
        self.features = nn.Sequential(nn.Conv3d(1, num_init_features, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm3d(num_init_features), nn.ReLU(inplace=True), nn.MaxPool3d(kernel_size=2, stride=2))
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
            if isinstance(m, nn.Conv3d): nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): nn.init.constant_(m.bias, 0)
    def _make_dense_layer(self, in_channels, growth_rate):
        return nn.Sequential(nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True), nn.Conv3d(in_channels, 4 * growth_rate, kernel_size=1, stride=1, bias=False), nn.BatchNorm3d(4 * growth_rate), nn.ReLU(inplace=True), nn.Conv3d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
    def _make_dense_block(self, in_channels, growth_rate, num_layers):
        layers = []
        for i in range(num_layers): layers.append(self._make_dense_layer(in_channels + i * growth_rate, growth_rate))
        return nn.ModuleList(layers)
    def _make_transition(self, in_channels, out_channels):
        return nn.Sequential(nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True), nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False), nn.AvgPool3d(kernel_size=2, stride=2))
    def forward(self, x):
        features = self.features(x)
        for layer in self.dense1: features = torch.cat([features, layer(features)], 1)
        features = self.trans1(features)
        for layer in self.dense2: features = torch.cat([features, layer(features)], 1)
        features = self.trans2(features)
        features = self.norm(features)
        features = self.relu(features)
        return torch.flatten(F.adaptive_avg_pool3d(features, (1, 1, 1)), 1)


class TwoStageGCN(nn.Module):
    """两阶段图卷积网络 (调整了阶段顺序)
    第一阶段：
        - 第一层使用提供的边矩阵进行图卷积
        - 第二层使用单位矩阵作为邻接矩阵，只进行线性变换，不进行节点聚合
    第二阶段：使用提供的边矩阵进行图卷积 (原来的第一阶段)
              只有在提供了第二个邻接矩阵(edge2)时才执行
    """
    def __init__(self, in_features, hidden):
        super(TwoStageGCN, self).__init__()
        self.hidden = hidden
        
        # 第一阶段 (原来的第二阶段) - 第一层卷积
        self.stage1_conv1 = GCN(in_features=in_features, out_features=self.hidden[0])
        self.stage1_bn1 = nn.BatchNorm1d(self.hidden[0])
        
        # 第一阶段 (原来的第二阶段) - 第二层卷积 (使用单位矩阵, 只做线性变换)
        self.stage1_conv2 = GCN(in_features=self.hidden[0], out_features=self.hidden[-1])
        self.stage1_bn2 = nn.BatchNorm1d(self.hidden[-1])
        
        # 第二阶段 (原来的第一阶段) - 维度不变
        self.stage2_conv = GCN(in_features=self.hidden[-1], out_features=self.hidden[-1])
        self.stage2_bn = nn.BatchNorm1d(self.hidden[-1])

    def forward(self, x, edge1, edge2=None):
        """
        执行两阶段图卷积 (调整了阶段顺序)
        参数:
            x: 输入节点特征 [B, P, d]
            edge1: 第一阶段边矩阵 [B, P, P]
            edge2: 第二阶段边矩阵 [B, P, P]，如果提供则执行第二阶段，否则只执行第一阶段
        """
        # 第一阶段第一层：使用提供的边矩阵进行消息传递
        x = self.stage1_conv1(x, edge1)
        x = F.relu(self.stage1_bn1(x.transpose(1, 2)).transpose(1, 2))
        
        # 第一阶段第二层：使用单位矩阵作为邻接矩阵，只进行线性变换
        batch_size, num_nodes = x.shape[0], x.shape[1]
        identity_matrix = torch.eye(num_nodes, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        x = self.stage1_conv2(x, identity_matrix)
        x = F.relu(self.stage1_bn2(x.transpose(1, 2)).transpose(1, 2))
        
        # 第二阶段：只有在提供了edge2时才执行 (原来的第一阶段)
        if edge2 is not None:
            x = torch.bmm(edge2, x)
        return x
    

class CausalNet(nn.Module):
    def __init__(self, num_class, hidden1, hidden2, kernels=None, pretrained_path=None, freeze_extractor=True):
        super(CausalNet, self).__init__()

        self.hidden1 = hidden1
        
        # 特征提取器保持不变
        self.feature_extractor = LightDenseNet3D(growth_rate=8, num_init_features=24)
        
        # 加载预训练权重
        if pretrained_path is not None:
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            print(f"Loaded pre-trained weights from {pretrained_path}")
        
        # 冻结特征提取器
        if freeze_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print("Feature extractor has been frozen")
        
        # 更新特征维度
        #feature_dim = self.feature_extractor.feature_dim
        feature_dim = 28
        # 使用新的两阶段GCN替代原有的GCN
        self.gcns2_causal = TwoStageGCN(in_features=feature_dim, hidden=hidden2)
        
        # MLP分类器保持不变
        
        
        P = 288  # 患者的patch数量
        
        
        
        feature_size = P * hidden2[-1]
        self.mlp_causal = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(64, num_class)
        )

    # 特征提取函数保持不变
    def emb(self, x):
        B, P = x.shape[:2]
        x_np = x.reshape(-1, *x.shape[2:]).unsqueeze(1)
        with torch.set_grad_enabled(self.feature_extractor.training):
            x_features = self.feature_extractor(x_np)
        x_emb = x_features.reshape(B, P, -1)
        return x_emb
        
    def prediction_whole(self, x_new, edge, is_large_graph=True):
        """全图预测 - 使用相同的边矩阵"""
        if is_large_graph:
            # 从大图中提取原始样本
            batch_size = x_new.shape[0]
            P = x_new.shape[1] // 5
            x_orig = x_new[:, :P, :]
            edge_orig = edge[:, :P, :P]
            # 两阶段卷积，使用相同的边矩阵
            xs = self.gcns2_causal(x_orig, edge_orig)
        else:
            # 直接使用相同的边矩阵进行两阶段卷积
            xs = self.gcns2_causal(x_new, edge)
            
        graph = readout(xs)
        return self.mlp_causal(graph)
    
    def prediction_causal(self, x_new, edge, masks, is_large_graph=True):
        """因果子图预测 - [最终优化版]"""
        node_mask, edge_mask = masks
        
        if is_large_graph:
            batch_size = x_new.shape[0]
            large_P = x_new.shape[1]
            P = large_P // 5
            num_neg_samples = 4
                        
            # 1. 创建跨样本边矩阵 (cross_edges) - 这部分逻辑不变
            cross_edges = torch.zeros_like(edge).to(edge.device)
            edge_weight = 1.0 / num_neg_samples        

            for neg_idx in range(1, 5):
                for p in range(P):
                    neg_p = neg_idx * P + p
                    cross_edges[:, p, neg_p] = edge_weight * (1 - node_mask[p])
                    cross_edges[:, neg_p, p] = edge_weight * (1 - node_mask[p])

            for p in range(P):
                cross_edges[:, p, p] = node_mask[p]

            for neg_idx in range(1, 5):
                for p in range(P):
                    diag_idx = neg_idx * P + p
                    cross_edges[:, diag_idx, diag_idx] = node_mask[p]

            # 2. 创建内部边矩阵 (internal_edges) - 逻辑不变
            internal_edges = torch.zeros_like(edge).to(edge.device)
            for i in range(batch_size):
                for s_idx in range(5): # 对5个子图都应用mask
                    sub_range = slice(s_idx*P, (s_idx+1)*P)
                    # 注意：这里我们假设 edge 的每个 5P x 5P 矩阵的对角块是相同的
                    # 如果不同，则需要切片 edge[i, sub_range, sub_range]
                    orig_edge_block = edge[i, :P, :P] 
                    internal_edges[i, sub_range, sub_range] = orig_edge_block * edge_mask
            
            # 3. [核心修复] 为内部边矩阵添加自环
            identity_large = torch.eye(large_P, device=edge.device).unsqueeze(0)
            final_internal_edges = internal_edges + identity_large
            
            # 4. 使用修复后的矩阵进行两阶段卷积处理
            xs_all = self.gcns2_causal(x_new, final_internal_edges, cross_edges)
            
            # 只取原始样本特征
            orig_features = xs_all[:, :P, :]
            graph = readout(orig_features)
            
        else:
            # 小图模式
            x_new_masked = x_new * node_mask.unsqueeze(0).unsqueeze(-1)
            inter_node_adj = edge * edge_mask.unsqueeze(0)
            
            # [核心修复] 为小图添加自环
            P = x_new.shape[1]
            identity = torch.eye(P, device=edge.device).unsqueeze(0)
            final_adj = inter_node_adj + identity
            
            xs = self.gcns2_causal(x_new_masked, final_adj)
            graph = readout(xs)
            
        return self.mlp_causal(graph)
    
    def prediction_counterfactual(self, x_new, edge, masks, is_large_graph=True):
        """反事实子图预测 - [最终优化版]"""
        node_mask, edge_mask = masks
        node_mask_inv = 1 - node_mask
        edge_mask_inv = 1 - edge_mask
        
        if is_large_graph:
            batch_size = x_new.shape[0]
            large_P = x_new.shape[1]
            P = large_P // 5
            num_neg_samples = 4

            # 1. 创建跨样本边矩阵 (cross_edges) - 逻辑不变
            cross_edges = torch.zeros_like(edge).to(edge.device)
            edge_weight = 1.0 / num_neg_samples

            for neg_idx in range(1, 5):
                for p in range(P):
                    neg_p = neg_idx * P + p
                    cross_edges[:, p, neg_p] = edge_weight * node_mask[p]
                    cross_edges[:, neg_p, p] = edge_weight * node_mask[p]

            for p in range(P):
                cross_edges[:, p, p] = 1 - node_mask[p]

            for neg_idx in range(1, 5):
                for p in range(P):
                    diag_idx = neg_idx * P + p
                    cross_edges[:, diag_idx, diag_idx] = 1 - node_mask[p]
            
            # 2. 创建内部边矩阵 (internal_edges) - 逻辑不变
            internal_edges = torch.zeros_like(edge).to(edge.device)
            for i in range(batch_size):
                for s_idx in range(5):
                    sub_range = slice(s_idx*P, (s_idx+1)*P)
                    orig_edge_block = edge[i, :P, :P]
                    internal_edges[i, sub_range, sub_range] = orig_edge_block * edge_mask_inv

            # 3. [核心修复] 为内部边矩阵添加自环
            identity_large = torch.eye(large_P, device=edge.device).unsqueeze(0)
            final_internal_edges = internal_edges + identity_large
            
            # 4. 使用修复后的矩阵进行两阶段卷积处理
            xs_all = self.gcns2_causal(x_new, final_internal_edges, cross_edges)
            
            # 只取原始样本特征
            orig_features = xs_all[:, :P, :]
            graph = readout(orig_features)
            
        else:
            # 小图模式
            x_new_masked = x_new * node_mask_inv.unsqueeze(0).unsqueeze(-1)
            inter_node_adj = edge * edge_mask_inv.unsqueeze(0)

            # [核心修复] 为小图添加自环
            P = x_new.shape[1]
            identity = torch.eye(P, device=edge.device).unsqueeze(0)
            final_adj = inter_node_adj + identity
            
            xs = self.gcns2_causal(x_new_masked, final_adj)
            graph = readout(xs)
            
        return self.mlp_causal(graph)
    
    def prediction_causal_invariance(self, x_new, edge, masks, is_large_graph=True):
            """因果不变性预测 - [最小改动优化版]"""
            node_mask, edge_mask = masks

            if is_large_graph:
                # 从大图中提取原始样本
                batch_size = x_new.shape[0]
                P = x_new.shape[1] // 5
                x_orig = x_new[:, :P, :].clone()
                edge_orig = edge[:, :P, :P].clone()

                # 应用节点掩码
                x_masked = x_orig * node_mask.unsqueeze(0).unsqueeze(-1)

                # [核心修改] 创建带有自环的最终邻接矩阵
                # 1. 应用学习到的边掩码
                inter_node_adj = edge_orig * edge_mask.unsqueeze(0)

                # 2. 创建单位矩阵代表自环
                identity = torch.eye(P, device=inter_node_adj.device).unsqueeze(0)

                # 3. 将自环加到图中，得到最终传入GCN的邻接矩阵
                final_adj = inter_node_adj + identity

                # 小图两阶段卷积
                xs = self.gcns2_causal(x_masked, final_adj)
            else:
                # 直接应用掩码
                x_masked = x_new * node_mask.unsqueeze(0).unsqueeze(-1)

                # [核心修改] 创建带有自环的最终邻接矩阵
                inter_node_adj = edge * edge_mask.unsqueeze(0)
                P = x_new.shape[1]
                identity = torch.eye(P, device=inter_node_adj.device).unsqueeze(0)
                final_adj = inter_node_adj + identity

                xs = self.gcns2_causal(x_masked, final_adj)

            graph = readout(xs)
            return self.mlp_causal(graph)
    
    def prediction_causal_variability(self, x_new, edge, masks, is_large_graph=True):
        """因果变异性预测 - [最小改动优化版]"""
        node_mask, edge_mask = masks
        
        if is_large_graph:
            # 提取原始样本
            batch_size = x_new.shape[0]
            P = x_new.shape[1] // 5
            x_orig = x_new[:, :P, :].clone()
            edge_orig = edge[:, :P, :P].clone()

            # 对非因果部分进行操作
            causal_mask = node_mask.unsqueeze(0).unsqueeze(-1)
            x_perturbed = x_orig * (1 - causal_mask)
            
            # [核心修改] 对扰动后的图也添加自环，保证稳定
            # 1. 应用反转的边掩码
            causal_edge_mask = edge_mask.unsqueeze(0)
            edge_perturbed = edge_orig * (1 - causal_edge_mask)

            # 2. 创建单位矩阵代表自环
            identity = torch.eye(P, device=edge_perturbed.device).unsqueeze(0)

            # 3. 将自环加到图中
            final_adj_perturbed = edge_perturbed + identity
            
            # 小图两阶段卷积
            xs = self.gcns2_causal(x_perturbed, final_adj_perturbed)
        else:
            # 对非因果部分进行操作
            causal_mask = node_mask.unsqueeze(0).unsqueeze(-1)
            x_perturbed = x_new * (1 - causal_mask)
            
            # [核心修改] 对扰动后的图也添加自环，保证稳定
            causal_edge_mask = edge_mask.unsqueeze(0)
            edge_perturbed = edge * (1 - causal_edge_mask)
            
            P = x_new.shape[1]
            identity = torch.eye(P, device=edge_perturbed.device).unsqueeze(0)
            final_adj_perturbed = edge_perturbed + identity
            
            xs = self.gcns2_causal(x_perturbed, final_adj_perturbed)
            
        graph = readout(xs)
        return self.mlp_causal(graph)

def readout(x):  # x:[B,P,d]
    # 从 [B,P,d] 形状重塑为 [B,P*d]
    batch_size = x.shape[0]
    return x.reshape(batch_size, -1)

class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, X, adj):  # X:[B*P,d]=[1600,4], adj:[B*P,B*P]=[1600,1600]
        XW = self.W(X)
        AXW = torch.matmul(adj, XW)  # AXW
        return AXW

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_features, self.out_features)
