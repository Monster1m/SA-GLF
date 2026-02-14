import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import logging
import random

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GCNEncoder(nn.Module):
    """GCN编码器 - 生成节点嵌入"""

    def __init__(self, input_dim, hidden_dim=512, output_dim=1536, num_layers=2):
        """
        初始化GCN编码器

        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
            num_layers: GCN层数
        """
        super().__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim

        # 输入投影层
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        # GCN层
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = GCNConv(hidden_dim, hidden_dim)
            self.convs.append(conv)

        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x, edge_index):
        # 输入投影
        x = self.input_proj(x)

        # GCN层
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # 输出投影
        return self.output_proj(x)


class DualGraphContrastiveModel:
    """双图对比学习节点嵌入模型"""

    def __init__(self, graph1, graph2):
        self.graph1 = graph1
        self.graph2 = graph2

        # 准备数据
        self.data1 = self._prepare_data(graph1)
        self.data2 = self._prepare_data(graph2)

        # 获取特征维度
        input_dim = max(self.data1.x.size(1), self.data2.x.size(1))
        logger.info(f"使用输入维度: {input_dim}")

        # 初始化GCN编码器
        self.encoder = GCNEncoder(input_dim, output_dim=1536)

        # 创建索引映射
        self.idx_map1 = {node: i for i, node in enumerate(graph1.nodes)}
        self.idx_map2 = {node: i for i, node in enumerate(graph2.nodes)}

        # 训练历史记录
        self.loss_history = []

    def _prepare_data(self, graph):
        """准备图数据 - 修复字典类型特征问题"""
        # 确保图有节点
        if not graph.nodes:
            raise ValueError("图没有节点")

        # 提取所有节点特征
        features = []
        for node in graph.nodes:
            attrs = graph.nodes[node]

            # 检查是否有features属性
            if 'features' in attrs:
                feat = attrs['features']

                # 如果特征是字典，提取值
                if isinstance(feat, dict):
                    # 按键排序以确保顺序一致
                    sorted_keys = sorted(feat.keys())
                    feat = [feat[k] for k in sorted_keys]
                # 如果特征是单个值，转换为列表
                elif not isinstance(feat, list):
                    feat = [feat]
            else:
                # 使用节点度作为默认特征
                feat = [graph.degree(node)]

            features.append(feat)

        # 确保所有特征长度相同
        max_len = max(len(f) for f in features)
        padded_features = [f + [0.0] * (max_len - len(f)) for f in features]

        # 创建特征张量
        x = torch.tensor(padded_features, dtype=torch.float)

        # 创建边索引
        edge_index = []
        for edge in graph.edges:
            src = list(graph.nodes).index(edge[0])
            dst = list(graph.nodes).index(edge[1])
            edge_index.append([src, dst])

        # 转换为张量
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # 创建PyG数据对象
        data = Data(x=x, edge_index=edge_index)

        logger.info(f"节点特征维度: {data.x.size(1)}")
        return data

    def train(self, epochs=100, lr=0.001, temperature=0.1, num_negatives=10):
        """训练模型 - 对比学习优化"""
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

        # 训练循环
        for epoch in range(epochs):
            optimizer.zero_grad()

            # 计算图1的损失
            loss1 = self._graph_contrastive_loss(self.data1, temperature, num_negatives)

            # 计算图2的损失
            loss2 = self._graph_contrastive_loss(self.data2, temperature, num_negatives)

            # 总损失
            total_loss = loss1 + loss2

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 记录损失
            self.loss_history.append(total_loss.item())

            logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss.item():.4f}")

    def _graph_contrastive_loss(self, data, temperature, num_negatives):
        """计算单图的对比损失"""
        # 生成节点嵌入
        embeddings = self.encoder(data.x, data.edge_index)

        # 归一化嵌入
        embeddings = F.normalize(embeddings, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.mm(embeddings, embeddings.t())

        # 正样本对 - 对角线元素
        positives = torch.diag(sim_matrix)

        # 负样本 - 随机采样其他节点
        batch_size = embeddings.size(0)
        negative_indices = torch.randint(0, batch_size, (batch_size, num_negatives))
        negatives = sim_matrix[torch.arange(batch_size).unsqueeze(1), negative_indices]

        # InfoNCE损失
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1) / temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=embeddings.device)
        loss = F.cross_entropy(logits, labels)

        return loss

    def generate_embeddings(self):
        """生成节点嵌入字典"""
        # 生成图1嵌入
        with torch.no_grad():
            emb1 = self.encoder(self.data1.x, self.data1.edge_index).detach().numpy()

        # 生成图2嵌入
        with torch.no_grad():
            emb2 = self.encoder(self.data2.x, self.data2.edge_index).detach().numpy()

        # 创建嵌入字典
        emb_dict1 = {}
        for node, idx in self.idx_map1.items():
            emb_dict1[node] = emb1[idx]

        emb_dict2 = {}
        for node, idx in self.idx_map2.items():
            emb_dict2[node] = emb2[idx]

        return emb_dict1, emb_dict2


def gcn_subg_node_emb(g1, g2):
    """生成子图节点嵌入"""
    # 创建模型
    model = DualGraphContrastiveModel(g1, g2)

    # 训练模型
    model.train(epochs=1, lr=0.0001, temperature=0.1, num_negatives=2)

    # 生成嵌入字典
    emb_dict1, emb_dict2 = model.generate_embeddings()
    return emb_dict1, emb_dict2