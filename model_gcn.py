"""
带对齐优化的双网络GCN嵌入生成器
功能：处理两个NetworkX图对象，使用对齐字典进行对比损失优化，生成节点嵌入并保存为CSV文件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os


class GCNLayer(nn.Module):
    """GCN层"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # 添加自环
        adj = adj + torch.eye(adj.size(0), device=x.device)

        # 计算归一化邻接矩阵
        degree = adj.sum(dim=1, keepdim=True)
        degree_sqrt = torch.sqrt(degree + 1e-8)
        norm_adj = adj / (degree_sqrt * degree_sqrt.t() + 1e-8)

        # 图卷积
        x = self.linear(x)
        x = torch.matmul(norm_adj, x)

        return x


class DualGCN(nn.Module):
    """双网络共享参数GCN模型"""

    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        # 构建隐藏层
        current_dim = in_dim
        for h_dim in hidden_dims:
            self.layers.append(GCNLayer(current_dim, h_dim, dropout))
            current_dim = h_dim

        # 输出层
        self.layers.append(GCNLayer(current_dim, out_dim, dropout=0.0))

    def forward(self, x1: torch.Tensor, adj1: torch.Tensor,
                x2: torch.Tensor, adj2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 处理第一个图
        h1 = x1
        for layer in self.layers[:-1]:
            h1 = layer(h1, adj1)
            h1 = F.relu(h1)
            h1 = F.dropout(h1, self.dropout, training=self.training)

        # 处理第二个图
        h2 = x2
        for layer in self.layers[:-1]:
            h2 = layer(h2, adj2)
            h2 = F.relu(h2)
            h2 = F.dropout(h2, self.dropout, training=self.training)

        # 输出层
        out1 = self.layers[-1](h1, adj1)
        out2 = self.layers[-1](h2, adj2)

        return out1, out2


class ContrastiveLoss(nn.Module):
    """对比损失函数，用于优化对齐节点对的嵌入"""

    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor,
                idx1: torch.Tensor, idx2: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失

        参数:
            emb1: 第一个图的嵌入 [n1, dim]
            emb2: 第二个图的嵌入 [n2, dim]
            idx1: 第一个图中对齐节点的索引
            idx2: 第二个图中对齐节点的索引
        """
        # 获取对齐节点的嵌入
        align_emb1 = emb1[idx1]  # [n_align, dim]
        align_emb2 = emb2[idx2]  # [n_align, dim]

        # 计算对齐节点对之间的余弦相似度
        align_emb1_norm = F.normalize(align_emb1, dim=1)
        align_emb2_norm = F.normalize(align_emb2, dim=1)

        # 正样本对相似度（对角线）
        pos_sim = torch.sum(align_emb1_norm * align_emb2_norm, dim=1)  # [n_align]

        # 负样本对相似度（所有非对角线对）
        sim_matrix = torch.mm(align_emb1_norm, align_emb2_norm.t())  # [n_align, n_align]

        # 计算对比损失 (InfoNCE)
        labels = torch.arange(align_emb1.size(0), device=emb1.device)
        loss = F.cross_entropy(sim_matrix / self.temperature, labels)

        return loss


class DualGCNEmbedder:
    """带对齐优化的双网络GCN嵌入生成器"""

    def __init__(self, hidden_dims: List[int] = [64, 32], out_dim: int = 16, dropout: float = 0.2,
                 lr: float = 0.01, temperature: float = 0.1):
        """
        参数:
            hidden_dims: 隐藏层维度列表，控制GCN层数
            out_dim: 输出嵌入维度
            dropout: Dropout率
            lr: 学习率
            temperature: 对比损失温度参数
        """
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.dropout = dropout
        self.lr = lr
        self.temperature = temperature
        self.model = None
        self.in_dim = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.contrastive_loss = ContrastiveLoss(temperature)

    def extract_features(self, G: nx.Graph) -> Tuple[torch.Tensor, List]:
        """从NetworkX图中提取节点特征"""
        node_ids = list(G.nodes())
        features = []

        for node in node_ids:
            node_data = G.nodes[node]

            # 获取特征
            feat = None
            for attr in ['features', 'feature', 'attr', 'attributes']:
                if attr in node_data:
                    feat = node_data[attr]
                    break

            # 处理特征
            if feat is None:
                feat = [G.degree(node), 0.0]  # 默认特征
            elif isinstance(feat, (int, float)):
                feat = [float(feat)]
            elif isinstance(feat, list):
                feat = [float(x) for x in feat]
            elif isinstance(feat, str):
                try:
                    # 尝试解析字符串
                    if '[' in feat and ']' in feat:
                        content = feat[feat.find('[') + 1:feat.find(']')]
                        feat = [float(x.strip()) for x in content.split(',') if x.strip()]
                    else:
                        feat = [float(feat)]
                except:
                    feat = [0.0]
            else:
                feat = [0.0]

            features.append(feat)

        # 确定最大维度并填充
        max_dim = max(len(f) for f in features) if features else 1
        padded_features = [f + [0.0] * (max_dim - len(f)) if len(f) < max_dim else f[:max_dim]
                           for f in features]

        return torch.tensor(padded_features, dtype=torch.float), node_ids

    def build_adjacency(self, G: nx.Graph, node_ids: List) -> torch.Tensor:
        """构建邻接矩阵"""
        n = len(node_ids)
        node_index = {node: i for i, node in enumerate(node_ids)}
        adj = torch.zeros((n, n), dtype=torch.float)

        for u, v in G.edges():
            if u in node_index and v in node_index:
                i, j = node_index[u], node_index[v]
                adj[i, j] = 1.0
                adj[j, i] = 1.0

        return adj

    def align_features(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """对齐两个图的特征维度"""
        d1, d2 = x1.shape[1], x2.shape[1]

        if d1 == d2:
            return x1, x2

        max_dim = max(d1, d2)

        if d1 < max_dim:
            padding = torch.zeros(x1.shape[0], max_dim - d1, dtype=x1.dtype)
            x1 = torch.cat([x1, padding], dim=1)

        if d2 < max_dim:
            padding = torch.zeros(x2.shape[0], max_dim - d2, dtype=x2.dtype)
            x2 = torch.cat([x2, padding], dim=1)

        return x1, x2

    def get_alignment_indices(self, ids1: List, ids2: List, alignment_dict: Dict) -> Tuple[List, List]:
        """根据对齐字典获取节点索引"""
        idx1, idx2 = [], []
        idx_map1 = {node_id: idx for idx, node_id in enumerate(ids1)}
        idx_map2 = {node_id: idx for idx, node_id in enumerate(ids2)}

        for node1, node2 in alignment_dict.items():
            if node1 in idx_map1 and node2 in idx_map2:
                idx1.append(idx_map1[node1])
                idx2.append(idx_map2[node2])

        return idx1, idx2

    def train_with_alignment(self, x1: torch.Tensor, adj1: torch.Tensor,
                             x2: torch.Tensor, adj2: torch.Tensor,
                             idx1: List, idx2: List, epochs: int = 100) -> List[float]:
        """使用对齐字典训练模型"""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        idx1_tensor = torch.tensor(idx1, device=self.device, dtype=torch.long)
        idx2_tensor = torch.tensor(idx2, device=self.device, dtype=torch.long)

        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()

            # 前向传播
            emb1, emb2 = self.model(x1, adj1, x2, adj2)

            # 计算对比损失
            loss = self.contrastive_loss(emb1, emb2, idx1_tensor, idx2_tensor)

            # 反向传播
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        return losses

    def generate_embeddings(self, G1: nx.Graph, G2: nx.Graph,
                            alignment_dict: Optional[Dict] = None,
                            train_epochs: int = 0) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
        """为两个图生成节点嵌入（可选对齐优化）"""
        # 提取特征
        x1, ids1 = self.extract_features(G1)
        x2, ids2 = self.extract_features(G2)

        # 构建邻接矩阵
        adj1 = self.build_adjacency(G1, ids1)
        adj2 = self.build_adjacency(G2, ids2)

        # 对齐特征维度
        x1, x2 = self.align_features(x1, x2)

        # 移动到设备
        x1, x2 = x1.to(self.device), x2.to(self.device)
        adj1, adj2 = adj1.to(self.device), adj2.to(self.device)

        # 初始化模型
        if self.model is None:
            self.in_dim = x1.shape[1]
            self.model = DualGCN(
                in_dim=self.in_dim,
                hidden_dims=self.hidden_dims,
                out_dim=self.out_dim,
                dropout=self.dropout
            ).to(self.device)

        # 对齐优化训练
        if alignment_dict and train_epochs > 0:
            idx1, idx2 = self.get_alignment_indices(ids1, ids2, alignment_dict)
            if idx1 and idx2:
                print(f"对齐优化训练: {len(idx1)} 个对齐节点对, {train_epochs} 轮")
                losses = self.train_with_alignment(x1, adj1, x2, adj2, idx1, idx2, train_epochs)
                print(f"最终损失: {losses[-1]:.4f}")

        # 生成嵌入
        self.model.eval()
        with torch.no_grad():
            emb1, emb2 = self.model(x1, adj1, x2, adj2)
            emb1, emb2 = emb1.cpu(), emb2.cpu()

        return emb1, emb2, ids1, ids2

    def save_to_csv(self, embeddings: torch.Tensor, node_ids: List, filename: str):
        """保存嵌入为CSV文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

        # 准备数据
        n_nodes, embed_dim = embeddings.shape
        data = {'node_id': node_ids}

        for i in range(embed_dim):
            data[f'dim_{i + 1}'] = embeddings[:, i].numpy()

        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

        print(f"保存嵌入到: {filename} ({df.shape[0]} 个节点, {df.shape[1] - 1} 个维度)")

    def generate_and_save(self, G1: nx.Graph, G2: nx.Graph,
                          file1: str = "graph1_embeddings.csv",
                          file2: str = "graph2_embeddings.csv",
                          alignment_dict: Optional[Dict] = None,
                          train_epochs: int = 0):
        """主函数：生成并保存嵌入（可选对齐优化）"""
        print(f"处理图1: {G1.number_of_nodes()} 节点, {G1.number_of_edges()} 边")
        print(f"处理图2: {G2.number_of_nodes()} 节点, {G2.number_of_edges()} 边")
        print(f"GCN配置: 隐藏层={self.hidden_dims}, 输出维度={self.out_dim}")

        if alignment_dict:
            print(f"对齐优化: {len(alignment_dict)} 个对齐节点对, 训练轮数={train_epochs}")

        # 生成嵌入
        emb1, emb2, ids1, ids2 = self.generate_embeddings(G1, G2, alignment_dict, train_epochs)

        # 保存为CSV
        self.save_to_csv(emb1, ids1, file1)
        self.save_to_csv(emb2, ids2, file2)

        return emb1, emb2


def create_test_graph1() -> nx.Graph:
    """创建测试图1"""
    G = nx.karate_club_graph()
    for node in G.nodes():
        G.nodes[node]['features'] = [node * 0.1, node * 0.2, node * 0.3]
    return G


def create_test_graph2() -> nx.Graph:
    """创建测试图2"""
    G = nx.erdos_renyi_graph(20, 0.3, seed=42)
    for node in G.nodes():
        G.nodes[node]['features'] = [node, node ** 2, node % 5]
    return G


def create_alignment_dict(G1: nx.Graph, G2: nx.Graph, n_pairs: int = 8) -> Dict:
    """创建测试用的节点对齐字典"""
    nodes1 = list(G1.nodes())[:n_pairs]
    nodes2 = list(G2.nodes())[:n_pairs]
    return {nodes1[i]: nodes2[i] for i in range(min(len(nodes1), len(nodes2)))}


def test_alignment_optimization():
    """测试对齐优化功能"""
    print("测试带对齐优化的双网络GCN")
    print("=" * 50)

    # 创建测试图
    G1 = create_test_graph1()
    G2 = create_test_graph2()

    # 创建对齐字典
    alignment_dict = create_alignment_dict(G1, G2, 8)
    print(f"对齐字典: {alignment_dict}")

    # 测试1: 无对齐优化
    print("\n1. 测试无对齐优化...")
    embedder1 = DualGCNEmbedder(
        hidden_dims=[64, 32],
        out_dim=16,
        dropout=0.2
    )
    emb1, emb2 = embedder1.generate_and_save(
        G1, G2,
        "no_alignment_g1.csv",
        "no_alignment_g2.csv"
    )

    # 测试2: 带对齐优化
    print("\n2. 测试带对齐优化...")
    embedder2 = DualGCNEmbedder(
        hidden_dims=[64, 32],
        out_dim=16,
        dropout=0.2,
        lr=0.01,
        temperature=0.1
    )
    emb1, emb2 = embedder2.generate_and_save(
        G1, G2,
        "with_alignment_g1.csv",
        "with_alignment_g2.csv",
        alignment_dict=alignment_dict,
        train_epochs=50
    )

    # 测试3: 不同层数的GCN
    print("\n3. 测试4层GCN带对齐优化...")
    embedder3 = DualGCNEmbedder(
        hidden_dims=[128, 64, 32],
        out_dim=16,
        dropout=0.3,
        lr=0.01,
        temperature=0.1
    )
    emb1, emb2 = embedder3.generate_and_save(
        G1, G2,
        "deep_alignment_g1.csv",
        "deep_alignment_g2.csv",
        alignment_dict=alignment_dict,
        train_epochs=30
    )

    print("\n" + "=" * 50)
    print("测试完成!")

    # 检查生成的文件
    files = ["no_alignment_g1.csv", "no_alignment_g2.csv",
             "with_alignment_g1.csv", "with_alignment_g2.csv",
             "deep_alignment_g1.csv", "deep_alignment_g2.csv"]

    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            print(f"{f}: {df.shape[0]} 行, {df.shape[1] - 1} 列")


if __name__ == "__main__":
    test_alignment_optimization()

# """
# 简洁高效的双网络共享参数GCN嵌入生成器
# 功能：处理两个NetworkX图对象，生成节点嵌入并保存为CSV文件
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import networkx as nx
# import pandas as pd
# import numpy as np
# from typing import List, Dict, Tuple
# import os
#
#
# class GCNLayer(nn.Module):
#     """GCN层"""
#
#     def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
#         super().__init__()
#         self.linear = nn.Linear(in_dim, out_dim, bias=False)
#         self.dropout = dropout
#         nn.init.xavier_uniform_(self.linear.weight)
#
#     def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
#         # 添加自环
#         adj = adj + torch.eye(adj.size(0), device=x.device)
#
#         # 计算归一化邻接矩阵
#         degree = adj.sum(dim=1, keepdim=True)
#         degree_sqrt = torch.sqrt(degree + 1e-8)
#         norm_adj = adj / (degree_sqrt * degree_sqrt.t() + 1e-8)
#
#         # 图卷积
#         x = self.linear(x)
#         x = torch.matmul(norm_adj, x)
#
#         return x
#
#
# class DualGCN(nn.Module):
#     """双网络共享参数GCN模型"""
#
#     def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, dropout: float = 0.2):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         self.dropout = dropout
#
#         # 构建隐藏层
#         current_dim = in_dim
#         for h_dim in hidden_dims:
#             self.layers.append(GCNLayer(current_dim, h_dim, dropout))
#             current_dim = h_dim
#
#         # 输出层
#         self.layers.append(GCNLayer(current_dim, out_dim, dropout=0.0))
#
#     def forward(self, x1: torch.Tensor, adj1: torch.Tensor,
#                 x2: torch.Tensor, adj2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         # 处理第一个图
#         h1 = x1
#         for layer in self.layers[:-1]:
#             h1 = layer(h1, adj1)
#             h1 = F.relu(h1)
#             h1 = F.dropout(h1, self.dropout, training=self.training)
#
#         # 处理第二个图
#         h2 = x2
#         for layer in self.layers[:-1]:
#             h2 = layer(h2, adj2)
#             h2 = F.relu(h2)
#             h2 = F.dropout(h2, self.dropout, training=self.training)
#
#         # 输出层
#         out1 = self.layers[-1](h1, adj1)
#         out2 = self.layers[-1](h2, adj2)
#
#         return out1, out2
#
#
# class DualGCNEmbedder:
#     """双网络GCN嵌入生成器"""
#
#     def __init__(self, hidden_dims: List[int] = [64, 32], out_dim: int = 16, dropout: float = 0.2):
#         """
#         参数:
#             hidden_dims: 隐藏层维度列表，控制GCN层数
#             out_dim: 输出嵌入维度
#             dropout: Dropout率
#         """
#         self.hidden_dims = hidden_dims
#         self.out_dim = out_dim
#         self.dropout = dropout
#         self.model = None
#         self.in_dim = None
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     def extract_features(self, G: nx.Graph) -> Tuple[torch.Tensor, List]:
#         """从NetworkX图中提取节点特征"""
#         node_ids = list(G.nodes())
#         features = []
#
#         for node in node_ids:
#             node_data = G.nodes[node]
#
#             # 获取特征
#             feat = None
#             for attr in ['features', 'feature', 'attr', 'attributes']:
#                 if attr in node_data:
#                     feat = node_data[attr]
#                     break
#
#             # 处理特征
#             if feat is None:
#                 feat = [G.degree(node), 0.0]  # 默认特征
#             elif isinstance(feat, (int, float)):
#                 feat = [float(feat)]
#             elif isinstance(feat, list):
#                 feat = [float(x) for x in feat]
#             elif isinstance(feat, str):
#                 try:
#                     # 尝试解析字符串
#                     if '[' in feat and ']' in feat:
#                         content = feat[feat.find('[') + 1:feat.find(']')]
#                         feat = [float(x.strip()) for x in content.split(',') if x.strip()]
#                     else:
#                         feat = [float(feat)]
#                 except:
#                     feat = [0.0]
#             else:
#                 feat = [0.0]
#
#             features.append(feat)
#
#         # 确定最大维度并填充
#         max_dim = max(len(f) for f in features) if features else 1
#         padded_features = [f + [0.0] * (max_dim - len(f)) if len(f) < max_dim else f[:max_dim]
#                            for f in features]
#
#         return torch.tensor(padded_features, dtype=torch.float), node_ids
#
#     def build_adjacency(self, G: nx.Graph, node_ids: List) -> torch.Tensor:
#         """构建邻接矩阵"""
#         n = len(node_ids)
#         node_index = {node: i for i, node in enumerate(node_ids)}
#         adj = torch.zeros((n, n), dtype=torch.float)
#
#         for u, v in G.edges():
#             if u in node_index and v in node_index:
#                 i, j = node_index[u], node_index[v]
#                 adj[i, j] = 1.0
#                 adj[j, i] = 1.0
#
#         return adj
#
#     def align_features(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """对齐两个图的特征维度"""
#         d1, d2 = x1.shape[1], x2.shape[1]
#
#         if d1 == d2:
#             return x1, x2
#
#         max_dim = max(d1, d2)
#
#         if d1 < max_dim:
#             padding = torch.zeros(x1.shape[0], max_dim - d1, dtype=x1.dtype)
#             x1 = torch.cat([x1, padding], dim=1)
#
#         if d2 < max_dim:
#             padding = torch.zeros(x2.shape[0], max_dim - d2, dtype=x2.dtype)
#             x2 = torch.cat([x2, padding], dim=1)
#
#         return x1, x2
#
#     def generate_embeddings(self, G1: nx.Graph, G2: nx.Graph) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
#         """为两个图生成节点嵌入"""
#         # 提取特征
#         x1, ids1 = self.extract_features(G1)
#         x2, ids2 = self.extract_features(G2)
#
#         # 构建邻接矩阵
#         adj1 = self.build_adjacency(G1, ids1)
#         adj2 = self.build_adjacency(G2, ids2)
#
#         # 对齐特征维度
#         x1, x2 = self.align_features(x1, x2)
#
#         # 移动到设备
#         x1, x2 = x1.to(self.device), x2.to(self.device)
#         adj1, adj2 = adj1.to(self.device), adj2.to(self.device)
#
#         # 初始化模型
#         if self.model is None:
#             self.in_dim = x1.shape[1]
#             self.model = DualGCN(
#                 in_dim=self.in_dim,
#                 hidden_dims=self.hidden_dims,
#                 out_dim=self.out_dim,
#                 dropout=self.dropout
#             ).to(self.device)
#
#         # 生成嵌入
#         self.model.eval()
#         with torch.no_grad():
#             emb1, emb2 = self.model(x1, adj1, x2, adj2)
#             emb1, emb2 = emb1.cpu(), emb2.cpu()
#
#         return emb1, emb2, ids1, ids2
#
#     def save_to_csv(self, embeddings: torch.Tensor, node_ids: List, filename: str):
#         """保存嵌入为CSV文件"""
#         # 确保目录存在
#         os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
#
#         # 准备数据
#         n_nodes, embed_dim = embeddings.shape
#         data = {'node_id': node_ids}
#
#         for i in range(embed_dim):
#             data[f'dim_{i + 1}'] = embeddings[:, i].numpy()
#
#         # 创建DataFrame并保存
#         df = pd.DataFrame(data)
#         df.to_csv(filename, index=False)
#
#         print(f"保存嵌入到: {filename} ({df.shape[0]} 个节点, {df.shape[1] - 1} 个维度)")
#
#     def generate_and_save(self, G1: nx.Graph, G2: nx.Graph,
#                           file1: str = "graph1_embeddings.csv",
#                           file2: str = "graph2_embeddings.csv"):
#         """主函数：生成并保存嵌入"""
#         print(f"处理图1: {G1.number_of_nodes()} 节点, {G1.number_of_edges()} 边")
#         print(f"处理图2: {G2.number_of_nodes()} 节点, {G2.number_of_edges()} 边")
#         print(f"GCN配置: 隐藏层={self.hidden_dims}, 输出维度={self.out_dim}")
#
#         # 生成嵌入
#         emb1, emb2, ids1, ids2 = self.generate_embeddings(G1, G2)
#
#         # 保存为CSV
#         self.save_to_csv(emb1, ids1, file1)
#         self.save_to_csv(emb2, ids2, file2)
#
#         return emb1, emb2
#
#
# def create_test_graph1() -> nx.Graph:
#     """创建测试图1"""
#     G = nx.karate_club_graph()
#     for node in G.nodes():
#         G.nodes[node]['features'] = [node * 0.1, node * 0.2, node * 0.3]
#     return G
#
#
# def create_test_graph2() -> nx.Graph:
#     """创建测试图2"""
#     G = nx.erdos_renyi_graph(20, 0.3, seed=42)
#     for node in G.nodes():
#         G.nodes[node]['features'] = [node, node ** 2, node % 5]
#     return G
#
#
# def test_dual_gcn_embedder():
#     """测试函数"""
#     print("测试双网络GCN嵌入生成器")
#     print("=" * 50)
#
#     # 创建测试图
#     G1 = create_test_graph1()
#     G2 = create_test_graph2()
#
#     # 测试1: 2层GCN
#     print("\n测试2层GCN...")
#     embedder1 = DualGCNEmbedder(
#         hidden_dims=[64],  # 1个隐藏层，共2层
#         out_dim=16,
#         dropout=0.2
#     )
#     emb1, emb2 = embedder1.generate_and_save(
#         G1, G2,
#         "test_2layer_g1.csv",
#         "test_2layer_g2.csv"
#     )
#
#     # 测试2: 3层GCN
#     print("\n测试3层GCN...")
#     embedder2 = DualGCNEmbedder(
#         hidden_dims=[128, 64],  # 2个隐藏层，共3层
#         out_dim=32,
#         dropout=0.3
#     )
#     emb1, emb2 = embedder2.generate_and_save(
#         G1, G2,
#         "test_3layer_g1.csv",
#         "test_3layer_g2.csv"
#     )
#
#     # 测试3: 4层GCN
#     print("\n测试4层GCN...")
#     embedder3 = DualGCNEmbedder(
#         hidden_dims=[256, 128, 64],  # 3个隐藏层，共4层
#         out_dim=64,
#         dropout=0.4
#     )
#     emb1, emb2 = embedder3.generate_and_save(
#         G1, G2,
#         "test_4layer_g1.csv",
#         "test_4layer_g2.csv"
#     )
#
#     print("\n" + "=" * 50)
#     print("测试完成!")
#
#     # 检查生成的文件
#     files = ["test_2layer_g1.csv", "test_2layer_g2.csv",
#              "test_3layer_g1.csv", "test_3layer_g2.csv",
#              "test_4layer_g1.csv", "test_4layer_g2.csv"]
#
#     for f in files:
#         if os.path.exists(f):
#             df = pd.read_csv(f)
#             print(f"{f}: {df.shape[0]} 行, {df.shape[1] - 1} 列")
#
#
# if __name__ == "__main__":
#     test_dual_gcn_embedder()
#
# # """
# # 增强的双网络共享参数GCN嵌入生成器
# # 添加节点对齐字典输入和对齐损失优化功能
# # """
# #
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torch.optim as optim
# # import networkx as nx
# # import numpy as np
# # import pandas as pd
# # import os
# # import re
# # from typing import List, Dict, Tuple, Optional, Set, Union
# # import warnings
# # from collections import defaultdict
# # import time
# #
# # warnings.filterwarnings("ignore", category=UserWarning)
# #
# #
# # class SharedGCNLayer(nn.Module):
# #     """共享参数的GCN层"""
# #
# #     def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
# #         super().__init__()
# #         self.in_features = in_features
# #         self.out_features = out_features
# #         self.dropout = dropout
# #
# #         # 线性变换
# #         self.linear = nn.Linear(in_features, out_features, bias=False)
# #
# #         # 初始化参数
# #         self._init_weights()
# #
# #     def _init_weights(self):
# #         """初始化权重"""
# #         nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
# #
# #     def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
# #         """
# #         前向传播
# #
# #         参数:
# #             x: 节点特征 [n_nodes, in_features]
# #             adj: 邻接矩阵 [n_nodes, n_nodes]
# #
# #         返回:
# #             更新后的节点特征 [n_nodes, out_features]
# #         """
# #         # 添加自环
# #         adj = adj + torch.eye(adj.size(0), device=adj.device)
# #
# #         # 计算度矩阵
# #         degree = torch.sum(adj, dim=1, keepdim=True)
# #         degree_sqrt = torch.sqrt(degree + 1e-8)  # 防止除零
# #
# #         # 对称归一化
# #         norm_adj = adj / (degree_sqrt * degree_sqrt.t() + 1e-8)
# #
# #         # 特征变换
# #         x = self.linear(x)
# #
# #         # 图卷积
# #         x = torch.matmul(norm_adj, x)
# #
# #         return x
# #
# #
# # class DualSharedGCN(nn.Module):
# #     """双网络共享参数的GCN模型"""
# #
# #     def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.2):
# #         """
# #         初始化双网络共享参数GCN
# #
# #         参数:
# #             input_dim: 输入特征维度
# #             hidden_dims: 隐藏层维度列表，例如[64, 32]表示两个隐藏层
# #             output_dim: 输出嵌入维度
# #             dropout: Dropout率
# #         """
# #         super().__init__()
# #         self.layers = nn.ModuleList()
# #         self.dropout = dropout
# #         self.num_layers = len(hidden_dims) + 1  # 隐藏层 + 输出层
# #
# #         # 构建隐藏层
# #         current_dim = input_dim
# #         for hidden_dim in hidden_dims:
# #             self.layers.append(
# #                 SharedGCNLayer(current_dim, hidden_dim, dropout)
# #             )
# #             current_dim = hidden_dim
# #
# #         # 输出层
# #         self.layers.append(
# #             SharedGCNLayer(current_dim, output_dim, dropout=0.0)
# #         )
# #
# #     def forward(self, features1: torch.Tensor, adj1: torch.Tensor,
# #                 features2: torch.Tensor, adj2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
# #         """
# #         前向传播（同时处理两个图）
# #
# #         参数:
# #             features1: 第一个图的节点特征 [n_nodes1, input_dim]
# #             adj1: 第一个图的邻接矩阵 [n_nodes1, n_nodes1]
# #             features2: 第二个图的节点特征 [n_nodes2, input_dim]
# #             adj2: 第二个图的邻接矩阵 [n_nodes2, n_nodes2]
# #
# #         返回:
# #             emb1: 第一个图的节点嵌入 [n_nodes1, output_dim]
# #             emb2: 第二个图的节点嵌入 [n_nodes2, output_dim]
# #         """
# #         # 处理第一个图
# #         x1 = features1
# #         for i, layer in enumerate(self.layers[:-1]):
# #             x1 = layer(x1, adj1)
# #             x1 = F.relu(x1)
# #             x1 = F.dropout(x1, p=self.dropout, training=self.training)
# #
# #         # 处理第二个图
# #         x2 = features2
# #         for i, layer in enumerate(self.layers[:-1]):
# #             x2 = layer(x2, adj2)
# #             x2 = F.relu(x2)
# #             x2 = F.dropout(x2, p=self.dropout, training=self.training)
# #
# #         # 应用输出层
# #         emb1 = self.layers[-1](x1, adj1)
# #         emb2 = self.layers[-1](x2, adj2)
# #
# #         return emb1, emb2
# #
# #
# # class AlignmentLoss(nn.Module):
# #     """对齐损失计算模块"""
# #
# #     def __init__(self, loss_type: str = 'mse', margin: float = 1.0):
# #         """
# #         初始化对齐损失
# #
# #         参数:
# #             loss_type: 损失类型，'mse'、'cosine' 或 'contrastive'
# #             margin: 对比损失的边界值
# #         """
# #         super().__init__()
# #         self.loss_type = loss_type
# #         self.margin = margin
# #         self.mse_loss = nn.MSELoss()
# #         self.cos_sim = nn.CosineSimilarity(dim=1)
# #
# #     def forward(self, emb1: torch.Tensor, emb2: torch.Tensor,
# #                 idx1: torch.Tensor, idx2: torch.Tensor) -> torch.Tensor:
# #         """
# #         计算对齐损失
# #
# #         参数:
# #             emb1: 第一个图的嵌入 [n_nodes1, embed_dim]
# #             emb2: 第二个图的嵌入 [n_nodes2, embed_dim]
# #             idx1: 第一个图中对齐节点的索引 [n_aligned_pairs]
# #             idx2: 第二个图中对齐节点的索引 [n_aligned_pairs]
# #
# #         返回:
# #             对齐损失值
# #         """
# #         # 获取对齐节点的嵌入
# #         aligned_emb1 = emb1[idx1]  # [n_aligned_pairs, embed_dim]
# #         aligned_emb2 = emb2[idx2]  # [n_aligned_pairs, embed_dim]
# #
# #         if self.loss_type == 'mse':
# #             # MSE损失：鼓励对齐节点嵌入相似
# #             return self.mse_loss(aligned_emb1, aligned_emb2)
# #
# #         elif self.loss_type == 'cosine':
# #             # 余弦相似度损失：鼓励对齐节点嵌入方向一致
# #             cos_sim = self.cos_sim(aligned_emb1, aligned_emb2)  # [n_aligned_pairs]
# #             # 我们希望余弦相似度接近1
# #             return torch.mean(1.0 - cos_sim)
# #
# #         elif self.loss_type == 'contrastive':
# #             # 对比损失：鼓励对齐节点接近，不相关节点远离
# #             pos_dist = F.pairwise_distance(aligned_emb1, aligned_emb2)  # 正样本对距离
# #             pos_loss = pos_dist.pow(2)
# #
# #             # 生成负样本对（随机采样）
# #             n_pairs = len(idx1)
# #             neg_idx1 = torch.randint(0, emb1.size(0), (n_pairs,), device=emb1.device)
# #             neg_idx2 = torch.randint(0, emb2.size(0), (n_pairs,), device=emb2.device)
# #
# #             neg_emb1 = emb1[neg_idx1]
# #             neg_emb2 = emb2[neg_idx2]
# #             neg_dist = F.pairwise_distance(neg_emb1, neg_emb2)
# #
# #             # 对比损失
# #             neg_loss = F.relu(self.margin - neg_dist).pow(2)
# #
# #             return torch.mean(pos_loss + neg_loss)
# #
# #         else:
# #             raise ValueError(f"不支持的损失类型: {self.loss_type}")
# #
# #
# # class DualSharedGCNEmbedder:
# #     """双网络共享参数GCN嵌入生成器（支持节点对齐优化）"""
# #
# #     def __init__(self, hidden_dims: List[int] = [64, 32], output_dim: int = 16, dropout: float = 0.2,
# #                  alignment_loss_type: str = 'mse', alignment_loss_weight: float = 1.0,
# #                  learning_rate: float = 0.01, debug_mode: bool = True):
# #         """
# #         初始化双网络共享参数GCN嵌入器（带对齐优化）
# #
# #         参数:
# #             hidden_dims: 隐藏层维度列表，控制GCN层数
# #             output_dim: 输出嵌入维度
# #             dropout: Dropout率
# #             alignment_loss_type: 对齐损失类型，'mse'、'cosine' 或 'contrastive'
# #             alignment_loss_weight: 对齐损失权重
# #             learning_rate: 学习率
# #             debug_mode: 调试模式，输出详细处理信息
# #         """
# #         self.hidden_dims = hidden_dims
# #         self.output_dim = output_dim
# #         self.dropout = dropout
# #         self.alignment_loss_type = alignment_loss_type
# #         self.alignment_loss_weight = alignment_loss_weight
# #         self.learning_rate = learning_rate
# #         self.debug_mode = debug_mode
# #
# #         # 模型将在首次使用时初始化
# #         self.model = None
# #         self.input_dim = None
# #         self.alignment_loss = None
# #         self.optimizer = None
# #
# #         # 设备
# #         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #
# #         print(f"双网络共享参数GCN配置: 隐藏层维度={hidden_dims}, 输出维度={output_dim}, 总层数={len(hidden_dims) + 1}")
# #         print(f"对齐损失配置: 类型={alignment_loss_type}, 权重={alignment_loss_weight}")
# #         print(f"优化器配置: 学习率={learning_rate}")
# #         print(f"设备: {self.device}")
# #         print(f"调试模式: {debug_mode}")
# #
# #     def extract_node_features(self, G: nx.Graph) -> Tuple[torch.Tensor, List]:
# #         """
# #         从NetworkX图中提取节点特征（修复节点丢失问题）
# #
# #         参数:
# #             G: NetworkX图对象
# #
# #         返回:
# #             features: 节点特征矩阵 [n_nodes, n_features]
# #             node_ids: 节点ID列表
# #         """
# #         node_ids = list(G.nodes())
# #         n_nodes = len(node_ids)
# #
# #         if self.debug_mode:
# #             print(f"  提取特征: 原始图有 {n_nodes} 个节点")
# #             print(f"  节点ID示例: {node_ids[:10]}")
# #
# #         # 收集所有节点的特征
# #         features_list = []
# #         valid_node_ids = []  # 记录实际处理的节点ID
# #         skipped_nodes = 0  # 记录跳过的节点数
# #
# #         for node in node_ids:
# #             node_data = G.nodes[node]
# #
# #             # 尝试获取特征
# #             feat = None
# #             for attr_name in ['features', 'feature', 'attr', 'attributes']:
# #                 if attr_name in node_data:
# #                     feat = node_data[attr_name]
# #                     if self.debug_mode and node in list(G.nodes())[:5]:  # 只输出前5个节点的特征信息
# #                         print(f"    节点 {node}: 从字段 '{attr_name}' 获取特征")
# #                     break
# #
# #             # 处理特征
# #             processed_feat = None
# #             if feat is None:
# #                 # 如果没有特征，使用度数和聚类系数
# #                 try:
# #                     degree = G.degree(node)
# #                     clustering = nx.clustering(G, node) if n_nodes > 1 else 0.0
# #                     processed_feat = [float(degree), float(clustering)]
# #                     if self.debug_mode and node in list(G.nodes())[:5]:
# #                         print(f"    节点 {node}: 无特征，使用度数和聚类系数: {processed_feat}")
# #                 except Exception as e:
# #                     if self.debug_mode:
# #                         print(f"    节点 {node}: 特征计算失败: {e}")
# #                     skipped_nodes += 1
# #                     continue
# #             elif isinstance(feat, (int, float)):
# #                 processed_feat = [float(feat)]
# #             elif isinstance(feat, list):
# #                 # 已经是列表，确保是数值
# #                 try:
# #                     processed_feat = [float(x) for x in feat]
# #                 except Exception as e:
# #                     if self.debug_mode:
# #                         print(f"    节点 {node}: 列表特征转换失败: {e}")
# #                     skipped_nodes += 1
# #                     continue
# #             elif isinstance(feat, str):
# #                 # 尝试解析字符串
# #                 try:
# #                     if '[' in feat and ']' in feat:
# #                         # 提取方括号内的内容
# #                         match = re.search(r'\[(.*?)\]', feat)
# #                         if match:
# #                             content = match.group(1)
# #                             processed_feat = [float(x.strip()) for x in content.split(',')]
# #                     else:
# #                         processed_feat = [float(feat)]
# #                 except Exception as e:
# #                     if self.debug_mode:
# #                         print(f"    节点 {node}: 字符串特征解析失败: {e}")
# #                     skipped_nodes += 1
# #                     continue
# #             else:
# #                 if self.debug_mode:
# #                     print(f"    节点 {node}: 未知特征类型: {type(feat)}")
# #                 skipped_nodes += 1
# #                 continue
# #
# #             # 添加到列表
# #             features_list.append(processed_feat)
# #             valid_node_ids.append(node)
# #
# #         if self.debug_mode:
# #             print(f"  成功处理 {len(valid_node_ids)} 个节点，跳过 {skipped_nodes} 个节点")
# #             if skipped_nodes > 0:
# #                 print(f"  警告: 跳过了 {skipped_nodes} 个节点的特征提取")
# #
# #         if not features_list:
# #             if self.debug_mode:
# #                 print("  错误: 没有提取到任何节点特征")
# #             return torch.tensor([], dtype=torch.float), []
# #
# #         # 确定最大特征维度
# #         max_dim = max(len(f) for f in features_list) if features_list else 1
# #
# #         if self.debug_mode:
# #             print(f"  最大特征维度: {max_dim}")
# #
# #         # 填充特征向量
# #         padded_features = []
# #         for f in features_list:
# #             if len(f) < max_dim:
# #                 padded_features.append(f + [0.0] * (max_dim - len(f)))
# #             else:
# #                 padded_features.append(f[:max_dim])
# #
# #         features_tensor = torch.tensor(padded_features, dtype=torch.float)
# #
# #         if self.debug_mode:
# #             print(f"  特征矩阵形状: {features_tensor.shape}")
# #             print(f"  有效节点ID数量: {len(valid_node_ids)}")
# #             if len(valid_node_ids) < n_nodes:
# #                 missing_nodes = set(node_ids) - set(valid_node_ids)
# #                 print(f"  缺失节点: {list(missing_nodes)[:10] if len(missing_nodes) > 10 else list(missing_nodes)}")
# #
# #         return features_tensor, valid_node_ids
# #
# #     def nx_to_adjacency(self, G: nx.Graph, node_ids: List) -> torch.Tensor:
# #         """
# #         将NetworkX图转换为邻接矩阵
# #
# #         参数:
# #             G: NetworkX图对象
# #             node_ids: 节点ID列表
# #
# #         返回:
# #             邻接矩阵 [n_nodes, n_nodes]
# #         """
# #         n = len(node_ids)
# #         node_index = {node: i for i, node in enumerate(node_ids)}
# #
# #         if self.debug_mode:
# #             print(f"  构建邻接矩阵: {n} 个节点")
# #
# #         # 创建邻接矩阵
# #         adj = torch.zeros((n, n), dtype=torch.float)
# #
# #         edges_added = 0
# #         for u, v in G.edges():
# #             if u in node_index and v in node_index:
# #                 i, j = node_index[u], node_index[v]
# #                 adj[i, j] = 1.0
# #                 adj[j, i] = 1.0  # 假设无向图
# #                 edges_added += 1
# #             else:
# #                 if self.debug_mode and edges_added < 10:  # 只输出前10条边的问题
# #                     if u not in node_index:
# #                         print(f"    警告: 边 ({u}, {v}) 中的节点 {u} 不在有效节点列表中")
# #                     if v not in node_index:
# #                         print(f"    警告: 边 ({u}, {v}) 中的节点 {v} 不在有效节点列表中")
# #
# #         if self.debug_mode:
# #             print(f"  添加了 {edges_added} 条边到邻接矩阵")
# #             print(f"  邻接矩阵形状: {adj.shape}")
# #
# #         return adj
# #
# #     def align_features(self, features1: torch.Tensor, features2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
# #         """
# #         对齐两个图的特征维度
# #
# #         参数:
# #             features1: 第一个图的特征 [n_nodes1, dim1]
# #             features2: 第二个图的特征 [n_nodes2, dim2]
# #
# #         返回:
# #             aligned_features1: 对齐后的第一个图特征
# #             aligned_features2: 对齐后的第二个图特征
# #         """
# #         dim1 = features1.shape[1]
# #         dim2 = features2.shape[1]
# #
# #         if self.debug_mode:
# #             print(f"  对齐特征维度: 图1={dim1}, 图2={dim2}")
# #
# #         if dim1 == dim2:
# #             if self.debug_mode:
# #                 print(f"  特征维度相同，无需对齐")
# #             return features1, features2
# #
# #         # 以较大的维度为准
# #         max_dim = max(dim1, dim2)
# #
# #         if self.debug_mode:
# #             print(f"  对齐到最大维度: {max_dim}")
# #
# #         # 对齐第一个图
# #         if dim1 < max_dim:
# #             padding = torch.zeros(features1.shape[0], max_dim - dim1, dtype=features1.dtype)
# #             features1 = torch.cat([features1, padding], dim=1)
# #             if self.debug_mode:
# #                 print(f"  图1特征从 {dim1} 填充到 {max_dim}")
# #
# #         # 对齐第二个图
# #         if dim2 < max_dim:
# #             padding = torch.zeros(features2.shape[0], max_dim - dim2, dtype=features2.dtype)
# #             features2 = torch.cat([features2, padding], dim=1)
# #             if self.debug_mode:
# #                 print(f"  图2特征从 {dim2} 填充到 {max_dim}")
# #
# #         return features1, features2
# #
# #     def get_alignment_indices(self, node_ids1: List, node_ids2: List, alignment_dict: Dict) -> Tuple[List, List]:
# #         """
# #         根据对齐字典获取节点索引
# #
# #         参数:
# #             node_ids1: 第一个图的节点ID列表
# #             node_ids2: 第二个图的节点ID列表
# #             alignment_dict: 节点对齐字典，键为图1节点ID，值为图2节点ID
# #
# #         返回:
# #             idx1: 第一个图中对齐节点的索引列表
# #             idx2: 第二个图中对齐节点的索引列表
# #         """
# #         # 创建节点ID到索引的映射
# #         idx_map1 = {node_id: idx for idx, node_id in enumerate(node_ids1)}
# #         idx_map2 = {node_id: idx for idx, node_id in enumerate(node_ids2)}
# #
# #         # 获取对齐节点的索引
# #         idx1, idx2 = [], []
# #         for node1, node2 in alignment_dict.items():
# #             if node1 in idx_map1 and node2 in idx_map2:
# #                 idx1.append(idx_map1[node1])
# #                 idx2.append(idx_map2[node2])
# #             else:
# #                 if self.debug_mode:
# #                     if node1 not in idx_map1:
# #                         print(f"  警告: 对齐节点 {node1} 不在图1的有效节点中")
# #                     if node2 not in idx_map2:
# #                         print(f"  警告: 对齐节点 {node2} 不在图2的有效节点中")
# #
# #         if self.debug_mode:
# #             print(f"  对齐节点对: {len(idx1)} 对 (总共 {len(alignment_dict)} 对)")
# #             if idx1:
# #                 print(f"  对齐节点示例: 图1节点 {node_ids1[idx1[0]]} ↔ 图2节点 {node_ids2[idx2[0]]}")
# #
# #         return idx1, idx2
# #
# #     def optimize_with_alignment(self, G1: nx.Graph, G2: nx.Graph, alignment_dict: Dict,
# #                                 epochs: int = 100, patience: int = 10) -> Dict:
# #         """
# #         使用对齐损失优化模型
# #
# #         参数:
# #             G1: 第一个NetworkX图对象
# #             G2: 第二个NetworkX图对象
# #             alignment_dict: 节点对齐字典，键为图1节点ID，值为图2节点ID
# #             epochs: 训练轮数
# #             patience: 早停耐心值
# #
# #         返回:
# #             training_stats: 训练统计信息
# #         """
# #         print(f"\n{'=' * 60}")
# #         print(f"开始对齐优化训练...")
# #         print(f"训练轮数: {epochs}, 对齐节点对: {len(alignment_dict)}")
# #
# #         # 提取特征
# #         features1, node_ids1 = self.extract_node_features(G1)
# #         features2, node_ids2 = self.extract_node_features(G2)
# #
# #         n_nodes1 = len(node_ids1)
# #         n_nodes2 = len(node_ids2)
# #
# #         if self.debug_mode:
# #             print(f"\n特征提取结果:")
# #             print(f"  图1: {n_nodes1} 个节点 (原始: {G1.number_of_nodes()})")
# #             print(f"  图2: {n_nodes2} 个节点 (原始: {G2.number_of_nodes()})")
# #
# #         if n_nodes1 == 0 or n_nodes2 == 0:
# #             print(f"  错误: 至少一个图没有有效节点")
# #             return {}
# #
# #         # 转换为邻接矩阵
# #         adj1 = self.nx_to_adjacency(G1, node_ids1)
# #         adj2 = self.nx_to_adjacency(G2, node_ids2)
# #
# #         # 对齐特征维度
# #         features1, features2 = self.align_features(features1, features2)
# #
# #         if self.debug_mode:
# #             print(f"\n特征对齐结果:")
# #             print(f"  图1特征形状: {features1.shape}")
# #             print(f"  图2特征形状: {features2.shape}")
# #             print(f"  图1邻接矩阵形状: {adj1.shape}")
# #             print(f"  图2邻接矩阵形状: {adj2.shape}")
# #
# #         # 获取对齐节点的索引
# #         idx1, idx2 = self.get_alignment_indices(node_ids1, node_ids2, alignment_dict)
# #
# #         if len(idx1) == 0:
# #             print(f"  警告: 没有有效的对齐节点对，跳过训练")
# #             return {}
# #
# #         # 移动到设备
# #         features1 = features1.to(self.device)
# #         features2 = features2.to(self.device)
# #         adj1 = adj1.to(self.device)
# #         adj2 = adj2.to(self.device)
# #         idx1_tensor = torch.tensor(idx1, dtype=torch.long, device=self.device)
# #         idx2_tensor = torch.tensor(idx2, dtype=torch.long, device=self.device)
# #
# #         # 初始化模型（如果需要）
# #         if self.model is None:
# #             self.input_dim = features1.shape[1]
# #             if self.debug_mode:
# #                 print(f"\n初始化模型，输入维度: {self.input_dim}")
# #             self.model = DualSharedGCN(
# #                 input_dim=self.input_dim,
# #                 hidden_dims=self.hidden_dims,
# #                 output_dim=self.output_dim,
# #                 dropout=self.dropout
# #             ).to(self.device)
# #
# #         # 初始化对齐损失
# #         if self.alignment_loss is None:
# #             self.alignment_loss = AlignmentLoss(loss_type=self.alignment_loss_type)
# #
# #         # 初始化优化器
# #         if self.optimizer is None:
# #             self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
# #
# #         # 训练模型
# #         self.model.train()
# #         best_loss = float('inf')
# #         patience_counter = 0
# #         loss_history = []
# #
# #         print(f"\n开始训练...")
# #         start_time = time.time()
# #
# #         for epoch in range(epochs):
# #             self.optimizer.zero_grad()
# #
# #             # 前向传播
# #             emb1, emb2 = self.model(features1, adj1, features2, adj2)
# #
# #             # 计算对齐损失
# #             alignment_loss_value = self.alignment_loss(emb1, emb2, idx1_tensor, idx2_tensor)
# #             total_loss = alignment_loss_value * self.alignment_loss_weight
# #
# #             # 反向传播
# #             total_loss.backward()
# #             self.optimizer.step()
# #
# #             loss_history.append(total_loss.item())
# #
# #             # 打印训练信息
# #             if (epoch + 1) % 10 == 0 or epoch == 0:
# #                 print(f"  Epoch {epoch + 1:3d}/{epochs} | Loss: {total_loss.item():.6f}")
# #
# #             # 早停检查
# #             if total_loss.item() < best_loss:
# #                 best_loss = total_loss.item()
# #                 patience_counter = 0
# #             else:
# #                 patience_counter += 1
# #                 if patience_counter >= patience:
# #                     print(f"  早停在第 {epoch + 1} 轮")
# #                     break
# #
# #         training_time = time.time() - start_time
# #         print(f"训练完成! 总时间: {training_time:.2f}秒, 最终损失: {total_loss.item():.6f}")
# #
# #         # 切换到评估模式
# #         self.model.eval()
# #
# #         # 计算对齐嵌入的相似度
# #         with torch.no_grad():
# #             emb1, emb2 = self.model(features1, adj1, features2, adj2)
# #             aligned_emb1 = emb1[idx1_tensor]
# #             aligned_emb2 = emb2[idx2_tensor]
# #
# #             # 计算对齐节点嵌入的相似度
# #             mse = F.mse_loss(aligned_emb1, aligned_emb2).item()
# #             cos_sim = F.cosine_similarity(aligned_emb1, aligned_emb2).mean().item()
# #
# #             if self.debug_mode:
# #                 print(f"\n对齐嵌入统计:")
# #                 print(f"  MSE距离: {mse:.6f}")
# #                 print(f"  余弦相似度: {cos_sim:.6f}")
# #
# #         # 训练统计信息
# #         training_stats = {
# #             'epochs_trained': epoch + 1,
# #             'total_time': training_time,
# #             'final_loss': total_loss.item(),
# #             'best_loss': best_loss,
# #             'aligned_pairs': len(idx1),
# #             'mse_distance': mse,
# #             'cosine_similarity': cos_sim,
# #             'loss_history': loss_history
# #         }
# #
# #         return training_stats
# #
# #     def generate_embeddings(self, G1: nx.Graph, G2: nx.Graph) -> Tuple[Dict, Dict, torch.Tensor, torch.Tensor]:
# #         """
# #         为两个图生成节点嵌入（共享参数模型）
# #
# #         参数:
# #             G1: 第一个NetworkX图对象
# #             G2: 第二个NetworkX图对象
# #
# #         返回:
# #             emb1_dict: 第一个图的节点嵌入字典
# #             emb2_dict: 第二个图的节点嵌入字典
# #             emb1_tensor: 第一个图的嵌入张量
# #             emb2_tensor: 第二个图的嵌入张量
# #         """
# #         print(f"\n{'=' * 60}")
# #         print(f"开始生成嵌入...")
# #
# #         # 提取特征
# #         features1, node_ids1 = self.extract_node_features(G1)
# #         features2, node_ids2 = self.extract_node_features(G2)
# #
# #         n_nodes1 = len(node_ids1)
# #         n_nodes2 = len(node_ids2)
# #
# #         if self.debug_mode:
# #             print(f"\n特征提取结果:")
# #             print(f"  图1: {n_nodes1} 个节点 (原始: {G1.number_of_nodes()})")
# #             print(f"  图2: {n_nodes2} 个节点 (原始: {G2.number_of_nodes()})")
# #
# #         if n_nodes1 == 0 or n_nodes2 == 0:
# #             if self.debug_mode:
# #                 print(f"  错误: 至少一个图没有有效节点")
# #             return {}, {}, torch.empty(0, self.output_dim), torch.empty(0, self.output_dim)
# #
# #         # 转换为邻接矩阵
# #         adj1 = self.nx_to_adjacency(G1, node_ids1)
# #         adj2 = self.nx_to_adjacency(G2, node_ids2)
# #
# #         # 对齐特征维度
# #         features1, features2 = self.align_features(features1, features2)
# #
# #         if self.debug_mode:
# #             print(f"\n特征对齐结果:")
# #             print(f"  图1特征形状: {features1.shape}")
# #             print(f"  图2特征形状: {features2.shape}")
# #             print(f"  图1邻接矩阵形状: {adj1.shape}")
# #             print(f"  图2邻接矩阵形状: {adj2.shape}")
# #
# #         # 移动到设备
# #         features1 = features1.to(self.device)
# #         features2 = features2.to(self.device)
# #         adj1 = adj1.to(self.device)
# #         adj2 = adj2.to(self.device)
# #
# #         # 初始化模型（如果需要）
# #         if self.model is None:
# #             self.input_dim = features1.shape[1]
# #             if self.debug_mode:
# #                 print(f"\n初始化模型，输入维度: {self.input_dim}")
# #             self.model = DualSharedGCN(
# #                 input_dim=self.input_dim,
# #                 hidden_dims=self.hidden_dims,
# #                 output_dim=self.output_dim,
# #                 dropout=self.dropout
# #             ).to(self.device)
# #
# #         # 生成嵌入
# #         self.model.eval()
# #         with torch.no_grad():
# #             emb1, emb2 = self.model(features1, adj1, features2, adj2)
# #             emb1 = emb1.cpu()
# #             emb2 = emb2.cpu()
# #
# #         if self.debug_mode:
# #             print(f"\n嵌入生成结果:")
# #             print(f"  图1嵌入形状: {emb1.shape}")
# #             print(f"  图2嵌入形状: {emb2.shape}")
# #
# #             # 检查嵌入是否为NaN或Inf
# #             if torch.isnan(emb1).any() or torch.isinf(emb1).any():
# #                 print(f"  警告: 图1嵌入包含NaN或Inf值")
# #             if torch.isnan(emb2).any() or torch.isinf(emb2).any():
# #                 print(f"  警告: 图2嵌入包含NaN或Inf值")
# #
# #         # 创建节点ID到嵌入的映射
# #         emb1_dict = {node_id: emb.numpy() for node_id, emb in zip(node_ids1, emb1)}
# #         emb2_dict = {node_id: emb.numpy() for node_id, emb in zip(node_ids2, emb2)}
# #
# #         if self.debug_mode:
# #             print(f"\n嵌入字典大小:")
# #             print(f"  图1嵌入字典: {len(emb1_dict)} 个节点")
# #             print(f"  图2嵌入字典: {len(emb2_dict)} 个节点")
# #             print(f"{'=' * 60}")
# #
# #         return emb1_dict, emb2_dict, emb1, emb2
# #
# #     def save_embeddings_to_csv(self, embeddings: torch.Tensor, node_ids: List, filename: str) -> None:
# #         """
# #         将嵌入保存为CSV文件
# #
# #         参数:
# #             embeddings: 嵌入张量 [n_nodes, embed_dim]
# #             node_ids: 节点ID列表
# #             filename: 输出文件名
# #         """
# #         # 确保目录存在
# #         if '/' in filename or '\\' in filename:
# #             os.makedirs(os.path.dirname(filename), exist_ok=True)
# #
# #         # 获取嵌入维度
# #         n_nodes, embed_dim = embeddings.shape
# #
# #         if self.debug_mode:
# #             print(f"\n保存嵌入到文件: {filename}")
# #             print(f"  嵌入张量形状: {embeddings.shape}")
# #             print(f"  节点ID数量: {len(node_ids)}")
# #
# #         # 检查节点ID和嵌入数量是否一致
# #         if len(node_ids) != n_nodes:
# #             print(f"  警告: 节点ID数量 ({len(node_ids)}) 与嵌入数量 ({n_nodes}) 不匹配")
# #             # 取较小值
# #             min_len = min(len(node_ids), n_nodes)
# #             node_ids = node_ids[:min_len]
# #             embeddings = embeddings[:min_len]
# #             print(f"  已截断到 {min_len} 个节点")
# #
# #         # 创建DataFrame
# #         data = []
# #         for i, node_id in enumerate(node_ids):
# #             row = {'node_id': node_id}
# #             for j in range(embed_dim):
# #                 row[f'dim_{j + 1}'] = embeddings[i, j].item()
# #             data.append(row)
# #
# #         df = pd.DataFrame(data)
# #
# #         # 保存为CSV
# #         df.to_csv(filename, index=False)
# #         print(f"嵌入已保存到: {filename}")
# #         print(f"  节点数: {df.shape[0]}, 嵌入维度: {df.shape[1] - 1}")
# #
# #         if self.debug_mode:
# #             # 验证保存的数据
# #             loaded_df = pd.read_csv(filename)
# #             print(f"  验证: 从文件加载了 {loaded_df.shape[0]} 行数据")
# #
# #             # 检查是否有NaN值
# #             nan_count = loaded_df.isna().sum().sum()
# #             if nan_count > 0:
# #                 print(f"  警告: 文件中包含 {nan_count} 个NaN值")
# #
# #     def generate_and_save_embeddings(self, G1: nx.Graph, G2: nx.Graph,
# #                                      output_file1: str = "graph1_embeddings.csv",
# #                                      output_file2: str = "graph2_embeddings.csv",
# #                                      alignment_dict: Optional[Dict] = None,
# #                                      train_epochs: int = 0) -> Tuple[Dict, Dict, Dict]:
# #         """
# #         为两个图生成节点嵌入并保存为CSV文件（可选对齐优化）
# #
# #         参数:
# #             G1: 第一个NetworkX图对象
# #             G2: 第二个NetworkX图对象
# #             output_file1: 第一个图的嵌入输出文件
# #             output_file2: 第二个图的嵌入输出文件
# #             alignment_dict: 节点对齐字典，键为图1节点ID，值为图2节点ID
# #             train_epochs: 训练轮数，0表示不训练
# #
# #         返回:
# #             emb1_dict: 第一个图的节点嵌入字典
# #             emb2_dict: 第二个图的节点嵌入字典
# #             training_stats: 训练统计信息字典
# #         """
# #         print(f"\n{'=' * 60}")
# #         print(f"处理第一个图: {G1.number_of_nodes()} 个节点, {G1.number_of_edges()} 条边")
# #         print(f"处理第二个图: {G2.number_of_nodes()} 个节点, {G2.number_of_edges()} 条边")
# #         print(f"使用共享参数的双网络GCN模型")
# #
# #         training_stats = {}
# #
# #         # 对齐优化训练
# #         if alignment_dict is not None and train_epochs > 0:
# #             print(f"\n使用节点对齐进行模型优化...")
# #             print(f"对齐节点对: {len(alignment_dict)} 对")
# #             print(f"训练轮数: {train_epochs}")
# #
# #             training_stats = self.optimize_with_alignment(
# #                 G1, G2, alignment_dict, epochs=train_epochs
# #             )
# #
# #         elif alignment_dict is not None and train_epochs == 0:
# #             print(f"\n对齐字典已提供但训练轮数为0，跳过对齐优化")
# #         else:
# #             print(f"\n未提供对齐字典，跳过对齐优化")
# #
# #         # 为图生成嵌入
# #         print("\n生成两个图的节点嵌入...")
# #         emb1_dict, emb2_dict, emb1_tensor, emb2_tensor = self.generate_embeddings(G1, G2)
# #
# #         # 获取节点ID列表
# #         node_ids1 = list(G1.nodes())
# #         node_ids2 = list(G2.nodes())
# #
# #         # 保存嵌入
# #         print(f"\n保存第一个图的嵌入到: {output_file1}")
# #         self.save_embeddings_to_csv(emb1_tensor, node_ids1, output_file1)
# #
# #         print(f"\n保存第二个图的嵌入到: {output_file2}")
# #         self.save_embeddings_to_csv(emb2_tensor, node_ids2, output_file2)
# #
# #         # 计算对齐嵌入的相似度（如果有对齐字典）
# #         if alignment_dict is not None and emb1_dict and emb2_dict:
# #             print(f"\n计算对齐嵌入的相似度...")
# #             valid_pairs = 0
# #             similarities = []
# #
# #             for node1, node2 in alignment_dict.items():
# #                 if node1 in emb1_dict and node2 in emb2_dict:
# #                     vec1 = torch.tensor(emb1_dict[node1])
# #                     vec2 = torch.tensor(emb2_dict[node2])
# #                     similarity = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
# #                     similarities.append(similarity)
# #                     valid_pairs += 1
# #
# #             if similarities:
# #                 avg_similarity = sum(similarities) / len(similarities)
# #                 min_similarity = min(similarities)
# #                 max_similarity = max(similarities)
# #
# #                 print(f"  对齐嵌入统计:")
# #                 print(f"    有效对齐对: {valid_pairs}/{len(alignment_dict)}")
# #                 print(f"    平均余弦相似度: {avg_similarity:.6f}")
# #                 print(f"    最小余弦相似度: {min_similarity:.6f}")
# #                 print(f"    最大余弦相似度: {max_similarity:.6f}")
# #
# #                 training_stats['post_alignment_similarity'] = {
# #                     'valid_pairs': valid_pairs,
# #                     'avg_similarity': avg_similarity,
# #                     'min_similarity': min_similarity,
# #                     'max_similarity': max_similarity
# #                 }
# #
# #         return emb1_dict, emb2_dict, training_stats
# #
# #
# # def create_test_graph1() -> nx.Graph:
# #     """创建测试图1"""
# #     G = nx.karate_club_graph()
# #
# #     # 添加节点特征
# #     for node in G.nodes():
# #         # 为节点添加不同类型的特征
# #         if node % 3 == 0:
# #             G.nodes[node]['features'] = [node * 0.1, node * 0.2, node * 0.3]
# #         elif node % 3 == 1:
# #             G.nodes[node]['features'] = f"[{node * 2}, {node * 3}]"
# #         else:
# #             G.nodes[node]['features'] = float(node)
# #
# #     return G
# #
# #
# # def create_test_graph2() -> nx.Graph:
# #     """创建测试图2"""
# #     G = nx.erdos_renyi_graph(20, 0.3, seed=42)
# #
# #     # 添加节点特征
# #     for node in G.nodes():
# #         # 为节点添加不同类型的特征
# #         if node % 4 == 0:
# #             G.nodes[node]['features'] = [node, node ** 2, node ** 3, node % 5]
# #         elif node % 4 == 1:
# #             G.nodes[node]['features'] = float(node) * 0.5
# #         elif node % 4 == 2:
# #             G.nodes[node]['features'] = f"feature is [{node}, {node * 2}]"
# #         else:
# #             G.nodes[node]['features'] = [float(node)]
# #
# #     return G
# #
# #
# # def create_alignment_dict(G1: nx.Graph, G2: nx.Graph, n_pairs: int = 10) -> Dict:
# #     """创建测试用的节点对齐字典"""
# #     # 随机选择一些节点对进行对齐
# #     nodes1 = list(G1.nodes())
# #     nodes2 = list(G2.nodes())
# #
# #     # 确保节点数量足够
# #     n1, n2 = min(len(nodes1), n_pairs), min(len(nodes2), n_pairs)
# #
# #     # 创建对齐字典
# #     alignment_dict = {}
# #     for i in range(n1):
# #         if i < len(nodes1) and i < len(nodes2):
# #             alignment_dict[nodes1[i]] = nodes2[i]
# #
# #     return alignment_dict
# #
# #
# # def test_alignment_optimization():
# #     """测试对齐优化功能"""
# #     print("=" * 60)
# #     print("测试带对齐优化的双网络共享参数GCN嵌入生成器")
# #     print("=" * 60)
# #
# #     # 创建测试图
# #     print("\n1. 创建测试图...")
# #     G1 = create_test_graph1()
# #     G2 = create_test_graph2()
# #     print(f"  图1: {G1.number_of_nodes()} 个节点, {G1.number_of_edges()} 条边")
# #     print(f"  图2: {G2.number_of_nodes()} 个节点, {G2.number_of_edges()} 条边")
# #
# #     # 创建节点对齐字典
# #     print("\n2. 创建节点对齐字典...")
# #     alignment_dict = create_alignment_dict(G1, G2, n_pairs=8)
# #     print(f"  对齐节点对数量: {len(alignment_dict)}")
# #     print(f"  对齐节点对示例: {list(alignment_dict.items())[:5]}")
# #
# #     # 测试1: 无对齐优化的GCN
# #     print("\n" + "=" * 40)
# #     print("3. 测试无对齐优化的GCN...")
# #     print("=" * 40)
# #     try:
# #         embedder_no_align = DualSharedGCNEmbedder(
# #             hidden_dims=[64, 32],
# #             output_dim=16,
# #             dropout=0.2,
# #             debug_mode=False
# #         )
# #
# #         emb1_dict_no, emb2_dict_no, stats_no = embedder_no_align.generate_and_save_embeddings(
# #             G1, G2,
# #             output_file1="no_alignment_graph1.csv",
# #             output_file2="no_alignment_graph2.csv",
# #             alignment_dict=alignment_dict,
# #             train_epochs=0
# #         )
# #
# #         print(f"\n  成功生成嵌入:")
# #         print(f"    图1: {len(emb1_dict_no)} 个节点的嵌入")
# #         print(f"    图2: {len(emb2_dict_no)} 个节点的嵌入")
# #
# #     except Exception as e:
# #         print(f"  测试1失败: {e}")
# #         import traceback
# #         traceback.print_exc()
# #
# #     # 测试2: 带对齐优化的GCN (MSE损失)
# #     print("\n" + "=" * 40)
# #     print("4. 测试带MSE对齐优化的GCN...")
# #     print("=" * 40)
# #     try:
# #         embedder_mse = DualSharedGCNEmbedder(
# #             hidden_dims=[64, 32],
# #             output_dim=16,
# #             dropout=0.2,
# #             alignment_loss_type='mse',
# #             alignment_loss_weight=1.0,
# #             learning_rate=0.01,
# #             debug_mode=False
# #         )
# #
# #         emb1_dict_mse, emb2_dict_mse, stats_mse = embedder_mse.generate_and_save_embeddings(
# #             G1, G2,
# #             output_file1="mse_alignment_graph1.csv",
# #             output_file2="mse_alignment_graph2.csv",
# #             alignment_dict=alignment_dict,
# #             train_epochs=50
# #         )
# #
# #         print(f"\n  MSE对齐优化统计:")
# #         if stats_mse:
# #             print(f"    训练轮数: {stats_mse.get('epochs_trained', 0)}")
# #             print(f"    最终损失: {stats_mse.get('final_loss', 0):.6f}")
# #             print(f"    对齐节点对: {stats_mse.get('aligned_pairs', 0)}")
# #
# #             if 'post_alignment_similarity' in stats_mse:
# #                 sim = stats_mse['post_alignment_similarity']
# #                 print(f"    对齐嵌入余弦相似度:")
# #                 print(f"      有效对齐对: {sim.get('valid_pairs', 0)}")
# #                 print(f"      平均相似度: {sim.get('avg_similarity', 0):.6f}")
# #
# #     except Exception as e:
# #         print(f"  测试2失败: {e}")
# #         import traceback
# #         traceback.print_exc()
# #
# #     # 测试3: 带对齐优化的GCN (余弦损失)
# #     print("\n" + "=" * 40)
# #     print("5. 测试带余弦对齐优化的GCN...")
# #     print("=" * 40)
# #     try:
# #         embedder_cos = DualSharedGCNEmbedder(
# #             hidden_dims=[64, 32],
# #             output_dim=16,
# #             dropout=0.2,
# #             alignment_loss_type='cosine',
# #             alignment_loss_weight=1.0,
# #             learning_rate=0.01,
# #             debug_mode=False
# #         )
# #
# #         emb1_dict_cos, emb2_dict_cos, stats_cos = embedder_cos.generate_and_save_embeddings(
# #             G1, G2,
# #             output_file1="cosine_alignment_graph1.csv",
# #             output_file2="cosine_alignment_graph2.csv",
# #             alignment_dict=alignment_dict,
# #             train_epochs=50
# #         )
# #
# #         print(f"\n  余弦对齐优化统计:")
# #         if stats_cos:
# #             print(f"    训练轮数: {stats_cos.get('epochs_trained', 0)}")
# #             print(f"    最终损失: {stats_cos.get('final_loss', 0):.6f}")
# #             print(f"    对齐节点对: {stats_cos.get('aligned_pairs', 0)}")
# #
# #             if 'post_alignment_similarity' in stats_cos:
# #                 sim = stats_cos['post_alignment_similarity']
# #                 print(f"    对齐嵌入余弦相似度:")
# #                 print(f"      有效对齐对: {sim.get('valid_pairs', 0)}")
# #                 print(f"      平均相似度: {sim.get('avg_similarity', 0):.6f}")
# #
# #     except Exception as e:
# #         print(f"  测试3失败: {e}")
# #         import traceback
# #         traceback.print_exc()
# #
# #     # 测试4: 带对齐优化的GCN (对比损失)
# #     print("\n" + "=" * 40)
# #     print("6. 测试带对比对齐优化的GCN...")
# #     print("=" * 40)
# #     try:
# #         embedder_contrast = DualSharedGCNEmbedder(
# #             hidden_dims=[64, 32],
# #             output_dim=16,
# #             dropout=0.2,
# #             alignment_loss_type='contrastive',
# #             alignment_loss_weight=1.0,
# #             learning_rate=0.01,
# #             debug_mode=False
# #         )
# #
# #         emb1_dict_contrast, emb2_dict_contrast, stats_contrast = embedder_contrast.generate_and_save_embeddings(
# #             G1, G2,
# #             output_file1="contrastive_alignment_graph1.csv",
# #             output_file2="contrastive_alignment_graph2.csv",
# #             alignment_dict=alignment_dict,
# #             train_epochs=50
# #         )
# #
# #         print(f"\n  对比对齐优化统计:")
# #         if stats_contrast:
# #             print(f"    训练轮数: {stats_contrast.get('epochs_trained', 0)}")
# #             print(f"    最终损失: {stats_contrast.get('final_loss', 0):.6f}")
# #             print(f"    对齐节点对: {stats_contrast.get('aligned_pairs', 0)}")
# #
# #             if 'post_alignment_similarity' in stats_contrast:
# #                 sim = stats_contrast['post_alignment_similarity']
# #                 print(f"    对齐嵌入余弦相似度:")
# #                 print(f"      有效对齐对: {sim.get('valid_pairs', 0)}")
# #                 print(f"      平均相似度: {sim.get('avg_similarity', 0):.6f}")
# #
# #     except Exception as e:
# #         print(f"  测试4失败: {e}")
# #         import traceback
# #         traceback.print_exc()
# #
# #     print("\n" + "=" * 60)
# #     print("测试完成!")
# #     print("=" * 60)
# #
# #     # 显示生成的文件
# #     print("\n生成的文件:")
# #     for filename in [
# #         "no_alignment_graph1.csv", "no_alignment_graph2.csv",
# #         "mse_alignment_graph1.csv", "mse_alignment_graph2.csv",
# #         "cosine_alignment_graph1.csv", "cosine_alignment_graph2.csv",
# #         "contrastive_alignment_graph1.csv", "contrastive_alignment_graph2.csv"
# #     ]:
# #         if os.path.exists(filename):
# #             df = pd.read_csv(filename)
# #             print(f"  {filename}: {df.shape[0]} 行, {df.shape[1] - 1} 个维度")
# #
# #
# # def test_alignment_effectiveness():
# #     """测试对齐优化的有效性"""
# #     print("\n" + "=" * 60)
# #     print("测试对齐优化有效性")
# #     print("=" * 60)
# #
# #     # 创建两个相似的图
# #     G1 = nx.erdos_renyi_graph(30, 0.3, seed=1)
# #     G2 = nx.erdos_renyi_graph(30, 0.3, seed=2)
# #
# #     # 添加相似的节点特征
# #     for node in G1.nodes():
# #         G1.nodes[node]['features'] = [float(node), float(node ** 2), float(node % 5)]
# #
# #     for node in G2.nodes():
# #         G2.nodes[node]['features'] = [float(node), float(node ** 2), float(node % 5)]
# #
# #     # 创建完美的对齐字典（前20个节点完全对齐）
# #     alignment_dict = {}
# #     for i in range(20):
# #         alignment_dict[i] = i
# #
# #     print(f"图1: {G1.number_of_nodes()} 个节点, {G1.number_of_edges()} 条边")
# #     print(f"图2: {G2.number_of_nodes()} 个节点, {G2.number_of_edges()} 条边")
# #     print(f"对齐节点对: {len(alignment_dict)}")
# #
# #     # 测试带对齐优化的GCN
# #     print("\n训练带对齐优化的模型...")
# #     embedder = DualSharedGCNEmbedder(
# #         hidden_dims=[32, 16],
# #         output_dim=8,
# #         dropout=0.1,
# #         alignment_loss_type='cosine',
# #         alignment_loss_weight=1.0,
# #         learning_rate=0.01,
# #         debug_mode=False
# #     )
# #
# #     # 训练模型
# #     training_stats = embedder.optimize_with_alignment(
# #         G1, G2, alignment_dict, epochs=100, patience=20
# #     )
# #
# #     if training_stats:
# #         print(f"\n训练统计:")
# #         print(f"  训练轮数: {training_stats.get('epochs_trained', 0)}")
# #         print(f"  最终损失: {training_stats.get('final_loss', 0):.6f}")
# #         print(f"  MSE距离: {training_stats.get('mse_distance', 0):.6f}")
# #         print(f"  余弦相似度: {training_stats.get('cosine_similarity', 0):.6f}")
# #
# #         # 生成并保存嵌入
# #         emb1_dict, emb2_dict, _ = embedder.generate_and_save_embeddings(
# #             G1, G2,
# #             output_file1="test_alignment1.csv",
# #             output_file2="test_alignment2.csv"
# #         )
# #
# #         # 计算对齐节点的相似度
# #         similarities = []
# #         for node1, node2 in alignment_dict.items():
# #             if node1 in emb1_dict and node2 in emb2_dict:
# #                 vec1 = torch.tensor(emb1_dict[node1])
# #                 vec2 = torch.tensor(emb2_dict[node2])
# #                 similarity = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
# #                 similarities.append(similarity)
# #
# #         if similarities:
# #             print(f"\n对齐节点嵌入相似度:")
# #             print(f"  平均余弦相似度: {sum(similarities) / len(similarities):.6f}")
# #             print(f"  最小余弦相似度: {min(similarities):.6f}")
# #             print(f"  最大余弦相似度: {max(similarities):.6f}")
# #
# #
# # if __name__ == "__main__":
# #     # 运行测试
# #     test_alignment_optimization()
# #     test_alignment_effectiveness()
# #
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import pandas as pd
import os
import re
from typing import List, Dict, Tuple, Optional, Set
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class SharedGCNLayer(nn.Module):
    """共享参数的GCN层"""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        # 线性变换
        self.linear = nn.Linear(in_features, out_features, bias=False)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.linear.weight, gain=1.0)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 节点特征 [n_nodes, in_features]
            adj: 邻接矩阵 [n_nodes, n_nodes]

        返回:
            更新后的节点特征 [n_nodes, out_features]
        """
        # 添加自环
        adj = adj + torch.eye(adj.size(0), device=adj.device)

        # 计算度矩阵
        degree = torch.sum(adj, dim=1, keepdim=True)
        degree_sqrt = torch.sqrt(degree + 1e-8)  # 防止除零

        # 对称归一化
        norm_adj = adj / (degree_sqrt * degree_sqrt.t() + 1e-8)

        # 特征变换
        x = self.linear(x)

        # 图卷积
        x = torch.matmul(norm_adj, x)

        return x


class DualSharedGCN(nn.Module):
    """双网络共享参数的GCN模型"""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.2):
        """
        初始化双网络共享参数GCN

        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表，例如[64, 32]表示两个隐藏层
            output_dim: 输出嵌入维度
            dropout: Dropout率
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.num_layers = len(hidden_dims) + 1  # 隐藏层 + 输出层

        # 构建隐藏层
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(
                SharedGCNLayer(current_dim, hidden_dim, dropout)
            )
            current_dim = hidden_dim

        # 输出层
        self.layers.append(
            SharedGCNLayer(current_dim, output_dim, dropout=0.0)
        )

    def forward(self, features1: torch.Tensor, adj1: torch.Tensor,
                features2: torch.Tensor, adj2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（同时处理两个图）

        参数:
            features1: 第一个图的节点特征 [n_nodes1, input_dim]
            adj1: 第一个图的邻接矩阵 [n_nodes1, n_nodes1]
            features2: 第二个图的节点特征 [n_nodes2, input_dim]
            adj2: 第二个图的邻接矩阵 [n_nodes2, n_nodes2]

        返回:
            emb1: 第一个图的节点嵌入 [n_nodes1, output_dim]
            emb2: 第二个图的节点嵌入 [n_nodes2, output_dim]
        """
        # 处理第一个图
        x1 = features1
        for i, layer in enumerate(self.layers[:-1]):
            x1 = layer(x1, adj1)
            x1 = F.relu(x1)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # 处理第二个图
        x2 = features2
        for i, layer in enumerate(self.layers[:-1]):
            x2 = layer(x2, adj2)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # 应用输出层
        emb1 = self.layers[-1](x1, adj1)
        emb2 = self.layers[-1](x2, adj2)

        return emb1, emb2


class DualSharedGCNEmbedder:
    """双网络共享参数GCN嵌入生成器（修复节点丢失问题）"""

    def __init__(self, hidden_dims: List[int] = [64, 32], output_dim: int = 16, dropout: float = 0.2,
                 debug_mode: bool = True):
        """
        初始化双网络共享参数GCN嵌入器

        参数:
            hidden_dims: 隐藏层维度列表，控制GCN层数
            output_dim: 输出嵌入维度
            dropout: Dropout率
            debug_mode: 调试模式，输出详细处理信息
        """
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.debug_mode = debug_mode

        # 模型将在首次使用时初始化
        self.model = None
        self.input_dim = None

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"双网络共享参数GCN配置: 隐藏层维度={hidden_dims}, 输出维度={output_dim}, 总层数={len(hidden_dims) + 1}")
        print(f"设备: {self.device}")
        print(f"调试模式: {debug_mode}")

    def extract_node_features(self, G: nx.Graph) -> Tuple[torch.Tensor, List]:
        """
        从NetworkX图中提取节点特征（修复节点丢失问题）

        参数:
            G: NetworkX图对象

        返回:
            features: 节点特征矩阵 [n_nodes, n_features]
            node_ids: 节点ID列表
        """
        node_ids = list(G.nodes())
        n_nodes = len(node_ids)

        if self.debug_mode:
            print(f"  提取特征: 原始图有 {n_nodes} 个节点")
            print(f"  节点ID示例: {node_ids[:10]}")

        # 收集所有节点的特征
        features_list = []
        valid_node_ids = []  # 记录实际处理的节点ID
        skipped_nodes = 0  # 记录跳过的节点数

        for node in node_ids:
            node_data = G.nodes[node]

            # 尝试获取特征
            feat = None
            for attr_name in ['features', 'feature', 'attr', 'attributes']:
                if attr_name in node_data:
                    feat = node_data[attr_name]
                    if self.debug_mode and node in list(G.nodes())[:5]:  # 只输出前5个节点的特征信息
                        print(f"    节点 {node}: 从字段 '{attr_name}' 获取特征")
                    break

            # 处理特征
            processed_feat = None
            if feat is None:
                # 如果没有特征，使用度数和聚类系数
                try:
                    degree = G.degree(node)
                    clustering = nx.clustering(G, node) if n_nodes > 1 else 0.0
                    processed_feat = [float(degree), float(clustering)]
                    if self.debug_mode and node in list(G.nodes())[:5]:
                        print(f"    节点 {node}: 无特征，使用度数和聚类系数: {processed_feat}")
                except Exception as e:
                    if self.debug_mode:
                        print(f"    节点 {node}: 特征计算失败: {e}")
                    skipped_nodes += 1
                    continue
            elif isinstance(feat, (int, float)):
                processed_feat = [float(feat)]
            elif isinstance(feat, list):
                # 已经是列表，确保是数值
                try:
                    processed_feat = [float(x) for x in feat]
                except Exception as e:
                    if self.debug_mode:
                        print(f"    节点 {node}: 列表特征转换失败: {e}")
                    skipped_nodes += 1
                    continue
            elif isinstance(feat, str):
                # 尝试解析字符串
                try:
                    if '[' in feat and ']' in feat:
                        # 提取方括号内的内容
                        match = re.search(r'\[(.*?)\]', feat)
                        if match:
                            content = match.group(1)
                            processed_feat = [float(x.strip()) for x in content.split(',')]
                    else:
                        processed_feat = [float(feat)]
                except Exception as e:
                    if self.debug_mode:
                        print(f"    节点 {node}: 字符串特征解析失败: {e}")
                    skipped_nodes += 1
                    continue
            else:
                if self.debug_mode:
                    print(f"    节点 {node}: 未知特征类型: {type(feat)}")
                skipped_nodes += 1
                continue

            # 添加到列表
            features_list.append(processed_feat)
            valid_node_ids.append(node)

        if self.debug_mode:
            print(f"  成功处理 {len(valid_node_ids)} 个节点，跳过 {skipped_nodes} 个节点")
            if skipped_nodes > 0:
                print(f"  警告: 跳过了 {skipped_nodes} 个节点的特征提取")

        if not features_list:
            if self.debug_mode:
                print("  错误: 没有提取到任何节点特征")
            return torch.tensor([], dtype=torch.float), []

        # 确定最大特征维度
        max_dim = max(len(f) for f in features_list) if features_list else 1

        if self.debug_mode:
            print(f"  最大特征维度: {max_dim}")

        # 填充特征向量
        padded_features = []
        for f in features_list:
            if len(f) < max_dim:
                padded_features.append(f + [0.0] * (max_dim - len(f)))
            else:
                padded_features.append(f[:max_dim])

        features_tensor = torch.tensor(padded_features, dtype=torch.float)

        if self.debug_mode:
            print(f"  特征矩阵形状: {features_tensor.shape}")
            print(f"  有效节点ID数量: {len(valid_node_ids)}")
            if len(valid_node_ids) < n_nodes:
                missing_nodes = set(node_ids) - set(valid_node_ids)
                print(f"  缺失节点: {list(missing_nodes)[:10] if len(missing_nodes) > 10 else list(missing_nodes)}")

        return features_tensor, valid_node_ids

    def nx_to_adjacency(self, G: nx.Graph, node_ids: List) -> torch.Tensor:
        """
        将NetworkX图转换为邻接矩阵

        参数:
            G: NetworkX图对象
            node_ids: 节点ID列表

        返回:
            邻接矩阵 [n_nodes, n_nodes]
        """
        n = len(node_ids)
        node_index = {node: i for i, node in enumerate(node_ids)}

        if self.debug_mode:
            print(f"  构建邻接矩阵: {n} 个节点")

        # 创建邻接矩阵
        adj = torch.zeros((n, n), dtype=torch.float)

        edges_added = 0
        for u, v in G.edges():
            if u in node_index and v in node_index:
                i, j = node_index[u], node_index[v]
                adj[i, j] = 1.0
                adj[j, i] = 1.0  # 假设无向图
                edges_added += 1
            else:
                if self.debug_mode and edges_added < 10:  # 只输出前10条边的问题
                    if u not in node_index:
                        print(f"    警告: 边 ({u}, {v}) 中的节点 {u} 不在有效节点列表中")
                    if v not in node_index:
                        print(f"    警告: 边 ({u}, {v}) 中的节点 {v} 不在有效节点列表中")

        if self.debug_mode:
            print(f"  添加了 {edges_added} 条边到邻接矩阵")
            print(f"  邻接矩阵形状: {adj.shape}")

        return adj

    def align_features(self, features1: torch.Tensor, features2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对齐两个图的特征维度

        参数:
            features1: 第一个图的特征 [n_nodes1, dim1]
            features2: 第二个图的特征 [n_nodes2, dim2]

        返回:
            aligned_features1: 对齐后的第一个图特征
            aligned_features2: 对齐后的第二个图特征
        """
        dim1 = features1.shape[1]
        dim2 = features2.shape[1]

        if self.debug_mode:
            print(f"  对齐特征维度: 图1={dim1}, 图2={dim2}")

        if dim1 == dim2:
            if self.debug_mode:
                print(f"  特征维度相同，无需对齐")
            return features1, features2

        # 以较大的维度为准
        max_dim = max(dim1, dim2)

        if self.debug_mode:
            print(f"  对齐到最大维度: {max_dim}")

        # 对齐第一个图
        if dim1 < max_dim:
            padding = torch.zeros(features1.shape[0], max_dim - dim1, dtype=features1.dtype)
            features1 = torch.cat([features1, padding], dim=1)
            if self.debug_mode:
                print(f"  图1特征从 {dim1} 填充到 {max_dim}")

        # 对齐第二个图
        if dim2 < max_dim:
            padding = torch.zeros(features2.shape[0], max_dim - dim2, dtype=features2.dtype)
            features2 = torch.cat([features2, padding], dim=1)
            if self.debug_mode:
                print(f"  图2特征从 {dim2} 填充到 {max_dim}")

        return features1, features2

    def generate_embeddings(self, G1: nx.Graph, G2: nx.Graph) -> Tuple[Dict, Dict, torch.Tensor, torch.Tensor]:
        """
        为两个图生成节点嵌入（共享参数模型）

        参数:
            G1: 第一个NetworkX图对象
            G2: 第二个NetworkX图对象

        返回:
            emb1_dict: 第一个图的节点嵌入字典
            emb2_dict: 第二个图的节点嵌入字典
            emb1_tensor: 第一个图的嵌入张量
            emb2_tensor: 第二个图的嵌入张量
        """
        print(f"\n{'=' * 60}")
        print(f"开始生成嵌入...")

        # 提取特征
        features1, node_ids1 = self.extract_node_features(G1)
        features2, node_ids2 = self.extract_node_features(G2)

        n_nodes1 = len(node_ids1)
        n_nodes2 = len(node_ids2)

        if self.debug_mode:
            print(f"\n特征提取结果:")
            print(f"  图1: {n_nodes1} 个节点 (原始: {G1.number_of_nodes()})")
            print(f"  图2: {n_nodes2} 个节点 (原始: {G2.number_of_nodes()})")

        if n_nodes1 == 0 or n_nodes2 == 0:
            if self.debug_mode:
                print(f"  错误: 至少一个图没有有效节点")
            return {}, {}, torch.empty(0, self.output_dim), torch.empty(0, self.output_dim)

        # 转换为邻接矩阵
        adj1 = self.nx_to_adjacency(G1, node_ids1)
        adj2 = self.nx_to_adjacency(G2, node_ids2)

        # 对齐特征维度
        features1, features2 = self.align_features(features1, features2)

        if self.debug_mode:
            print(f"\n特征对齐结果:")
            print(f"  图1特征形状: {features1.shape}")
            print(f"  图2特征形状: {features2.shape}")
            print(f"  图1邻接矩阵形状: {adj1.shape}")
            print(f"  图2邻接矩阵形状: {adj2.shape}")

        # 移动到设备
        features1 = features1.to(self.device)
        features2 = features2.to(self.device)
        adj1 = adj1.to(self.device)
        adj2 = adj2.to(self.device)

        # 初始化模型（如果需要）
        if self.model is None:
            self.input_dim = features1.shape[1]
            if self.debug_mode:
                print(f"\n初始化模型，输入维度: {self.input_dim}")
            self.model = DualSharedGCN(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                output_dim=self.output_dim,
                dropout=self.dropout
            ).to(self.device)

        # 生成嵌入
        self.model.eval()
        with torch.no_grad():
            emb1, emb2 = self.model(features1, adj1, features2, adj2)
            emb1 = emb1.cpu()
            emb2 = emb2.cpu()

        if self.debug_mode:
            print(f"\n嵌入生成结果:")
            print(f"  图1嵌入形状: {emb1.shape}")
            print(f"  图2嵌入形状: {emb2.shape}")

            # 检查嵌入是否为NaN或Inf
            if torch.isnan(emb1).any() or torch.isinf(emb1).any():
                print(f"  警告: 图1嵌入包含NaN或Inf值")
            if torch.isnan(emb2).any() or torch.isinf(emb2).any():
                print(f"  警告: 图2嵌入包含NaN或Inf值")

        # 创建节点ID到嵌入的映射
        emb1_dict = {node_id: emb.numpy() for node_id, emb in zip(node_ids1, emb1)}
        emb2_dict = {node_id: emb.numpy() for node_id, emb in zip(node_ids2, emb2)}

        if self.debug_mode:
            print(f"\n嵌入字典大小:")
            print(f"  图1嵌入字典: {len(emb1_dict)} 个节点")
            print(f"  图2嵌入字典: {len(emb2_dict)} 个节点")
            print(f"{'=' * 60}")

        return emb1_dict, emb2_dict, emb1, emb2

    def save_embeddings_to_csv(self, embeddings: torch.Tensor, node_ids: List, filename: str) -> None:
        """
        将嵌入保存为CSV文件

        参数:
            embeddings: 嵌入张量 [n_nodes, embed_dim]
            node_ids: 节点ID列表
            filename: 输出文件名
        """
        # 确保目录存在
        if '/' in filename or '\\' in filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        # 获取嵌入维度
        n_nodes, embed_dim = embeddings.shape

        if self.debug_mode:
            print(f"\n保存嵌入到文件: {filename}")
            print(f"  嵌入张量形状: {embeddings.shape}")
            print(f"  节点ID数量: {len(node_ids)}")

        # 检查节点ID和嵌入数量是否一致
        if len(node_ids) != n_nodes:
            print(f"  警告: 节点ID数量 ({len(node_ids)}) 与嵌入数量 ({n_nodes}) 不匹配")
            # 取较小值
            min_len = min(len(node_ids), n_nodes)
            node_ids = node_ids[:min_len]
            embeddings = embeddings[:min_len]
            print(f"  已截断到 {min_len} 个节点")

        # 创建DataFrame
        data = []
        for i, node_id in enumerate(node_ids):
            row = {'node_id': node_id}
            for j in range(embed_dim):
                row[f'dim_{j + 1}'] = embeddings[i, j].item()
            data.append(row)

        df = pd.DataFrame(data)

        # 保存为CSV
        df.to_csv(filename, index=False)
        print(f"嵌入已保存到: {filename}")
        print(f"  节点数: {df.shape[0]}, 嵌入维度: {df.shape[1] - 1}")

        if self.debug_mode:
            # 验证保存的数据
            loaded_df = pd.read_csv(filename)
            print(f"  验证: 从文件加载了 {loaded_df.shape[0]} 行数据")

            # 检查是否有NaN值
            nan_count = loaded_df.isna().sum().sum()
            if nan_count > 0:
                print(f"  警告: 文件中包含 {nan_count} 个NaN值")

    def generate_and_save_embeddings(self, G1: nx.Graph, G2: nx.Graph,
                                     output_file1: str = "graph1_embeddings.csv",
                                     output_file2: str = "graph2_embeddings.csv") -> Tuple[Dict, Dict]:
        """
        为两个图生成节点嵌入并保存为CSV文件

        参数:
            G1: 第一个NetworkX图对象
            G2: 第二个NetworkX图对象
            output_file1: 第一个图的嵌入输出文件
            output_file2: 第二个图的嵌入输出文件

        返回:
            emb1_dict: 第一个图的节点嵌入字典
            emb2_dict: 第二个图的节点嵌入字典
        """
        print(f"\n{'=' * 60}")
        print(f"处理第一个图: {G1.number_of_nodes()} 个节点, {G1.number_of_edges()} 条边")
        print(f"处理第二个图: {G2.number_of_nodes()} 个节点, {G2.number_of_edges()} 条边")
        print(f"使用共享参数的双网络GCN模型")

        # 为图生成嵌入
        print("\n生成两个图的节点嵌入...")
        emb1_dict, emb2_dict, emb1_tensor, emb2_tensor = self.generate_embeddings(G1, G2)

        # 获取节点ID列表
        node_ids1 = list(G1.nodes())
        node_ids2 = list(G2.nodes())

        # 保存嵌入
        print(f"\n保存第一个图的嵌入到: {output_file1}")
        self.save_embeddings_to_csv(emb1_tensor, node_ids1, output_file1)

        print(f"\n保存第二个图的嵌入到: {output_file2}")
        self.save_embeddings_to_csv(emb2_tensor, node_ids2, output_file2)

        return emb1_dict, emb2_dict


def create_test_graph1() -> nx.Graph:
    """创建测试图1"""
    G = nx.karate_club_graph()

    # 添加节点特征
    for node in G.nodes():
        # 为节点添加不同类型的特征
        if node % 3 == 0:
            G.nodes[node]['features'] = [node * 0.1, node * 0.2, node * 0.3]
        elif node % 3 == 1:
            G.nodes[node]['features'] = f"[{node * 2}, {node * 3}]"
        else:
            G.nodes[node]['features'] = float(node)

    return G


def create_test_graph2() -> nx.Graph:
    """创建测试图2"""
    G = nx.erdos_renyi_graph(20, 0.3, seed=42)

    # 添加节点特征
    for node in G.nodes():
        # 为节点添加不同类型的特征
        if node % 4 == 0:
            G.nodes[node]['features'] = [node, node ** 2, node ** 3, node % 5]
        elif node % 4 == 1:
            G.nodes[node]['features'] = float(node) * 0.5
        elif node % 4 == 2:
            G.nodes[node]['features'] = f"feature is [{node}, {node * 2}]"
        else:
            G.nodes[node]['features'] = [float(node)]

    return G


def test_fixed_dual_shared_gcn_embedder():
    """测试修复后的双网络共享参数GCN嵌入器"""
    print("=" * 60)
    print("测试修复节点丢失问题的双网络共享参数GCN嵌入生成器")
    print("=" * 60)

    # 创建测试图
    print("\n1. 创建测试图...")
    G1 = create_test_graph1()
    G2 = create_test_graph2()
    print(f"  图1: {G1.number_of_nodes()} 个节点, {G1.number_of_edges()} 条边")
    print(f"  图2: {G2.number_of_nodes()} 个节点, {G2.number_of_edges()} 条边")

    # 测试1: 2层GCN
    print("\n" + "=" * 40)
    print("2. 测试2层GCN（共享参数）...")
    print("=" * 40)
    try:
        embedder1 = DualSharedGCNEmbedder(
            hidden_dims=[64],  # 1个隐藏层，共2层GCN
            output_dim=16,
            dropout=0.2,
            debug_mode=True
        )

        emb1_dict, emb2_dict = embedder1.generate_and_save_embeddings(
            G1, G2,
            output_file1="fixed_dual_2layer_gcn_graph1.csv",
            output_file2="fixed_dual_2layer_gcn_graph2.csv"
        )

        print(f"\n  成功生成嵌入:")
        print(f"    图1: {len(emb1_dict)} 个节点的嵌入")
        print(f"    图2: {len(emb2_dict)} 个节点的嵌入")

        # 验证文件
        if os.path.exists("fixed_dual_2layer_gcn_graph1.csv"):
            df1 = pd.read_csv("fixed_dual_2layer_gcn_graph1.csv")
            print(f"    图1文件: {df1.shape[0]} 行, 原始图: {G1.number_of_nodes()} 节点")
            if df1.shape[0] < G1.number_of_nodes():
                print(f"    ⚠️ 警告: 图1嵌入文件缺失 {G1.number_of_nodes() - df1.shape[0]} 个节点")

        if os.path.exists("fixed_dual_2layer_gcn_graph2.csv"):
            df2 = pd.read_csv("fixed_dual_2layer_gcn_graph2.csv")
            print(f"    图2文件: {df2.shape[0]} 行, 原始图: {G2.number_of_nodes()} 节点")
            if df2.shape[0] < G2.number_of_nodes():
                print(f"    ⚠️ 警告: 图2嵌入文件缺失 {G2.number_of_nodes() - df2.shape[0]} 个节点")

    except Exception as e:
        print(f"  测试1失败: {e}")
        import traceback
        traceback.print_exc()

    # 测试2: 创建一个有问题的图，验证修复
    print("\n" + "=" * 40)
    print("3. 测试有问题的图（验证修复）...")
    print("=" * 40)

    # 创建一个包含各种问题的测试图
    G3 = nx.Graph()
    G3.add_nodes_from(range(10))
    G3.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    # 添加各种类型的特征
    G3.nodes[0]['features'] = [1.0, 2.0, 3.0]  # 正常特征
    G3.nodes[1]['features'] = "invalid_string"  # 无法解析的字符串
    G3.nodes[2]['features'] = 5.0  # 数值特征
    G3.nodes[3]['features'] = "[1, 2, 3, 4]"  # 字符串列表
    G3.nodes[4]['features'] = []  # 空列表
    # 节点5-9没有特征

    print(f"  测试图3: {G3.number_of_nodes()} 个节点")

    try:
        embedder2 = DualSharedGCNEmbedder(
            hidden_dims=[32],
            output_dim=8,
            dropout=0.1,
            debug_mode=True
        )

        # 测试单个图的处理
        features, node_ids = embedder2.extract_node_features(G3)
        print(f"  特征提取结果: {len(node_ids)} 个有效节点")

    except Exception as e:
        print(f"  测试2失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    # 显示生成的文件
    print("\n生成的文件:")
    for filename in [
        "fixed_dual_2layer_gcn_graph1.csv", "fixed_dual_2layer_gcn_graph2.csv"
    ]:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            print(f"  {filename}: {df.shape[0]} 行, {df.shape[1] - 1} 个维度")

            # 检查是否有缺失值
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                print(f"    ⚠️ 包含 {nan_count} 个缺失值")


if __name__ == "__main__":
    test_fixed_dual_shared_gcn_embedder()
