import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import logging
import networkx as nx

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_subg_node_alignment_metrics_with_dicts(
        src_emb_dict1, src_emb_dict2, tgt_emb_dict1, tgt_emb_dict2,
        true_align_dict, source_subgraph=None, target_subgraph=None,
        batch_size=1000, fusion_method='concatenation',
        src_weight1=1, src_weight2=0, tgt_weight1=1, tgt_weight2=0
):
    # 如果提供了子图，过滤节点
    if source_subgraph:
        source_nodes = set(source_subgraph.nodes())
        src_fused_emb = {node: emb for node, emb in src_fused_emb.items() if node in source_nodes}

    if target_subgraph:
        target_nodes = set(target_subgraph.nodes())
        tgt_fused_emb = {node: emb for node, emb in tgt_fused_emb.items() if node in target_nodes}

    # 筛选有效对齐对
    valid_alignment = {}
    for src_node, tgt_node in true_align_dict.items():
        # 检查源节点是否在源子图（如果提供了子图）
        if source_subgraph is not None and src_node not in source_subgraph.nodes:
            continue

        # 检查目标节点是否在目标子图（如果提供了子图）
        if target_subgraph is not None and tgt_node not in target_subgraph.nodes:
            continue

        # 检查节点是否在融合嵌入中
        if src_node in src_fused_emb and tgt_node in tgt_fused_emb:
            valid_alignment[src_node] = tgt_node

    valid_count = len(valid_alignment)
    if valid_count == 0:
        logger.error("无有效对齐节点对可用")
        return {}, {}, np.array([])

    logger.info(f"使用 {valid_count} 个有效对齐节点对进行评估")

    # 创建源节点列表和嵌入矩阵
    src_nodes = list(valid_alignment.keys())
    src_emb_matrix = np.array([src_fused_emb[node] for node in src_nodes])

    # 创建目标节点列表和嵌入矩阵（仅包含有效目标节点）
    tgt_nodes = list(set(valid_alignment.values()))
    tgt_emb_matrix = np.array([tgt_fused_emb[node] for node in tgt_nodes])
    tgt_idx_map = {node: idx for idx, node in enumerate(tgt_nodes)}

    # 初始化结果
    metrics = {'acc': 0, 'hits@5': 0, 'hits@10': 0, 'mrr': 0.0, 'avg_rank': 0.0}
    predicted_alignment = {}
    all_ranks = []

    # 分批计算相似性
    num_batches = (valid_count + batch_size - 1) // batch_size
    similarity_matrices = []

    for batch_idx in tqdm(range(num_batches), desc="计算相似性"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, valid_count)

        # 获取当前批次的源节点
        batch_src_nodes = src_nodes[start_idx:end_idx]
        batch_src_emb = src_emb_matrix[start_idx:end_idx]

        # 计算批次相似性矩阵（仅与有效目标节点）
        batch_sim_matrix = np.dot(batch_src_emb, tgt_emb_matrix.T)
        similarity_matrices.append(batch_sim_matrix)

        # 处理每个源节点
        for i, src_node in enumerate(batch_src_nodes):
            true_tgt_node = valid_alignment[src_node]
            true_idx = tgt_idx_map[true_tgt_node]

            # 获取相似度向量
            sim_vector = batch_sim_matrix[i]

            # 计算排名
            sorted_indices = np.argsort(sim_vector)[::-1]  # 降序排列
            rank = np.where(sorted_indices == true_idx)[0][0] + 1  # 排名从1开始
            all_ranks.append(rank)

            # 更新指标
            if rank == 1:
                metrics['acc'] += 1
                metrics['hits@5'] += 1
                metrics['hits@10'] += 1
            elif rank <= 5:
                metrics['hits@5'] += 1
                metrics['hits@10'] += 1
            elif rank <= 10:
                metrics['hits@10'] += 1

            metrics['mrr'] += 1.0 / rank

            # 记录预测 (使用top1)
            top1_idx = sorted_indices[0]
            pred_tgt_node = tgt_nodes[top1_idx]
            predicted_alignment[src_node] = pred_tgt_node

    # 计算最终指标
    metrics['acc'] /= valid_count
    metrics['hits@5'] /= valid_count
    metrics['hits@10'] /= valid_count
    metrics['mrr'] /= valid_count
    metrics['avg_rank'] = np.mean(all_ranks)

    # 合并相似性矩阵
    full_sim_matrix = np.vstack(similarity_matrices) if similarity_matrices else np.array([])

    return metrics

import numpy as np

# 普通聚合方式实现
def simple_average(emb1, emb2):
    return (emb1 + emb2) / 2

def weighted_average(emb1, emb2, weight1=0.5, weight2=0.5):
    return weight1 * emb1 + weight2 * emb2

def max_pooling(emb1, emb2):
    return np.maximum(emb1, emb2)

def min_pooling(emb1, emb2):
    return np.minimum(emb1, emb2)

def concatenation(emb1, emb2):
    return np.concatenate([emb1, emb2])

def elementwise_product(emb1, emb2):
    return emb1 * emb2

def elementwise_max(emb1, emb2):
    return np.maximum(emb1, emb2)

def elementwise_min(emb1, emb2):
    return np.minimum(emb1, emb2)

def weighted_sum(emb1, emb2, weight1=1.0, weight2=1.0):
    return weight1 * emb1 + weight2 * emb2

def normalized_weighted_average(emb1, emb2, weight1=0.5, weight2=0.5):
    fused_emb = weight1 * emb1 + weight2 * emb2
    norm = np.linalg.norm(fused_emb)
    return fused_emb / norm if norm > 0 else fused_emb

def median(emb1, emb2):
    return np.median([emb1, emb2], axis=0)

def geometric_mean(emb1, emb2):
    return np.sqrt(emb1 * emb2)


# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from tqdm import tqdm
# import logging
# import networkx as nx
#
# # 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
#
# def evaluate_subg_node_alignment_metrics(emb_file1, emb_file2, true_align_dict,
#                                     source_subgraph=None, target_subgraph=None,
#                                     batch_size=1000):
#     """
#     计算两个图节点的相似性矩阵并评估多种指标，只考虑子图中的节点
#
#     参数:
#         emb_file1 (str): 源图嵌入文件路径
#         emb_file2 (str): 目标图嵌入文件路径
#         true_align_dict (dict): 真实对齐字典 {source_node: target_node}
#         source_subgraph (nx.Graph): 源子图对象 (可选)
#         target_subgraph (nx.Graph): 目标子图对象 (可选)
#         batch_size (int): 批处理大小 (用于内存优化)
#
#     返回:
#         dict: 包含所有评估指标的字典
#         dict: 预测的对齐字典
#         np.ndarray: 相似性矩阵 (部分或完整)
#     """
#
#     # 1. 加载嵌入文件
#     def load_embeddings(file_path):
#         """安全加载嵌入文件并进行ID检查和去重"""
#         df = pd.read_csv(file_path)
#         # 处理常见的不必要列
#         drop_cols = [c for c in df.columns if c.lower() in ['unnamed:0', 'graph_id', 'node_index']]
#         if drop_cols:
#             df = df.drop(columns=drop_cols)
#
#         # 检查node_id列是否存在
#         if 'node_id' not in df.columns:
#             raise ValueError(f"嵌入文件 {file_path} 缺少 'node_id' 列")
#
#         # 节点ID去重
#         if df['node_id'].duplicated().any():
#             # logger.warning(f"发现重复节点ID: {len(df) - len(df.drop_duplicates('node_id'))}")
#             df = df.drop_duplicates('node_id', keep='first')
#
#         df = df.set_index('node_id')
#         # logger.info(f"成功加载: {len(df)} 个节点, {df.shape[1]} 维嵌入")
#         return df
#
#     try:
#         df1 = load_embeddings(emb_file1)
#         df2 = load_embeddings(emb_file2)
#     except Exception as e:
#         logger.error(f"嵌入加载失败: {str(e)}")
#         raise
#
#     # 2. 如果提供了子图，过滤只包含子图中的节点
#     if source_subgraph is not None:
#         source_nodes = set(source_subgraph.nodes())
#         df1 = df1[df1.index.isin(source_nodes)]
#         logger.info(f"源子图过滤后节点数: {len(df1)}")
#
#     if target_subgraph is not None:
#         target_nodes = set(target_subgraph.nodes())
#         df2 = df2[df2.index.isin(target_nodes)]
#         logger.info(f"目标子图过滤后节点数: {len(df2)}")
#
#     # 3. 准备嵌入矩阵
#     src_emb = df1.values.astype(np.float32)
#     tgt_emb = df2.values.astype(np.float32)
#
#     # 4. 创建索引映射
#     src_idx_map = {node: idx for idx, node in enumerate(df1.index)}
#     src_node_map = {idx: node for node, idx in src_idx_map.items()}
#     tgt_idx_map = {node: idx for idx, node in enumerate(df2.index)}
#     tgt_node_map = {idx: node for node, idx in tgt_idx_map.items()}
#
#     # 5. 筛选有效对齐对 (只考虑在子图中的节点)
#     valid_alignment = {}
#     for src_node, tgt_node in true_align_dict.items():
#         # 检查源节点是否在源子图（如果提供了子图）
#         if source_subgraph is not None and src_node not in source_subgraph.nodes:
#             continue
#
#         # 检查目标节点是否在目标子图（如果提供了子图）
#         if target_subgraph is not None and tgt_node not in target_subgraph.nodes:
#             continue
#
#         # 检查节点是否在嵌入文件中
#         if src_node in src_idx_map and tgt_node in tgt_idx_map:
#             valid_alignment[src_node] = tgt_node
#
#     valid_count = len(valid_alignment)
#     if valid_count == 0:
#         logger.error("无有效对齐节点对可用")
#         return {}, {}, np.array([])
#
#     logger.info(f"使用 {valid_count} 个有效对齐节点对进行评估")
#
#     # 6. 初始化结果
#     metrics = {
#         'acc': 0,  # 准确率 (Hits@1)
#         'hits@5': 0,  # Hits@5
#         'hits@10': 0,  # Hits@10
#         'mrr': 0.0,  # 平均倒数排名
#         'avg_rank': 0.0  # 平均排名
#     }
#     predicted_alignment = {}
#     all_ranks = []  # 存储所有节点的排名
#
#     # 7. 分批计算相似性矩阵
#     num_batches = (valid_count + batch_size - 1) // batch_size
#     similarity_matrices = []
#
#     logger.info(f"开始分批计算相似性矩阵 (批大小={batch_size})...")
#     for batch_idx in tqdm(range(num_batches), desc="计算相似性"):
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, valid_count)
#
#         # 获取当前批次的源节点
#         batch_nodes = list(valid_alignment.keys())[start_idx:end_idx]
#         batch_idxs = [src_idx_map[node] for node in batch_nodes]
#
#         # 提取批次嵌入
#         src_batch = src_emb[batch_idxs]
#
#         # 计算批次相似性矩阵
#         batch_sim_matrix = cosine_similarity(src_batch, tgt_emb)
#         similarity_matrices.append(batch_sim_matrix)
#
#         # 处理每个源节点
#         for i, src_node in enumerate(batch_nodes):
#             # 获取真实目标节点索引
#             true_tgt_node = valid_alignment[src_node]
#             true_idx = tgt_idx_map[true_tgt_node]
#
#             # 获取当前节点的相似度向量
#             sim_vector = batch_sim_matrix[i]
#
#             # 计算排名
#             sorted_indices = np.argsort(sim_vector)[::-1]  # 降序排列
#             rank = np.where(sorted_indices == true_idx)[0][0] + 1  # 排名从1开始
#
#             # 更新指标
#             if rank == 1:
#                 metrics['acc'] += 1
#                 metrics['hits@5'] += 1
#                 metrics['hits@10'] += 1
#             elif rank <= 5:
#                 metrics['hits@5'] += 1
#                 metrics['hits@10'] += 1
#             elif rank <= 10:
#                 metrics['hits@10'] += 1
#
#             metrics['mrr'] += 1.0 / rank
#             all_ranks.append(rank)
#
#             # 记录预测 (使用top1)
#             top1_idx = sorted_indices[0]
#             pred_tgt_node = tgt_node_map[top1_idx]
#             predicted_alignment[src_node] = pred_tgt_node
#
#     # 8. 计算最终指标
#     if valid_count > 0:
#         metrics['acc'] /= valid_count
#         metrics['hits@5'] /= valid_count
#         metrics['hits@10'] /= valid_count
#         metrics['mrr'] /= valid_count
#         metrics['avg_rank'] = np.mean(all_ranks)
#
#     # 9. 合并相似性矩阵
#     full_sim_matrix = np.vstack(similarity_matrices) if similarity_matrices else np.array([])
#
#     return metrics