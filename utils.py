from torch_geometric.utils.convert import *
from torch_geometric.data import NeighborSampler as RawNeighborSampler

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import copy
import networkx as nx 
import os 
import random as rnd
from random import sample
import json
from networkx.readwrite import json_graph
import graph_utils
import math
from data import *

'''methods for main alg'''




def cal_degree_dict(G_list, G, layer):
    G_degree = G.degree()
    degree_dict = {}
    degree_dict[0] = {}
    for node in G_list:
        degree_dict[0][node] = {node}
    for i in range(1, layer + 1):
        degree_dict[i] = {}
        for node in G_list:
            neighbor_set = []
            for neighbor in degree_dict[i - 1][node]:
                neighbor_set += nx.neighbors(G, neighbor)
            neighbor_set = set(neighbor_set)
            for j in range(i - 1, -1, -1):
                neighbor_set -= degree_dict[j][node]
            degree_dict[i][node] = neighbor_set
    for i in range(layer + 1):
        for node in G_list:
            if len(degree_dict[i][node]) == 0:
                degree_dict[i][node] = [0]
            else:
                degree_dict[i][node] = node_to_degree(G_degree, degree_dict[i][node])
    return degree_dict
    
def seed_link(seed_list1, seed_list2, G1, G2):
    k = 0
    for i in range(len(seed_list1) - 1):
        for j in range(np.max([1, i + 1]), len(seed_list1)):
            if G1.has_edge(seed_list1[i], seed_list1[j]) and not G2.has_edge(seed_list2[i], seed_list2[j]):
                G2.add_edges_from([[seed_list2[i], seed_list2[j]]])
                k += 1
            if not G1.has_edge(seed_list1[i], seed_list1[j]) and G2.has_edge(seed_list2[i], seed_list2[j]):
                G1.add_edges_from([[seed_list1[i], seed_list1[j]]])
                k += 1
    print('Add seed links : {}'.format(k), end = '\t')
    return G1, G2

def node_to_degree(G_degree, SET):
    SET = list(SET)
    SET = sorted([G_degree[x] for x in SET])
    return SET
    





def clip(x):
    if x <= 0:
        return 0
    else:
        return x
    
    
    
def calculate_Tversky_fortest(setA, setB, alpha, beta):

    setA = set(setA)
    setB = set(setB)   
    ep = 0.01
        
    inter = len(setA & setB) + ep
    diffA = len(setA - setB) 
    diffB = len(setB - setA)
     
    Tver = inter / (inter + alpha*diffA + beta*diffB)
    
    bool = (inter<diffB)
    #print(f"ACNs:{inter},diffB:{diffB}")
    
    return Tver,bool





''' preproc '''

def preprocessing(G1, G2, alignment_dict):
    '''
    Parameters
    ----------
    G1 : source graph
    G2 : target graph
    alignment_dict : grth dict

    '''
    # shift index for constructing union
    # construct shifted dict
    shift = G1.number_of_nodes()
    G2_list = list(G2.nodes())
    G2_shiftlist = list(idx + shift for idx in list(G2.nodes()))
    shifted_dict = dict(zip(G2_list,G2_shiftlist))
    
    #relable idx for G2
    G2 = nx.relabel_nodes(G2, shifted_dict)
    
    #update alignment dict
    align1list = list(alignment_dict.keys())
    align2list = list(alignment_dict.values())   
    shifted_align2list = [a+shift for a in align2list]
    
    groundtruth_dict = dict(zip(align1list, shifted_align2list))
    groundtruth_dict, groundtruth_dict_reversed = get_reversed(groundtruth_dict)
    
    return G2, groundtruth_dict, groundtruth_dict_reversed


def create_idx_dict_pair(G1,G2,alignment_dict):
    '''
    Make sure that this function is followed after preprocessing dict.

    '''
    
    G1list = list(G1.nodes())
    #G1list.sort()
    idx1_list = list(range(G1.number_of_nodes()))
    #make dict for G1
    idx1_dict = {a : b for b, a in zip(idx1_list,G1list)}

    
    G2list = list(G2.nodes())
    #G2list.sort()
    idx2_list = list(range(G2.number_of_nodes()))
    #make dict for G2
    idx2_dict = {c : d for d, c in zip(idx2_list,G2list)}
    
    return idx1_dict, idx2_dict
    
def normalized_adj(G):
    # make sure ordering has ascending order
    deg = dict(G.degree)
    deg = sorted(deg.items())
    deglist = [math.pow(b, -0.5) for (a,b) in deg]
    degarr = np.array(deglist)
    degarr = np.expand_dims(degarr, axis = 0)
    return degarr.T

def greedy_match(X, G1, G2):
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    m, n = X.shape
    x = np.array(X.flatten()).reshape(-1, )
    minSize = min(m, n)
    usedRows = np.zeros(n)
    usedCols = np.zeros(m)
    maxList = np.zeros(minSize)
    row = np.zeros(minSize)
    col = np.zeros(minSize)
    ix = np.argsort(-np.array(x))
    matched = 0
    index = 0
    while (matched < minSize):
        ipos = ix[index]
        jc = int(np.floor(ipos / n))
        ic = int(ipos - jc * n)
        if (usedRows[ic] != 1 and usedCols[jc] != 1):
            row[matched] = G1_nodes[ic]
            col[matched] = G2_nodes[jc]
            maxList[matched] = x[index]
            usedRows[ic] = 1
            usedCols[jc] = 1
            matched += 1
        index += 1;
    row = row.astype(int)
    col = col.astype(int)
    return zip(col, row)


def random_key_value(dictionary):
    """
    从字典中随机选择一个元素，返回其键和值

    参数:
    dictionary: dict - 输入字典

    返回:
    tuple: (随机键, 对应值) 或 (None, None) 如果字典为空
    """
    if not dictionary:  # 检查字典是否为空
        return None, None

    # 高效随机选择键值对
    key = random.choice(list(dictionary.keys()))
    return key, dictionary[key]


from collections import deque
import networkx as nx


def centrality_aware_bfs_expansion2(center_node, G, n, centrality_dict, node_list=None):
    """
    以中心节点为中心，使用改进的BFS算法扩展子图，按中心性排序添加节点，
    确保生成的子图是连通图且边数最小（树状结构），并限制扩展节点必须在node_list中

    参数:
    center_node: 中心节点的ID
    G: NetworkX 图对象 (全图)
    n: 目标子图节点数
    centrality_dict: 节点中心性字典 {node_id: centrality_score}
    node_list: 可选，允许添加的节点列表

    返回:
    subgraph: 扩展后的子图 (NetworkX Graph)，保留所有节点和边属性
    """
    # 如果只需要中心节点，直接返回包含该节点的子图
    if n <= 1:
        return G.subgraph([center_node])

    # 检查中心节点是否在允许的节点列表中（如果提供了node_list）
    if node_list is not None and center_node not in node_list:
        # 如果中心节点不在允许列表中，则返回空图或只包含中心节点的图？
        # 根据需求，这里返回只包含中心节点的图
        return G.subgraph([center_node])

    # 初始化数据结构
    visited = set([center_node])  # 已访问节点集合
    result_nodes = [center_node]  # 已添加节点列表
    parent_map = {center_node: None}  # 父节点映射，用于构建树结构
    queue = deque([center_node])  # BFS队列

    # 当节点数不足时继续扩展
    while len(result_nodes) < n and queue:
        current_node = queue.popleft()

        # 获取当前节点的所有未访问邻居，并过滤不在node_list中的节点
        neighbors = [
            nbr for nbr in G.neighbors(current_node)
            if nbr not in visited and (node_list is None or nbr in node_list)
        ]

        # 如果没有邻居可添加，跳过
        if not neighbors:
            continue

        # 按中心性对邻居排序（降序）
        sorted_neighbors = sorted(
            neighbors,
            key=lambda x: centrality_dict.get(x, 0),
            reverse=True
        )

        # 添加邻居直到满足数量或邻居耗尽
        for neighbor in sorted_neighbors:
            if len(result_nodes) < n:
                # 添加节点
                visited.add(neighbor)
                result_nodes.append(neighbor)
                parent_map[neighbor] = current_node
                queue.append(neighbor)
            else:
                break  # 已达到目标节点数

    # 创建最小连通子图（树状结构）
    tree_graph = nx.Graph()

    # 添加节点并保留属性
    for node in result_nodes:
        tree_graph.add_node(node, **G.nodes[node])

    # 只添加必要的边（树边），确保连通且边数最小
    for node in result_nodes:
        if node != center_node:  # 中心节点没有父节点
            parent = parent_map[node]
            # 获取原图中的边属性
            edge_data = G.get_edge_data(parent, node)
            # 添加边并保留属性
            if edge_data:
                tree_graph.add_edge(parent, node, **edge_data)
            else:
                tree_graph.add_edge(parent, node)

    return tree_graph


def split_dict(input_dict):
    keys_list = list(input_dict.keys())
    values_list = list(input_dict.values())
    return keys_list, values_list



def count_aligned_nodes(G1: nx.Graph, G2: nx.Graph, alignment_dict: dict) :
    """
    计算两个图中存在的对齐节点对数量

    参数:
        G1: 第一个图对象 (NetworkX Graph)
        G2: 第二个图对象 (NetworkX Graph)
        alignment_dict: 对齐字典 {G1节点: G2节点}

    返回:
        两个图中都存在且对齐的节点对数量
    """
    # 验证输入
    if not isinstance(G1, nx.Graph) or not isinstance(G2, nx.Graph):
        raise TypeError("输入必须是NetworkX图对象")
    if not isinstance(alignment_dict, dict):
        raise TypeError("对齐字典必须是字典类型")

    # 初始化计数器
    valid_pairs = 0
    nodes_list=[]
    # 遍历对齐字典
    for node1, node2 in alignment_dict.items():
        # 检查节点是否在两个图中都存在
        if node1 in G1 and node2 in G2:
            nodes_list.append(node1)
            valid_pairs += 1

    return valid_pairs,nodes_list


def find_all_max_indexes(lst):
    """
    查找列表中所有最大值元素的索引

    参数:
        lst: 输入列表（可包含数字、字符串等可比较元素）

    返回:
        list: 包含所有最大值索引的列表（从0开始）
        []: 如果列表为空

    说明:
        - 如果列表为空，返回空列表
        - 如果列表中有多个最大值，返回所有索引
        - 只返回索引，不返回值
    """
    if not lst:  # 检查列表是否为空
        return []

    max_value = lst[0]  # 初始化最大值为第一个元素
    max_indexes = [0]  # 初始化最大索引列表

    # 遍历列表（从第二个元素开始）
    for i in range(1, len(lst)):
        current_value = lst[i]

        if current_value > max_value:
            # 找到新的最大值
            max_value = current_value
            max_indexes = [i]  # 重置索引列表
        elif current_value == max_value:
            # 找到另一个最大值
            max_indexes.append(i)

    return max_indexes

from typing import Dict
def calculate_alignment_accuracy(
        node_list: List[Any],
        pred_alignment: Dict[Any, Any],
        true_alignment: Dict[Any, Any],
        n: int
) -> float:
    """
    计算预测准确率：预测正确的对齐对数量占n的百分比

    参数:
        node_list: 节点列表 (来自图1)
        pred_alignment: 预测的对齐字典 {图1节点: 图2节点}
        true_alignment: 真实的对齐字典 {图1节点: 图2节点}
        n: 整数，表示要考虑的节点数量

    返回:
        准确率 (0.0-1.0)
    """
    # 验证输入
    if n <= 0:
        raise ValueError("n必须大于0")

    # 确保n不超过节点列表长度
    n = min(n, len(node_list))

    # 计算预测正确的数量
    correct_count = 0

    # 遍历节点列表中的前n个节点
    for i in range(n):
        node = node_list[i]

        # 检查节点是否在预测字典和真实字典中
        if node in pred_alignment and node in true_alignment:
            # 检查预测是否正确
            if pred_alignment[node] == true_alignment[node]:
                correct_count += 1

    # 计算准确率
    accuracy = correct_count / n if n > 0 else 0.0

    return accuracy


def evaluate_alignment(pred_dict, true_dict):
    """
    评估预测对齐字典与真实对齐字典的准确性

    参数:
        pred_dict: 预测对齐字典 {源图节点: 目标图节点}
        true_dict: 真实对齐字典 {源图节点: 目标图节点}

    返回:
        包含各种评估指标的字典
    """
    # 确保输入是字典
    if not isinstance(pred_dict, dict) or not isinstance(true_dict, dict):
        raise ValueError("输入必须是字典类型")

    # 初始化统计变量
    correct_count = 0
    total_count = 0
    precision_at_1 = 0.0
    precision_at_5 = 0.0
    precision_at_10 = 0.0
    reciprocal_ranks = []

    # 遍历真实对齐字典
    for source_node, true_target in true_dict.items():
        # 检查源节点是否在预测字典中
        if source_node in pred_dict:
            total_count += 1

            # 获取预测的目标节点
            pred_target = pred_dict[source_node]

            # 检查预测是否正确
            if pred_target == true_target:
                correct_count += 1
                reciprocal_ranks.append(1.0)  # 排名第一
            else:
                reciprocal_ranks.append(0.0)  # 不在前k名

    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    # 计算MRR
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    # 返回评估结果
    return {
        "accuracy": accuracy,
        "precision@1": accuracy,  # 对于1对1对齐，accuracy就是precision@1
        "mrr": mrr,
        "correct_count": correct_count,
        "total_count": total_count,
        "coverage": total_count / len(true_dict) if true_dict else 0.0
    }

def merge_attributes_with_embeddings(attribute_matrix, embedding_csv_path, output_csv_path):
    """
    将节点属性拼接到嵌入向量后生成新的CSV文件

    参数:
        attribute_matrix (np.ndarray): 节点属性矩阵，形状为[节点数, 属性维度]
        embedding_csv_path (str): 嵌入向量CSV文件路径
        output_csv_path (str): 输出文件路径
    """
    # 读取嵌入向量CSV文件
    embedding_df = pd.read_csv(embedding_csv_path)

    # 验证数据一致性
    if embedding_df.shape[0] != attribute_matrix.shape[0]:
        raise ValueError(
            f"节点数量不匹配: 嵌入文件包含{embedding_df.shape[0]}个节点，"
            f"属性矩阵包含{attribute_matrix.shape[0]}个节点"
        )

    # 获取属性列名 (attr_1, attr_2, ...)
    num_attributes = attribute_matrix.shape[1]
    attribute_columns = [f'attr_{i + 1}' for i in range(num_attributes)]

    # 创建属性DataFrame
    attribute_df = pd.DataFrame(
        attribute_matrix,
        columns=attribute_columns
    )

    # 合并嵌入向量和属性
    # 使用iloc确保按位置合并而不依赖索引
    result_df = pd.concat([
        embedding_df.iloc[:, :],  # 所有原始嵌入列
        attribute_df.iloc[:, :]  # 所有属性列
    ], axis=1)

    # 保存结果到CSV
    result_df.to_csv(output_csv_path, index=False)
    print(f"成功生成合并文件: {output_csv_path}")
    print(f"合并后维度: {embedding_df.shape[1]}嵌入 + {num_attributes}属性 = {result_df.shape[1]}列")




def emb_to_matrix(emb1, emb2):
    """读取两个嵌入CSV文件并返回属性矩阵和节点ID映射

    参数:
        emb1: 第一个图的嵌入文件路径
        emb2: 第二个图的嵌入文件路径

    返回:
        attribute_matrix1: 第一个图的嵌入矩阵
        attribute_matrix2: 第二个图的嵌入矩阵
        id_to_idx1: 第一个图的节点ID到矩阵索引的映射
        id_to_idx2: 第二个图的节点ID到矩阵索引的映射
    """
    # 读取嵌入文件
    attribute_matrix1, id_to_idx1 = read_embeddings_to_matrix(emb1)
    attribute_matrix2, id_to_idx2 = read_embeddings_to_matrix(emb2)

    return attribute_matrix1, attribute_matrix2, id_to_idx1, id_to_idx2


def read_embeddings_to_matrix(csv_path):
    """读取节点嵌入CSV文件并转换为属性矩阵和ID映射

    参数:
        csv_path: CSV文件路径

    返回:
        attribute_matrix: 属性矩阵 (n_nodes, emb_dim)
        id_to_idx: 节点ID到矩阵索引的映射字典
    """
    # 使用pandas读取CSV文件
    df = pd.read_csv(csv_path)

    # 确保有node_id列
    if 'node_id' not in df.columns:
        raise ValueError(f"CSV文件 {csv_path} 缺少 'node_id' 列")

    # 提取节点ID和嵌入向量
    node_ids = df['node_id'].values
    embeddings = df.drop(columns=['node_id']).values

    # 创建ID到索引的映射
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    return embeddings, id_to_idx

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple
def refine_embeddings_with_alignment(
        source_emb: np.ndarray,
        target_emb: np.ndarray,
        id_to_idx_source: dict,  # 新增：源图节点ID到索引的映射
        id_to_idx_target: dict,  # 新增：目标图节点ID到索引的映射
        alignment_dict: dict,
        output_source_path: str,
        output_target_path: str,
        epochs: int = 10000,
        lr: float = 0.0001,
        margin: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用节点对齐预测字典优化节点嵌入表示，并保存结果到CSV文件

    参数:
        source_emb: 源图节点嵌入矩阵 (N, D)
        target_emb: 目标图节点嵌入矩阵 (M, D)
        id_to_idx_source: 源图节点ID到索引的映射字典
        id_to_idx_target: 目标图节点ID到索引的映射字典
        alignment_dict: 预测的对齐字典 {源节点ID: 目标节点ID}
        output_source_path: 优化后源图嵌入保存路径
        output_target_path: 优化后目标图嵌入保存路径
        epochs: 训练轮数
        lr: 学习率
        margin: 三元组损失的边界值

    返回:
        Tuple: 优化后的源图和目标图嵌入 (Numpy数组)
    """
    # 转换为PyTorch张量
    source_tensor = torch.tensor(source_emb, dtype=torch.float32, requires_grad=True)
    target_tensor = torch.tensor(target_emb, dtype=torch.float32, requires_grad=True)

    # 提取有效的对齐对索引
    valid_pairs = []
    for src_id, tgt_id in alignment_dict.items():
        # 使用映射字典获取索引
        src_idx = id_to_idx_source.get(src_id, None)
        tgt_idx = id_to_idx_target.get(tgt_id, None)

        # 检查索引是否有效
        if src_idx is not None and tgt_idx is not None:
            if src_idx < source_emb.shape[0] and tgt_idx < target_emb.shape[0]:
                valid_pairs.append((src_idx, tgt_idx))

    if not valid_pairs:
        print("警告: 对齐字典中没有有效的节点对，跳过优化")
        # 仍保存未优化的嵌入
        save_embeddings_to_csv(source_emb, output_source_path, id_to_idx_source)
        save_embeddings_to_csv(target_emb, output_target_path, id_to_idx_target)
        return source_emb, target_emb

    print(f"使用 {len(valid_pairs)} 个对齐节点对优化嵌入...")

    # 转换列表为张量以便批量处理
    src_indices = torch.LongTensor([pair[0] for pair in valid_pairs])
    tgt_indices = torch.LongTensor([pair[1] for pair in valid_pairs])

    # 定义优化器
    optimizer = optim.Adam([source_tensor, target_tensor], lr=lr)

    # 定义三元组损失
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 批量获取锚点、正样本和负样本
        anchors = source_tensor[src_indices]
        positives = target_tensor[tgt_indices]

        # 随机选择负样本 (来自不同节点)
        # 确保负样本索引在有效范围内
        n_target = target_tensor.shape[0]
        negative_indices = torch.randint(0, n_target, (len(anchors),))
        negatives = target_tensor[negative_indices]

        # 计算三元组损失
        loss = triplet_loss(anchors, positives, negatives)

        # 添加嵌入正则化防止过度变化
        reg_loss = 0.001 * (torch.norm(source_tensor) + torch.norm(target_tensor))
        total_loss = loss + reg_loss

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

    # 嵌入归一化
    refined_source = nn.functional.normalize(source_tensor, p=2, dim=1).detach().numpy()
    refined_target = nn.functional.normalize(target_tensor, p=2, dim=1).detach().numpy()

    print("嵌入优化完成")

    # 保存优化后的嵌入到CSV文件
    save_embeddings_to_csv(refined_source, output_source_path, id_to_idx_source)
    save_embeddings_to_csv(refined_target, output_target_path, id_to_idx_target)

    return refined_source, refined_target


def save_embeddings_to_csv(
        embeddings: np.ndarray,
        file_path: str,
        id_to_idx: dict
):
    """
    将嵌入保存为CSV文件，格式为：node_id,dim_1,dim_2,dim_3,...

    参数:
        embeddings: 嵌入矩阵 (n_nodes, embedding_dim)
        file_path: 保存路径
        id_to_idx: 节点ID到索引的映射字典
    """
    try:
        # 创建反向映射：索引到节点ID
        idx_to_id = {idx: node_id for node_id, idx in id_to_idx.items()}

        # 确保索引范围正确
        n_nodes = embeddings.shape[0]
        if len(idx_to_id) != n_nodes:
            print(f"警告: 映射字典长度({len(idx_to_id)})与嵌入矩阵行数({n_nodes})不匹配")
            # 使用可用索引
            valid_indices = [idx for idx in range(n_nodes) if idx in idx_to_id]
        else:
            valid_indices = range(n_nodes)

        # 创建数据框
        data = []
        for idx in valid_indices:
            node_id = idx_to_id.get(idx, idx)  # 如果找不到映射，使用索引作为ID
            row = [node_id] + embeddings[idx].tolist()
            data.append(row)

        # 创建列名
        dims = embeddings.shape[1]
        columns = ['node_id'] + [f'dim_{i + 1}' for i in range(dims)]

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(file_path, index=False)

        print(f"已保存嵌入到 {file_path}，包含 {len(data)} 个节点，{dims} 维嵌入")

    except Exception as e:
        print(f"保存嵌入到CSV失败: {str(e)}")


def centrality_aware_bfs_expansion4(center_node, G, n, centrality_dict, node_list=None):
    """
    以中心节点为中心，使用改进的BFS算法扩展子图，按中心性排序添加节点，
    保留所有发现的边（不再生成最小连通图），并限制扩展节点必须在node_list中

    参数:
    center_node: 中心节点的ID
    G: NetworkX 图对象 (全图)
    n: 目标子图节点数
    centrality_dict: 节点中心性字典 {node_id: centrality_score}
    node_list: 可选，允许添加的节点列表

    返回:
    subgraph: 扩展后的子图 (NetworkX Graph)，包含所有节点和它们之间的边
    """
    # 如果只需要中心节点，直接返回包含该节点的子图
    if n <= 1:
        return G.subgraph([center_node])

    # 检查中心节点是否在允许的节点列表中（如果提供了node_list）
    if node_list is not None and center_node not in node_list:
        # 如果中心节点不在允许列表中，则返回只包含中心节点的图
        return G.subgraph([center_node])

    # 初始化数据结构
    visited = set([center_node])  # 已访问节点集合
    result_nodes = set([center_node])  # 已添加节点集合
    queue = deque([center_node])  # BFS队列

    # 当节点数不足时继续扩展
    while len(result_nodes) < n and queue:
        current_node = queue.popleft()

        # 获取当前节点的所有未访问邻居，并过滤不在node_list中的节点
        neighbors = [
            nbr for nbr in G.neighbors(current_node)
            if nbr not in visited and (node_list is None or nbr in node_list)
        ]

        # 如果没有邻居可添加，跳过
        if not neighbors:
            continue

        # 按中心性对邻居排序（降序）
        sorted_neighbors = sorted(
            neighbors,
            key=lambda x: centrality_dict.get(x, 0),
            reverse=True
        )

        # 添加邻居直到满足数量或邻居耗尽
        for neighbor in sorted_neighbors:
            if len(result_nodes) < n:
                # 添加节点
                visited.add(neighbor)
                result_nodes.add(neighbor)
                queue.append(neighbor)
            else:
                break  # 已达到目标节点数

    # 创建子图视图 - 保留所有节点和它们之间的边
    subgraph = G.subgraph(result_nodes)
    return subgraph


def get_subgraph_embedding2(subgraph, embedding_csv_path, pooling='sum', return_matrix=False):
    """
    从节点嵌入CSV文件中提取子图节点的嵌入，并进行池化生成子图嵌入

    参数:
    subgraph: NetworkX图对象 - 子图
    embedding_csv_path: str - 节点嵌入CSV文件路径
    pooling: str - 池化方法 ('mean', 'max', 'sum', 'concat')
    return_matrix: bool - 是否返回嵌入矩阵而非池化向量

    返回:
    np.array: 子图嵌入向量或嵌入矩阵
    """
    # 1. 读取节点嵌入CSV文件
    try:
        # 读取CSV文件，第一列作为索引
        embeddings_df = pd.read_csv(embedding_csv_path, index_col=0)
    except Exception as e:
        raise FileNotFoundError(f"无法读取嵌入文件: {str(e)}")

    # 2. 获取子图节点ID
    subgraph_nodes = list(subgraph.nodes())

    # 3. 提取子图节点的嵌入
    subgraph_embeddings = []
    missing_nodes = []
    valid_nodes = []

    for node in subgraph_nodes:
        try:
            # 尝试获取节点的嵌入向量
            node_embedding = embeddings_df.loc[node].values
            subgraph_embeddings.append(node_embedding)
            valid_nodes.append(node)
        except KeyError:
            missing_nodes.append(node)

    # 4. 处理缺失节点
    if missing_nodes:
        print(f"警告: 子图中 {len(missing_nodes)} 个节点在嵌入文件中缺失")
        # 获取嵌入维度
        embedding_dim = embeddings_df.shape[1]
        # 使用零向量替代缺失节点
        for node in missing_nodes:
            subgraph_embeddings.append(np.zeros(embedding_dim))
        print(f"为缺失节点使用零向量替代")

    # 5. 转换为NumPy数组
    embedding_matrix = np.array(subgraph_embeddings)

    # 6. 如果请求返回嵌入矩阵
    if return_matrix:
        return embedding_matrix

    # 7. 应用池化操作
    if pooling == 'mean':
        # 平均池化
        subgraph_embedding = np.mean(embedding_matrix, axis=0)
    elif pooling == 'max':
        # 最大池化
        subgraph_embedding = np.max(embedding_matrix, axis=0)
    elif pooling == 'sum':
        # 求和池化
        subgraph_embedding = np.sum(embedding_matrix, axis=0)
    elif pooling == 'concat':
        # 拼接所有节点嵌入
        subgraph_embedding = np.concatenate(embedding_matrix)
    else:
        raise ValueError(f"不支持的池化方法: {pooling}。请使用 'mean', 'max', 'sum' 或 'concat'")

    return subgraph_embedding



import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_most_similar_graph_max(source_emb, candidate_embs_list):
    """
    计算源图与候选图之间的相似度（使用最大节点相似度平均）

    参数:
        source_emb: 源图的嵌入矩阵 (n_source_nodes x embedding_dim)
        candidate_embs_list: 候选图嵌入矩阵列表
            [candidate1_matrix (n_cand1_nodes x embedding_dim),
             candidate2_matrix, ...]

    返回:
        max_index: 最相似候选图的索引 (从0开始)
        max_similarity: 最大相似度值
        all_similarities: 所有候选图的相似度列表
    """
    # 1. 验证输入
    if not candidate_embs_list:
        raise ValueError("候选图列表不能为空")

    # 2. 初始化结果
    all_similarities = []
    max_similarity = -np.inf
    max_index = -1

    # 3. 遍历所有候选图
    for idx, candidate_emb in enumerate(candidate_embs_list):
        # 3.1 计算节点相似度矩阵
        node_similarities = cosine_similarity(source_emb, candidate_emb)

        # 3.2 计算每个源图节点的最大相似度
        max_per_source_node = np.max(node_similarities, axis=1)

        # 3.3 计算平均最大相似度
        graph_similarity = np.mean(max_per_source_node)
        all_similarities.append(graph_similarity)

        # 3.4 更新最佳匹配
        if graph_similarity > max_similarity:
            max_similarity = graph_similarity
            max_index = idx

    return [max_index, max_similarity, all_similarities]



def find_most_similar_graph_max2(source_vector, candidate_vectors):
    """
    在候选向量列表中查找与源向量最相似的向量（基于余弦相似度）
    如果向量长度不同，对较短的向量补零以保证维度一致

    参数:
        source_vector: numpy数组 (1D) - 源嵌入向量
        candidate_vectors: 列表 - 包含多个候选嵌入向量 (每个都是1D numpy数组)

    返回:
        max_index: int - 最相似候选向量的索引 (从0开始)
        max_similarity: float - 最大余弦相似度值
        all_similarities: list - 所有候选向量的相似度列表
    """
    # 1. 验证输入
    if not candidate_vectors:
        raise ValueError("候选向量列表不能为空")

    # 2. 确保源向量是1D数组
    source_vector = np.asarray(source_vector).flatten()
    source_len = len(source_vector)

    # 3. 计算源向量的范数（避免重复计算）
    norm_source = np.linalg.norm(source_vector)

    # 4. 初始化结果
    all_similarities = []
    max_similarity = -np.inf
    max_index = -1

    # 5. 遍历所有候选向量
    for idx, candidate in enumerate(candidate_vectors):
        # 5.1 确保候选向量是1D数组
        candidate_vector = np.asarray(candidate).flatten()
        candidate_len = len(candidate_vector)

        # 5.2 处理长度不一致的情况
        if source_len != candidate_len:
            # 确定最大长度
            max_len = max(source_len, candidate_len)

            # 创建新的源向量副本（如果需要补零）
            source_vec = source_vector.copy()
            if len(source_vec) < max_len:
                source_vec = np.pad(source_vec, (0, max_len - len(source_vec)), 'constant')

            # 创建新的候选向量副本（如果需要补零）
            cand_vec = candidate_vector.copy()
            if len(cand_vec) < max_len:
                cand_vec = np.pad(cand_vec, (0, max_len - len(cand_vec)), 'constant')
        else:
            source_vec = source_vector
            cand_vec = candidate_vector

        # 5.3 计算点积
        dot_product = np.dot(source_vec, cand_vec)

        # 5.4 计算候选向量的范数
        norm_candidate = np.linalg.norm(cand_vec)

        # 5.5 避免除以零错误
        if norm_source == 0 or norm_candidate == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm_source * norm_candidate)

        # 5.6 存储相似度
        all_similarities.append(similarity)

        # 5.7 更新最佳匹配
        if similarity > max_similarity:
            max_similarity = similarity
            max_index = idx

    return [max_index, max_similarity, all_similarities]

def filter_valid_alignment(
        g1: nx.Graph,
        g2: nx.Graph,
        alignment_dict: Dict[Any, Any]
) -> Dict[Any, Any]:
    """
    过滤有效的对齐节点对，确保节点在两个图中都存在

    参数:
        g1: 第一个图对象
        g2: 第二个图对象
        alignment_dict: 原始对齐字典 {图1节点ID: 图2节点ID}

    返回:
        新的对齐字典，只包含在两个图中都存在的节点对
    """
    valid_alignment = {}

    for node1, node2 in alignment_dict.items():
        # 检查节点1是否在图1中存在
        if node1 in g1.nodes:
            # 检查节点2是否在图2中存在
            if node2 in g2.nodes:
                valid_alignment[node1] = node2

    return valid_alignment

from typing import Dict, Any, Tuple


def calculate_alignment_metrics(
        length,
        predicted_alignment: Dict[Any, Any],
        prior_alignment: Dict[Any, Any]
) -> Dict[str, float]:
    """
    计算预测对齐与先验对齐的匹配指标，特别处理子图场景

    参数:
        predicted_alignment: 预测的子图对齐字典 {节点A: 节点B}
        prior_alignment: 全图的对齐字典 {节点A: 节点B}

    返回:
        dict: 包含三个指标的字典 {
            'precision': 准确率 (0.0-1.0),
            'recall': 召回率 (基于子图相关节点),
            'f1': F1分数 (0.0-1.0)
        }
    """
    # 获取子图节点集
    subgraph_nodes = set(predicted_alignment.keys())

    # 计算实际相关的正样本节点
    relevant_nodes = subgraph_nodes & set(prior_alignment.keys())

    # 计算匹配数（只考虑子图中实际存在的对齐）
    match_count = sum(
        1 for node in subgraph_nodes
        if node in prior_alignment and predicted_alignment[node] == prior_alignment[node]
    )

    # 计算各类指标的分母
    total_predictions = len(predicted_alignment)  # 预测总数
    total_relevant = len(relevant_nodes)  # 子图中实际存在的对齐数

    # 计算准确率（预测的正确比例）
    #precision = match_count / total_predictions if total_predictions > 0 else 0.0
    precision = match_count / length if length > 0 else 0.0

    # 计算召回率（子图中实际存在的对齐被预测到的比例）
    recall = match_count / total_relevant if total_relevant > 0 else 0.0

    # 计算F1分数
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from typing import Dict, Any, Tuple, Optional


def compute_subgraph_similarity(
        subgraph1: nx.Graph,
        subgraph2: nx.Graph,
        embedding_file1: str,
        embedding_file2: str,
        true_alignment_dict: Optional[Dict[Any, Any]] = None,  # 新增：真实对齐字典
        similarity_threshold: float = 0.0,
        enforce_one_to_one: bool = True
):
    """
    计算两个子图的节点相似度矩阵，并根据映射字典提升特定位置的相似度
    确保生成的对齐对是一一映射（使用匈牙利算法）
    新增：如果提供真实对齐字典，计算评估指标

    参数:
        subgraph1: 第一个子图对象
        subgraph2: 第二个子图对象
        embedding_file1: 第一个图的节点嵌入CSV文件
        embedding_file2: 第二个图的节点嵌入CSV文件
        true_alignment_dict: 真实对齐字典 {subgraph1节点: subgraph2节点} (可选)
        similarity_threshold: 最小相似度阈值 (默认0.0)
        enforce_one_to_one: 是否强制一一映射 (默认True)

    返回:
        字典: 每个subgraph1节点在subgraph2中最相似的节点（一一映射）
        如果提供真实对齐字典，同时返回评估结果
    """
    # 1. 从CSV文件加载嵌入向量
    embeddings1 = load_embeddings(embedding_file1)
    embeddings2 = load_embeddings(embedding_file2)

    # 2. 获取子图节点
    nodes1 = list(subgraph1.nodes())
    nodes2 = list(subgraph2.nodes())

    # 3. 提取子图节点的嵌入向量
    emb_matrix1 = get_subgraph_embeddings(nodes1, embeddings1)
    emb_matrix2 = get_subgraph_embeddings(nodes2, embeddings2)

    # 4. 创建节点映射关系
    node_map1, reverse_map1 = create_node_mapping(nodes1)
    node_map2, reverse_map2 = create_node_mapping(nodes2)

    # 5. 计算相似度矩阵
    sim_matrix = cosine_similarity(emb_matrix1, emb_matrix2)

    # 7. 找到每个节点的最佳匹配
    if enforce_one_to_one:
        # 使用匈牙利算法确保一一映射
        best_matches = find_one_to_one_matches(
            sim_matrix, node_map1, node_map2, reverse_map1, reverse_map2, similarity_threshold
        )
    else:
        # 原始方法（可能多对一）
        best_matches = find_best_matches(sim_matrix, node_map1, reverse_map2, similarity_threshold)

    # 8. 如果有真实对齐字典，计算评估指标
    evaluation_results = {}
    if true_alignment_dict:
        # 过滤真实对齐字典，只保留在两个图中都存在的节点对
        valid_true_alignment = {}
        for src_node, tgt_node in true_alignment_dict.items():
            if src_node in node_map1 and tgt_node in node_map2:
                valid_true_alignment[src_node] = tgt_node

        # 计算评估指标
        evaluation_results = evaluate_alignment_subg(
            best_matches, valid_true_alignment, len(nodes2)
        )

        # 返回预测结果和评估指标
        return best_matches, evaluation_results

    return best_matches


def evaluate_alignment_subg(
        pred_alignment: Dict[Any, Any],
        true_alignment: Dict[Any, Any],
        n_target_nodes: int
) -> Dict[str, float]:
    """
    评估对齐结果的准确性

    参数:
        pred_alignment: 预测的对齐字典 {源节点: 目标节点}
        true_alignment: 真实的对齐字典 {源节点: 目标节点}
        n_target_nodes: 目标图的节点数

    返回:
        包含评估指标的字典
    """
    # 初始化指标
    metrics = {
        "accuracy": 0.0,
        "hits@1": 0.0,
        "hits@5": 0.0,
        "hits@10": 0.0
    }

    # 计算准确率
    correct_count = 0
    total_count = 0

    for src_node, true_tgt in true_alignment.items():
        if src_node in pred_alignment:
            total_count += 1
            if pred_alignment[src_node] == true_tgt:
                correct_count += 1

    if total_count > 0:
        metrics["accuracy"] = correct_count / total_count

    # 根据目标节点数决定计算哪些Hits@k指标
    if n_target_nodes < 5:
        # 只计算Hits@1
        metrics["hits@5"] = None
        metrics["hits@10"] = None
    elif n_target_nodes < 10:
        # 计算Hits@1和Hits@5
        metrics["hits@10"] = None
    # 否则计算所有指标

    return metrics


def find_one_to_one_matches(
        sim_matrix: np.ndarray,
        node_map1: Dict[Any, int],
        node_map2: Dict[Any, int],
        reverse_map1: Dict[int, Any],
        reverse_map2: Dict[int, Any],
        similarity_threshold: float = 0.0
) -> Dict[Any, Any]:
    """
    使用匈牙利算法找到最优的一一映射

    参数:
        sim_matrix: 相似度矩阵 (n1 x n2)
        node_map1: 子图1节点到索引的映射
        node_map2: 子图2节点到索引的映射
        reverse_map1: 索引到子图1节点的映射
        reverse_map2: 索引到子图2节点的映射
        similarity_threshold: 最小相似度阈值

    返回:
        字典: {subgraph1节点: subgraph2节点} (一一映射)
    """
    n1, n2 = sim_matrix.shape

    # 使用匈牙利算法求解最大权匹配
    # 将相似度转换为成本（最小化问题）
    cost_matrix = -sim_matrix  # 最大化相似度等价于最小化负相似度

    # 添加虚拟节点使矩阵成为方阵
    n = max(n1, n2)
    padded_cost = np.zeros((n, n))
    padded_cost[:n1, :n2] = cost_matrix

    # 执行匈牙利算法
    row_ind, col_ind = linear_sum_assignment(padded_cost)

    # 构建匹配字典
    matches = {}
    for i, j in zip(row_ind, col_ind):
        # 只处理实际节点（非虚拟节点）
        if i < n1 and j < n2:
            similarity = sim_matrix[i, j]
            if similarity >= similarity_threshold:
                node1 = reverse_map1[i]
                node2 = reverse_map2[j]
                matches[node1] = node2

    return matches


def load_embeddings(embedding_file: str) -> Dict[Any, np.ndarray]:
    """
    从CSV文件加载节点嵌入向量

    参数:
        embedding_file: CSV文件路径

    返回:
        字典: {节点ID: 嵌入向量}
    """
    df = pd.read_csv(embedding_file)
    if 'node_id' not in df.columns:
        raise ValueError("CSV文件必须包含'node_id'列")

    # 提取嵌入向量列
    embedding_cols = [col for col in df.columns if col.startswith('dim_')]

    embeddings = {}
    for _, row in df.iterrows():
        node_id = row['node_id']
        emb_vector = row[embedding_cols].values.astype(float)
        embeddings[node_id] = emb_vector

    return embeddings


def get_subgraph_embeddings(
        nodes: List[Any],
        embeddings: Dict[Any, np.ndarray]
) -> np.ndarray:
    """
    获取子图节点的嵌入矩阵

    参数:
        nodes: 节点ID列表
        embeddings: 嵌入向量字典

    返回:
        嵌入矩阵 [n_nodes, emb_dim]
    """
    emb_list = []
    for node in nodes:
        if node in embeddings:
            emb_list.append(embeddings[node])
        else:
            # 如果节点没有嵌入，使用零向量
            emb_dim = len(next(iter(embeddings.values())))
            emb_list.append(np.zeros(emb_dim))

    return np.array(emb_list)


def create_node_mapping(nodes: List[Any]) -> Tuple[Dict[Any, int], Dict[int, Any]]:
    """
    创建节点映射关系

    参数:
        nodes: 节点ID列表

    返回:
        node_map: {原始ID: 新索引}
        reverse_map: {新索引: 原始ID}
    """
    node_map = {}
    reverse_map = {}
    for idx, node in enumerate(nodes):
        node_map[node] = idx
        reverse_map[idx] = node

    return node_map, reverse_map


def find_best_matches(
        sim_matrix: np.ndarray,
        subgraph1: nx.Graph,
        subgraph2: nx.Graph,
        node_map1: Dict[Any, int],
        reverse_map2: Dict[int, Any],
        similarity_threshold: float = 0.0,
        attribute_name: str = 'features'
) -> Dict[Any, Any]:
    """
    找到每个节点的最佳匹配（可能多对一），优先选择属性相同的节点

    参数:
        sim_matrix: 相似度矩阵
        subgraph1: 第一个子图对象
        subgraph2: 第二个子图对象
        node_map1: 子图1节点到索引的映射
        reverse_map2: 索引到子图2节点的映射
        similarity_threshold: 最小相似度阈值
        attribute_name: 用于比较的节点属性名

    返回:
        字典: 每个subgraph1节点在subgraph2中最相似的节点
    """
    best_matches = {}
    n1, n2 = sim_matrix.shape

    # 安全比较属性值函数
    def are_attributes_equal(attr1, attr2) -> bool:
        """安全比较两个属性值是否相等"""
        if attr1 is None and attr2 is None:
            return True
        if attr1 is None or attr2 is None:
            return False
        if isinstance(attr1, np.ndarray) and isinstance(attr2, np.ndarray):
            return attr1.shape == attr2.shape and np.array_equal(attr1, attr2)
        if isinstance(attr1, list) and isinstance(attr2, list):
            return len(attr1) == len(attr2) and all(a == b for a, b in zip(attr1, attr2))
        return attr1 == attr2

    for node1, idx1 in node_map1.items():
        # 获取节点1的属性
        attr1 = subgraph1.nodes[node1].get(attribute_name, None)

        # 找到所有属性相同的候选节点
        candidate_indices = []
        for idx2 in range(n2):
            node2 = reverse_map2[idx2]
            attr2 = subgraph2.nodes[node2].get(attribute_name, None)

            if are_attributes_equal(attr1, attr2):
                candidate_indices.append(idx2)

        # 如果有属性相同的候选节点
        if candidate_indices:
            # 选择相似度最高的候选节点
            best_idx2 = candidate_indices[np.argmax(sim_matrix[idx1, candidate_indices])]
            best_similarity = sim_matrix[idx1, best_idx2]

            if best_similarity >= similarity_threshold:
                best_node2 = reverse_map2[best_idx2]
                best_matches[node1] = best_node2
                continue  # 找到匹配，跳过后续处理

        # 如果没有属性相同的候选节点，或者所有候选节点都不满足阈值
        # 选择相似度最高的节点（不考虑属性）
        best_idx2 = np.argmax(sim_matrix[idx1])
        best_similarity = sim_matrix[idx1, best_idx2]

        if best_similarity >= similarity_threshold:
            best_node2 = reverse_map2[best_idx2]
            best_matches[node1] = best_node2

    return best_matches


import networkx as nx


def get_valid_alignment_dict(
        G1: nx.Graph,
        G2: nx.Graph,
        true_alignment_dict: dict
) -> dict:
    """
    找出两个图中真实对齐的节点对并生成新的字典

    参数:
        G1: 第一个图对象
        G2: 第二个图对象
        true_alignment_dict: 真实对齐字典 {G1节点ID: G2节点ID}

    返回:
        新的对齐字典，只包含在两个图中都存在的节点对
    """
    # 创建新的有效对齐字典
    valid_alignment = {}

    # 遍历原始对齐字典
    for g1_node, g2_node in true_alignment_dict.items():
        # 检查节点是否在两个图中都存在
        if G1.has_node(g1_node) and G2.has_node(g2_node):
            valid_alignment[g1_node] = g2_node


    return valid_alignment


import networkx as nx
from typing import List, Tuple, Union


def get_graph_nodes(
        G1: Union[nx.Graph, nx.DiGraph, nx.MultiGraph],
        G2: Union[nx.Graph, nx.DiGraph, nx.MultiGraph]
) -> Tuple[List[Any], List[Any]]:
    """
    从两个图对象中提取节点列表

    参数:
        G1: 第一个图对象 (NetworkX Graph, DiGraph, MultiGraph等)
        G2: 第二个图对象 (NetworkX Graph, DiGraph, MultiGraph等)

    返回:
        (nodes1, nodes2): 两个图的节点列表元组
    """
    # 确保输入是有效的图对象
    if not hasattr(G1, 'nodes') or not hasattr(G2, 'nodes'):
        raise ValueError("输入必须是有效的图对象")

    # 获取节点列表
    nodes1 = list(G1.nodes())
    nodes2 = list(G2.nodes())

    return nodes1, nodes2


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_node_alignment_metrics(emb_file1, emb_file2, true_align_dict, batch_size=1000):
    """
    计算两个图节点的相似性矩阵并评估多种指标

    参数:
        emb_file1 (str): 源图嵌入文件路径
        emb_file2 (str): 目标图嵌入文件路径
        true_align_dict (dict): 真实对齐字典 {source_node: target_node}
        batch_size (int): 批处理大小 (用于内存优化)

    返回:
        dict: 包含所有评估指标的字典
        dict: 预测的对齐字典
        np.ndarray: 相似性矩阵 (部分或完整)
    """
    logger.info("开始节点对齐评估...")

    # 1. 加载嵌入文件
    def load_embeddings(file_path):
        """安全加载嵌入文件并进行ID检查和去重"""
        df = pd.read_csv(file_path)
        # 处理常见的不必要列
        drop_cols = [c for c in df.columns if c.lower() in ['unnamed:0', 'graph_id', 'node_index']]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # 检查node_id列是否存在
        if 'node_id' not in df.columns:
            raise ValueError(f"嵌入文件 {file_path} 缺少 'node_id' 列")

        # 节点ID去重
        if df['node_id'].duplicated().any():
            logger.warning(f"发现重复节点ID: {len(df) - len(df.drop_duplicates('node_id'))}")
            df = df.drop_duplicates('node_id', keep='first')

        df = df.set_index('node_id')
        logger.info(f"成功加载: {len(df)} 个节点, {df.shape[1]} 维嵌入")
        return df

    try:
        df1 = load_embeddings(emb_file1)
        df2 = load_embeddings(emb_file2)
    except Exception as e:
        logger.error(f"嵌入加载失败: {str(e)}")
        raise

    # 2. 准备嵌入矩阵
    src_emb = df1.values.astype(np.float32)
    tgt_emb = df2.values.astype(np.float32)

    # 3. 创建索引映射
    src_idx_map = {node: idx for idx, node in enumerate(df1.index)}
    src_node_map = {idx: node for node, idx in src_idx_map.items()}
    tgt_idx_map = {node: idx for idx, node in enumerate(df2.index)}
    tgt_node_map = {idx: node for node, idx in tgt_idx_map.items()}

    # 4. 筛选有效对齐对
    valid_alignment = {}
    for src_node, tgt_node in true_align_dict.items():
        if src_node in src_idx_map and tgt_node in tgt_idx_map:
            valid_alignment[src_node] = tgt_node

    valid_count = len(valid_alignment)
    if valid_count == 0:
        logger.error("无有效对齐节点对可用")
        return {}, {}, np.array([])

    logger.info(f"使用 {valid_count} 个有效对齐节点对进行评估")

    # 5. 初始化结果
    metrics = {
        'acc': 0,  # 准确率 (Hits@1)
        'hits@5': 0,  # Hits@5
        'hits@10': 0,  # Hits@10
        'mrr': 0.0,  # 平均倒数排名
        'avg_rank': 0.0  # 平均排名
    }
    predicted_alignment = {}
    all_ranks = []  # 存储所有节点的排名

    # 6. 分批计算相似性矩阵
    num_batches = (valid_count + batch_size - 1) // batch_size
    similarity_matrices = []

    logger.info(f"开始分批计算相似性矩阵 (批大小={batch_size})...")
    for batch_idx in tqdm(range(num_batches), desc="计算相似性"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, valid_count)

        # 获取当前批次的源节点
        batch_nodes = list(valid_alignment.keys())[start_idx:end_idx]
        batch_idxs = [src_idx_map[node] for node in batch_nodes]

        # 提取批次嵌入
        src_batch = src_emb[batch_idxs]

        # 计算批次相似性矩阵
        batch_sim_matrix = cosine_similarity(src_batch, tgt_emb)
        similarity_matrices.append(batch_sim_matrix)

        # 处理每个源节点
        for i, src_node in enumerate(batch_nodes):
            # 获取真实目标节点索引
            true_tgt_node = valid_alignment[src_node]
            true_idx = tgt_idx_map[true_tgt_node]

            # 获取当前节点的相似度向量
            sim_vector = batch_sim_matrix[i]

            # 计算排名
            sorted_indices = np.argsort(sim_vector)[::-1]  # 降序排列
            rank = np.where(sorted_indices == true_idx)[0][0] + 1  # 排名从1开始

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
            all_ranks.append(rank)

            # 记录预测 (使用top1)
            top1_idx = sorted_indices[0]
            pred_tgt_node = tgt_node_map[top1_idx]
            predicted_alignment[src_node] = pred_tgt_node

    # 7. 计算最终指标
    if valid_count > 0:
        metrics['acc'] /= valid_count
        metrics['hits@5'] /= valid_count
        metrics['hits@10'] /= valid_count
        metrics['mrr'] /= valid_count
        metrics['avg_rank'] = np.mean(all_ranks)

    # 8. 合并相似性矩阵
    full_sim_matrix = np.vstack(similarity_matrices) if similarity_matrices else np.array([])

    logger.info("评估完成:")
    logger.info(f"  ACC (Hits@1): {metrics['acc']:.4f}")
    logger.info(f"  Hits@5: {metrics['hits@5']:.4f}")
    logger.info(f"  Hits@10: {metrics['hits@10']:.4f}")
    logger.info(f"  MRR: {metrics['mrr']:.4f}")
    logger.info(f"  平均排名: {metrics['avg_rank']:.1f}")

    return metrics, predicted_alignment, full_sim_matrix


import networkx as nx
import numpy as np


def add_node_attributes_from_matrix(graph, attribute_matrix, attribute_name='features'):
    """
    将属性矩阵添加到图的节点中

    参数:
        graph (nx.Graph): NetworkX图对象
        attribute_matrix (np.ndarray): 属性矩阵，每行对应一个节点
        attribute_name (str): 属性名称，默认为'features'

    返回:
        nx.Graph: 添加属性后的图对象
    """
    # 检查输入有效性
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph必须是NetworkX图对象")

    if not isinstance(attribute_matrix, np.ndarray):
        raise TypeError("attribute_matrix必须是NumPy数组")

    if attribute_matrix.ndim != 2:
        raise ValueError("attribute_matrix必须是2维数组")

    # 获取节点数和属性维度
    num_nodes = graph.number_of_nodes()
    num_attributes = attribute_matrix.shape[1]

    # 验证节点数匹配
    if attribute_matrix.shape[0] != num_nodes:
        raise ValueError(f"属性矩阵行数({attribute_matrix.shape[0]})与图节点数({num_nodes})不匹配")

    # 获取排序后的节点ID
    sorted_nodes = sorted(graph.nodes())

    # 添加属性到每个节点
    for i, node_id in enumerate(sorted_nodes):
        # 获取当前节点的属性向量
        attributes = attribute_matrix[i].tolist()

        # 添加或更新节点属性
        if attribute_name in graph.nodes[node_id]:
            # 如果已有该属性，则追加新属性
            existing_attrs = graph.nodes[node_id][attribute_name]
            if not isinstance(existing_attrs, list):
                existing_attrs = [existing_attrs]
            graph.nodes[node_id][attribute_name] = existing_attrs + attributes
        else:
            # 如果无该属性，则创建新属性
            graph.nodes[node_id][attribute_name] = attributes

    return graph


# 高级版本：支持多种属性类型和验证
def enhanced_add_attributes(graph, attribute_matrix, attribute_name='features', validate=True):
    """
    增强版属性添加函数 - 支持多种属性类型和验证

    参数:
        graph (nx.Graph): NetworkX图对象
        attribute_matrix (np.ndarray): 属性矩阵
        attribute_name (str): 属性名称
        validate (bool): 是否执行严格验证

    返回:
        nx.Graph: 添加属性后的图对象
    """
    # 输入验证
    if validate:
        if not isinstance(graph, nx.Graph):
            raise TypeError("graph必须是NetworkX图对象")

        if not isinstance(attribute_matrix, np.ndarray):
            raise TypeError("attribute_matrix必须是NumPy数组")

        if attribute_matrix.ndim != 2:
            raise ValueError("attribute_matrix必须是2维数组")

        num_nodes = graph.number_of_nodes()
        if attribute_matrix.shape[0] != num_nodes:
            raise ValueError(f"属性矩阵行数({attribute_matrix.shape[0]})与图节点数({num_nodes})不匹配")

    # 获取排序后的节点ID
    sorted_nodes = sorted(graph.nodes())

    # 添加属性
    for i, node_id in enumerate(sorted_nodes):
        # 获取属性值
        attributes = attribute_matrix[i]

        # 处理不同数据类型
        if attributes.size == 1:
            # 标量值
            attr_value = attributes.item()
        else:
            # 向量值
            attr_value = attributes.tolist()

        # 设置节点属性
        graph.nodes[node_id][attribute_name] = attr_value

    return graph


# 批量添加多个属性矩阵
def add_multiple_attributes(graph, attribute_dict):
    """
    批量添加多个属性矩阵到图中

    参数:
        graph (nx.Graph): NetworkX图对象
        attribute_dict (dict): 属性字典，格式为{属性名: 属性矩阵}

    返回:
        nx.Graph: 添加属性后的图对象
    """
    # 验证输入
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph必须是NetworkX图对象")

    if not isinstance(attribute_dict, dict):
        raise TypeError("attribute_dict必须是字典")

    # 获取排序后的节点ID
    sorted_nodes = sorted(graph.nodes())
    num_nodes = len(sorted_nodes)

    # 添加每个属性
    for attr_name, attr_matrix in attribute_dict.items():
        # 验证属性矩阵
        if not isinstance(attr_matrix, np.ndarray):
            raise TypeError(f"属性'{attr_name}'的矩阵必须是NumPy数组")

        if attr_matrix.ndim != 2:
            raise ValueError(f"属性'{attr_name}'的矩阵必须是2维数组")

        if attr_matrix.shape[0] != num_nodes:
            raise ValueError(f"属性'{attr_name}'矩阵行数({attr_matrix.shape[0]})与图节点数({num_nodes})不匹配")

        # 添加属性
        for i, node_id in enumerate(sorted_nodes):
            # 获取属性值
            attributes = attr_matrix[i]

            # 处理不同数据类型
            if attributes.size == 1:
                attr_value = attributes.item()
            else:
                attr_value = attributes.tolist()

            # 设置节点属性
            graph.nodes[node_id][attr_name] = attr_value

    return graph


import numpy as np
import networkx as nx
import pandas as pd
import os

def save_node_features_with_adjacency(graph, output_path, feature_key='features'):
    """
    将节点属性向量和邻接矩阵行向量拼接后保存为CSV文件

    参数:
        graph: NetworkX图对象
        output_path: 输出CSV文件路径
        feature_key: 节点属性键名，默认为'features'
    """
    # 验证输入
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph必须是NetworkX图对象")

    # 获取排序后的节点ID
    sorted_nodes = sorted(graph.nodes())
    num_nodes = len(sorted_nodes)

    # 从图对象获取属性矩阵
    attr_matrix = []
    for node in sorted_nodes:
        # 获取节点属性
        node_attrs = graph.nodes[node].get(feature_key, [])

        # 确保属性是列表形式
        if not isinstance(node_attrs, list):
            node_attrs = [node_attrs]

        attr_matrix.append(node_attrs)

    # 转换为NumPy数组
    max_dim = max(len(attrs) for attrs in attr_matrix) if attr_matrix else 0
    if max_dim == 0:
        print("未找到节点特征属性，使用节点度作为默认特征")
        attr_matrix = [[graph.degree(node)] for node in sorted_nodes]
        max_dim = 1

    # 填充属性矩阵
    padded_attr_matrix = np.zeros((num_nodes, max_dim))
    for i, attrs in enumerate(attr_matrix):
        padded_attr_matrix[i, :len(attrs)] = attrs

    # 创建邻接矩阵
    adj_matrix = nx.to_numpy_array(graph, nodelist=sorted_nodes)

    # 拼接属性向量和邻接向量
    combined_features = np.hstack([padded_attr_matrix, adj_matrix])

    # 创建列名
    num_attrs = padded_attr_matrix.shape[1]
    num_nodes = adj_matrix.shape[1]
    columns = ['node_id'] + [f'dim_{i}' for i in range(num_attrs + num_nodes)]

    # 创建数据
    data = []
    for i, node in enumerate(sorted_nodes):
        row = [node] + combined_features[i].tolist()
        data.append(row)

    # 创建DataFrame
    df = pd.DataFrame(data, columns=columns)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存为CSV
    df.to_csv(output_path, index=False)
    print(f"成功保存拼接特征至 {output_path}, 包含 {len(df)} 个节点, 总维度: {combined_features.shape[1]}")

    return df


import pandas as pd
import networkx as nx
import logging
import os

def update_graph_features_from_csv(graph, csv_path, feature_key='features', sample_node=None):
    """
    从CSV文件读取节点嵌入特征并更新图对象

    参数:
        graph: NetworkX图对象
        csv_path: CSV文件路径
        feature_key: 节点特征属性键名 (默认为'features')
        sample_node: 验证节点ID (可选)

    返回:
        NetworkX图对象: 更新后的图对象
    """
    # 验证输入
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph必须是NetworkX图对象")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取CSV文件失败: {str(e)}")
        raise

    # 验证CSV格式
    if 'node_id' not in df.columns:
        raise ValueError("CSV文件必须包含'node_id'列")

    # 获取特征列
    feature_columns = [col for col in df.columns if col.startswith('dim_')]
    if not feature_columns:
        raise ValueError("CSV文件未找到任何'dim_'开头的特征列")

    print(f"找到 {len(feature_columns)} 个特征维度")

    # 创建节点ID到特征向量的映射
    node_features = {}
    for _, row in df.iterrows():
        node_id = row['node_id']
        # 确保节点ID类型与图中一致
        if node_id in graph.nodes:
            # 获取特征向量
            features = row[feature_columns].values.tolist()
            node_features[node_id] = features

    # 更新图节点特征
    for node_id, features in node_features.items():
        graph.nodes[node_id][feature_key] = features

    # 验证更新
    if sample_node:
        if sample_node in graph.nodes:
            print(f"验证节点 {sample_node} 的特征:")
            print(f"特征维度: {len(graph.nodes[sample_node][feature_key])}")
            print(f"前5个特征值: {graph.nodes[sample_node][feature_key][:5]}")
        else:
            print(f"验证节点 {sample_node} 不在图中")

    # 统计更新情况
    updated_nodes = len(node_features)
    total_nodes = graph.number_of_nodes()
    print(f"成功更新 {updated_nodes}/{total_nodes} 个节点的特征")

    return graph


def load_embeddings_from_csv(node_list1, node_list2, csv_path1, csv_path2):
    """
    从CSV文件加载节点嵌入并生成嵌入字典

    参数:
        node_list1: 图1的节点ID列表
        node_list2: 图2的节点ID列表
        csv_path1: 图1的节点嵌入CSV文件路径
        csv_path2: 图2的节点嵌入CSV文件路径

    返回:
        emb_dict1: 图1的节点嵌入字典 {node_id: embedding_vector}
        emb_dict2: 图2的节点嵌入字典 {node_id: embedding_vector}
    """
    # 读取CSV文件
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)

    # 提取嵌入列
    emb_cols1 = [col for col in df1.columns if col.startswith('dim_')]
    emb_cols2 = [col for col in df2.columns if col.startswith('dim_')]

    # 转换节点ID为整型
    df1['node_id'] = df1['node_id'].apply(convert_to_int)
    df2['node_id'] = df2['node_id'].apply(convert_to_int)

    # 过滤有效节点
    valid_df1 = df1[df1['node_id'].isin(node_list1)]
    valid_df2 = df2[df2['node_id'].isin(node_list2)]

    # 创建嵌入字典
    emb_dict1 = dict(zip(valid_df1['node_id'], valid_df1[emb_cols1].values))
    emb_dict2 = dict(zip(valid_df2['node_id'], valid_df2[emb_cols2].values))

    return emb_dict1, emb_dict2


def convert_to_int(raw_id):
    """将原始ID转换为整型"""
    try:
        # 如果是浮点数且是整数，转换为整数
        if isinstance(raw_id, float) and raw_id.is_integer():
            return int(raw_id)
        # 如果是字符串，尝试转换为整数
        if isinstance(raw_id, str):
            return int(float(raw_id)) if '.' in raw_id else int(raw_id)
        # 其他情况直接转换
        return int(raw_id)
    except:
        # 转换失败返回None
        return None


def extract_node_embedding(center_node_id,csv_path,d_percent):
    df = pd.read_csv(csv_path)
    embedding_dict = df.set_index('node_id').to_dict(orient='index')

    # 将每个节点的Series转换为列表（按列顺序）
    for node in embedding_dict:
        embedding_dict[node] = list(embedding_dict[node].values())

    # 4. 提取目标节点的嵌入向量
    embedding = embedding_dict[center_node_id]

    # 5. 计算边界列表
    bounds = []
    d = d_percent / 100.0
    for i in range(len(embedding)):
        bound1 = embedding[i] * (1 - d)  # 可能的下界或上界
        bound2 = embedding[i] * (1 + d)  # 可能的上界或下界
        # bound1=val+d
        # bound2=val-d
        lower_bound = min(bound1, bound2)  # 取较小值作为下界
        upper_bound = max(bound1, bound2)  # 取较大值作为上界
        bounds.append((lower_bound, upper_bound))
    # print(bounds)
    #
    count=0
    count_nodes=set()
    for node, _ in embedding_dict.items():
        is_True=True
        emb=embedding_dict[node]
        for j in range(len(emb)):
            low,up=bounds[j]
            if emb[j]<=up and emb[j]>=low:
                # print(emb[j],low,up,"True")
                is_True=True
            else:
                # print(emb[j], low, up, "False")
                is_True=False
        if is_True==True:
            count+=1
            count_nodes.add(node)
    return count_nodes,count


import pandas as pd
import numpy as np
from collections import defaultdict


def load_node_embeddings(csv_path, node_list):
    """
    从CSV文件加载指定节点的嵌入向量

    参数:
        csv_path: CSV文件路径
        node_list: 节点列表，只加载这些节点的嵌入

    返回:
        字典: {node_id: embedding_vector}
    """
    df = pd.read_csv(csv_path)

    # 构建嵌入字典
    embedding_dict = {}

    for _, row in df.iterrows():
        node_id = row['node_id']
        if node_id in node_list:
            # 提取所有维度值（跳过node_id列）
            embedding = row.iloc[1:].values.astype(float)
            embedding_dict[node_id] = embedding

    return embedding_dict


def cosine_similarity_1(vec1, vec2):
    """
    计算两个向量的余弦相似度

    参数:
        vec1: 第一个向量
        vec2: 第二个向量

    返回:
        余弦相似度值
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def find_max_similarity_pairs(node_list1, node_list2, csv_path1, csv_path2):
    """
    找到两个节点列表中每个节点的最大相似度匹配

    参数:
        node_list1: 第一个节点列表
        node_list2: 第二个节点列表
        csv_path1: 第一个嵌入CSV文件路径
        csv_path2: 第二个嵌入CSV文件路径

    返回:
        字典: {(node1, node2): similarity_score}
    """
    # 1. 加载两个节点列表的嵌入向量
    embeddings1 = load_node_embeddings(csv_path1, node_list1)
    embeddings2 = load_node_embeddings(csv_path2, node_list2)

    # 2. 检查哪些节点成功加载
    valid_nodes1 = [node for node in node_list1 if node in embeddings1]
    valid_nodes2 = [node for node in node_list2 if node in embeddings2]

    if not valid_nodes1:
        print(f"警告: 第一个节点列表中没有节点在 {csv_path1} 中找到")
        return {}

    if not valid_nodes2:
        print(f"警告: 第二个节点列表中没有节点在 {csv_path2} 中找到")
        return {}

    # 3. 构建相似度矩阵
    n = len(valid_nodes1)
    m = len(valid_nodes2)

    # 创建相似度矩阵
    similarity_matrix = np.zeros((n, m))

    # 填充相似度矩阵
    for i, node1 in enumerate(valid_nodes1):
        for j, node2 in enumerate(valid_nodes2):
            similarity_matrix[i, j] = cosine_similarity_1(
                embeddings1[node1], embeddings2[node2]
            )

    # 4. 找到每一行的最大相似度及其索引
    max_similarity_dict = {}

    for i, node1 in enumerate(valid_nodes1):
        # 找到当前行的最大相似度
        max_sim = np.max(similarity_matrix[i, :])
        max_indices = np.where(similarity_matrix[i, :] == max_sim)[0]

        # 如果有多个相同最大相似度的节点，选择第一个
        if len(max_indices) > 0:
            max_index = max_indices[0]
            node2 = valid_nodes2[max_index]

            # 添加到字典
            max_similarity_dict[(node1, node2)] = max_sim

    return max_similarity_dict


import pandas as pd
import numpy as np
from collections import defaultdict


def compute_evaluation_metrics_with_debug(similarity_matrix, valid_nodes1, valid_nodes2, prior_alignment_dict=None):
    """
    计算评估指标：hits@1, hits@5, hits@10, MRR, Accuracy, Recall，并返回调试信息

    参数:
        similarity_matrix: 相似度矩阵，形状为 (n, m)
        valid_nodes1: 有效的节点列表1
        valid_nodes2: 有效的节点列表2
        prior_alignment_dict: 先验对齐字典，格式为 {node1: node2}

    返回:
        字典: 包含各项评估指标和调试信息
    """
    if prior_alignment_dict is None or not prior_alignment_dict:
        return {
            'hits@1': 0.0,
            'hits@5': 0.0,
            'hits@10': 0.0,
            'mrr': 0.0,
            'accuracy': 0.0,  # 准确率指标
            'recall': 0.0,  # 召回率指标
            'num_evaluated_nodes': 0,
            'total_possible_nodes': 0,
            'skipped_nodes': [],
            'skipped_reasons': defaultdict(int)
        }

    n, m = similarity_matrix.shape
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    evaluated_nodes = 0

    # 调试信息
    skipped_nodes = []
    skipped_reasons = defaultdict(int)

    # 计算总可能的评估节点数
    total_possible_nodes = 0

    # 计算召回率相关的统计
    total_true_positives = 0
    total_false_negatives = 0

    # 为每个节点计算排名
    for i, node1 in enumerate(valid_nodes1):
        # 如果节点1不在先验对齐字典中
        if node1 not in prior_alignment_dict:
            total_possible_nodes += 1
            skipped_nodes.append({
                'node1': node1,
                'reason': 'node1 not in prior_alignment_dict',
                'details': f'节点 {node1} 不在先验对齐字典中'
            })
            skipped_reasons['node1_not_in_prior'] += 1
            continue

        # 获取真实匹配的节点2
        true_node2 = prior_alignment_dict[node1]
        total_possible_nodes += 1

        # 如果真实匹配节点不在节点列表2中
        if true_node2 not in valid_nodes2:
            skipped_nodes.append({
                'node1': node1,
                'true_node2': true_node2,
                'reason': 'true_node2 not in valid_nodes2',
                'details': f'节点 {node1} 的真实匹配节点 {true_node2} 不在有效节点列表2中'
            })
            skipped_reasons['true_node2_not_in_valid_nodes2'] += 1
            continue

        # 获取当前节点在节点列表2中的索引
        true_index = valid_nodes2.index(true_node2)

        # 获取当前行（节点1与所有节点2的相似度）
        row_similarities = similarity_matrix[i, :]

        # 计算当前行中每个相似度的排名（从高到低排序）
        sorted_indices = np.argsort(-row_similarities)

        # 查找真实匹配的排名（从0开始）
        rank = np.where(sorted_indices == true_index)[0][0] + 1

        # 计算hits指标
        if rank == 1:
            hits_at_1 += 1
            total_true_positives += 1
        else:
            total_false_negatives += 1

        if rank <= 5:
            hits_at_5 += 1
        if rank <= 10:
            hits_at_10 += 1

        # 计算倒数排名
        reciprocal_ranks.append(1.0 / rank)
        evaluated_nodes += 1

    # 计算指标
    if evaluated_nodes > 0:
        hits_at_1_rate = hits_at_1 / evaluated_nodes
        hits_at_5_rate = hits_at_5 / evaluated_nodes
        hits_at_10_rate = hits_at_10 / evaluated_nodes
        mrr_score = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        accuracy_score = hits_at_1_rate

        # 计算召回率
        if total_true_positives + total_false_negatives > 0:
            recall_score = total_true_positives / (total_true_positives + total_false_negatives)
        else:
            recall_score = 0.0
    else:
        hits_at_1_rate = 0.0
        hits_at_5_rate = 0.0
        hits_at_10_rate = 0.0
        mrr_score = 0.0
        accuracy_score = 0.0
        recall_score = 0.0

    return {
        'hits@1': hits_at_1_rate,
        'hits@5': hits_at_5_rate,
        'hits@10': hits_at_10_rate,
        'mrr': mrr_score,
        'accuracy': accuracy_score,  # 准确率指标
        'recall': recall_score,  # 召回率指标
        'num_evaluated_nodes': evaluated_nodes,
        'total_possible_nodes': total_possible_nodes,
        'skipped_nodes': skipped_nodes,
        'skipped_reasons': dict(skipped_reasons),
        'true_positives': total_true_positives,
        'false_negatives': total_false_negatives
    }


def find_max_similarity_pairs_with_debug(node_list1, node_list2, csv_path1, csv_path2, prior_alignment_dict=None):
    """
    找到两个节点列表中每个节点的最大相似度匹配，并计算评估指标（带调试信息）

    参数:
        node_list1: 第一个节点列表
        node_list2: 第二个节点列表
        csv_path1: 第一个嵌入CSV文件路径
        csv_path2: 第二个嵌入CSV文件路径
        prior_alignment_dict: 先验对齐字典，格式为 {node1: node2}

    返回:
        tuple: (max_similarity_dict, evaluation_metrics)
        max_similarity_dict: {(node1, node2): similarity_score}
        evaluation_metrics: 包含评估指标和调试信息的字典
    """
    # 1. 加载两个节点列表的嵌入向量
    embeddings1 = load_node_embeddings(csv_path1, node_list1)
    embeddings2 = load_node_embeddings(csv_path2, node_list2)

    # 2. 检查哪些节点成功加载
    valid_nodes1 = [node for node in node_list1 if node in embeddings1]
    valid_nodes2 = [node for node in node_list2 if node in embeddings2]

    # 打印加载信息
    print("节点加载情况:")
    print(f"  节点列表1: 总节点数={len(node_list1)}, 成功加载={len(valid_nodes1)}")
    print(f"  节点列表2: 总节点数={len(node_list2)}, 成功加载={len(valid_nodes2)}")

    if not valid_nodes1:
        print(f"警告: 第一个节点列表中没有节点在 {csv_path1} 中找到")
        return {}, {}

    if not valid_nodes2:
        print(f"警告: 第二个节点列表中没有节点在 {csv_path2} 中找到")
        return {}, {}

    # 3. 构建相似度矩阵
    n = len(valid_nodes1)
    m = len(valid_nodes2)

    # 创建相似度矩阵
    similarity_matrix = np.zeros((n, m))

    # 填充相似度矩阵
    for i, node1 in enumerate(valid_nodes1):
        for j, node2 in enumerate(valid_nodes2):
            similarity_matrix[i, j] = cosine_similarity_1(
                embeddings1[node1], embeddings2[node2]
            )

    # 4. 计算评估指标（带调试信息）
    evaluation_metrics = compute_evaluation_metrics_with_debug(
        similarity_matrix, valid_nodes1, valid_nodes2, prior_alignment_dict
    )

    # 5. 找到每一行的最大相似度及其索引
    max_similarity_dict = {}

    for i, node1 in enumerate(valid_nodes1):
        # 找到当前行的最大相似度
        max_sim = np.max(similarity_matrix[i, :])
        max_indices = np.where(similarity_matrix[i, :] == max_sim)[0]

        # 如果有多个相同最大相似度的节点，选择第一个
        if len(max_indices) > 0:
            max_index = max_indices[0]
            node2 = valid_nodes2[max_index]

            # 添加到字典
            max_similarity_dict[(node1, node2)] = max_sim

    return max_similarity_dict, evaluation_metrics


"""
NetworkX图节点特征提取与CSV保存工具
功能：提取NetworkX图中节点的features属性，并保存为CSV文件
第一列：node_id，后续列：特征维度（dim_1, dim_2, ...）
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import warnings
import re
import os

warnings.filterwarnings("ignore", category=UserWarning)


class GraphFeatureExtractor:
    """NetworkX图节点特征提取器"""

    def __init__(self, debug_mode: bool = False):
        """
        初始化图特征提取器

        参数:
            debug_mode: 是否开启调试模式，输出详细信息
        """
        self.debug_mode = debug_mode

    def extract_single_node_features(self, node_data: Dict) -> Optional[List[float]]:
        """
        提取单个节点的特征

        参数:
            node_data: 节点属性字典

        返回:
            特征向量列表，如果提取失败则返回None
        """
        # 尝试从不同属性名中获取特征
        feat = None
        for attr_name in ['features', 'feature', 'attr', 'attributes', 'feat']:
            if attr_name in node_data:
                feat = node_data[attr_name]
                break

        # 如果节点没有特征属性
        if feat is None:
            if self.debug_mode:
                print(f"    节点没有特征属性，返回空特征向量")
            return []  # 返回空列表

        # 处理不同类型的数据
        try:
            if isinstance(feat, (int, float)):
                # 标量特征
                return [float(feat)]

            elif isinstance(feat, list):
                # 列表特征
                if len(feat) == 0:
                    return []  # 空列表

                # 尝试转换为浮点数
                processed = []
                for item in feat:
                    if isinstance(item, (int, float)):
                        processed.append(float(item))
                    elif isinstance(item, str):
                        # 尝试将字符串转换为数字
                        try:
                            processed.append(float(item))
                        except:
                            if self.debug_mode:
                                print(f"    列表中的字符串无法转换为数字: {item}")
                            processed.append(0.0)
                    else:
                        if self.debug_mode:
                            print(f"    列表包含无法处理的类型: {type(item)}")
                        processed.append(0.0)
                return processed

            elif isinstance(feat, np.ndarray):
                # numpy数组
                return feat.astype(float).tolist()

            elif isinstance(feat, str):
                # 字符串特征
                # 尝试解析多种格式

                # 1. 包含方括号的列表字符串，例如 "[1.0, 2.0, 3.0]"
                if '[' in feat and ']' in feat:
                    match = re.search(r'\[(.*?)\]', feat)
                    if match:
                        content = match.group(1)
                        # 分割并转换为浮点数
                        try:
                            elements = [x.strip() for x in content.split(',')]
                            return [float(x) for x in elements if x != '']
                        except:
                            if self.debug_mode:
                                print(f"    无法解析列表字符串: {feat}")
                            return []

                # 2. 用逗号或其他分隔符分隔的字符串
                separators = [',', ';', '|', ' ']
                for sep in separators:
                    if sep in feat:
                        try:
                            elements = [x.strip() for x in feat.split(sep)]
                            return [float(x) for x in elements if x != '']
                        except:
                            continue

                # 3. 尝试直接转换为浮点数
                try:
                    return [float(feat)]
                except:
                    if self.debug_mode:
                        print(f"    无法解析字符串: {feat}")
                    return []

            elif hasattr(feat, '__len__'):
                # 其他可迭代对象
                try:
                    return [float(x) for x in feat]
                except:
                    if self.debug_mode:
                        print(f"    无法处理的可迭代对象: {type(feat)}")
                    return []

            else:
                # 其他类型
                if self.debug_mode:
                    print(f"    无法处理的类型: {type(feat)}")
                return []

        except Exception as e:
            if self.debug_mode:
                print(f"    特征提取异常: {e}")
            return None

    def extract_graph_features(self, G: nx.Graph) -> Tuple[List, List[List[float]], Dict[str, int]]:
        """
        从图中提取所有节点的特征

        参数:
            G: NetworkX图对象

        返回:
            node_ids: 节点ID列表
            features: 特征矩阵列表
            stats: 统计信息字典
        """
        node_ids = list(G.nodes())
        n_nodes = len(node_ids)

        if self.debug_mode:
            print(f"图信息: {n_nodes} 个节点, {G.number_of_edges()} 条边")
            print(f"前5个节点ID: {node_ids[:5]}")

        features = []
        failed_nodes = 0
        empty_features = 0

        for i, node in enumerate(node_ids):
            node_data = G.nodes[node]

            # 提取节点特征
            node_features = self.extract_single_node_features(node_data)

            if node_features is None:
                # 提取失败
                failed_nodes += 1
                if self.debug_mode and failed_nodes <= 3:  # 只显示前3个失败的节点
                    print(f"  节点 {node}: 特征提取失败")
                continue
            elif len(node_features) == 0:
                # 空特征
                empty_features += 1
                if self.debug_mode and empty_features <= 3:  # 只显示前3个空特征的节点
                    print(f"  节点 {node}: 空特征向量")

            features.append(node_features)

        # 统计信息
        stats = {
            'total_nodes': n_nodes,
            'nodes_with_features': len(features),
            'failed_nodes': failed_nodes,
            'empty_features': empty_features,
            'success_rate': len(features) / n_nodes if n_nodes > 0 else 0
        }

        if self.debug_mode:
            print(f"\n特征提取统计:")
            print(f"  总节点数: {stats['total_nodes']}")
            print(f"  成功提取特征的节点数: {stats['nodes_with_features']}")
            print(f"  提取失败的节点数: {stats['failed_nodes']}")
            print(f"  空特征节点数: {stats['empty_features']}")
            print(f"  成功率: {stats['success_rate']:.2%}")

        return node_ids, features, stats

    def align_features(self, features: List[List[float]]) -> List[List[float]]:
        """
        对齐特征维度（填充到最大维度）

        参数:
            features: 特征列表

        返回:
            对齐后的特征列表
        """
        if not features:
            return features

        # 找到最大特征维度
        max_dim = max(len(f) for f in features)

        if self.debug_mode:
            print(f"对齐特征维度: 最大维度 = {max_dim}")

        # 填充特征
        aligned_features = []
        for f in features:
            if len(f) < max_dim:
                # 填充0
                aligned_features.append(f + [0.0] * (max_dim - len(f)))
            else:
                # 截断到最大维度
                aligned_features.append(f[:max_dim])

        return aligned_features

    def save_features_to_csv(self, node_ids: List, features: List[List[float]],
                             filename: str) -> bool:
        """
        将特征保存为CSV文件

        参数:
            node_ids: 节点ID列表
            features: 特征矩阵
            filename: 输出文件名

        返回:
            是否保存成功
        """
        if not features:
            print(f"错误: 没有特征可保存")
            return False

        n_nodes = len(node_ids)
        n_features = len(features)

        if n_nodes != n_features:
            print(f"警告: 节点数 ({n_nodes}) 与特征数 ({n_features}) 不匹配")
            # 取较小值
            min_len = min(n_nodes, n_features)
            node_ids = node_ids[:min_len]
            features = features[:min_len]
            print(f"  已截断到 {min_len} 个节点")

        # 对齐特征维度
        aligned_features = self.align_features(features)
        n_dim = len(aligned_features[0]) if aligned_features else 0

        if self.debug_mode:
            print(f"保存到文件: {filename}")
            print(f"  节点数: {len(node_ids)}")
            print(f"  特征维度: {n_dim}")
            if aligned_features:
                print(f"  特征示例: {aligned_features[0]}")

        # 创建DataFrame
        data = []
        for i, node_id in enumerate(node_ids):
            row = {'node_id': node_id}
            for j in range(n_dim):
                row[f'dim_{j + 1}'] = aligned_features[i][j]
            data.append(row)

        df = pd.DataFrame(data)

        # 确保目录存在
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

        # 保存CSV
        df.to_csv(filename, index=False)

        # 验证文件
        if os.path.exists(filename):
            loaded_df = pd.read_csv(filename)
            success = len(loaded_df) == len(df)

            if success:
                if self.debug_mode:
                    print(f"文件保存成功: {filename}")
                    print(f"  行数: {len(loaded_df)}, 列数: {len(loaded_df.columns)}")

                    # 检查NaN值
                    nan_count = loaded_df.isna().sum().sum()
                    if nan_count > 0:
                        print(f"  警告: 文件中包含 {nan_count} 个NaN值")
            else:
                print(f"警告: 文件保存后验证失败")

            return success
        else:
            print(f"错误: 文件未成功创建")
            return False

    def extract_and_save_features(self, G: nx.Graph, filename: str) -> Dict[str, int]:
        """
        提取并保存图的特征到CSV文件

        参数:
            G: NetworkX图对象
            filename: 输出文件名

        返回:
            统计信息字典
        """
        print(f"\n{'=' * 60}")
        print(f"开始处理图: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")

        # 提取特征
        node_ids, features, stats = self.extract_graph_features(G)

        if stats['nodes_with_features'] == 0:
            print(f"警告: 没有提取到任何节点特征")
            return stats

        # 保存到CSV
        success = self.save_features_to_csv(node_ids, features, filename)

        if success:
            print(f"✓ 特征已保存到: {filename}")
            print(f"  保存了 {stats['nodes_with_features']} 个节点的特征")

            if stats['failed_nodes'] > 0:
                print(f"  ⚠️  {stats['failed_nodes']} 个节点特征提取失败")
            if stats['empty_features'] > 0:
                print(f"  ⚠️  {stats['empty_features']} 个节点特征为空")
        else:
            print(f"✗ 特征保存失败")

        print(f"{'=' * 60}")

        return stats


def save_graph_features_to_csv(G1: nx.Graph, G2: nx.Graph,
                               filename1: str = "graph1_features.csv",
                               filename2: str = "graph2_features.csv",
                               debug_mode: bool = False) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    主函数：提取两个图的节点特征并保存为CSV文件

    参数:
        G1: 第一个NetworkX图对象
        G2: 第二个NetworkX图对象
        filename1: 第一个图的输出文件名
        filename2: 第二个图的输出文件名
        debug_mode: 是否开启调试模式

    返回:
        两个图的统计信息字典
    """
    print("=" * 60)
    print("NetworkX图节点特征提取与保存工具")
    print("=" * 60)

    # 创建特征提取器
    extractor = GraphFeatureExtractor(debug_mode=debug_mode)

    # 处理第一个图
    print(f"\n处理第一个图:")
    stats1 = extractor.extract_and_save_features(G1, filename1)

    # 处理第二个图
    print(f"\n处理第二个图:")
    stats2 = extractor.extract_and_save_features(G2, filename2)

    # 总体统计
    print(f"\n{'=' * 60}")
    print("处理完成!")
    print(f"图1: 保存了 {stats1.get('nodes_with_features', 0)}/{stats1.get('total_nodes', 0)} 个节点特征")
    print(f"图2: 保存了 {stats2.get('nodes_with_features', 0)}/{stats2.get('total_nodes', 0)} 个节点特征")
    print(f"{'=' * 60}")

    return stats1, stats2


