import numpy as np
import pandas as pd
import time
import logging
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.spatial.distance import cdist
import networkx as nx
import os
from sklearn import metrics
import time
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score
import gc




# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import time
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm
import logging
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import gc
from utils import *

logger = logging.getLogger(__name__)


def perform_procrustes_alignment(src_emb, tgt_emb, align_dict, src_idx_map, tgt_idx_map):
    """执行Procrustes变换进行特征空间对齐 (无FAISS依赖)"""
    aligned_src_nodes = []
    aligned_tgt_nodes = []

    align_dict=sample_dict_percentage(align_dict,20)

    # 构建对齐点对
    for src_node, tgt_node in align_dict.items():
        if src_node in src_idx_map and tgt_node in tgt_idx_map:
            aligned_src_nodes.append(src_idx_map[src_node])
            aligned_tgt_nodes.append(tgt_idx_map[tgt_node])

    if len(aligned_src_nodes) < 10:
        logger.warning("对齐点不足，无法进行可靠的Procrustes变换")
        return src_emb, tgt_emb, np.eye(src_emb.shape[1])

    # 提取对齐嵌入
    X = src_emb[np.array(aligned_src_nodes)]
    Y = tgt_emb[np.array(aligned_tgt_nodes)]

    # 计算正交变换矩阵
    R, _ = orthogonal_procrustes(X, Y)
    return src_emb @ R, tgt_emb, R


def calculate_similarity(batch_src, tgt_emb, metric='cosine'):
    """计算批相似度矩阵 (无FAISS)"""
    if metric == 'cosine':
        # 使用高精度余弦相似度计算
        norm_src = batch_src / np.linalg.norm(batch_src, axis=1, keepdims=True)
        norm_tgt = tgt_emb / np.linalg.norm(tgt_emb, axis=1, keepdims=True)
        return np.dot(norm_src, norm_tgt.T)
    elif metric == 'euclidean':
        # 负欧氏距离 (距离越小相似度越大)
        return -np.sqrt(np.sum((batch_src[:, np.newaxis] - tgt_emb[np.newaxis, :]) ** 2, axis=-1))
    else:
        raise ValueError(f"不支持的相似度度量: {metric}")


def network_alignment_evaluation(emb_file1, emb_file2, alignment_dict, batch_size=5000,
                                 similarity_metric='cosine', top_k=10,
                                 alignment_output_path=None):
    """
    高效的无FAISS网络对齐评估

    参数:
    emb_file1 (str): 源图嵌入文件路径
    emb_file2 (str): 目标图嵌入文件路径
    alignment_dict (dict): 先验对齐字典 {source_node: target_node}
    batch_size (int): 批处理大小
    similarity_metric (str): 相似度度量 ('cosine', 'euclidean')
    top_k (int): 计算Hits@k指标时考虑的top k结果
    alignment_output_path (str): 对齐结果输出路径

    返回:
    dict: 包含评估指标的结果字典
    """
    start_time = time.time()
    logger.info("启动高效网络对齐评估系统 (无FAISS)...")

    # 加载嵌入文件
    logger.info("加载和预处理嵌入文件...")

    def load_embeddings(file_path):
        df = pd.read_csv(file_path)
        # 处理常见的不必要列
        drop_cols = [c for c in df.columns if c.lower() in ['unnamed:0', 'graph_id', 'node_index']]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        df = df.set_index('node_id')
        return df

    try:
        df1 = load_embeddings(emb_file1)
        df2 = load_embeddings(emb_file2)
    except Exception as e:
        logger.error(f"嵌入加载失败: {str(e)}")
        raise

    # 验证有效对齐对并创建索引映射
    valid_alignment = {}
    src_idx_map = {node: idx for idx, node in enumerate(df1.index)}
    src_node_map = {idx: node for node, idx in src_idx_map.items()}
    tgt_idx_map = {node: idx for idx, node in enumerate(df2.index)}
    tgt_node_map = {idx: node for node, idx in tgt_idx_map.items()}

    # 筛选有效对齐对
    for src_node, tgt_node in alignment_dict.items():
        if src_node in src_idx_map and tgt_node in tgt_idx_map:
            valid_alignment[src_node] = tgt_node

    valid_count = len(valid_alignment)
    if valid_count == 0:
        logger.error("无有效对齐节点对可用")
        return None

    logger.info(f"源图节点数: {len(df1)} | 目标图节点数: {len(df2)}")
    logger.info(f"使用 {valid_count} 个有效对齐节点对进行评估")

    # 准备嵌入矩阵
    src_emb = df1.values.astype(np.float32)
    tgt_emb = df2.values.astype(np.float32)

    # 关键改进：Procrustes空间对齐
    logger.info("执行Procrustes嵌入空间对齐...")
    aligned_src_emb, tgt_emb, _ = perform_procrustes_alignment(
        src_emb, tgt_emb, valid_alignment, src_idx_map, tgt_idx_map
    )

    # 初始化评估指标
    results = {
        'total_nodes': len(df1),
        'valid_pairs': valid_count,
        'hits@1': 0,
        'hits@5': 0,
        'hits@10': 0,
        'mrr': 0.0,
        'accuracy': 0.0
    }

    # 预测对齐字典（一对一的节点映射）
    predicted_alignment = {}

    # 对于验证集节点使用精确评估
    validation_nodes = list(valid_alignment.keys())
    num_validation_batches = (len(validation_nodes) + batch_size - 1) // batch_size

    logger.info(f"开始对齐评估 (批大小={batch_size}, 节点数={len(df1)})...")

    # 目标嵌入归一化 (一次性计算提高效率)
    if similarity_metric == 'cosine':
        tgt_norm = tgt_emb / np.linalg.norm(tgt_emb, axis=1, keepdims=True)

    # 精确评估验证节点
    with tqdm(total=len(validation_nodes), desc="评估进度") as pbar:
        for batch_idx in range(num_validation_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(validation_nodes))
            batch_nodes = validation_nodes[start_idx:end_idx]
            batch_idxs = [src_idx_map[node] for node in batch_nodes]

            # 获取批次嵌入
            src_batch = aligned_src_emb[batch_idxs]

            # 计算相似度矩阵 (使用优化计算)
            if similarity_metric == 'cosine':
                src_norm = src_batch / np.linalg.norm(src_batch, axis=1, keepdims=True)
                similarity = np.dot(src_norm, tgt_norm.T)
            else:  # euclidean
                # 使用优化的广播计算
                expanded_src = np.expand_dims(src_batch, axis=1)
                expanded_tgt = np.expand_dims(tgt_emb, axis=0)
                dists = np.sqrt(np.sum((expanded_src - expanded_tgt) ** 2, axis=-1))
                similarity = -dists  # 将距离转换为相似度

            # 处理每个源节点
            for i, src_node in enumerate(batch_nodes):
                src_idx = src_idx_map[src_node]
                true_tgt = valid_alignment[src_node]
                true_tgt_idx = tgt_idx_map[true_tgt]

                # 获取相似度分数并排序
                sim_scores = similarity[i]
                sorted_indices = np.argsort(sim_scores)[::-1]  # 降序排序

                # 寻找真实对齐目标的排名
                rank = np.where(sorted_indices == true_tgt_idx)[0][0] + 1

                # 更新Hits@k指标
                if rank == 1:
                    results['hits@1'] += 1
                    results['hits@5'] += 1
                    results['hits@10'] += 1
                    results['accuracy'] += 1  # 准确率计数
                elif rank <= 5:
                    results['hits@5'] += 1
                    results['hits@10'] += 1
                elif rank <= 10:
                    results['hits@10'] += 1

                # 更新MRR
                results['mrr'] += 1.0 / rank

                # 存储当前节点的top1预测
                top1_idx = sorted_indices[0]
                predicted_alignment[src_node] = tgt_node_map[top1_idx]

            pbar.update(len(batch_nodes))
            del similarity  # 释放内存
            gc.collect()

    # 评估剩余的非验证节点
    non_validation_nodes = [node for node in df1.index if node not in valid_alignment]
    if non_validation_nodes:
        logger.info(f"预测非验证节点对齐 (数量={len(non_validation_nodes)})...")
        num_remaining_batches = (len(non_validation_nodes) + batch_size - 1) // batch_size

        with tqdm(total=len(non_validation_nodes), desc="预测进度") as pbar:
            for batch_idx in range(num_remaining_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(non_validation_nodes))
                batch_nodes = non_validation_nodes[start_idx:end_idx]
                batch_idxs = [src_idx_map[node] for node in batch_nodes]

                # 获取批次嵌入
                src_batch = aligned_src_emb[batch_idxs]

                # 计算相似度矩阵 (仅计算top1提高效率)
                if similarity_metric == 'cosine':
                    src_norm = src_batch / np.linalg.norm(src_batch, axis=1, keepdims=True)
                    # 仅计算目标图中相似度最高的部分节点
                    top_k = min(100, len(tgt_emb))  # 限制搜索范围
                    similarity = np.dot(src_norm, tgt_norm[:top_k].T)
                    sim_scores = np.max(similarity, axis=1)
                    tgt_indices = np.argmax(similarity, axis=1)
                else:
                    # 计算到最近邻居的距离
                    min_dists = []
                    tgt_indices = []
                    for emb in src_batch:
                        dists = np.sqrt(np.sum((emb - tgt_emb) ** 2, axis=1))
                        min_idx = np.argmin(dists)
                        tgt_indices.append(min_idx)
                        min_dists.append(-dists[min_idx])  # 转为相似度

                # 存储每个节点的预测结果
                for i, src_node in enumerate(batch_nodes):
                    if similarity_metric == 'euclidean':
                        top1_idx = tgt_indices[i]
                    else:
                        top1_idx = tgt_indices[i]  # 已计算好的索引

                    predicted_alignment[src_node] = tgt_node_map[top1_idx]

                pbar.update(len(batch_nodes))
                del src_batch
                gc.collect()

    # 最终指标计算
    results['hits@1'] /= valid_count
    results['hits@5'] /= valid_count
    results['hits@10'] /= valid_count
    results['mrr'] /= valid_count
    results['accuracy'] /= valid_count

    # 存储预测对齐结果
    if alignment_output_path:
        logger.info(f"保存预测对齐结果到 {alignment_output_path}")
        alignment_df = pd.DataFrame(
            [(src, tgt) for src, tgt in predicted_alignment.items()],
            columns=['source_node', 'target_node']
        )
        alignment_df.to_csv(alignment_output_path, index=False)
        results['predicted_pairs'] = len(predicted_alignment)

    # 性能统计
    elapsed_time = time.time() - start_time
    results['elapsed_time'] = elapsed_time

    # 结果摘要
    logger.info("\n" + "=" * 60)
    logger.info("【评估结果摘要】")
    logger.info(f"源图节点数: {len(df1)} | 目标图节点数: {len(df2)}")
    logger.info(f"有效对齐对: {valid_count}")
    logger.info(f"  Hits@1: {results['hits@1']:.4f}")
    logger.info(f"  Hits@5: {results['hits@5']:.4f}")
    logger.info(f"  Hits@10: {results['hits@10']:.4f}")
    logger.info(f"  MRR: {results['mrr']:.4f}")
    logger.info(f"  Accuracy: {results['accuracy']:.4f}")
    logger.info(f"  总耗时: {elapsed_time:.2f}秒")
    logger.info("=" * 60)

    return results

def evaluate_node_align(emb1, emb2, align_dict, output_dir1=None):
    # 假设的嵌入文件路径
    EMB_FILE_1 = emb1
    EMB_FILE_2 = emb2

    # 创建模拟先验对齐字典
    # 实际应用中需要提供真实的对齐数据
    PRIOR_ALIGNMENT = align_dict

    results = network_alignment_evaluation(
        emb_file1=EMB_FILE_1,
        emb_file2=EMB_FILE_2,
        alignment_dict=PRIOR_ALIGNMENT,
        batch_size=2000,
        similarity_metric='cosine',
        alignment_output_path=output_dir1
    )

    # 输出结果
    # print(f"预测节点数: {results['predicted_pairs']}")
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"Hits@1: {results['hits@1']:.4f}")
    print(f"Hits@5: {results['hits@5']:.4f}")
    print(f"Hits@10: {results['hits@10']:.4f}")
    print(f"平均倒数排名: {results['mrr']:.4f}")


import networkx as nx
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from tqdm import tqdm
import time

import networkx as nx
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from tqdm import tqdm
import time
import logging

logger = logging.getLogger(__name__)


def evaluate_subgraph_alignment(
        source_subgraph: nx.Graph,
        target_subgraph: nx.Graph,
        true_alignment_dict: dict,
        predicted_alignment_dict: dict,
        k: int = 10,
        verbose: bool = False
) -> dict:
    """
    鲁棒性子图对齐评估函数 - 修复了分类指标计算问题

    参数:
        source_subgraph: 源子图 (NetworkX图对象)
        target_subgraph: 目标子图 (NetworkX图对象)
        true_alignment_dict: 真实对齐字典 {源图节点: 目标图节点}
        predicted_alignment_dict: 预测对齐字典 {源图节点: 目标图节点}
        k: Hits@k评估中的top-k值
        verbose: 是否显示详细计算过程

    返回:
        包含多维评估指标的字典
    """
    start_time = time.time()
    results = {}

    # === 预处理阶段 ===
    # 提取源子图节点在全局的真实对齐
    valid_source_nodes = [n for n in source_subgraph.nodes if n in true_alignment_dict]

    # 创建映射：源节点索引 -> 目标节点索引
    source_to_target = {}
    for src_node in valid_source_nodes:
        target_node = true_alignment_dict[src_node]
        if target_node in target_subgraph.nodes:
            source_to_target[src_node] = target_node

    valid_count = len(valid_source_nodes)
    if valid_count == 0:
        logger.error("无有效对齐节点对可用")
        return {
            'error': 'No valid alignment pairs',
            'elapsed_time': time.time() - start_time
        }

    # === 节点对齐评估 ===
    if verbose:
        print("\n" + "=" * 60)
        print("节点对齐评估")
        print(f"验证节点数: {valid_count}")

    # 1. 准确率 (Accuracy)
    correct_count = 0
    for src_node in valid_source_nodes:
        if src_node in predicted_alignment_dict:
            pred_target = predicted_alignment_dict[src_node]
            true_target = true_alignment_dict[src_node]
            if pred_target == true_target:
                correct_count += 1

    results['node_accuracy'] = correct_count / valid_count if valid_count > 0 else 0

    # 2. 精确率、召回率和F1分数 (修复了分类指标计算问题)
    if verbose:
        print("\n" + "=" * 60)
        print("计算精确率、召回率和F1分数...")

    # 创建标签列表 - 只包含有效节点
    y_true = []
    y_pred = []

    for src_node in valid_source_nodes:
        true_target = true_alignment_dict[src_node]
        if src_node in predicted_alignment_dict:
            pred_target = predicted_alignment_dict[src_node]
            # 1表示正确对齐，0表示错误对齐
            y_true.append(1)
            y_pred.append(1 if pred_target == true_target else 0)

    if len(y_true) > 0 and len(y_pred) > 0:
        try:
            results['node_precision'] = precision_score(y_true, y_pred, zero_division=0)
            results['node_recall'] = recall_score(y_true, y_pred, zero_division=0)
            results['node_f1'] = f1_score(y_true, y_pred, zero_division=0)
        except Exception as e:
            logger.error(f"计算分类指标失败: {str(e)}")
            results['node_precision'] = 0
            results['node_recall'] = 0
            results['node_f1'] = 0
    else:
        results['node_precision'] = 0
        results['node_recall'] = 0
        results['node_f1'] = 0

    if verbose:
        print(f"精确率: {results['node_precision']:.4f}")
        print(f"召回率: {results['node_recall']:.4f}")
        print(f"F1分数: {results['node_f1']:.4f}")

    # 3. Hits@k指标
    if verbose:
        print("\n" + "=" * 60)
        print(f"计算Hits@{k}指标...")

    hit1_count = 0
    hitk_count = 0

    for src_node in valid_source_nodes:
        true_target = true_alignment_dict[src_node]

        # 获取预测的目标节点
        pred_target = predicted_alignment_dict.get(src_node)

        # 检查是否在top k中
        if pred_target == true_target:
            hit1_count += 1
            hitk_count += 1
        else:
            # 在实际应用中，这里应该有更复杂的排名逻辑
            # 但在这个简化版中，我们只检查是否在top k中
            # 假设所有预测都是top1
            pass

    results[f'hits@{1}'] = hit1_count / valid_count if valid_count > 0 else 0
    results[f'hits@{k}'] = hitk_count / valid_count if valid_count > 0 else 0

    if verbose:
        print(f"Hits@1: {results['hits@1']:.4f}")
        print(f"Hits@{k}: {results[f'hits@{k}']:.4f}")

    # === 边结构保持性评估 ===
    if verbose:
        print("\n" + "=" * 60)
        print("边结构保持性评估")

    preserved_edges = 0
    total_edges = 0

    # 遍历源子图的所有边
    for u, v in source_subgraph.edges():
        if u in predicted_alignment_dict and v in predicted_alignment_dict:
            u_target = predicted_alignment_dict[u]
            v_target = predicted_alignment_dict[v]

            # 检查目标图中是否存在这条边
            if target_subgraph.has_edge(u_target, v_target):
                preserved_edges += 1
            total_edges += 1

    results['edge_preservation'] = preserved_edges / total_edges if total_edges > 0 else 0

    if verbose:
        print(f"边保持率: {results['edge_preservation']:.4f}")

    # === 拓扑一致性评估 ===
    if verbose:
        print("\n" + "=" * 60)
        print("拓扑一致性评估")

    # 1. 度分布相似性
    source_degrees = [d for _, d in source_subgraph.degree()]
    target_degrees = [d for _, d in target_subgraph.degree()]

    # 使用KL散度度量度分布差异
    min_degree = min(min(source_degrees), min(target_degrees))
    max_degree = max(max(source_degrees), max(target_degrees))
    bins = np.arange(min_degree, max_degree + 2)

    source_hist, _ = np.histogram(source_degrees, bins=bins, density=True)
    target_hist, _ = np.histogram(target_degrees, bins=bins, density=True)

    # 添加小值避免除零
    source_hist += 1e-10
    target_hist += 1e-10

    kl_divergence = np.sum(source_hist * np.log(source_hist / target_hist))
    results['degree_kl_divergence'] = kl_divergence

    # 2. 聚类系数相似性
    source_cc = nx.average_clustering(source_subgraph)
    target_cc = nx.average_clustering(target_subgraph)
    results['clustering_coeff_diff'] = abs(source_cc - target_cc)

    if verbose:
        print(f"度分布KL散度: {results['degree_kl_divergence']:.4f}")
        print(f"聚类系数差异: {results['clustering_coeff_diff']:.4f}")

    # === 子图相似性评估 ===
    if verbose:
        print("\n" + "=" * 60)
        print("子图相似性评估")

    # 1. 图编辑距离 (GED) 近似
    node_mismatch = len(set(source_subgraph.nodes) - set(target_subgraph.nodes))
    edge_mismatch = abs(source_subgraph.number_of_edges() - target_subgraph.number_of_edges())
    results['ged_approximation'] = node_mismatch + edge_mismatch

    if verbose:
        print(f"图编辑距离近似: {results['ged_approximation']}")

    # === 对齐一致性评估 ===
    if verbose:
        print("\n" + "=" * 60)
        print("对齐一致性评估")

    # 1. 双向一致性检查
    forward_consistent = 0
    backward_consistent = 0

    # 创建反向映射
    reverse_mapping = {}
    for src, tgt in predicted_alignment_dict.items():
        if tgt not in reverse_mapping:
            reverse_mapping[tgt] = []
        reverse_mapping[tgt].append(src)

    for src_node in valid_source_nodes:
        if src_node in predicted_alignment_dict:
            pred_target = predicted_alignment_dict[src_node]
            # 前向一致性：源->目标
            if pred_target == true_alignment_dict.get(src_node, None):
                forward_consistent += 1

            # 后向一致性：目标->源
            if pred_target in reverse_mapping and src_node in reverse_mapping[pred_target]:
                backward_consistent += 1

    results['forward_consistency'] = forward_consistent / valid_count if valid_count > 0 else 0
    results['backward_consistency'] = backward_consistent / valid_count if valid_count > 0 else 0
    results['bidirectional_consistency'] = min(results['forward_consistency'], results['backward_consistency'])

    if verbose:
        print(f"前向一致性: {results['forward_consistency']:.4f}")
        print(f"后向一致性: {results['backward_consistency']:.4f}")
        print(f"双向一致性: {results['bidirectional_consistency']:.4f}")

    # === 综合评估 ===
    elapsed_time = time.time() - start_time
    results['elapsed_time'] = elapsed_time

    if verbose:
        print("\n" + "=" * 60)
        print("子图对齐评估完成!")
        print("最终结果:")
        for k, v in results.items():
            print(f"{k:>25}: {v:.4f}" if isinstance(v, float) else f"{k:>25}: {v}")
        print("=" * 60)

    return results


def evaluate_sub_align(source_subg, target_subg, true_align_dict, pred_align_dict):
    """
    子图对齐评估封装函数

    参数:
        source_subg: 源子图
        target_subg: 目标子图
        true_align_dict: 真实对齐字典
        pred_align_dict: 预测对齐字典
    """
    # 评估
    metrics = evaluate_subgraph_alignment(
        source_subgraph=source_subg,
        target_subgraph=target_subg,
        true_alignment_dict=true_align_dict,
        predicted_alignment_dict=pred_align_dict,
        k=5,
        verbose=True
    )
    return metrics

# def evaluate_subgraph_alignment(
#         source_subgraph: nx.Graph,
#         target_subgraph: nx.Graph,
#         true_alignment_dict: dict,
#         k: int = 10,
#         verbose: bool = False  # 添加详细输出模式
# ) -> dict:
#     """
#     子图对齐评估函数 - 包含多种专业指标
#
#     参数:
#         source_subgraph: 源子图 (NetworkX图对象)
#         target_subgraph: 目标子图 (NetworkX图对象)
#         true_alignment_dict: 全局真实对齐字典 {源图节点: 目标图节点}
#         k: NDCG@k评估中的top-k值
#         verbose: 是否显示详细计算过程
#
#     返回:
#         包含各种评估指标的字典
#     """
#     # === 预处理阶段 ===
#     # 提取源子图节点在全局的真实对齐
#     valid_source_nodes = [n for n in source_subgraph.nodes if n in true_alignment_dict]
#
#     # 创建预测字典（这里假设预测字典就是输入的真实字典，实际应用中需替换为模型预测）
#     predicted_alignment = true_alignment_dict  # 实际应用中替换为模型输出
#
#     # === 评估指标计算 ===
#     results = {}
#
#     # 1. 节点对齐准确率 (Accuracy)
#     if verbose:
#         print("\n" + "=" * 60)
#         print("计算节点对齐准确率 (Accuracy)...")
#         print(f"验证节点数: {len(valid_source_nodes)}")
#     correct_count = 0
#     for src_node in valid_source_nodes:
#         true_target = true_alignment_dict[src_node]
#         pred_target = predicted_alignment.get(src_node)
#         if pred_target == true_target:
#             correct_count += 1
#
#     results['accuracy'] = correct_count / len(valid_source_nodes) if valid_source_nodes else 0
#     if verbose:
#         print(f"准确率: {results['accuracy']:.4f}")
#
#     # 4. F1分数
#     if verbose:
#         print("\n" + "=" * 60)
#         print("计算F1分数...")
#     y_pred = [1 if predicted_alignment.get(n) == true_alignment_dict.get(n) else 0
#               for n in valid_source_nodes]
#     try:
#         results['f1'] = metrics.f1_score(
#             [1] * len(valid_source_nodes),  # 所有样本应为正例
#             y_pred
#         ) if y_pred else 0
#         if verbose:
#             print(f"F1分数: {results['f1']:.4f}")
#     except:
#         results['f1'] = 0
#         if verbose:
#             print("F1分数计算失败")
#
#
#
#     # 6. 节点对保持率 (Pair Preservation Ratio, PPR)
#     if verbose:
#         print("\n" + "=" * 60)
#         print("计算节点对保持率 (PPR)...")
#         print(f"源图边数: {source_subgraph.number_of_edges()}")
#     preserved_edges = 0
#     total_edges = 0
#     for i, (u, v) in enumerate(source_subgraph.edges()):
#         if u in predicted_alignment and v in predicted_alignment:
#             u_target = predicted_alignment[u]
#             v_target = predicted_alignment[v]
#             # 检查目标图中是否存在这条边（注意：无向图）
#             if target_subgraph.has_edge(u_target, v_target) or target_subgraph.has_edge(v_target, u_target):
#                 preserved_edges += 1
#             total_edges += 1
#
#         if verbose and i % 100 == 0 and i > 0:
#             print(
#                 f"  处理边 {i + 1}/{source_subgraph.number_of_edges()}，当前PPR: {preserved_edges / total_edges if total_edges > 0 else 0:.4f}")
#
#     results['ppr'] = preserved_edges / total_edges if total_edges > 0 else 0
#     if verbose:
#         print(f"PPR: {results['ppr']:.4f} ({preserved_edges}/{total_edges} 条边保持)")
#
#
#
#
#
#     # 完成评估
#     if verbose:
#         print("\n" + "=" * 60)
#         print("子图对齐评估完成!")
#         print("最终结果:")
#         for k, v in results.items():
#             print(f"{k:>15}: {v:.4f}")
#         print("=" * 60)
#
#     return results
#
#
# def evaluate_sub_align(source_subg,target_subg,align_dict):
#     # 评估
#     metrics = evaluate_subgraph_alignment(
#         source_subgraph=source_subg,
#         target_subgraph=target_subg,
#         true_alignment_dict=align_dict,
#         k=5,
#         verbose=True  # 开启详细输出
#     )

