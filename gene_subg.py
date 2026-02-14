import networkx as nx
import random
import numpy as np
from collections import deque
from typing import Set, Dict, List, Any
import time
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import expm_multiply
import heapq


def ensure_connected_subgraph(G: nx.Graph, nodes: Set, center_node: int) -> nx.Graph:
    """
    确保子图是连通的且边数最少（最小生成树）

    参数:
        G: 原始图对象
        nodes: 节点集合
        center_node: 中心节点ID

    返回:
        连通子图（最小生成树）
    """
    # 创建子图
    subgraph = G.subgraph(nodes)

    # 检查是否连通
    if nx.is_connected(subgraph):
        # 如果连通，返回最小生成树
        mst = nx.minimum_spanning_tree(subgraph)
        # 复制节点属性
        for node in mst.nodes:
            mst.nodes[node].update(G.nodes[node])
        return mst

    # 如果不连通，找到包含中心节点的连通分量
    components = list(nx.connected_components(subgraph))
    main_component = None

    # 找到包含中心节点的分量
    for comp in components:
        if center_node in comp:
            main_component = comp
            break

    # 如果找不到包含中心节点的分量，使用第一个分量
    if main_component is None:
        main_component = components[0]

    # 添加必要节点使子图连通
    added_nodes = set()
    for comp in components:
        if comp == main_component:
            continue

        # 找到两个分量之间的最短路径
        source = next(iter(comp))
        target = next(iter(main_component))
        try:
            path = nx.shortest_path(G, source=source, target=target)
            # 添加路径上的所有节点
            added_nodes.update(path)
        except nx.NetworkXNoPath:
            # 如果没有路径，跳过这个分量
            continue

    # 合并所有节点
    all_nodes = nodes | added_nodes

    # 创建新的连通子图
    connected_subgraph = G.subgraph(all_nodes)

    # 创建最小生成树（边数最少）
    mst = nx.minimum_spanning_tree(connected_subgraph)

    # 复制节点属性
    for node in mst.nodes:
        mst.nodes[node].update(G.nodes[node])

    return mst


def generate_rwr_subgraph(
        G: nx.Graph,
        center_node: int,
        target_node_count: int,
        candidate_nodes: set,
        alpha: float = 0.15,
        max_total_steps: int = 10000,
        patience: int = 200
) -> nx.Graph:
    """
    使用带重启的自适应随机游走 (RWR) 生成指定大小的连通子图
    确保子图是连通的且边数最少

    参数:
        G: 原始图对象 (NetworkX Graph)
        center_node: 中心节点ID
        target_node_count: 目标子图节点数
        candidate_nodes: 候选节点集合
        alpha: 重启概率 (默认0.15)
        max_total_steps: 最大总步数限制 (防止无限循环)
        patience: 无新节点收集时的容忍步数

    返回:
        包含目标节点数的连通子图 (保留所有原始属性)
    """
    # 验证输入
    if center_node not in G:
        raise ValueError(f"中心节点 {center_node} 不在图中")
    if center_node not in candidate_nodes:
        raise ValueError(f"中心节点 {center_node} 不在候选节点集合中")
    if target_node_count < 1:
        raise ValueError("目标节点数必须大于0")

    # 如果目标节点数大于图大小，返回完整图
    if target_node_count >= len(G):
        subgraph = G.copy()
        subgraph.graph["collection_metrics"] = {
            "status": "full_graph",
            "collected_nodes": len(G),
            "total_steps": 0
        }
        return subgraph

    # 初始化数据结构
    collected_nodes = set([center_node])  # 已收集节点集合
    current_node = center_node  # 当前节点
    total_steps = 0  # 总步数计数器
    steps_without_new = 0  # 无新节点步数计数器
    restart_count = 0  # 重启次数计数器

    # 节点访问频率记录 (用于智能重启)
    visit_frequency: Dict[int, int] = {center_node: 1}

    # 自适应游走直到收集足够节点
    while len(collected_nodes) < target_node_count and total_steps < max_total_steps:
        # 重启判定
        if random.random() < alpha:
            current_node = center_node
            restart_count += 1
        else:
            # 获取当前节点的邻居
            neighbors = list(G.neighbors(current_node))

            # 如果没有邻居，强制重启
            if not neighbors:
                current_node = center_node
                restart_count += 1
            else:
                # 只考虑候选节点集合中的邻居
                valid_neighbors = [n for n in neighbors if n in candidate_nodes]

                if not valid_neighbors:
                    current_node = center_node
                    restart_count += 1
                    continue

                # 智能邻居选择：优先选择未收集节点
                unvisited_neighbors = [n for n in valid_neighbors if n not in collected_nodes]

                if unvisited_neighbors:
                    # 优先选择未访问节点
                    current_node = random.choice(unvisited_neighbors)
                else:
                    # 所有邻居都已收集，选择访问频率最低的邻居
                    least_visited = min(valid_neighbors, key=lambda n: visit_frequency.get(n, 0))
                    current_node = least_visited

        # 更新访问频率
        visit_frequency[current_node] = visit_frequency.get(current_node, 0) + 1

        # 检查是否新节点
        if current_node not in collected_nodes:
            collected_nodes.add(current_node)
            steps_without_new = 0  # 重置无新节点计数器
        else:
            steps_without_new += 1

        total_steps += 1

        # 无新节点容忍机制
        if steps_without_new > patience:
            # 自适应调整：降低重启概率以扩大探索范围
            alpha = max(0.05, alpha * 0.8)
            steps_without_new = 0

    # 确保子图连通且边数最少
    connected_subgraph = ensure_connected_subgraph(G, collected_nodes, center_node)

    # 添加中心节点标记
    connected_subgraph.nodes[center_node]["is_center"] = True

    # 添加收集指标元数据
    connected_subgraph.graph["collection_metrics"] = {
        "requested_nodes": target_node_count,
        "collected_nodes": len(connected_subgraph),
        "total_steps": total_steps,
        "restart_count": restart_count,
        "avg_restart_rate": restart_count / total_steps if total_steps > 0 else 0,
        "success": len(connected_subgraph) >= target_node_count,
        "final_alpha": alpha,
        "added_nodes": len(connected_subgraph) - len(collected_nodes)
    }

    return connected_subgraph


def generate_bfs_subgraph(
        G: nx.Graph,
        center_node: int,
        target_node_count: int,
        candidate_nodes: set,
        shuffle_neighbors: bool = True
) -> nx.Graph:
    """
    使用BFS生成以指定节点为中心的连通子图
    确保子图是连通的且边数最少

    参数:
        G: 原始图对象 (NetworkX Graph)
        center_node: 中心节点ID
        target_node_count: 目标子图节点数
        candidate_nodes: 候选节点集合
        shuffle_neighbors: 是否随机打乱邻居顺序 (默认True，增加随机性)

    返回:
        包含目标节点数的连通子图 (保留所有原始属性)
    """
    # 验证输入
    if center_node not in G:
        raise ValueError(f"中心节点 {center_node} 不在图中")
    if center_node not in candidate_nodes:
        raise ValueError(f"中心节点 {center_node} 不在候选节点集合中")
    if target_node_count < 1:
        raise ValueError("目标节点数必须大于0")

    # 如果目标节点数大于图大小，返回完整图
    if target_node_count >= len(G):
        subgraph = G.copy()
        subgraph.graph["collection_metrics"] = {
            "status": "full_graph",
            "collected_nodes": len(G),
            "target_nodes": target_node_count
        }
        return subgraph

    # 初始化数据结构
    visited = set([center_node])  # 已访问节点集合
    queue = deque([center_node])  # BFS队列

    # BFS遍历直到收集足够节点
    while queue and len(visited) < target_node_count:
        current = queue.popleft()

        # 获取当前节点的邻居
        neighbors = list(G.neighbors(current))

        # 只考虑候选节点集合中的邻居
        valid_neighbors = [n for n in neighbors if n in candidate_nodes]

        # 随机打乱邻居顺序 (增加随机性)
        if shuffle_neighbors:
            random.shuffle(valid_neighbors)

        # 遍历邻居
        for neighbor in valid_neighbors:
            if len(visited) >= target_node_count:
                break

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # 确保子图连通且边数最少
    connected_subgraph = ensure_connected_subgraph(G, visited, center_node)

    # 添加中心节点标记属性
    for node in connected_subgraph.nodes:
        connected_subgraph.nodes[node]["is_center"] = (node == center_node)

    # 添加收集指标元数据
    connected_subgraph.graph["collection_metrics"] = {
        "method": "BFS",
        "center_node": center_node,
        "target_nodes": target_node_count,
        "collected_nodes": len(connected_subgraph),
        "success": len(connected_subgraph) >= target_node_count,
        "shuffle_neighbors": shuffle_neighbors,
        "added_nodes": len(connected_subgraph) - len(visited)
    }

    return connected_subgraph


def generate_dfs_subgraph(
        G: nx.Graph,
        center_node: int,
        target_node_count: int,
        candidate_nodes: set,
        shuffle_neighbors: bool = True,
        max_depth: int = None
) -> nx.Graph:
    """
    使用DFS生成以指定节点为中心的连通子图
    确保子图是连通的且边数最少

    参数:
        G: 原始图对象 (NetworkX Graph)
        center_node: 中心节点ID
        target_node_count: 目标子图节点数
        candidate_nodes: 候选节点集合
        shuffle_neighbors: 是否随机打乱邻居顺序 (默认True，增加随机性)
        max_depth: 最大搜索深度 (可选)

    返回:
        包含目标节点数的连通子图 (保留所有原始属性)
    """
    # 验证输入
    if center_node not in G:
        raise ValueError(f"中心节点 {center_node} 不在图中")
    if center_node not in candidate_nodes:
        raise ValueError(f"中心节点 {center_node} 不在候选节点集合中")
    if target_node_count < 1:
        raise ValueError("目标节点数必须大于0")

    # 如果目标节点数大于图大小，返回完整图
    if target_node_count >= len(G):
        subgraph = G.copy()
        subgraph.graph["collection_metrics"] = {
            "status": "full_graph",
            "collected_nodes": len(G),
            "target_nodes": target_node_count
        }
        return subgraph

    # 初始化数据结构
    visited = set([center_node])  # 已访问节点集合
    stack = deque([center_node])  # DFS栈

    # DFS遍历直到收集足够节点
    while stack and len(visited) < target_node_count:
        current = stack.pop()

        # 获取当前节点的邻居
        neighbors = list(G.neighbors(current))

        # 只考虑候选节点集合中的邻居
        valid_neighbors = [n for n in neighbors if n in candidate_nodes]

        # 随机打乱邻居顺序 (增加随机性)
        if shuffle_neighbors:
            random.shuffle(valid_neighbors)

        # 遍历邻居
        for neighbor in valid_neighbors:
            if len(visited) >= target_node_count:
                break

            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)  # 将邻居压入栈中

                # 检查深度限制
                if max_depth is not None:
                    # 计算当前深度
                    depth = nx.shortest_path_length(G, source=center_node, target=neighbor)
                    if depth > max_depth:
                        break

    # 确保子图连通且边数最少
    connected_subgraph = ensure_connected_subgraph(G, visited, center_node)

    # 添加中心节点标记属性
    for node in connected_subgraph.nodes:
        connected_subgraph.nodes[node]["is_center"] = (node == center_node)

    # 添加收集指标元数据
    connected_subgraph.graph["collection_metrics"] = {
        "method": "DFS",
        "center_node": center_node,
        "target_nodes": target_node_count,
        "collected_nodes": len(connected_subgraph),
        "success": len(connected_subgraph) >= target_node_count,
        "shuffle_neighbors": shuffle_neighbors,
        "max_depth": max_depth,
        "added_nodes": len(connected_subgraph) - len(visited)
    }

    return connected_subgraph


def generate_diffusion_subgraph(
        G: nx.Graph,
        center_node: int,
        target_node_count: int,
        candidate_nodes: set,
        diffusion_time: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6
) -> nx.Graph:
    """
    使用图扩散算法生成以指定节点为中心的连通子图
    确保子图是连通的且边数最少

    参数:
        G: 原始图对象 (NetworkX Graph)
        center_node: 中心节点ID
        target_node_count: 目标子图节点数
        candidate_nodes: 候选节点集合
        diffusion_time: 扩散时间参数 (默认1.0)
        max_iter: 最大迭代次数 (默认100)
        tol: 收敛容差 (默认1e-6)

    返回:
        包含目标节点数的连通子图 (保留所有原始属性)
    """
    # 验证输入
    if center_node not in G:
        raise ValueError(f"中心节点 {center_node} 不在图中")
    if center_node not in candidate_nodes:
        raise ValueError(f"中心节点 {center_node} 不在候选节点集合中")
    if target_node_count < 1:
        raise ValueError("目标节点数必须大于0")

    # 如果目标节点数大于图大小，返回完整图
    if target_node_count >= len(G):
        subgraph = G.copy()
        subgraph.graph["collection_metrics"] = {
            "status": "full_graph",
            "collected_nodes": len(G),
            "target_nodes": target_node_count
        }
        return subgraph

    # 记录开始时间
    start_time = time.time()

    # 创建图的邻接矩阵表示
    n = len(G)
    node_index = {node: i for i, node in enumerate(G.nodes)}
    index_node = {i: node for i, node in enumerate(G.nodes)}

    # 构建邻接矩阵 (稀疏格式)
    rows, cols = [], []
    for u, v in G.edges():
        i, j = node_index[u], node_index[v]
        rows.append(i)
        cols.append(j)
        rows.append(j)
        cols.append(i)

    data = np.ones(len(rows))
    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    # 计算度矩阵
    degrees = np.array([d for _, d in G.degree()])
    D = diags(degrees, format='csr')

    # 计算图拉普拉斯矩阵 L = D - A
    L = D - A

    # 创建初始热源向量 (中心节点)
    heat_vector = np.zeros(n)
    center_idx = node_index[center_node]
    heat_vector[center_idx] = 1.0

    # 使用热核扩散计算扩散分数
    # H = exp(-t * L) * heat_vector
    diffusion_scores = expm_multiply(-diffusion_time * L, heat_vector)

    # 将扩散分数映射回节点
    node_scores = {}
    for i, score in enumerate(diffusion_scores):
        node = index_node[i]
        # 只考虑候选节点集合中的节点
        if node in candidate_nodes:
            node_scores[node] = score

    # 选择扩散分数最高的target_node_count个节点
    top_nodes = heapq.nlargest(target_node_count, node_scores.items(), key=lambda x: x[1])
    selected_nodes = set(node for node, score in top_nodes)

    # 确保中心节点总是包含在子图中
    if center_node not in selected_nodes:
        # 如果中心节点不在选中的节点中，添加它
        selected_nodes.add(center_node)

        # 如果添加后节点数超过目标数，移除分数最低的节点
        if len(selected_nodes) > target_node_count:
            # 找到分数最低的节点
            min_score_node = min(selected_nodes, key=lambda n: node_scores.get(n, 0))
            selected_nodes.remove(min_score_node)

    # 确保子图连通且边数最少
    connected_subgraph = ensure_connected_subgraph(G, selected_nodes, center_node)

    # 添加扩散分数作为节点属性
    for node in connected_subgraph.nodes:
        connected_subgraph.nodes[node]["diffusion_score"] = node_scores.get(node, 0)  # 使用get避免KeyError

    # 添加中心节点标记属性
    if center_node in connected_subgraph.nodes:  # 再次检查确保中心节点在子图中
        connected_subgraph.nodes[center_node]["is_center"] = True

    # 添加收集指标元数据
    end_time = time.time()
    connected_subgraph.graph["collection_metrics"] = {
        "method": "Heat Kernel Diffusion",
        "center_node": center_node,
        "target_nodes": target_node_count,
        "collected_nodes": len(connected_subgraph),
        "diffusion_time": diffusion_time,
        "computation_time": end_time - start_time,
        "max_score": max(node_scores.values()) if node_scores else 0,
        "min_score": min(node_scores.values()) if node_scores else 0,
        "success": len(connected_subgraph) >= target_node_count,
        "added_nodes": len(connected_subgraph) - len(selected_nodes)
    }

    return connected_subgraph


def gene_candiate_subg(G, center_node, n, candidate_nodes):
    """
    生成四种类型的候选连通子图
    确保每个子图都是连通的且边数最少

    参数:
        G: 原始图对象
        center_node: 中心节点ID
        n: 目标子图节点数
        candidate_nodes: 候选节点集合

    返回:
        四种候选连通子图: (rwr_subg, bfs_subg, dfs_subg, diff_subg)
    """
    rwr_subg = generate_rwr_subgraph(G, center_node, n, candidate_nodes)
    bfs_subg = generate_bfs_subgraph(G, center_node, n, candidate_nodes)
    dfs_subg = generate_dfs_subgraph(G, center_node, n, candidate_nodes)
    diff_subg = generate_diffusion_subgraph(G, center_node, n, candidate_nodes)
    return rwr_subg, bfs_subg, dfs_subg, diff_subg









# import networkx as nx
# import random
# import numpy as np
# from collections import deque
# from typing import Set, Dict, List
# import networkx as nx
# import numpy as np
# from scipy.sparse import csr_matrix, diags
# from scipy.sparse.linalg import expm_multiply
# import heapq
# import time
# def generate_rwr_subgraph(
#         G: nx.Graph,
#         center_node: int,
#         target_node_count: int,
#         candidate_nodes: set,
#         alpha: float = 0.15,
#         max_total_steps: int = 10000,
#         patience: int = 200
# ) -> nx.Graph:
#     """
#     使用带重启的自适应随机游走 (RWR) 生成指定大小的子图
#     确保子图包含原图中的所有边
#
#     参数:
#         G: 原始图对象 (NetworkX Graph)
#         center_node: 中心节点ID
#         target_node_count: 目标子图节点数
#         candidate_nodes: 候选节点集合
#         alpha: 重启概率 (默认0.15)
#         max_total_steps: 最大总步数限制 (防止无限循环)
#         patience: 无新节点收集时的容忍步数
#
#     返回:
#         包含目标节点数的子图 (保留所有原始属性)
#     """
#     # 验证输入
#     if center_node not in G:
#         raise ValueError(f"中心节点 {center_node} 不在图中")
#     if center_node not in candidate_nodes:
#         raise ValueError(f"中心节点 {center_node} 不在候选节点集合中")
#     if target_node_count < 1:
#         raise ValueError("目标节点数必须大于0")
#
#     # 如果目标节点数大于图大小，返回完整图
#     if target_node_count >= len(G):
#         subgraph = G.copy()
#         subgraph.graph["collection_metrics"] = {
#             "status": "full_graph",
#             "collected_nodes": len(G),
#             "total_steps": 0
#         }
#         return subgraph
#
#     # 初始化数据结构
#     collected_nodes = set([center_node])  # 已收集节点集合
#     current_node = center_node  # 当前节点
#     total_steps = 0  # 总步数计数器
#     steps_without_new = 0  # 无新节点步数计数器
#     restart_count = 0  # 重启次数计数器
#
#     # 节点访问频率记录 (用于智能重启)
#     visit_frequency: Dict[int, int] = {center_node: 1}
#
#     # 自适应游走直到收集足够节点
#     while len(collected_nodes) < target_node_count and total_steps < max_total_steps:
#         # 重启判定
#         if random.random() < alpha:
#             current_node = center_node
#             restart_count += 1
#         else:
#             # 获取当前节点的邻居
#             neighbors = list(G.neighbors(current_node))
#
#             # 如果没有邻居，强制重启
#             if not neighbors:
#                 current_node = center_node
#                 restart_count += 1
#             else:
#                 # 只考虑候选节点集合中的邻居
#                 valid_neighbors = [n for n in neighbors if n in candidate_nodes]
#
#                 if not valid_neighbors:
#                     current_node = center_node
#                     restart_count += 1
#                     continue
#
#                 # 智能邻居选择：优先选择未收集节点
#                 unvisited_neighbors = [n for n in valid_neighbors if n not in collected_nodes]
#
#                 if unvisited_neighbors:
#                     # 优先选择未访问节点
#                     current_node = random.choice(unvisited_neighbors)
#                 else:
#                     # 所有邻居都已收集，选择访问频率最低的邻居
#                     least_visited = min(valid_neighbors, key=lambda n: visit_frequency.get(n, 0))
#                     current_node = least_visited
#
#         # 更新访问频率
#         visit_frequency[current_node] = visit_frequency.get(current_node, 0) + 1
#
#         # 检查是否新节点
#         if current_node not in collected_nodes:
#             collected_nodes.add(current_node)
#             steps_without_new = 0  # 重置无新节点计数器
#         else:
#             steps_without_new += 1
#
#         total_steps += 1
#
#         # 无新节点容忍机制
#         if steps_without_new > patience:
#             # 自适应调整：降低重启概率以扩大探索范围
#             alpha = max(0.05, alpha * 0.8)
#             steps_without_new = 0
#
#     # 创建子图并保留原始属性
#     subgraph = G.subgraph(collected_nodes).copy()
#
#     # 添加中心节点标记
#     subgraph.nodes[center_node]["is_center"] = True
#
#     # 添加收集指标元数据
#     subgraph.graph["collection_metrics"] = {
#         "requested_nodes": target_node_count,
#         "collected_nodes": len(collected_nodes),
#         "total_steps": total_steps,
#         "restart_count": restart_count,
#         "avg_restart_rate": restart_count / total_steps if total_steps > 0 else 0,
#         "success": len(collected_nodes) >= target_node_count,
#         "final_alpha": alpha
#     }
#
#     return subgraph
#
#
# def generate_bfs_subgraph(
#         G: nx.Graph,
#         center_node: int,
#         target_node_count: int,
#         candidate_nodes: set,
#         shuffle_neighbors: bool = True
# ) -> nx.Graph:
#     """
#     使用BFS生成以指定节点为中心的子图
#     确保子图包含原图中的所有边
#
#     参数:
#         G: 原始图对象 (NetworkX Graph)
#         center_node: 中心节点ID
#         target_node_count: 目标子图节点数
#         candidate_nodes: 候选节点集合
#         shuffle_neighbors: 是否随机打乱邻居顺序 (默认True，增加随机性)
#
#     返回:
#         包含目标节点数的子图 (保留所有原始属性)
#     """
#     # 验证输入
#     if center_node not in G:
#         raise ValueError(f"中心节点 {center_node} 不在图中")
#     if center_node not in candidate_nodes:
#         raise ValueError(f"中心节点 {center_node} 不在候选节点集合中")
#     if target_node_count < 1:
#         raise ValueError("目标节点数必须大于0")
#
#     # 如果目标节点数大于图大小，返回完整图
#     if target_node_count >= len(G):
#         subgraph = G.copy()
#         subgraph.graph["collection_metrics"] = {
#             "status": "full_graph",
#             "collected_nodes": len(G),
#             "target_nodes": target_node_count
#         }
#         return subgraph
#
#     # 初始化数据结构
#     visited = set([center_node])  # 已访问节点集合
#     queue = deque([center_node])  # BFS队列
#
#     # BFS遍历直到收集足够节点
#     while queue and len(visited) < target_node_count:
#         current = queue.popleft()
#
#         # 获取当前节点的邻居
#         neighbors = list(G.neighbors(current))
#
#         # 只考虑候选节点集合中的邻居
#         valid_neighbors = [n for n in neighbors if n in candidate_nodes]
#
#         # 随机打乱邻居顺序 (增加随机性)
#         if shuffle_neighbors:
#             random.shuffle(valid_neighbors)
#
#         # 遍历邻居
#         for neighbor in valid_neighbors:
#             if len(visited) >= target_node_count:
#                 break
#
#             if neighbor not in visited:
#                 visited.add(neighbor)
#                 queue.append(neighbor)
#
#     # 创建子图并保留原始属性
#     subgraph = G.subgraph(visited).copy()
#
#     # 添加中心节点标记属性
#     for node in subgraph.nodes:
#         subgraph.nodes[node]["is_center"] = (node == center_node)
#
#     # 添加收集指标元数据
#     subgraph.graph["collection_metrics"] = {
#         "method": "BFS",
#         "center_node": center_node,
#         "target_nodes": target_node_count,
#         "collected_nodes": len(visited),
#         "success": len(visited) >= target_node_count,
#         "shuffle_neighbors": shuffle_neighbors
#     }
#
#     return subgraph
#
#
# def generate_dfs_subgraph(
#         G: nx.Graph,
#         center_node: int,
#         target_node_count: int,
#         candidate_nodes: set,
#         shuffle_neighbors: bool = True,
#         max_depth: int = None
# ) -> nx.Graph:
#     """
#     使用DFS生成以指定节点为中心的子图
#     确保子图包含原图中的所有边
#
#     参数:
#         G: 原始图对象 (NetworkX Graph)
#         center_node: 中心节点ID
#         target_node_count: 目标子图节点数
#         candidate_nodes: 候选节点集合
#         shuffle_neighbors: 是否随机打乱邻居顺序 (默认True，增加随机性)
#         max_depth: 最大搜索深度 (可选)
#
#     返回:
#         包含目标节点数的子图 (保留所有原始属性)
#     """
#     # 验证输入
#     if center_node not in G:
#         raise ValueError(f"中心节点 {center_node} 不在图中")
#     if center_node not in candidate_nodes:
#         raise ValueError(f"中心节点 {center_node} 不在候选节点集合中")
#     if target_node_count < 1:
#         raise ValueError("目标节点数必须大于0")
#
#     # 如果目标节点数大于图大小，返回完整图
#     if target_node_count >= len(G):
#         subgraph = G.copy()
#         subgraph.graph["collection_metrics"] = {
#             "status": "full_graph",
#             "collected_nodes": len(G),
#             "target_nodes": target_node_count
#         }
#         return subgraph
#
#     # 初始化数据结构
#     visited = set([center_node])  # 已访问节点集合
#     stack = deque([center_node])  # DFS栈
#
#     # DFS遍历直到收集足够节点
#     while stack and len(visited) < target_node_count:
#         current = stack.pop()
#
#         # 获取当前节点的邻居
#         neighbors = list(G.neighbors(current))
#
#         # 只考虑候选节点集合中的邻居
#         valid_neighbors = [n for n in neighbors if n in candidate_nodes]
#
#         # 随机打乱邻居顺序 (增加随机性)
#         if shuffle_neighbors:
#             random.shuffle(valid_neighbors)
#
#         # 遍历邻居
#         for neighbor in valid_neighbors:
#             if len(visited) >= target_node_count:
#                 break
#
#             if neighbor not in visited:
#                 visited.add(neighbor)
#                 stack.append(neighbor)  # 将邻居压入栈中
#
#                 # 检查深度限制
#                 if max_depth is not None:
#                     # 计算当前深度
#                     depth = nx.shortest_path_length(G, source=center_node, target=neighbor)
#                     if depth > max_depth:
#                         break
#
#     # 创建子图并保留原始属性
#     subgraph = G.subgraph(visited).copy()
#
#     # 添加中心节点标记属性
#     for node in subgraph.nodes:
#         subgraph.nodes[node]["is_center"] = (node == center_node)
#
#     # 添加收集指标元数据
#     subgraph.graph["collection_metrics"] = {
#         "method": "DFS",
#         "center_node": center_node,
#         "target_nodes": target_node_count,
#         "collected_nodes": len(visited),
#         "success": len(visited) >= target_node_count,
#         "shuffle_neighbors": shuffle_neighbors,
#         "max_depth": max_depth
#     }
#
#     return subgraph
#
#
# def generate_diffusion_subgraph(
#         G: nx.Graph,
#         center_node: int,
#         target_node_count: int,
#         candidate_nodes: set,
#         diffusion_time: float = 1.0,
#         max_iter: int = 100,
#         tol: float = 1e-6
# ) -> nx.Graph:
#     """
#     使用图扩散算法生成以指定节点为中心的子图
#     确保子图包含原图中的所有边
#
#     参数:
#         G: 原始图对象 (NetworkX Graph)
#         center_node: 中心节点ID
#         target_node_count: 目标子图节点数
#         candidate_nodes: 候选节点集合
#         diffusion_time: 扩散时间参数 (默认1.0)
#         max_iter: 最大迭代次数 (默认100)
#         tol: 收敛容差 (默认1e-6)
#
#     返回:
#         包含目标节点数的子图 (保留所有原始属性)
#     """
#     # 验证输入
#     if center_node not in G:
#         raise ValueError(f"中心节点 {center_node} 不在图中")
#     if center_node not in candidate_nodes:
#         raise ValueError(f"中心节点 {center_node} 不在候选节点集合中")
#     if target_node_count < 1:
#         raise ValueError("目标节点数必须大于0")
#
#     # 如果目标节点数大于图大小，返回完整图
#     if target_node_count >= len(G):
#         subgraph = G.copy()
#         subgraph.graph["collection_metrics"] = {
#             "status": "full_graph",
#             "collected_nodes": len(G),
#             "target_nodes": target_node_count
#         }
#         return subgraph
#
#     # 记录开始时间
#     start_time = time.time()
#
#     # 创建图的邻接矩阵表示
#     n = len(G)
#     node_index = {node: i for i, node in enumerate(G.nodes)}
#     index_node = {i: node for i, node in enumerate(G.nodes)}
#
#     # 构建邻接矩阵 (稀疏格式)
#     rows, cols = [], []
#     for u, v in G.edges():
#         i, j = node_index[u], node_index[v]
#         rows.append(i)
#         cols.append(j)
#         rows.append(j)
#         cols.append(i)
#
#     data = np.ones(len(rows))
#     A = csr_matrix((data, (rows, cols)), shape=(n, n))
#
#     # 计算度矩阵
#     degrees = np.array([d for _, d in G.degree()])
#     D = diags(degrees, format='csr')
#
#     # 计算图拉普拉斯矩阵 L = D - A
#     L = D - A
#
#     # 创建初始热源向量 (中心节点)
#     heat_vector = np.zeros(n)
#     center_idx = node_index[center_node]
#     heat_vector[center_idx] = 1.0
#
#     # 使用热核扩散计算扩散分数
#     # H = exp(-t * L) * heat_vector
#     diffusion_scores = expm_multiply(-diffusion_time * L, heat_vector)
#
#     # 将扩散分数映射回节点
#     node_scores = {}
#     for i, score in enumerate(diffusion_scores):
#         node = index_node[i]
#         # 只考虑候选节点集合中的节点
#         if node in candidate_nodes:
#             node_scores[node] = score
#
#     # 选择扩散分数最高的target_node_count个节点
#     top_nodes = heapq.nlargest(target_node_count, node_scores.items(), key=lambda x: x[1])
#     selected_nodes = set(node for node, score in top_nodes)
#
#     # 确保中心节点总是包含在子图中
#     if center_node not in selected_nodes:
#         # 如果中心节点不在选中的节点中，添加它
#         selected_nodes.add(center_node)
#
#         # 如果添加后节点数超过目标数，移除分数最低的节点
#         if len(selected_nodes) > target_node_count:
#             # 找到分数最低的节点
#             min_score_node = min(selected_nodes, key=lambda n: node_scores.get(n, 0))
#             selected_nodes.remove(min_score_node)
#
#     # 创建子图并保留原始属性
#     subgraph = G.subgraph(selected_nodes).copy()
#
#     # 添加扩散分数作为节点属性
#     for node in subgraph.nodes:
#         subgraph.nodes[node]["diffusion_score"] = node_scores.get(node, 0)  # 使用get避免KeyError
#
#     # 添加中心节点标记属性
#     if center_node in subgraph.nodes:  # 再次检查确保中心节点在子图中
#         subgraph.nodes[center_node]["is_center"] = True
#
#     # 添加收集指标元数据
#     end_time = time.time()
#     subgraph.graph["collection_metrics"] = {
#         "method": "Heat Kernel Diffusion",
#         "center_node": center_node,
#         "target_nodes": target_node_count,
#         "collected_nodes": len(selected_nodes),
#         "diffusion_time": diffusion_time,
#         "computation_time": end_time - start_time,
#         "max_score": max(node_scores.values()) if node_scores else 0,
#         "min_score": min(node_scores.values()) if node_scores else 0,
#         "success": len(selected_nodes) >= target_node_count
#     }
#
#     return subgraph
#
#
# def gene_candiate_subg(G, center_node, n, candidate_nodes):
#     """
#     生成四种类型的候选子图
#     确保每个子图都包含原图中的所有边
#
#     参数:
#         G: 原始图对象
#         center_node: 中心节点ID
#         n: 目标子图节点数
#         candidate_nodes: 候选节点集合
#
#     返回:
#         四种候选子图: (rwr_subg, bfs_subg, dfs_subg, diff_subg)
#     """
#     rwr_subg = generate_rwr_subgraph(G, center_node, n, candidate_nodes)
#     bfs_subg = generate_bfs_subgraph(G, center_node, n, candidate_nodes)
#     dfs_subg = generate_dfs_subgraph(G, center_node, n, candidate_nodes)
#     diff_subg = generate_diffusion_subgraph(G, center_node, n, candidate_nodes)
#     return rwr_subg, bfs_subg, dfs_subg, diff_subg
#
#
#
#
#
#
#
#
# # # import networkx as nx
# # # import random
# # # import numpy as np
# # # from collections import deque
# # # from typing import Set, Dict, List
# # #
# # #
# # # def generate_rwr_subgraph(
# # #         G: nx.Graph,
# # #         center_node: int,
# # #         target_node_count: int,
# # #         alpha: float = 0.15,
# # #         max_total_steps: int = 10000,
# # #         patience: int = 200
# # # ) -> nx.Graph:
# # #     """
# # #     使用带重启的自适应随机游走 (RWR) 生成指定大小的子图
# # #
# # #     参数:
# # #         G: 原始图对象 (NetworkX Graph)
# # #         center_node: 中心节点ID
# # #         target_node_count: 目标子图节点数
# # #         alpha: 重启概率 (默认0.15)
# # #         max_total_steps: 最大总步数限制 (防止无限循环)
# # #         patience: 无新节点收集时的容忍步数
# # #
# # #     返回:
# # #         包含目标节点数的子图 (保留所有原始属性)
# # #     """
# # #     # 验证输入
# # #     if center_node not in G:
# # #         raise ValueError(f"中心节点 {center_node} 不在图中")
# # #     if target_node_count < 1:
# # #         raise ValueError("目标节点数必须大于0")
# # #
# # #     # 如果目标节点数大于图大小，返回完整图
# # #     if target_node_count >= len(G):
# # #         subgraph = G.copy()
# # #         subgraph.graph["collection_metrics"] = {
# # #             "status": "full_graph",
# # #             "collected_nodes": len(G),
# # #             "total_steps": 0
# # #         }
# # #         return subgraph
# # #
# # #     # 初始化数据结构
# # #     collected_nodes = set([center_node])  # 已收集节点集合
# # #     current_node = center_node  # 当前节点
# # #     total_steps = 0  # 总步数计数器
# # #     steps_without_new = 0  # 无新节点步数计数器
# # #     restart_count = 0  # 重启次数计数器
# # #
# # #     # 节点访问频率记录 (用于智能重启)
# # #     visit_frequency: Dict[int, int] = {center_node: 1}
# # #
# # #     # 自适应游走直到收集足够节点
# # #     while len(collected_nodes) < target_node_count and total_steps < max_total_steps:
# # #         # 重启判定
# # #         if random.random() < alpha:
# # #             current_node = center_node
# # #             restart_count += 1
# # #         else:
# # #             # 获取当前节点的邻居
# # #             neighbors = list(G.neighbors(current_node))
# # #
# # #             # 如果没有邻居，强制重启
# # #             if not neighbors:
# # #                 current_node = center_node
# # #                 restart_count += 1
# # #             else:
# # #                 # 智能邻居选择：优先选择未收集节点
# # #                 unvisited_neighbors = [n for n in neighbors if n not in collected_nodes]
# # #
# # #                 if unvisited_neighbors:
# # #                     # 优先选择未访问节点
# # #                     current_node = random.choice(unvisited_neighbors)
# # #                 else:
# # #                     # 所有邻居都已收集，选择访问频率最低的邻居
# # #                     least_visited = min(neighbors, key=lambda n: visit_frequency.get(n, 0))
# # #                     current_node = least_visited
# # #
# # #         # 更新访问频率
# # #         visit_frequency[current_node] = visit_frequency.get(current_node, 0) + 1
# # #
# # #         # 检查是否新节点
# # #         if current_node not in collected_nodes:
# # #             collected_nodes.add(current_node)
# # #             steps_without_new = 0  # 重置无新节点计数器
# # #         else:
# # #             steps_without_new += 1
# # #
# # #         total_steps += 1
# # #
# # #         # 无新节点容忍机制
# # #         if steps_without_new > patience:
# # #             # 自适应调整：降低重启概率以扩大探索范围
# # #             alpha = max(0.05, alpha * 0.8)
# # #             steps_without_new = 0
# # #             # print(f"自适应调整: alpha={alpha:.3f}")
# # #
# # #     # 创建子图并保留原始属性
# # #     subgraph = G.subgraph(collected_nodes).copy()
# # #
# # #     # 添加中心节点标记
# # #     subgraph.nodes[center_node]["is_center"] = True
# # #
# # #     # 添加收集指标元数据
# # #     subgraph.graph["collection_metrics"] = {
# # #         "requested_nodes": target_node_count,
# # #         "collected_nodes": len(collected_nodes),
# # #         "total_steps": total_steps,
# # #         "restart_count": restart_count,
# # #         "avg_restart_rate": restart_count / total_steps if total_steps > 0 else 0,
# # #         "success": len(collected_nodes) >= target_node_count,
# # #         "final_alpha": alpha
# # #     }
# # #
# # #     return subgraph
# # #
# # #
# # # import networkx as nx
# # # from collections import deque
# # # import random
# # #
# # #
# # # def generate_bfs_subgraph(
# # #         G: nx.Graph,
# # #         center_node: int,
# # #         target_node_count: int,
# # #         shuffle_neighbors: bool = True
# # # ) -> nx.Graph:
# # #     """
# # #     使用BFS生成以指定节点为中心的子图
# # #
# # #     参数:
# # #         G: 原始图对象 (NetworkX Graph)
# # #         center_node: 中心节点ID
# # #         target_node_count: 目标子图节点数
# # #         shuffle_neighbors: 是否随机打乱邻居顺序 (默认True，增加随机性)
# # #
# # #     返回:
# # #         包含目标节点数的子图 (保留所有原始属性)
# # #     """
# # #     # 验证输入
# # #     if center_node not in G:
# # #         raise ValueError(f"中心节点 {center_node} 不在图中")
# # #     if target_node_count < 1:
# # #         raise ValueError("目标节点数必须大于0")
# # #
# # #     # 如果目标节点数大于图大小，返回完整图
# # #     if target_node_count >= len(G):
# # #         subgraph = G.copy()
# # #         subgraph.graph["collection_metrics"] = {
# # #             "status": "full_graph",
# # #             "collected_nodes": len(G),
# # #             "target_nodes": target_node_count
# # #         }
# # #         return subgraph
# # #
# # #     # 初始化数据结构
# # #     visited = set([center_node])  # 已访问节点集合
# # #     queue = deque([center_node])  # BFS队列
# # #
# # #     # BFS遍历直到收集足够节点
# # #     while queue and len(visited) < target_node_count:
# # #         current = queue.popleft()
# # #
# # #         # 获取当前节点的邻居
# # #         neighbors = list(G.neighbors(current))
# # #
# # #         # 随机打乱邻居顺序 (增加随机性)
# # #         if shuffle_neighbors:
# # #             random.shuffle(neighbors)
# # #
# # #         # 遍历邻居
# # #         for neighbor in neighbors:
# # #             if len(visited) >= target_node_count:
# # #                 break
# # #
# # #             if neighbor not in visited:
# # #                 visited.add(neighbor)
# # #                 queue.append(neighbor)
# # #
# # #     # 创建子图并保留原始属性
# # #     subgraph = G.subgraph(visited).copy()
# # #
# # #     # 添加中心节点标记属性
# # #     for node in subgraph.nodes:
# # #         subgraph.nodes[node]["is_center"] = (node == center_node)
# # #
# # #     # 添加收集指标元数据
# # #     subgraph.graph["collection_metrics"] = {
# # #         "method": "BFS",
# # #         "center_node": center_node,
# # #         "target_nodes": target_node_count,
# # #         "collected_nodes": len(visited),
# # #         "success": len(visited) >= target_node_count,
# # #         "shuffle_neighbors": shuffle_neighbors
# # #     }
# # #
# # #     return subgraph
# # #
# # #
# # # import networkx as nx
# # # import random
# # # from collections import deque
# # #
# # #
# # # def generate_dfs_subgraph(
# # #         G: nx.Graph,
# # #         center_node: int,
# # #         target_node_count: int,
# # #         shuffle_neighbors: bool = True,
# # #         max_depth: int = None
# # # ) -> nx.Graph:
# # #     """
# # #     使用DFS生成以指定节点为中心的子图
# # #
# # #     参数:
# # #         G: 原始图对象 (NetworkX Graph)
# # #         center_node: 中心节点ID
# # #         target_node_count: 目标子图节点数
# # #         shuffle_neighbors: 是否随机打乱邻居顺序 (默认True，增加随机性)
# # #         max_depth: 最大搜索深度 (可选)
# # #
# # #     返回:
# # #         包含目标节点数的子图 (保留所有原始属性)
# # #     """
# # #     # 验证输入
# # #     if center_node not in G:
# # #         raise ValueError(f"中心节点 {center_node} 不在图中")
# # #     if target_node_count < 1:
# # #         raise ValueError("目标节点数必须大于0")
# # #
# # #     # 如果目标节点数大于图大小，返回完整图
# # #     if target_node_count >= len(G):
# # #         subgraph = G.copy()
# # #         subgraph.graph["collection_metrics"] = {
# # #             "status": "full_graph",
# # #             "collected_nodes": len(G),
# # #             "target_nodes": target_node_count
# # #         }
# # #         return subgraph
# # #
# # #     # 初始化数据结构
# # #     visited = set([center_node])  # 已访问节点集合
# # #     stack = deque([center_node])  # DFS栈
# # #
# # #     # DFS遍历直到收集足够节点
# # #     while stack and len(visited) < target_node_count:
# # #         current = stack.pop()
# # #
# # #         # 获取当前节点的邻居
# # #         neighbors = list(G.neighbors(current))
# # #
# # #         # 随机打乱邻居顺序 (增加随机性)
# # #         if shuffle_neighbors:
# # #             random.shuffle(neighbors)
# # #
# # #         # 遍历邻居
# # #         for neighbor in neighbors:
# # #             if len(visited) >= target_node_count:
# # #                 break
# # #
# # #             if neighbor not in visited:
# # #                 visited.add(neighbor)
# # #                 stack.append(neighbor)  # 将邻居压入栈中
# # #
# # #                 # 检查深度限制
# # #                 if max_depth is not None:
# # #                     # 计算当前深度
# # #                     depth = nx.shortest_path_length(G, source=center_node, target=neighbor)
# # #                     if depth > max_depth:
# # #                         break
# # #
# # #     # 创建子图并保留原始属性
# # #     subgraph = G.subgraph(visited).copy()
# # #
# # #     # 添加中心节点标记属性
# # #     for node in subgraph.nodes:
# # #         subgraph.nodes[node]["is_center"] = (node == center_node)
# # #
# # #     # 添加收集指标元数据
# # #     subgraph.graph["collection_metrics"] = {
# # #         "method": "DFS",
# # #         "center_node": center_node,
# # #         "target_nodes": target_node_count,
# # #         "collected_nodes": len(visited),
# # #         "success": len(visited) >= target_node_count,
# # #         "shuffle_neighbors": shuffle_neighbors,
# # #         "max_depth": max_depth
# # #     }
# # #
# # #     return subgraph
# # #
# # #
# # # import networkx as nx
# # # import numpy as np
# # # from scipy.sparse import csr_matrix, diags
# # # from scipy.sparse.linalg import expm_multiply
# # # import heapq
# # # import time
# # #
# # #
# # # def generate_diffusion_subgraph(
# # #         G: nx.Graph,
# # #         center_node: int,
# # #         target_node_count: int,
# # #         diffusion_time: float = 1.0,
# # #         max_iter: int = 100,
# # #         tol: float = 1e-6
# # # ) -> nx.Graph:
# # #     """
# # #     使用图扩散算法生成以指定节点为中心的子图
# # #
# # #     参数:
# # #         G: 原始图对象 (NetworkX Graph)
# # #         center_node: 中心节点ID
# # #         target_node_count: 目标子图节点数
# # #         diffusion_time: 扩散时间参数 (默认1.0)
# # #         max_iter: 最大迭代次数 (默认100)
# # #         tol: 收敛容差 (默认1e-6)
# # #
# # #     返回:
# # #         包含目标节点数的子图 (保留所有原始属性)
# # #     """
# # #     # 验证输入
# # #     if center_node not in G:
# # #         raise ValueError(f"中心节点 {center_node} 不在图中")
# # #     if target_node_count < 1:
# # #         raise ValueError("目标节点数必须大于0")
# # #
# # #     # 如果目标节点数大于图大小，返回完整图
# # #     if target_node_count >= len(G):
# # #         subgraph = G.copy()
# # #         subgraph.graph["collection_metrics"] = {
# # #             "status": "full_graph",
# # #             "collected_nodes": len(G),
# # #             "target_nodes": target_node_count
# # #         }
# # #         return subgraph
# # #
# # #     # 记录开始时间
# # #     start_time = time.time()
# # #
# # #     # 创建图的邻接矩阵表示
# # #     n = len(G)
# # #     node_index = {node: i for i, node in enumerate(G.nodes)}
# # #     index_node = {i: node for i, node in enumerate(G.nodes)}
# # #
# # #     # 构建邻接矩阵 (稀疏格式)
# # #     rows, cols = [], []
# # #     for u, v in G.edges():
# # #         i, j = node_index[u], node_index[v]
# # #         rows.append(i)
# # #         cols.append(j)
# # #         rows.append(j)
# # #         cols.append(i)
# # #
# # #     data = np.ones(len(rows))
# # #     A = csr_matrix((data, (rows, cols)), shape=(n, n))
# # #
# # #     # 计算度矩阵
# # #     degrees = np.array([d for _, d in G.degree()])
# # #     D = diags(degrees, format='csr')
# # #
# # #     # 计算图拉普拉斯矩阵 L = D - A
# # #     L = D - A
# # #
# # #     # 创建初始热源向量 (中心节点)
# # #     heat_vector = np.zeros(n)
# # #     center_idx = node_index[center_node]
# # #     heat_vector[center_idx] = 1.0
# # #
# # #     # 使用热核扩散计算扩散分数
# # #     # H = exp(-t * L) * heat_vector
# # #     diffusion_scores = expm_multiply(-diffusion_time * L, heat_vector)
# # #
# # #     # 使用幂迭代法计算扩散分数 (替代方法)
# # #     """
# # #     # 归一化转移矩阵 P = D^{-1}A
# # #     D_inv = diags(1.0 / degrees, format='csr')
# # #     P = D_inv @ A
# # #
# # #     # 幂迭代计算扩散分数
# # #     diffusion_scores = np.zeros(n)
# # #     diffusion_scores[center_idx] = 1.0
# # #
# # #     for _ in range(max_iter):
# # #         prev_scores = diffusion_scores.copy()
# # #         diffusion_scores = (1 - diffusion_time) * (P @ diffusion_scores) + diffusion_time * heat_vector
# # #
# # #         # 检查收敛
# # #         diff = np.linalg.norm(diffusion_scores - prev_scores)
# # #         if diff < tol:
# # #             break
# # #     """
# # #
# # #     # 将扩散分数映射回节点
# # #     node_scores = {}
# # #     for i, score in enumerate(diffusion_scores):
# # #         node = index_node[i]
# # #         node_scores[node] = score
# # #
# # #     # 选择扩散分数最高的target_node_count个节点
# # #     top_nodes = heapq.nlargest(target_node_count, node_scores.items(), key=lambda x: x[1])
# # #     selected_nodes = set(node for node, score in top_nodes)
# # #
# # #     # 创建子图并保留原始属性
# # #     subgraph = G.subgraph(selected_nodes).copy()
# # #
# # #     # 添加扩散分数作为节点属性
# # #     for node in subgraph.nodes:
# # #         subgraph.nodes[node]["diffusion_score"] = node_scores[node]
# # #
# # #     # 添加中心节点标记属性
# # #     subgraph.nodes[center_node]["is_center"] = True
# # #
# # #     # 添加收集指标元数据
# # #     end_time = time.time()
# # #     subgraph.graph["collection_metrics"] = {
# # #         "method": "Heat Kernel Diffusion",
# # #         "center_node": center_node,
# # #         "target_nodes": target_node_count,
# # #         "collected_nodes": len(selected_nodes),
# # #         "diffusion_time": diffusion_time,
# # #         "computation_time": end_time - start_time,
# # #         "max_score": max(node_scores.values()),
# # #         "min_score": min(node_scores.values()),
# # #         "success": len(selected_nodes) >= target_node_count
# # #     }
# # #
# # #     return subgraph
# # #
# # #
# # # import networkx as nx
# # # import numpy as np
# # # from scipy.sparse import csr_matrix, diags
# # # from scipy.sparse.linalg import expm_multiply
# # # import heapq
# # # import time
# # #
# # #
# # # def generate_diffusion_subgraph(
# # #         G: nx.Graph,
# # #         center_node: int,
# # #         target_node_count: int,
# # #         diffusion_time: float = 1.0,
# # #         max_iter: int = 100,
# # #         tol: float = 1e-6
# # # ) -> nx.Graph:
# # #     """
# # #     使用图扩散算法生成以指定节点为中心的子图
# # #
# # #     参数:
# # #         G: 原始图对象 (NetworkX Graph)
# # #         center_node: 中心节点ID
# # #         target_node_count: 目标子图节点数
# # #         diffusion_time: 扩散时间参数 (默认1.0)
# # #         max_iter: 最大迭代次数 (默认100)
# # #         tol: 收敛容差 (默认1e-6)
# # #
# # #     返回:
# # #         包含目标节点数的子图 (保留所有原始属性)
# # #     """
# # #     # 验证输入
# # #     if center_node not in G:
# # #         raise ValueError(f"中心节点 {center_node} 不在图中")
# # #     if target_node_count < 1:
# # #         raise ValueError("目标节点数必须大于0")
# # #
# # #     # 如果目标节点数大于图大小，返回完整图
# # #     if target_node_count >= len(G):
# # #         subgraph = G.copy()
# # #         subgraph.graph["collection_metrics"] = {
# # #             "status": "full_graph",
# # #             "collected_nodes": len(G),
# # #             "target_nodes": target_node_count
# # #         }
# # #         return subgraph
# # #
# # #     # 记录开始时间
# # #     start_time = time.time()
# # #
# # #     # 创建图的邻接矩阵表示
# # #     n = len(G)
# # #     node_index = {node: i for i, node in enumerate(G.nodes)}
# # #     index_node = {i: node for i, node in enumerate(G.nodes)}
# # #
# # #     # 构建邻接矩阵 (稀疏格式)
# # #     rows, cols = [], []
# # #     for u, v in G.edges():
# # #         i, j = node_index[u], node_index[v]
# # #         rows.append(i)
# # #         cols.append(j)
# # #         rows.append(j)
# # #         cols.append(i)
# # #
# # #     data = np.ones(len(rows))
# # #     A = csr_matrix((data, (rows, cols)), shape=(n, n))
# # #
# # #     # 计算度矩阵
# # #     degrees = np.array([d for _, d in G.degree()])
# # #     D = diags(degrees, format='csr')
# # #
# # #     # 计算图拉普拉斯矩阵 L = D - A
# # #     L = D - A
# # #
# # #     # 创建初始热源向量 (中心节点)
# # #     heat_vector = np.zeros(n)
# # #     center_idx = node_index[center_node]
# # #     heat_vector[center_idx] = 1.0
# # #
# # #     # 使用热核扩散计算扩散分数
# # #     # H = exp(-t * L) * heat_vector
# # #     diffusion_scores = expm_multiply(-diffusion_time * L, heat_vector)
# # #
# # #     # 使用幂迭代法计算扩散分数 (替代方法)
# # #     """
# # #     # 归一化转移矩阵 P = D^{-1}A
# # #     D_inv = diags(1.0 / degrees, format='csr')
# # #     P = D_inv @ A
# # #
# # #     # 幂迭代计算扩散分数
# # #     diffusion_scores = np.zeros(n)
# # #     diffusion_scores[center_idx] = 1.0
# # #
# # #     for _ in range(max_iter):
# # #         prev_scores = diffusion_scores.copy()
# # #         diffusion_scores = (1 - diffusion_time) * (P @ diffusion_scores) + diffusion_time * heat_vector
# # #
# # #         # 检查收敛
# # #         diff = np.linalg.norm(diffusion_scores - prev_scores)
# # #         if diff < tol:
# # #             break
# # #     """
# # #
# # #     # 将扩散分数映射回节点
# # #     node_scores = {}
# # #     for i, score in enumerate(diffusion_scores):
# # #         node = index_node[i]
# # #         node_scores[node] = score
# # #
# # #     # 选择扩散分数最高的target_node_count个节点
# # #     top_nodes = heapq.nlargest(target_node_count, node_scores.items(), key=lambda x: x[1])
# # #     selected_nodes = set(node for node, score in top_nodes)
# # #
# # #     # 创建子图并保留原始属性
# # #     subgraph = G.subgraph(selected_nodes).copy()
# # #
# # #     # 添加扩散分数作为节点属性
# # #     for node in subgraph.nodes:
# # #         subgraph.nodes[node]["diffusion_score"] = node_scores[node]
# # #
# # #     # 添加中心节点标记属性
# # #     subgraph.nodes[center_node]["is_center"] = True
# # #
# # #     # 添加收集指标元数据
# # #     end_time = time.time()
# # #     subgraph.graph["collection_metrics"] = {
# # #         "method": "Heat Kernel Diffusion",
# # #         "center_node": center_node,
# # #         "target_nodes": target_node_count,
# # #         "collected_nodes": len(selected_nodes),
# # #         "diffusion_time": diffusion_time,
# # #         "computation_time": end_time - start_time,
# # #         "max_score": max(node_scores.values()),
# # #         "min_score": min(node_scores.values()),
# # #         "success": len(selected_nodes) >= target_node_count
# # #     }
# # #
# # #     return subgraph
# # #
# # # def gene_candiate_subg(G,center_node,n):
# # #     rwr_subg=generate_rwr_subgraph(G,center_node,n)
# # #     bfs_subg=generate_bfs_subgraph(G,center_node,n)
# # #     dfs_subg=generate_dfs_subgraph(G,center_node,n)
# # #     diff_subg=generate_diffusion_subgraph(G,center_node,n)
# # #     return rwr_subg,bfs_subg,dfs_subg,diff_subg
# #
# #
# # import networkx as nx
# # import random
# # import numpy as np
# # from collections import deque
# # from typing import Set, Dict, List
# # from scipy.sparse import csr_matrix, diags
# # from scipy.sparse.linalg import expm_multiply
# # import heapq
# # import time
# #
# #
# # def generate_rwr_subgraph(
# #         G: nx.Graph,
# #         center_node: int,
# #         target_node_count: int,
# #         alpha: float = 0.15,
# #         max_total_steps: int = 10000,
# #         patience: int = 200
# # ) -> nx.Graph:
# #     """
# #     使用带重启的自适应随机游走 (RWR) 生成指定大小的子图
# #     确保子图包含原图中的所有边
# #
# #     参数:
# #         G: 原始图对象 (NetworkX Graph)
# #         center_node: 中心节点ID
# #         target_node_count: 目标子图节点数
# #         alpha: 重启概率 (默认0.15)
# #         max_total_steps: 最大总步数限制 (防止无限循环)
# #         patience: 无新节点收集时的容忍步数
# #
# #     返回:
# #         包含目标节点数的子图 (保留所有原始属性)
# #     """
# #     # 验证输入
# #     if center_node not in G:
# #         raise ValueError(f"中心节点 {center_node} 不在图中")
# #     if target_node_count < 1:
# #         raise ValueError("目标节点数必须大于0")
# #
# #     # 如果目标节点数大于图大小，返回完整图
# #     if target_node_count >= len(G):
# #         subgraph = G.copy()
# #         subgraph.graph["collection_metrics"] = {
# #             "status": "full_graph",
# #             "collected_nodes": len(G),
# #             "total_steps": 0
# #         }
# #         return subgraph
# #
# #     # 初始化数据结构
# #     collected_nodes = set([center_node])  # 已收集节点集合
# #     current_node = center_node  # 当前节点
# #     total_steps = 0  # 总步数计数器
# #     steps_without_new = 0  # 无新节点步数计数器
# #     restart_count = 0  # 重启次数计数器
# #
# #     # 节点访问频率记录 (用于智能重启)
# #     visit_frequency: Dict[int, int] = {center_node: 1}
# #
# #     # 自适应游走直到收集足够节点
# #     while len(collected_nodes) < target_node_count and total_steps < max_total_steps:
# #         # 重启判定
# #         if random.random() < alpha:
# #             current_node = center_node
# #             restart_count += 1
# #         else:
# #             # 获取当前节点的邻居
# #             neighbors = list(G.neighbors(current_node))
# #
# #             # 如果没有邻居，强制重启
# #             if not neighbors:
# #                 current_node = center_node
# #                 restart_count += 1
# #             else:
# #                 # 智能邻居选择：优先选择未收集节点
# #                 unvisited_neighbors = [n for n in neighbors if n not in collected_nodes]
# #
# #                 if unvisited_neighbors:
# #                     # 优先选择未访问节点
# #                     current_node = random.choice(unvisited_neighbors)
# #                 else:
# #                     # 所有邻居都已收集，选择访问频率最低的邻居
# #                     least_visited = min(neighbors, key=lambda n: visit_frequency.get(n, 0))
# #                     current_node = least_visited
# #
# #         # 更新访问频率
# #         visit_frequency[current_node] = visit_frequency.get(current_node, 0) + 1
# #
# #         # 检查是否新节点
# #         if current_node not in collected_nodes:
# #             collected_nodes.add(current_node)
# #             steps_without_new = 0  # 重置无新节点计数器
# #         else:
# #             steps_without_new += 1
# #
# #         total_steps += 1
# #
# #         # 无新节点容忍机制
# #         if steps_without_new > patience:
# #             # 自适应调整：降低重启概率以扩大探索范围
# #             alpha = max(0.05, alpha * 0.8)
# #             steps_without_new = 0
# #
# #     # 创建子图并保留原始属性
# #     subgraph = G.subgraph(collected_nodes).copy()
# #
# #     # 添加中心节点标记
# #     subgraph.nodes[center_node]["is_center"] = True
# #
# #     # 添加收集指标元数据
# #     subgraph.graph["collection_metrics"] = {
# #         "requested_nodes": target_node_count,
# #         "collected_nodes": len(collected_nodes),
# #         "total_steps": total_steps,
# #         "restart_count": restart_count,
# #         "avg_restart_rate": restart_count / total_steps if total_steps > 0 else 0,
# #         "success": len(collected_nodes) >= target_node_count,
# #         "final_alpha": alpha
# #     }
# #
# #     return subgraph
# #
# #
# # def generate_bfs_subgraph(
# #         G: nx.Graph,
# #         center_node: int,
# #         target_node_count: int,
# #         shuffle_neighbors: bool = True
# # ) -> nx.Graph:
# #     """
# #     使用BFS生成以指定节点为中心的子图
# #     确保子图包含原图中的所有边
# #
# #     参数:
# #         G: 原始图对象 (NetworkX Graph)
# #         center_node: 中心节点ID
# #         target_node_count: 目标子图节点数
# #         shuffle_neighbors: 是否随机打乱邻居顺序 (默认True，增加随机性)
# #
# #     返回:
# #         包含目标节点数的子图 (保留所有原始属性)
# #     """
# #     # 验证输入
# #     if center_node not in G:
# #         raise ValueError(f"中心节点 {center_node} 不在图中")
# #     if target_node_count < 1:
# #         raise ValueError("目标节点数必须大于0")
# #
# #     # 如果目标节点数大于图大小，返回完整图
# #     if target_node_count >= len(G):
# #         subgraph = G.copy()
# #         subgraph.graph["collection_metrics"] = {
# #             "status": "full_graph",
# #             "collected_nodes": len(G),
# #             "target_nodes": target_node_count
# #         }
# #         return subgraph
# #
# #     # 初始化数据结构
# #     visited = set([center_node])  # 已访问节点集合
# #     queue = deque([center_node])  # BFS队列
# #
# #     # BFS遍历直到收集足够节点
# #     while queue and len(visited) < target_node_count:
# #         current = queue.popleft()
# #
# #         # 获取当前节点的邻居
# #         neighbors = list(G.neighbors(current))
# #
# #         # 随机打乱邻居顺序 (增加随机性)
# #         if shuffle_neighbors:
# #             random.shuffle(neighbors)
# #
# #         # 遍历邻居
# #         for neighbor in neighbors:
# #             if len(visited) >= target_node_count:
# #                 break
# #
# #             if neighbor not in visited:
# #                 visited.add(neighbor)
# #                 queue.append(neighbor)
# #
# #     # 创建子图并保留原始属性
# #     subgraph = G.subgraph(visited).copy()
# #
# #     # 添加中心节点标记属性
# #     for node in subgraph.nodes:
# #         subgraph.nodes[node]["is_center"] = (node == center_node)
# #
# #     # 添加收集指标元数据
# #     subgraph.graph["collection_metrics"] = {
# #         "method": "BFS",
# #         "center_node": center_node,
# #         "target_nodes": target_node_count,
# #         "collected_nodes": len(visited),
# #         "success": len(visited) >= target_node_count,
# #         "shuffle_neighbors": shuffle_neighbors
# #     }
# #
# #     return subgraph
# #
# #
# # def generate_dfs_subgraph(
# #         G: nx.Graph,
# #         center_node: int,
# #         target_node_count: int,
# #         shuffle_neighbors: bool = True,
# #         max_depth: int = None
# # ) -> nx.Graph:
# #     """
# #     使用DFS生成以指定节点为中心的子图
# #     确保子图包含原图中的所有边
# #
# #     参数:
# #         G: 原始图对象 (NetworkX Graph)
# #         center_node: 中心节点ID
# #         target_node_count: 目标子图节点数
# #         shuffle_neighbors: 是否随机打乱邻居顺序 (默认True，增加随机性)
# #         max_depth: 最大搜索深度 (可选)
# #
# #     返回:
# #         包含目标节点数的子图 (保留所有原始属性)
# #     """
# #     # 验证输入
# #     if center_node not in G:
# #         raise ValueError(f"中心节点 {center_node} 不在图中")
# #     if target_node_count < 1:
# #         raise ValueError("目标节点数必须大于0")
# #
# #     # 如果目标节点数大于图大小，返回完整图
# #     if target_node_count >= len(G):
# #         subgraph = G.copy()
# #         subgraph.graph["collection_metrics"] = {
# #             "status": "full_graph",
# #             "collected_nodes": len(G),
# #             "target_nodes": target_node_count
# #         }
# #         return subgraph
# #
# #     # 初始化数据结构
# #     visited = set([center_node])  # 已访问节点集合
# #     stack = deque([center_node])  # DFS栈
# #
# #     # DFS遍历直到收集足够节点
# #     while stack and len(visited) < target_node_count:
# #         current = stack.pop()
# #
# #         # 获取当前节点的邻居
# #         neighbors = list(G.neighbors(current))
# #
# #         # 随机打乱邻居顺序 (增加随机性)
# #         if shuffle_neighbors:
# #             random.shuffle(neighbors)
# #
# #         # 遍历邻居
# #         for neighbor in neighbors:
# #             if len(visited) >= target_node_count:
# #                 break
# #
# #             if neighbor not in visited:
# #                 visited.add(neighbor)
# #                 stack.append(neighbor)  # 将邻居压入栈中
# #
# #                 # 检查深度限制
# #                 if max_depth is not None:
# #                     # 计算当前深度
# #                     depth = nx.shortest_path_length(G, source=center_node, target=neighbor)
# #                     if depth > max_depth:
# #                         break
# #
# #     # 创建子图并保留原始属性
# #     subgraph = G.subgraph(visited).copy()
# #
# #     # 添加中心节点标记属性
# #     for node in subgraph.nodes:
# #         subgraph.nodes[node]["is_center"] = (node == center_node)
# #
# #     # 添加收集指标元数据
# #     subgraph.graph["collection_metrics"] = {
# #         "method": "DFS",
# #         "center_node": center_node,
# #         "target_nodes": target_node_count,
# #         "collected_nodes": len(visited),
# #         "success": len(visited) >= target_node_count,
# #         "shuffle_neighbors": shuffle_neighbors,
# #         "max_depth": max_depth
# #     }
# #
# #     return subgraph
# #
# #
# # def generate_diffusion_subgraph(
# #         G: nx.Graph,
# #         center_node: int,
# #         target_node_count: int,
# #         diffusion_time: float = 1.0,
# #         max_iter: int = 100,
# #         tol: float = 1e-6
# # ) -> nx.Graph:
# #     """
# #     使用图扩散算法生成以指定节点为中心的子图
# #     确保子图包含原图中的所有边
# #
# #     参数:
# #         G: 原始图对象 (NetworkX Graph)
# #         center_node: 中心节点ID
# #         target_node_count: 目标子图节点数
# #         diffusion_time: 扩散时间参数 (默认1.0)
# #         max_iter: 最大迭代次数 (默认100)
# #         tol: 收敛容差 (默认1e-6)
# #
# #     返回:
# #         包含目标节点数的子图 (保留所有原始属性)
# #     """
# #     # 验证输入
# #     if center_node not in G:
# #         raise ValueError(f"中心节点 {center_node} 不在图中")
# #     if target_node_count < 1:
# #         raise ValueError("目标节点数必须大于0")
# #
# #     # 如果目标节点数大于图大小，返回完整图
# #     if target_node_count >= len(G):
# #         subgraph = G.copy()
# #         subgraph.graph["collection_metrics"] = {
# #             "status": "full_graph",
# #             "collected_nodes": len(G),
# #             "target_nodes": target_node_count
# #         }
# #         return subgraph
# #
# #     # 记录开始时间
# #     start_time = time.time()
# #
# #     # 创建图的邻接矩阵表示
# #     n = len(G)
# #     node_index = {node: i for i, node in enumerate(G.nodes)}
# #     index_node = {i: node for i, node in enumerate(G.nodes)}
# #
# #     # 构建邻接矩阵 (稀疏格式)
# #     rows, cols = [], []
# #     for u, v in G.edges():
# #         i, j = node_index[u], node_index[v]
# #         rows.append(i)
# #         cols.append(j)
# #         rows.append(j)
# #         cols.append(i)
# #
# #     data = np.ones(len(rows))
# #     A = csr_matrix((data, (rows, cols)), shape=(n, n))
# #
# #     # 计算度矩阵
# #     degrees = np.array([d for _, d in G.degree()])
# #     D = diags(degrees, format='csr')
# #
# #     # 计算图拉普拉斯矩阵 L = D - A
# #     L = D - A
# #
# #     # 创建初始热源向量 (中心节点)
# #     heat_vector = np.zeros(n)
# #     center_idx = node_index[center_node]
# #     heat_vector[center_idx] = 1.0
# #
# #     # 使用热核扩散计算扩散分数
# #     # H = exp(-t * L) * heat_vector
# #     diffusion_scores = expm_multiply(-diffusion_time * L, heat_vector)
# #
# #     # 将扩散分数映射回节点
# #     node_scores = {}
# #     for i, score in enumerate(diffusion_scores):
# #         node = index_node[i]
# #         node_scores[node] = score
# #
# #     # 选择扩散分数最高的target_node_count个节点
# #     top_nodes = heapq.nlargest(target_node_count, node_scores.items(), key=lambda x: x[1])
# #     selected_nodes = set(node for node, score in top_nodes)
# #
# #     # 创建子图并保留原始属性
# #     subgraph = G.subgraph(selected_nodes).copy()
# #
# #     # 添加扩散分数作为节点属性
# #     for node in subgraph.nodes:
# #         subgraph.nodes[node]["diffusion_score"] = node_scores[node]
# #
# #     # 添加中心节点标记属性
# #     #subgraph.nodes[center_node]["is_center"] = True
# #
# #     # 添加收集指标元数据
# #     end_time = time.time()
# #     subgraph.graph["collection_metrics"] = {
# #         "method": "Heat Kernel Diffusion",
# #         "center_node": center_node,
# #         "target_nodes": target_node_count,
# #         "collected_nodes": len(selected_nodes),
# #         "diffusion_time": diffusion_time,
# #         "computation_time": end_time - start_time,
# #         "max_score": max(node_scores.values()),
# #         "min_score": min(node_scores.values()),
# #         "success": len(selected_nodes) >= target_node_count
# #     }
# #
# #     return subgraph
# #
# #
# # def gene_candiate_subg(G, center_node, n):
# #     """
# #     生成四种类型的候选子图
# #     确保每个子图都包含原图中的所有边
# #
# #     参数:
# #         G: 原始图对象
# #         center_node: 中心节点ID
# #         n: 目标子图节点数
# #
# #     返回:
# #         四种候选子图: (rwr_subg, bfs_subg, dfs_subg, diff_subg)
# #     """
# #     rwr_subg = generate_rwr_subgraph(G, center_node, n)
# #     bfs_subg = generate_bfs_subgraph(G, center_node, n)
# #     dfs_subg = generate_dfs_subgraph(G, center_node, n)
# #     diff_subg = generate_diffusion_subgraph(G, center_node, n)
# #     return rwr_subg, bfs_subg, dfs_subg, diff_subg
