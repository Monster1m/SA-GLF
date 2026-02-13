import csv
import random
import shutil
import os
import networkx as nx
from math import ceil
from collections import deque
from typing import Set, Tuple, List


def perturb_graph(original_edges_file, original_attrs_file,
                  new_edges_file, new_attrs_file, alignment_file,
                  delete_ratio=0.4, add_ratio=0.1):
    # 初始化图数据结构
    graph = {
        'nodes': set(),
        'edges': set(),
        'attributes': {}
    }

    # 创建NetworkX图对象用于连通性检查
    nx_graph = nx.Graph()

    # 读取原始边文件
    with open(original_edges_file, 'r') as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                src, tgt = row[0].strip(), row[1].strip()
                if src != tgt:  # 忽略自环
                    graph['nodes'].add(src)
                    graph['nodes'].add(tgt)
                    # 存储标准化边（排序确保无向图无重复）
                    edge = tuple(sorted([src, tgt]))
                    graph['edges'].add(edge)
                    # 添加到NetworkX图
                    nx_graph.add_edge(src, tgt)

    # 读取原始属性文件
    with open(original_attrs_file, 'r') as f:
        for row in csv.reader(f):
            if len(row) >= 1:
                node = row[0].strip()
                if node in graph['nodes']:
                    attrs = [float(x.strip()) for x in row[1:]]
                    graph['attributes'][node] = attrs

    # 计算基本指标
    num_original_edges = len(graph['edges'])
    num_delete = ceil(num_original_edges * delete_ratio)
    num_add = ceil(num_original_edges * add_ratio)

    # 1. 随机删边（确保不破坏连通性）
    if num_delete > 0:
        # 获取所有非桥接边（删除后不会断开图的边）
        non_bridge_edges = get_non_bridge_edges(nx_graph)

        # 确保有足够的非桥接边
        if len(non_bridge_edges) < num_delete:
            print(f"警告: 只有 {len(non_bridge_edges)} 个非桥接边可用，但需要删除 {num_delete} 条边")
            num_delete = min(num_delete, len(non_bridge_edges))
            print(f"将删除边数调整为: {num_delete}")

        if num_delete > 0:
            edges_to_delete = random.sample(non_bridge_edges, num_delete)

            # 删除边
            for edge in edges_to_delete:
                # 从图数据结构中删除
                sorted_edge = tuple(sorted(edge))
                if sorted_edge in graph['edges']:
                    graph['edges'].remove(sorted_edge)

                # 从NetworkX图中删除
                nx_graph.remove_edge(*edge)

    # 2. 随机增边（优先连接不同连通分量）
    if num_add > 0:
        # 获取当前连通分量
        components = list(nx.connected_components(nx_graph))

        # 如果图不连通，优先连接不同连通分量
        if len(components) > 1:
            print(f"图有 {len(components)} 个连通分量，优先连接不同分量")
            num_added = add_edges_between_components(nx_graph, graph, components, num_add)
            num_add -= num_added
            print(f"已添加 {num_added} 条跨分量边，还需添加 {num_add} 条边")

        # 添加剩余边（在连通分量内部）
        if num_add > 0:
            nodes_list = list(graph['nodes'])
            existing_edges = set(graph['edges'])
            new_edges = set()

            while len(new_edges) < num_add:
                # 随机选择两个不同的节点
                u, v = random.sample(nodes_list, 2)
                # 创建排序边
                edge = tuple(sorted([u, v]))

                # 如果不是自环且不是已存在边
                if u != v and edge not in existing_edges:
                    new_edges.add(edge)
                    existing_edges.add(edge)
                    nx_graph.add_edge(u, v)

            graph['edges'] |= new_edges

    # 3. 生成新边文件
    with open(new_edges_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for edge in graph['edges']:
            writer.writerow(edge)

    # 4. 生成新属性文件（与原始相同）
    shutil.copyfile(original_attrs_file, new_attrs_file)

    # 5. 生成对齐文件
    sorted_nodes = sorted(graph['nodes'], key=int)  # 按数值排序
    with open(alignment_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for node in sorted_nodes:
            writer.writerow([node, node])  # 原ID与新ID相同

    # 计算并返回统计数据
    num_nodes = len(graph['nodes'])
    num_edges = len(graph['edges'])

    # 检查连通性
    is_connected = nx.is_connected(nx_graph)
    num_components = nx.number_connected_components(nx_graph)

    # 获取任意节点的属性
    sample_attr = None
    for node in sorted_nodes:  # 使用排序列表获取确定性的示例节点
        if node in graph['attributes']:
            sample_attr = graph['attributes'][node]
            sample_node = node
            break

    # 输出结果
    print(f"原始边数: {num_original_edges}")
    print(f"删除边数: {num_delete}")
    print(f"增加边数: {num_add}")
    print(f"新图节点数: {num_nodes}")
    print(f"新图边数: {num_edges}")
    print(f"扰动率: {(num_delete + num_add) / num_original_edges:.2%}")
    print(f"连通性: {'连通' if is_connected else '不连通'}")
    print(f"连通分量数: {num_components}")

    if sample_attr is not None:
        preview = sample_attr[:5] + ['...'] if len(sample_attr) > 5 else sample_attr
        print(f"示例节点 '{sample_node}' 的属性向量 (前{len(preview)}个值): {preview}")
    else:
        print("未找到带属性的节点")

    print(f"\n文件已生成:")
    print(f"- 新边文件: {os.path.abspath(new_edges_file)}")
    print(f"- 新属性文件: {os.path.abspath(new_attrs_file)}")
    print(f"- 对齐文件: {os.path.abspath(alignment_file)}")

    return is_connected


def get_non_bridge_edges(G: nx.Graph) -> List[Tuple[str, str]]:
    """
    获取所有非桥接边（删除后不会断开图的边）

    参数:
        G: NetworkX图对象

    返回:
        非桥接边列表
    """
    # 使用BFS/DFS检测桥接边
    bridges = set()
    low = {}
    disc = {}
    parent = {}
    time = [0]  # 使用列表以便在递归中修改

    # 初始化
    for node in G.nodes():
        disc[node] = -1
        low[node] = -1
        parent[node] = None

    # DFS遍历
    def dfs(u):
        disc[u] = time[0]
        low[u] = time[0]
        time[0] += 1

        for v in G.neighbors(u):
            if disc[v] == -1:  # 未访问
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])

                # 检查是否为桥
                if low[v] > disc[u]:
                    bridges.add(tuple(sorted([u, v])))
            elif v != parent[u]:  # 回边
                low[u] = min(low[u], disc[v])

    # 从每个未访问节点开始
    for node in G.nodes():
        if disc[node] == -1:
            time[0] = 0
            dfs(node)

    # 所有边减去桥接边
    all_edges = set(tuple(sorted(edge)) for edge in G.edges())
    non_bridge_edges = list(all_edges - bridges)

    return non_bridge_edges


def add_edges_between_components(
        G: nx.Graph,
        graph_data: dict,
        components: List[Set[str]],
        num_edges: int
) -> int:
    """
    在不同连通分量之间添加边

    参数:
        G: NetworkX图对象
        graph_data: 图数据结构
        components: 连通分量列表
        num_edges: 要添加的边数

    返回:
        实际添加的边数
    """
    added = 0

    # 尝试连接所有分量对
    while len(components) > 1 and added < num_edges:
        # 随机选择两个不同的分量
        comp1, comp2 = random.sample(components, 2)

        # 从每个分量中随机选择一个节点
        node1 = random.choice(list(comp1))
        node2 = random.choice(list(comp2))

        # 创建排序边
        edge = tuple(sorted([node1, node2]))

        # 添加边（如果不存在）
        if edge not in graph_data['edges']:
            graph_data['edges'].add(edge)
            G.add_edge(node1, node2)
            added += 1

            # 合并分量
            new_comp = comp1 | comp2
            components.remove(comp1)
            components.remove(comp2)
            components.append(new_comp)

    return added

name="dl"
# 示例用法
is_connected = perturb_graph(
    original_edges_file=f"./dataset/graph/{name}1.edges",
    original_attrs_file=f"./dataset/attribute/{name}attr1.csv",
    new_edges_file=f"./dataset/graph/{name}2.edges",
    new_attrs_file=f"./dataset/attribute/{name}attr2.csv",
    alignment_file=f"./dataset/alignment/{name}.csv"
)

if not is_connected:
    print("警告: 扰动后图不连通")