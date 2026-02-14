import networkx as nx

def linear_normalize(x):
    min_x = min(x.values())
    max_x = max(x.values())
    normalized_x = {key: (value - min_x) / (max_x - min_x) for key, value in x.items()}
    return normalized_x

def node_weights(g):#, num_nodes_to_select
    pr = nx.pagerank(g)
    dc = nx.degree_centrality(g)

    # 对各种中心性度量进行线性归一化
    pr_normalized = linear_normalize(pr)
    dc_normalized = linear_normalize(dc)

    average_centralities = {}


    #加权求平均
    for node in g.nodes():
        average_centralities[node] = (pr_normalized[node]+dc_normalized[node]) / 2#

    return average_centralities
#生成种子节点
def generate_k_seed(g,k):

    # 生成选择平均中心性值最大的前 k 个节点
    average_centralities=node_weights(g)
    top_k_nodes = [node for node, _ in sorted(average_centralities.items(), key=lambda x: x[1], reverse=True)[:k]]#int(k)
    top_k_nodes_reverse = [node for node, _ in sorted(average_centralities.items(), key=lambda x: x[1])[:k]]

    return top_k_nodes,average_centralities,top_k_nodes_reverse


