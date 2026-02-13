import torch_geometric.utils.convert as cv
from torch_geometric.data import NeighborSampler as RawNeighborSampler
import pandas as pd
from utils import *
import warnings
import argparse
warnings.filterwarnings('ignore')
import collections
import networkx as nx
import copy 
from sklearn.metrics import roc_auc_score
import os
from models import *
import numpy as np
import random
import torch
from data import *
from gene_seed import generate_k_seed
from gene_subg import gene_candiate_subg
from model_gtn import *
from model_deepseek import model_deepseek
# from evaluate_align import evaluate_node_align
from evaluate_subg_node_align import *
from ttttt import *
from node2vec import *
from gcn_subg import gcn_subg_node_emb
from add_attr.community import leiden_community_detection
from add_attr.neighbors import add_neighbors_attribute,add_node_id_attribute
from add_attr.update_attr import modify_features_attribute
from model_gcn import *

def set_seeds(n):
    seed = int(n)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed = 22
set_seeds(seed)
print("set seed:", seed)


def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Run myProject.")
    parser.add_argument('--attribute_folder', nargs='?', default='dataset/attribute/')
    parser.add_argument('--data_folder', nargs='?', default='dataset/graph/')
    parser.add_argument('--alignment_folder', nargs='?', default='dataset/alignment/',
                         help="Make sure the alignment numbering start from 0")
    parser.add_argument('--k_hop', nargs='?', default=2)  
    parser.add_argument('--hid_dim', nargs='?', default=150) 
    parser.add_argument('--train_ratio', nargs='?', default= 0.1) 
    parser.add_argument('--graphname', nargs='?', default='douban')
    parser.add_argument('--mode', nargs='?', default='not_perturbed', help="not_perturbed or perturbed") 
    parser.add_argument('--edge_portion', nargs='?', default=0.05,  help="a param for the perturbation case")  
    
    return parser.parse_args()

args = parse_args()


''' ------------------------ Run Grad-Align -----------------------------  '''


if __name__ == "__main__":
        
    G1, G2, attr1, attr2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict = na_dataloader(args)
    print("图1、图2节点、边数",G1.number_of_nodes(),G1.number_of_edges(),G2.number_of_nodes(),G2.number_of_edges())

    g1 = add_node_attributes_from_matrix(G1, attr1, 'features')
    g2 = add_node_attributes_from_matrix(G2, attr2, 'features')

################################################
    gcn_emb_csv_path_A = f"./gcn_emb/{args.graphname}_G1_node_embeddings.csv"
    gcn_emb_csv_path_B = f"./gcn_emb/{args.graphname}_G2_node_embeddings.csv"
    # embedder1 = DualSharedGCNEmbedder(
    #     hidden_dims=[512,256],  # 1个隐藏层，共2层GCN
    #     output_dim=100,
    #     dropout=0.2,
    #     debug_mode=True
    # )
    #
    # emb1_dict, emb2_dict = embedder1.generate_and_save_embeddings(
    #     G1, G2,
    #     output_file1=gcn_emb_csv_path_A,
    #     output_file2=gcn_emb_csv_path_B
    # )
################################################
    refine_emb_csv_A = f"./refine_emb/{args.graphname}_G1_refine_node_emb.csv"
    refine_emb_csv_B = f"./refine_emb/{args.graphname}_G2_refine_node_emb.csv"
    #
    # GradAlign = GradAlign(G1, G2, attr1, attr2, args.k_hop, args.hid_dim, alignment_dict, alignment_dict_reversed, \
    #                                   args.train_ratio, idx1_dict, idx2_dict, alpha = G2.number_of_nodes() / G1.number_of_nodes(), beta = 1)
    # pre_align_dict=GradAlign.run_algorithm()
    # source_emb, target_emb, id_to_idx_source, id_to_idx_target = emb_to_matrix(gcn_emb_csv_path_A, gcn_emb_csv_path_B)
    # refined_source, refined_target = refine_embeddings_with_alignment(
    #     source_emb,
    #     target_emb,
    #     id_to_idx_source,
    #     id_to_idx_target,
    #     pre_align_dict,
    #     refine_emb_csv_A,
    #     refine_emb_csv_B
    # )
####################################################
    merge_data_csv_path_A = f"./merge_data/{args.graphname}_G1_merged_node_embeddings_with_attributes.csv"
    merge_data_csv_path_B = f"./merge_data/{args.graphname}_G2_merged_node_embeddings_with_attributes.csv"
    # merge_attributes_with_embeddings(
    #     attribute_matrix=attr1,
    #     embedding_csv_path=refine_emb_csv_A,
    #     output_csv_path=merge_data_csv_path_A
    # )
    # merge_attributes_with_embeddings(
    #     attribute_matrix=attr2,
    #     embedding_csv_path=refine_emb_csv_B,
    #     output_csv_path=merge_data_csv_path_B
    # )
#################################################
    att1 = f"./without_emb/{args.graphname}_G1.csv"
    att2 = f"./without_emb/{args.graphname}_G2.csv"
    # save_graph_features_to_csv(G1, G2,att1,att2)
    ds_emb_csv_path_A = f'./ds_emb/{args.graphname}_G1_deepseek_node_emb.csv'
    ds_emb_csv_path_B = f'./ds_emb/{args.graphname}_G2_deepseek_node_emb.csv'
    # model_deepseek(att1,att2,ds_emb_csv_path_A,ds_emb_csv_path_B)
    #model_deepseek(merge_data_csv_path_A, merge_data_csv_path_B, ds_emb_csv_path_A, ds_emb_csv_path_B)


    # hits_score, pred_align, sim_matrix =evaluate_node_alignment_metrics(
    #     ds_emb_csv_path_A,
    #     ds_emb_csv_path_B,
    #     alignment_dict,
    #     batch_size=5000
    # )



    t=300
    iter=0
    total_hits1=0
    total_hits5=0
    total_hits10=0
    total_mrr=0
    total_node_g2=[]
    total_subg_size=0
    while iter<t:
        center_node_A, center_node_B = random_key_value(alignment_dict)
        print(center_node_A, center_node_B)
        #douban:5/15;     econ
        source_subg, source_count = extract_node_embedding(center_node_A, ds_emb_csv_path_A, 5)
        target_subg, target_count = extract_node_embedding(center_node_B, ds_emb_csv_path_B, 15)
        print("源子图、目标子图节点数", source_count, target_count)

        count = 0
        for key, value in alignment_dict.items():
            if key in source_subg and value in target_subg:
                count += 1
        print("应对齐节点对数量",count)

        if count==0:
            continue

        result = find_max_similarity_pairs(
            source_subg, target_subg,
            ds_emb_csv_path_A,
            ds_emb_csv_path_B
        )

        if result:
            # 计算平均相似度
            avg_similarity = sum(result.values()) / len(result)
            print(f"平均相似度: {avg_similarity:.4f}")

            # 找到最高相似度对
            max_pair = max(result.items(), key=lambda x: x[1])
            print(f"最高相似度对: {max_pair[0]}，相似度: {max_pair[1]:.4f}")

            # 找到最低相似度对
            min_pair = min(result.items(), key=lambda x: x[1])
            print(f"最低相似度对: {min_pair[0]}，相似度: {min_pair[1]:.4f}")

        sorted_items = sorted(result.items(), key=lambda item: item[1], reverse=True)
        # print(sorted_items)

        source_subg_filter = []
        target_subg_filter = []
        for key, value in sorted_items:
            #douban:0.9999;   econ:0.9997
            if value > 0.9999:
                source_node, target_node = key
                source_subg_filter.append(source_node)
                target_subg_filter.append(target_node)
        print("过滤后节点数",len(source_subg_filter), len(target_subg_filter))
        total_subg_size=total_subg_size+len(source_subg_filter)
        if len(source_subg_filter)==0 or len(target_subg_filter)==0:
            continue



        #print("\n计算相似度匹配和评估指标...")
        max_similarity_dict, evaluation_metrics = find_max_similarity_pairs_with_debug(
            source_subg_filter, target_subg_filter,
            ds_emb_csv_path_A,
            ds_emb_csv_path_B,
            alignment_dict
        )

        print(f"第{iter}轮指标，Hits@1: {evaluation_metrics['hits@1']:.4f},"
              f"Hits@5: {evaluation_metrics['hits@5']:.4f},"
              f"Hits@10: {evaluation_metrics['hits@10']:.4f},"
              f"MRR: {evaluation_metrics['mrr']:.4f}")
        total_hits1 = total_hits1 + evaluation_metrics['hits@1']
        total_hits5 = total_hits5 + evaluation_metrics['hits@5']
        total_hits10 = total_hits10 + evaluation_metrics['hits@10']
        total_mrr = total_mrr + evaluation_metrics['mrr']
        for node in target_subg_filter:
            total_node_g2.append(node)
        iter=iter+1

        if iter%10==0:
            print(f"第{iter}轮平均值：hits1,hits5,hits10,mrr", total_hits1 / iter, total_hits5 / iter, total_hits10 / iter, total_mrr / iter)
    print("hits1,hits5,hits10,mrr",total_hits1/t,total_hits5/t,total_hits10/t,total_mrr/t)
    print("平均子图大小",total_subg_size/iter)

    node_count=0
    for n in total_node_g2:
        if n in G2.nodes():
            node_count=node_count+1
    per=node_count/G2.number_of_nodes()
    print(f"百分比{per}")
    #

