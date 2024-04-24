"""
使用kmeans对合成数据集进行测试
"""

import networkx as nx
import numpy as np
import scipy.sparse
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
from scipy.sparse import csc_matrix, save_npz, load_npz
from metrics import eva_metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
NMI = normalized_mutual_info_score
ARI  = adjusted_rand_score

def read_AG(adj_path, att_path, lab_path):
    adj = scipy.sparse.load_npz(adj_path)
    fea = np.load(att_path)
    labels = np.load(lab_path)
    return adj, fea, labels

if __name__ == '__main__':
    adj_path = r"dataset_4\adjacency_matrix.npz"
    att_path = r"dataset_4\attributes.npy"
    lab_path = r'dataset_4\labels.npy'

    # 读取
    adj, att, lab = read_AG(adj_path, att_path, lab_path)
    print(lab)
    print("==============")
    # 使用K均值聚类算法为节点分配标签
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(att)
    print(labels)


    res = eva_metrics(lab, labels)
    print(res)
