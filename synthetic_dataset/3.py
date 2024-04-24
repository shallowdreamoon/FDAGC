import networkx as nx
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, save_npz
import random
random.seed(42)
# 设置随机数生成器的种子
np.random.seed(40)

# 生成图的结构（这里使用Watts-Strogatz小世界网络）
G = nx.watts_strogatz_graph(n=100, k=4, p=0.2, seed=42)

# 绘制网络图
# pos = nx.spring_layout(G)  # 使用布局算法定义节点位置
# nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=300)
# plt.title("Watts-Strogatz Small World Network")
# plt.show()

# 生成属性特征
num_nodes = len(G.nodes)
num_attributes = 5
attributes = np.random.multivariate_normal([0, 0, 0, 0, 0], np.eye(num_attributes), num_nodes)

# 使用K均值聚类算法为节点分配标签
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(attributes)

# 添加噪音数据：将一些节点的标签更改为单独的标签
# 同时，更改噪音节点的属性特征
for i in range(5):
    labels[i] = 4  # 3 是一个新的标签，表示噪音集群
    attributes[i] = np.random.multivariate_normal([5, 5, 5, 5, 5], np.eye(num_attributes))

# 调整图的结构，确保连接紧密的节点属于同一集群
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if labels[i] == labels[j]:
            G.add_edge(i, j)

# 生成邻接矩阵
adjacency_matrix = nx.to_numpy_matrix(G)
# 将邻接矩阵转换为 CSC 格式
adjacency_csc = csc_matrix(adjacency_matrix)

# 保存 CSC 格式的矩阵为.npz文件
# save_npz(r'dataset_1\adjacency_matrix.npz', adjacency_csc)
#
#
# # 保存属性特征和邻接矩阵为 NumPy 数组
# np.save(r'dataset_1\attributes.npy', attributes)
# np.save(r'dataset_1\adjacency_matrix.npy', adjacency_matrix)
#
# # 保存标签数组为 NumPy 数组
# np.save(r'dataset_1\labels.npy', labels)

# 创建一个颜色映射，将标签映射到不同的颜色
color_map = {0: 'skyblue', 1: 'lightcoral', 2: 'lightgreen', 3: 'red', 4:'yellow'}
# 获取节点颜色
node_colors = [color_map[label] for label in labels]
# 绘制网络图
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=300, edge_color='gray', width=1.0, alpha=0.7)
plt.title("Watts-Strogatz Small World Network with Clustered Colors")
plt.show()


# # 绘制网络图
# pos = nx.spring_layout(G)  # 使用布局算法定义节点位置
# nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=300)
# plt.title("Watts-Strogatz Small World Network")
# plt.show()

