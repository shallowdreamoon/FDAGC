import networkx as nx
import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# 设置随机数生成器的种子
np.random.seed(40)

# 创建 Watts-Strogatz 小世界网络
n = 100
k = 4
p = 0.2
G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=42)

# 生成随机属性特征并为节点添加属性
for node in G.nodes():
    attribute_vector = [random.choice([0, 1]) for _ in range(10)]  # 随机生成长度为10的属性向量
    G.nodes[node]['attribute_vector'] = attribute_vector

# 提取属性特征并组合成属性矩阵
attribute_matrix = np.array([G.nodes[node]['attribute_vector'] for node in G.nodes()])

# 打印整个属性矩阵
print("Attribute Matrix:")
print(attribute_matrix)

# 绘制网络图
# pos = nx.spring_layout(G)  # 使用布局算法定义节点位置
# nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=300)
# plt.title("Watts-Strogatz Small World Network")
# plt.show()

# 使用K均值聚类算法为节点分配标签
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(attribute_matrix)


num_nodes = len(G.nodes)
# 调整图的结构，确保连接紧密的节点属于同一集群
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if labels[i] == labels[j]:
            G.add_edge(i, j)

# 创建一个颜色映射，将标签映射到不同的颜色
color_map = {0: 'skyblue', 1: 'lightcoral', 2: 'lightgreen', 3: 'red'}
# 获取节点颜色
node_colors = [color_map[label] for label in labels]
# 绘制网络图
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=300, edge_color='gray', width=1.0, alpha=0.7)
plt.title("Watts-Strogatz Small World Network with Clustered Colors")
plt.show()
