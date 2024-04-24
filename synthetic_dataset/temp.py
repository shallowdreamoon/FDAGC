import networkx as nx
import random
import networkx as nx
import numpy as np
import scipy.sparse
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
from scipy.sparse import csc_matrix, save_npz, load_npz
from metrics import eva_metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from metrics import eva_metrics

# 设置随机数生成器的种子
np.random.seed(40)
random.seed(42)


# 绘制网络图(不包含标签颜色的区分)
def plot_G_no_color(G):
    pos = nx.spring_layout(G)  # 使用布局算法定义节点位置
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=300)
    plt.title("Watts-Strogatz Small World Network")
    plt.show()

# 绘制网络图(包含标签颜色的区分)
def plot_G_color(G, labels):
    # 创建一个颜色映射，将标签映射到不同的颜色
    color_map = {}
    for label in labels:
        # 生成一个随机的RGB颜色
        color = "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color_map[label] = color
    #color_map = {0: 'skyblue', 1: 'lightcoral', 2: 'lightgreen', 3: 'red', 4: 'yellow'}

    # 获取节点颜色
    node_colors = [color_map[label] for label in labels]
    # 绘制网络图
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=300, edge_color='gray', width=1.0, alpha=0.7)
    plt.title("Watts-Strogatz Small World Network with Clustered Colors")
    plt.show()

if __name__ == '__main__':

    # 创建一个空的图
    G = nx.Graph()

    # 添加节点和属性向量
    num_nodes = 100  # 假设有 10 个节点
    vector_length = 20  # 属性向量的长度
    n_clu=4
    probability_of_one = 0.8  # 生成1的概率
    probability_of_one_inv = 0.8  # 生成1的概率

    my_list = list(range(num_nodes))
    chunk_size = int(num_nodes / n_clu)
    # 将列表等份划分
    chunks = [my_list[i:i + chunk_size] for i in range(0, len(my_list), chunk_size)]
    node_name = -1

    for i in range(len(chunks)):
        att_change_max = (i+1) * int(vector_length/n_clu)
        att_change_min = att_change_max - int(vector_length/n_clu)
        for j in range(len(chunks[i])):
            # 生成属性向量，前5个属性值设置为1，其余设置为0
            attribute_vector = [1 if att_change_min <= k and k < att_change_max and random.random() < probability_of_one else 0 for k in
                                range(vector_length)]

            # 添加节点，并将属性向量和标签添加为节点的属性
            node_name += 1
            G.add_node(node_name, attribute=attribute_vector, label=i)



    # for node in range(num_nodes):
    #
    #     # 生成属性向量，前5个属性值设置为1，其余设置为0
    #     attribute_vector = [1 if i < 5 and random.random() < probability_of_one else 0 for i in range(vector_length)]
    #
    #     # 添加节点，并将属性向量和标签添加为节点的属性
    #     G.add_node(node, attribute=attribute_vector, label=f"Node_{node}")

    # 根据标签添加边
    for i in G.nodes():
        for j in G.nodes():
            if i != j:  # 不连接节点自身
                if G.nodes[i]["label"] == G.nodes[j]["label"]:
                    # 如果标签相同，添加边的概率为0.8
                    if random.random() < 0.2:
                        G.add_edge(i, j)
                else:
                    # 如果标签不同，添加边的概率为0.1
                    if random.random() < 0.01:
                        G.add_edge(i, j)

    # 获取整个图的属性矩阵、邻接矩阵和标签信息
    adj_matrix = nx.to_numpy_matrix(G)  # 邻接矩阵
    attr_matrix = np.array([G.nodes[i]['attribute'] for i in range(num_nodes)])  # 属性矩阵
    labels = np.array([G.nodes[i]['label'] for i in range(num_nodes)])  # 标签信息
    print(adj_matrix)
    print(attr_matrix)
    print(labels)
    print(len(G.nodes))
    print(len(G.edges))
    #plot_G_no_color(G)
    plot_G_color(G, labels)


    kmeans = KMeans(n_clusters=4, random_state=42)
    pred = kmeans.fit_predict(adj_matrix)
    print(pred)
    res = eva_metrics(labels, pred)
    print(res)


