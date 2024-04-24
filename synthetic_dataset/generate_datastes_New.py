"""
与老师讨论后的最新版本；
"""

import networkx as nx
import numpy as np
import scipy.sparse
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
from scipy.sparse import csc_matrix, save_npz, load_npz
from metrics import  eva_metrics

# 设置随机数生成器的种子
np.random.seed(40)
random.seed(42)

"""
dataset_1: 节点数n=100，f=50, k=4, 噪音数据的属性值是0 or 1的，先加入5个噪声，再kmeans；
dataset_2: 节点数n=100，f=50, k=4，噪音数据的属性值是0 or 1的，先kmeans，再加入5个噪声，噪声变为1-2；
dataset_3: 节点数n=100，f=50, k=4,噪音数据的属性值是0 or 1的，先kmeans，再加入5个噪声，噪声变为10-20；
dataset_4: 节点数n=100，f=50, k=4,噪音数据的属性值是0 or 1的，先kmeans，再加入20个噪声，噪声变为10-20；
"""

#（1）先生成噪音数据，再kmeans聚类的话，可以使得后期基于邻接矩阵和属性矩阵的聚类结果都比较好；
def generate_AG(num_nodes, vector_length, n_clu, probability_of_att, probability_of_att_inv, probability_of_adj_intra, probability_of_adj_inter):
    # 设置随机数生成器的种子
    random.seed(42)
    np.random.seed(40)

    # 创建一个空的图
    G = nx.Graph()

    my_list = list(range(num_nodes))
    chunk_size = int(num_nodes / n_clu)
    # 将列表等份划分
    chunks = [my_list[i:i + chunk_size] for i in range(0, len(my_list), chunk_size)]
    node_name = -1

    for i in range(len(chunks)):
        att_change_max = (i + 1) * int(vector_length / n_clu)
        att_change_min = att_change_max - int(vector_length / n_clu)
        for j in range(len(chunks[i])):
            # 生成属性向量，前5个属性值设置为1，其余设置为0
            # attribute_vector = [
            #     1 if att_change_min <= k and k < att_change_max and random.random() < probability_of_att else 0 for k in
            #     range(vector_length)]
            attribute_vector = [0]*vector_length
            for k in range(vector_length):
                if att_change_min <= k and k < att_change_max:
                    if random.random() < probability_of_att:
                        attribute_vector[k] = 1
                else:
                    if random.random() < probability_of_att_inv:
                        attribute_vector[k] = 1


            # 添加节点，并将属性向量和标签添加为节点的属性
            node_name += 1
            G.add_node(node_name, attribute=attribute_vector, label=i)


    # 根据标签添加边
    for i in G.nodes():
        for j in G.nodes():
            if i != j:  # 不连接节点自身
                if G.nodes[i]["label"] == G.nodes[j]["label"]:
                    # 如果标签相同，添加边的概率为probability_of_adj_intra
                    if random.random() < probability_of_adj_intra:
                        G.add_edge(i, j)
                else:
                    # 如果标签不同，添加边的概率为probability_of_adj_inter
                    if random.random() < probability_of_adj_inter:
                        G.add_edge(i, j)

    # 获取整个图的属性矩阵、邻接矩阵和标签信息
    adj_matrix = nx.to_numpy_matrix(G)  # 邻接矩阵
    # 将邻接矩阵转换为 CSC 格式(稀疏格式)
    adj_matrix_csc = csc_matrix(adj_matrix)
    attr_matrix = np.array([G.nodes[i]['attribute'] for i in range(num_nodes)])  # 属性矩阵
    labels = np.array([G.nodes[i]['label'] for i in range(num_nodes)])  # 标签信息

    return adj_matrix_csc, attr_matrix, labels, G

#随机对xx%行的数据进行更改
def add_noise(matrix, noise_rate):
    # 计算矩阵的行数
    num_rows = matrix.shape[0]

    # 计算要反转的行数
    num_to_flip = int(noise_rate * num_rows)
    #print(num_to_flip)

    # 随机选择要反转的行
    rows_to_flip = np.random.choice(num_rows, num_to_flip, replace=False)
    #print(rows_to_flip)

    # 遍历选定的行，将0变为1，1变为0
    for row_idx in rows_to_flip:
        matrix[row_idx] = 1 - matrix[row_idx]
    return matrix

#随机更改整个矩阵中xx%的数据
def add_noise_1(matrix, noise_rate):

    matrix_size = matrix.shape[0]*matrix.shape[1]

    # 计算要反转的位置数，假设要反转20%的位置
    num_positions_to_flip = int(noise_rate * matrix_size)

    # 随机选择要反转的位置
    positions_to_flip = np.random.choice(matrix_size, num_positions_to_flip, replace=False)

    # 遍历选定的位置，将0变为1，1变为0
    for position in positions_to_flip:
        row = position // matrix.shape[0]
        col = position % matrix.shape[1]
        matrix[row, col] = 1 - matrix[row, col]
    return matrix

#随机对xx%行中xx%的数据进行更改
def add_noise_2(matrix, noise_rate):
    # 计算矩阵的行数
    num_rows = matrix.shape[0]

    # 计算要反转的行数
    num_to_flip = int(noise_rate * num_rows)
    #print(num_to_flip)

    # 随机选择要反转的行
    rows_to_flip = np.random.choice(num_rows, num_to_flip, replace=False)
    #print(rows_to_flip)



    # 遍历选定的行，再从每行中随机选取xx%的数据, 将0变为1，1变为0
    for row_idx in rows_to_flip:
        # 计算要反转的数据数量，假设要反转20%的数据
        num_data_to_flip = int(noise_rate * matrix.shape[1])
        # 随机选择要反转的数据的索引
        indices_to_flip = np.random.choice(matrix.shape[1], num_data_to_flip, replace=False)
        for idx in indices_to_flip:
            matrix[row_idx, idx] = 1 - matrix[row_idx, idx]
    return matrix


def save_AG(adjacency_csc, attributes, labels, adj_path, att_path, lab_path):
    # 保存 CSC 格式的矩阵为.npz文件
    save_npz(adj_path, adjacency_csc)

    # 保存属性特征和邻接矩阵为 NumPy 数组
    np.save(att_path, attributes)

    # 保存标签数组为 NumPy 数组
    np.save(lab_path, labels)
    print("Save over!")

def read_AG(adj_path, att_path, lab_path):
    adj = scipy.sparse.load_npz(adj_path)
    fea = np.load(att_path)
    labels = np.load(lab_path)
    return adj, fea, labels

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
    #ori_AG
    # adj_path = r"New_Version_1\ori_AG\adjacency_matrix.npz"
    # att_path = r"New_Version_1\ori_AG\attributes.npy"
    # lab_path = r"New_Version_1\ori_AG\labels.npy"

    #noise_AG_att
    # adj_path = r"New_Version_2\noise_att\80%\adjacency_matrix.npz"
    # att_path = r"New_Version_2\noise_att\80%\attributes.npy"
    # lab_path = r'New_Version_2\noise_att\80%\labels.npy'

    # noise_AG_adj
    adj_path = r"New_Version_2\noise_adj\50%\adjacency_matrix.npz"
    att_path = r"New_Version_2\noise_adj\50%\attributes.npy"
    lab_path = r'New_Version_2\noise_adj\50%\labels.npy'


    #生成
    # 添加节点和属性向量
    num_nodes = 1000  # 假设有 10 个节点
    vector_length = 200  # 属性向量的长度
    n_clu = 4
    probability_of_att = 0.20  # 同一集群中，属性值为1的概率(控制属性信息的稀疏性)
    probability_of_att_inv = 0.05  # 其余集群中，属性值为1的概率(控制属性信息的稀疏性)
    probability_of_adj_intra = 0.065  # 具有相同标签节点之间有边的概率(控制网络结构的稀疏性)  #0.12  #0.065（生成90%属性噪音时的设置）
    probability_of_adj_inter = 0.02  # 具有不同标签节点之间有边的概率(控制网络结构的稀疏性)  #0.02
    adjacency_csc, attributes, labels, AG = generate_AG(num_nodes, vector_length, n_clu, probability_of_att, probability_of_att_inv, probability_of_adj_intra, probability_of_adj_inter)
    print(len(AG.nodes))
    print(len(AG.edges))
    #print(np.sum(adjacency_csc.todense()))
    spa_att = np.sum(attributes)/(attributes.shape[0]*attributes.shape[1])
    spa_adj = np.sum(adjacency_csc.todense())/(adjacency_csc.todense().shape[0]*adjacency_csc.todense().shape[1])
    print("属性稀疏性：", spa_att)
    print("结构稀疏性：", spa_adj)
    plot_G_color(AG, labels)
    #print(attributes)

    kmeans = KMeans(n_clusters=4, random_state=42)
    pred = kmeans.fit_predict(adjacency_csc)
    #print(pred)
    res = eva_metrics(labels, pred)
    print(res)

    #attributes = add_noise(attributes, 0.2)
    #attributes = add_noise_2(attributes, 0.8)
    adjacency_noise = add_noise_2(adjacency_csc.todense(), 0.5)
    adjacency_csc = csc_matrix(adjacency_noise)

    kmeans = KMeans(n_clusters=4, random_state=42)
    pred = kmeans.fit_predict(attributes)
    # print(pred)
    res = eva_metrics(labels, pred)
    print(res)

    #存储
    #save_AG(adjacency_csc, attributes, labels, adj_path, att_path, lab_path)

    #读取
    #adj, att, lab = read_AG(adj_path, att_path, lab_path)
    #print(adj)
















