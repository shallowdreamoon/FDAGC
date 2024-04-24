import networkx as nx
import numpy as np
import scipy.sparse
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
from scipy.sparse import csc_matrix, save_npz, load_npz

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
def generate_AG(n, f, k, n_noise_att, n_noise_adj):
    # 设置随机数生成器的种子
    random.seed(42)
    np.random.seed(40)

    # 生成图的结构（这里使用Watts-Strogatz小世界网络）
    G = nx.watts_strogatz_graph(n=n, k=4, p=0.2, seed=42)

    # 生成属性特征
    num_nodes = len(G.nodes)
    num_attributes = f
    # mean = [0]*f
    # var = np.eye(num_attributes)
    # attributes = np.random.multivariate_normal(mean, var, num_nodes)

    #生成属性矩阵：保证属性矩阵中只包含0和1
    # 生成随机属性特征并为节点添加属性
    for node in G.nodes():
        attribute_vector = [random.choice([0, 1]) for _ in range(f)]  # 随机生成长度为f的属性向量
        G.nodes[node]['attribute_vector'] = attribute_vector
    # 提取属性特征并组合成属性矩阵
    attributes = np.array([G.nodes[node]['attribute_vector'] for node in G.nodes()])

    # 使用K均值聚类算法为节点分配标签
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(attributes)

    # 添加噪音数据：将一些节点的标签更改为单独的标签
    # 同时，更改噪音节点的属性特征
    s_list = list(range(0, num_nodes))
    random_selection = random.sample(s_list, k=n_noise_att)  #选取n_noise个噪音，将其作为单独的一个簇
    for i in random_selection:
        #labels[i] = k  # k+1 是一个新的标签，表示噪音簇
        #mean_1 = [5]*f  #5表示每个属性均值，为了与上面的0有所区别
        #attributes[i] = np.random.multivariate_normal(mean_1, np.eye(num_attributes))
        attributes[i] = [random.choice([10, 20]) for _ in range(f)]

    # # 使用K均值聚类算法为节点分配标签
    # kmeans = KMeans(n_clusters=k, random_state=42)
    # labels = kmeans.fit_predict(attributes)

    # 调整图的结构，确保连接紧密的节点属于同一集群
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if labels[i] == labels[j]:
                G.add_edge(i, j)

    # 添加噪音边（随机连接节点）
    num_noise_edges = n_noise_adj*n_noise_adj  # 你可以根据需要设置噪音边的数量
    for _ in range(num_noise_edges):
        node1 = random.choice(list(G.nodes))
        node2 = random.choice(list(G.nodes))
        G.add_edge(node1, node2)

    # 生成邻接矩阵
    adjacency_matrix = nx.to_numpy_matrix(G)
    # 将邻接矩阵转换为 CSC 格式(稀疏格式)
    adjacency_csc = csc_matrix(adjacency_matrix)

    return adjacency_csc, attributes, labels, G

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
    n = 1000 #节点个数
    f = 500  #特征维度
    k = 6   #正常簇的个数
    n_noise_att = 0 #噪音簇中的节点个数
    n_noise_adj = 0 #噪音簇中的边个数
    adj_path = r"noise_adj\80%\adjacency_matrix.npz"
    att_path = r"noise_adj\80%\attributes.npy"
    lab_path = r'noise_adj\80%\labels.npy'

    # adj_path = r"dataset_4\adjacency_matrix.npz"
    # att_path = r"dataset_4\attributes.npy"
    # lab_path = r'dataset_4\labels.npy'

    #生成
    adjacency_csc, attributes, labels = generate_AG(n, f, k, n_noise_att, n_noise_adj)[0:-1]
    #print(labels)
    AG = generate_AG(n, f, k, n_noise_att, n_noise_adj)[-1]
    print(len(AG.nodes))
    print(len(AG.edges))
    plot_G_color(AG, labels)
    #plot_G_no_color(AG)


    #存储
    #save_AG(adjacency_csc, attributes, labels, adj_path, att_path, lab_path)

    #读取
    #adj, att, lab = read_AG(adj_path, att_path, lab_path)
    #print(adj)
















