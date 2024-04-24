import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
from scipy import sparse
import scipy.sparse
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity



def prepare_graph_data(adj, add_self_loops=True, normalized=True):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    if add_self_loops:
        adj = adj + sp.eye(num_nodes)  # self-loop
    # data =  adj.tocoo().data
    #adj[adj > 0.0] = 1.0

    degree = np.squeeze(np.asarray(adj.sum(axis=1)))
    if normalized:
        with np.errstate(divide='ignore'):
            inverse_sqrt_degree = 1. / np.sqrt(degree)
        inverse_sqrt_degree[inverse_sqrt_degree == np.inf] = 0
        inverse_sqrt_degree = scipy.sparse.diags(inverse_sqrt_degree)
        adj = inverse_sqrt_degree @ adj @ inverse_sqrt_degree

    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    #return (indices, adj.data, adj.shape), adj.row, adj.col
    return (indices, adj.data, adj.shape)

def prepare_features_data(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    features = r_mat_inv.dot(features)
    #return features.todense(), sparse_to_tuple(features)
    return features

def adj_to_bias(adj):
    adj = adj.todense()
    adj = -1e9*(1-adj)
    adj = scipy.sparse.csc_matrix(adj)  # 稠密转稀疏

    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()  # 也是一种稀疏矩阵的存储方式
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape), adj.row, adj.col

def prepare_sparse_features(features):
    if not sp.isspmatrix_coo(features):
        features = sparse.csc_matrix(features).tocoo()
        features = features.astype(np.float32)
    indices = np.vstack((features.row, features.col)).transpose()
    return (indices, features.data, features.shape)

def conver_sparse_tf2np(input):
    # Convert Tensorflow sparse matrix to Numpy sparse matrix
    return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])), shape=(input[layer][2][0], input[layer][2][1])) for layer in input]

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("datasets/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("datasets/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    nx_graph = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(nx_graph)
    edges = nx_graph.edges()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    #
    # print(adj.shape)
    # print(features.shape)

    return sp.coo_matrix(adj), features.todense(), labels

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def get_processed_data(adjacency, features, method="cosine"):

    method = "cosine"
    if method == "cosine":
        #print(method)
        aff = cosine_similarity(features, features)
    if method == "pearson":
        print(method)
        aff = np.corrcoef(features, features)[0:features.shape[0],0:features.shape[0]]
    if method =="jaccard":
        print(method)
        aff = np.zeros((features.shape[0],features.shape[0]))
        for i in range(features.shape[0]):
            for j in range(features.shape[0]):
                tmp1 = (features[i]==features[j])
                tmp2 = np.matrix.tolist(tmp1)
                tmp3 = tmp2.count(True)
                aff[i][j]=tmp3/features.shape[1]

    aff = (np.array(adjacency.todense()) + np.eye(adjacency.shape[0])) * aff  #element-wise
    #aff = np.array(adjacency.todense()) * aff


    aff = scipy.sparse.csc_matrix(aff)  #dense to sparse
    #aff = convert_scipy_sparse_to_sparse_tensor(aff)
    #aff = tf.sparse.reorder(aff)

    if not sp.isspmatrix_coo(aff):
        aff = aff.tocoo()
    aff = aff.astype(np.float32)
    indices = np.vstack((aff.col, aff.row)).transpose()
    return (indices, aff.data, aff.shape)

def convert_scipy_sparse_to_sparse_tensor(matrix):
    matrix = matrix.tocoo()
    return tf.sparse.SparseTensor(np.vstack([matrix.row, matrix.col]).T,
                                   matrix.data.astype(np.float32),
                                   matrix.shape)




def save_result(filename, results):
    results = map(str, results)
    content = " ".join(results)
    with open(filename, "a+") as f:
        f.write('\n')
        f.writelines(content)
    f.close()

import scipy.io as sio
def load_mat_data(str):
    data = sio.loadmat('datasets/{}/ACM3025.mat'.format(str))
    if(str == 'large_cora'):
        X = data['X']
        A = data['G']
        gnd = data['labels']
        gnd = gnd[0, :]
    else:
        X = data['feature']
        A = data['PAP']
        B = data['PLP']
        av=[]
        av.append(A)
        av.append(B)
        gnd = data['label']
        #gnd = gnd.T
        #gnd = np.argmax(gnd, axis=0)

    adj = av[0].astype(np.float32)
    adj = scipy.sparse.csc_matrix(adj)
    X=X.astype(np.float32)
    gnd=gnd.astype(np.float32)
    return adj, X, gnd

def load_npz_data(filename):
    with np.load(open(filename, 'rb'), allow_pickle=True) as loader:
        loader = dict(loader)
        #indices表示所在列
        adjacency = scipy.sparse.csr_matrix((loader['adj_data'],      loader['adj_indices'],
                                             loader['adj_indptr']),   shape=loader['adj_shape'])
        features = scipy.sparse.csr_matrix(( loader['feature_data'],   loader['feature_indices'],
                                             loader['feature_indptr']),shape=loader['feature_shape'])
        label_indices = loader['label_indices']
        labels = loader['labels']
    assert adjacency.shape[0] == features.shape[0], 'Adjacency and feature size must be equal!'
    assert labels.shape[0] == label_indices.shape[0], 'Labels and label_indices size must be equal!'
    return adjacency, features, labels, label_indices

def load_txt_data(features_path, graph_path):
    data = np.loadtxt(features_path, dtype=float)
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(graph_path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #adj = adj + sp.eye(adj.shape[0])
    #adj = normalize(adj)

    return adj

def load_txt_data1(n, graph_path):
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(graph_path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #adj = adj + sp.eye(adj.shape[0])
    #adj = normalize(adj)

    return adj


def load_npz(filename):
    with np.load(open(filename, 'rb'), allow_pickle=True) as loader:
        loader = dict(loader)
        #indices表示所在列
        adjacency = scipy.sparse.csr_matrix((loader['adj_data'],      loader['adj_indices'],
                                             loader['adj_indptr']),   shape=loader['adj_shape'])
        features = scipy.sparse.csr_matrix(( loader['feature_data'],   loader['feature_indices'],
                                             loader['feature_indptr']),shape=loader['feature_shape'])
        label_indices = loader['label_indices']
        labels = loader['labels']
    assert adjacency.shape[0] == features.shape[0], 'Adjacency and feature size must be equal!'
    assert labels.shape[0] == label_indices.shape[0], 'Labels and label_indices size must be equal!'
    return adjacency, features, labels, label_indices

def read_AG(adj_path, att_path, lab_path):
    adj = scipy.sparse.load_npz(adj_path)
    fea = np.load(att_path)
    labels = np.load(lab_path)
    return adj, fea, labels




