import random
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import time
from fdagc import FDAGC
from utils import prepare_graph_data, get_processed_data, save_result, load_data, read_AG
import numpy as np
from easydict import EasyDict
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

config={
    'dataset': 'synthetic',
    'learning_rate': 0.001,          #0.001
    'hidden_dims': [64,32],         #[512, 512]
    'n_samples': 100,
    'n_clusters': 5,
    'n_epochs': 200,
    'lamda_1': 100,                        #100   #structure_res
    'lamda_2': 100,                        #100   #features_res
    'lamda_3': 1000,                     #1000  #self-supervised
    'lamda_4': 0.005,                        #0.005     #clustering_loss
    'seed': 0,
    'gradient_clipping': 5.0,
    'dropout': 0.,
}

FLAGS = EasyDict(config)

def task(i, subset):
    print("{}: {} is training...".format(i, subset))
    start1=time.time()
    #load data
    adj_path = r"E:\study\pycharm\Final_Coding\FDAGC\synthetic_dataset\New_Version_2\noise_adj\50%\adjacency_matrix.npz"
    att_path = r"E:\study\pycharm\Final_Coding\FDAGC\synthetic_dataset\New_Version_2\noise_adj\50%\attributes.npy"
    lab_path = r'E:\study\pycharm\Final_Coding\FDAGC\synthetic_dataset\New_Version_2\noise_adj\50%\labels.npy'

    #G, X, Y = load_data(FLAGS.dataset)
    G, X, Y = read_AG(adj_path, att_path, lab_path)
    Label = np.array(Y)
    #Label = np.array([np.argmax(l) for l in Y])
    feature_dim = X.shape[1]
    FLAGS.hidden_dims = [feature_dim] + FLAGS.hidden_dims
    FLAGS.n_samples = len(Y)
    FLAGS.n_clusters = np.max(Y)+1
    Label_indices = np.arange(G.shape[0])

    #process data
    G_tf = prepare_graph_data(G, add_self_loops=True, normalized=True)
    adj = prepare_graph_data(G, add_self_loops=False, normalized=False)
    aff = get_processed_data(G, X)
    print("get_data:", time.time()-start1)

    #create model
    start2 = time.time()
    model = FDAGC(FLAGS)
    print("create_model: ", time.time()-start2)

    start3 = time.time()
    #init model
    _ = model.assign(G_tf, X)
    print("init_model: ", time.time()-start3)

    start4 = time.time()
    #train model
    eva_res = model(G_tf, X, Label, Label_indices, adj, aff)
    eva_res.append(FLAGS.seed)
    save_result("result\\{}_evaluate.txt".format(FLAGS.dataset), eva_res)

    tf.compat.v1.reset_default_graph()
    print("train_model: ", time.time()-start4)
    print("{}: {} is Finished!".format(i, subset.split("\\")[-1]))

if __name__ == "__main__":
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #seed = 3384113
    #seed = 456483505
    seed = FLAGS.seed
    FLAGS.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.random.get_seed(seed)
    #print("This random sedd is {}.".format(seed))

    start = time.time()
    #singal_subset processing
    subset = "synthetic"
    task(1, subset)

    print("All subsets are finished!")

    print("Total time cost: {}".format(time.time()-start))
