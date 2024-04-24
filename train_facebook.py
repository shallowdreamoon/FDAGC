import random
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import time
from fdagc import FDAGC
from utils import prepare_graph_data, get_processed_data, load_npz, save_result
import numpy as np
from easydict import EasyDict
import warnings
warnings.filterwarnings("ignore")
import os
from multiprocessing import Pool
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


config={
    'dataset': 'facebook',
    'learning_rate': 0.001,          #0.001
    'hidden_dims': [512,128],         #[512, 128]
    'n_samples': 2708,
    'n_clusters': 7,
    'n_epochs': 200,
    'lamda_1': 100,                        #100   #structure_res
    'lamda_2': 100,                        #100   #features_res
    'lamda_3': 10000,                     #10000  #self-supervised
    'lamda_4': 0.005,                        #0.005     #clustering_loss
    'seed': 0,
    'gradient_clipping': 5.0,
    'dropout': 0.,
}

FLAGS = EasyDict(config)

def task(i, subset):
    print("{}: {} is training...".format(i, subset.split("\\")[-1]))
    start_temp = time.time()
    #load data
    G, X, Y, Y_indices = load_npz(subset)
    X = X.todense()
    Label = Y
    Label_indices = Y_indices
    feature_dim = X.shape[1]
    FLAGS.hidden_dims = [feature_dim] + FLAGS.hidden_dims
    FLAGS.n_samples = G.shape[0]
    FLAGS.n_clusters = np.max(Label)+1

    #process data
    G_tf = prepare_graph_data(G, add_self_loops=True, normalized=True)
    adj = prepare_graph_data(G, add_self_loops=False, normalized=False)
    aff = get_processed_data(G, X)

    #create model
    model = FDAGC(FLAGS)
    #init model
    _ = model.assign(G_tf, X)
    #train model
    eva_res = model(G_tf, X, Label, Label_indices, adj, aff)
    tmp_cost = [time.time()-start_temp]
    eva_res.append(subset.split("\\")[-1].strip(".npz"))
    save_result("result\\{}_evaluate0.txt".format(FLAGS.dataset), eva_res)
    save_result("result\\{}_time.txt".format(FLAGS.dataset), tmp_cost)

    FLAGS.hidden_dims = [512, 128]  #reset dimension
    tf.compat.v1.reset_default_graph()
    print("{}: {} is Finished!".format(i, subset.split("\\")[-1]))

if __name__ == "__main__":
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    seed = FLAGS.seed
    #seed = 456483505
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.random.get_seed(seed)
    #print("This random sedd is {}.".format(seed))

    #all_subsets processing
    dataset_path = r"datasets\Facebook_npz"
    subsets = os.listdir(dataset_path)
    po = Pool(5)
    for i in range(len(subsets)):
        subset = os.path.join(dataset_path,subsets[i])
        po.apply_async(task,(i, subset))
    po.close()
    po.join()
    print("All subsets are finished!")
