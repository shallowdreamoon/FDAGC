import tensorflow as tf
import tensorflow.compat.v1 as tf
from gcn_encoder import GCN_Encoder
from metrics import eva_metrics
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

# Evaluation Metric


class FDAGC():
    def __init__(self, args):
        self.args = args
        self.build_placeholders()
        self.gate = GCN_Encoder(args.hidden_dims, args.n_samples, args.n_clusters, args.lamda_1, args.lamda_2, args.lamda_3, args.lamda_4)
        self.loss, self.H, self.pred, self.membership = self.gate(self.A, self.X, self.adj, self.aff, self.P)
        self.optimize(self.loss)
        self.build_session()

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.adj = tf.sparse_placeholder(dtype=tf.float32)
        self.aff = tf.sparse_placeholder(dtype=tf.float32)
        self.P = tf.placeholder(tf.float32, shape=(None, self.args.n_clusters))

    def build_session(self, gpu=False):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if not gpu:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def __call__(self, A, X, true, indices, adj, aff):
        best_res = [0,0,0,0]
        con_epoch = 0
        for epoch in range(self.args.n_epochs):
            if con_epoch > 5:
                break
            else:
                results = self.run_epoch(epoch, A, X, true, indices, adj, aff)
                con_epoch += 1
                if results[0]==1:
                    best_res = results
                    break
                if results[0]>best_res[0]:
                    best_res = results
                    con_epoch = 0
        return best_res

    def run_epoch(self, epoch, A, X, true, indices, adj, aff):
        Q = self.session.run([self.gate.Q], feed_dict={self.A: A, self.X: X})[0]
        P = self.gate.target_distribution(Q)
        loss, pred, _, membership = self.session.run([self.loss, self.pred, self.train_op, self.membership], feed_dict={self.A: A, self.X: X, self.adj: adj, self.aff: aff, self.P: P})
        #print("epoch: {} loss: {}".format(epoch, loss))
        if epoch ==199 or epoch==99:
            #save fuzzy membership matrix
            #assignment = p.numpy()
            np.savetxt(r"result\membership.txt", membership, fmt="%.4f")
            print(pred)
        #保存评估结果
        results = eva_metrics(true, pred[indices])
        print(results)
        return results

    def infer(self, A, X):
        H = self.session.run([self.H], feed_dict={self.A: A, self.X: X})
        return H

    def assign(self, A, X):
        representations = self.infer(A, X)
        assign_mu_op = self.gate.get_assign_cluster_centers_op(1.0*representations[0])
        _ = self.session.run(assign_mu_op)