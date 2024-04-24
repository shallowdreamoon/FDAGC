import tensorflow as tf
import tensorflow.compat.v1 as tf
from sklearn.cluster import KMeans

class GCN_Encoder():
    def __init__(self, hidden_dims, n_samples, n_clusters, lamda_1 = 100, lamda_2 = 100, lamda_3 = 1000, lamda_4 = 1):
        self.n_layers1 = len(hidden_dims) - 1
        self.W, self.v = self.define_weights(hidden_dims)
        self.C = {}
        self.params = {"n_clusters": n_clusters, "encoder_dims": [hidden_dims[-1]], "alpha": 1.0, "n_samples": n_samples}
        self.mu = tf.Variable(tf.zeros(shape=(self.params["n_clusters"], hidden_dims[-1])), name="mu")
        self.kmeans = KMeans(n_clusters=self.params["n_clusters"], n_init=10)
        self.n_cluster = self.params["n_clusters"]
        self.input_batch_size = self.params["n_samples"]
        self.alpha = self.params['alpha']
        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2
        self.lamda_3 = lamda_3
        self.lamda_4 = lamda_4

    @tf.autograph.experimental.do_not_convert
    def __call__(self, A, X, adj, aff, P):
        forward = []
        backward = []

        # Encoder
        H = X
        for layer in range(self.n_layers1):
            H = self.__encoder(A, H, layer)
            #print(H)
            forward.append(H)
        # Final node representations
        self.H = H

        #Decoder
        for layer in range(self.n_layers1 - 1, -1, -1):
            H = self.__decoder(H, layer)
            backward.append(H)
        X_ = H

        with tf.name_scope("distribution"):
            self.Q = self._soft_assignment(1.0 * self.H, self.mu)
            self.P = P

        self.membership = tf.nn.softmax(self.Q)
        self.pred = tf.argmax(self.membership, axis=1)

        rec_loss = self.reconstructed_loss(adj, X, X_)
        self_loss = self.self_Mloss()
        fc_loss = self.fuzzy_closs(adj, aff)
        # Total loss
        self.loss = rec_loss+self_loss+fc_loss
        return self.loss, self.H, self.pred, self.membership

    def fuzzy_closs(self, adj, aff):
        # structure_loss
        graph_left = tf.transpose(tf.sparse.sparse_dense_matmul(adj, self.membership))
        graph_right = tf.matmul(graph_left, self.membership)
        structure_loss = -self.lamda_4 * 0.5 * tf.linalg.trace(graph_right)

        # content_loss
        content_left = tf.transpose(tf.sparse.sparse_dense_matmul(aff, self.membership))
        content_right = tf.matmul(content_left, self.membership)
        content_loss = -self.lamda_4 * 0.5 * tf.linalg.trace(content_right)

        # irrelevance
        S_irr = self.get_S(self.membership, adj)
        irr = tf.matmul(tf.transpose(self.membership), S_irr)
        irr_loss = self.lamda_4 * tf.linalg.trace(irr)

        # regularization
        cluster_sizes = tf.math.reduce_sum(self.membership, axis=0)  # Size [k].
        regu_loss = self.lamda_4 * 0 * tf.square(tf.norm(cluster_sizes))  # 0.005  #0.01
        # regu_loss = self.lamda_4*0.005*tf.square(tf.norm(assignment))

        # fuzzy-based clustering loss
        fc_loss = structure_loss + content_loss + irr_loss + regu_loss
        return fc_loss

    def reconstructed_loss(self, adj, X, X_):
        # The reconstruction loss
        structure_rec = tf.sigmoid(tf.matmul(self.H, self.H, transpose_b=True))
        structure_rec_loss = self.lamda_1 * tf.reduce_mean(
            (tf.sparse.to_dense(adj) - tf.eye(self.params["n_samples"]) - structure_rec) ** 2)
        features_rec_loss = self.lamda_2 * tf.reduce_mean((X - X_) ** 2)

        rec_loss = structure_rec_loss + features_rec_loss
        return rec_loss

    def self_Mloss(self):
        # self_supervised_loss
        self_Monitoring_loss = self.lamda_3 * self._kl_divergence(self.P, self.Q)
        return self_Monitoring_loss


    #add attention mechanism
    def __encoder(self, A, H, layer):
        H = tf.matmul(H, self.W[layer])
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        output = tf.sparse_tensor_dense_matmul(self.C[layer], H)
        return output

    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        output = tf.sparse_tensor_dense_matmul(self.C[layer], H)
        return output

    #Do not use attention mechanism
    def __encoder1(self, A, H, layer):
        H = tf.matmul(H, self.W[layer])
        output = tf.sparse_tensor_dense_matmul(A, H)
        return (output)

    def __decoder1(self, H, A, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        output = tf.sparse_tensor_dense_matmul(A, H)
        return (output)

    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers1):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att = {}
        for i in range(self.n_layers1):
            v = {}
            v[0] = tf.get_variable("v%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v[1] = tf.get_variable("v%s_1" % i, shape=(hidden_dims[i + 1], 1))
            Ws_att[i] = v

        return W, Ws_att

    def graph_attention_layer(self, A, M, v, layer):
        with tf.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)
            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                      values=tf.nn.sigmoid(logits.values),
                                                      dense_shape=logits.dense_shape)
            #unnormalized_attentions = tf.nn.leaky_relu(unnormalized_attentions)
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions



    def get_assign_cluster_centers_op(self, features):
        # init mu
        #print("Kmeans train start.")
        kmeans = self.kmeans.fit(features)
        #print("Kmeans train end.")
        return tf.assign(self.mu, kmeans.cluster_centers_)

    def _soft_assignment(self, embeddings, cluster_centers):
        def _pairwise_euclidean_distance(a, b):
            p1 = tf.matmul(
                tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
                tf.ones(shape=(1, self.n_cluster))
            )
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
                tf.ones(shape=(self.input_batch_size, 1)),
                transpose_b=True
            ))
            #res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True))  #之所以出现nan的原因：tf.add(p1,p2)-2*tf.matmul(a, b, transpose_b=True)的值为负值
            res = tf.sqrt(tf.abs(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True)))
            return res

        dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
        q = 1.0 / (1.0 + dist ** 2 / self.alpha) ** ((self.alpha + 1.0) / 2.0)
        q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def target_distribution_tensor(self, q):
        p = q ** 2 / tf.reduce_sum(q, axis=0)
        p = p / tf.reduce_sum(p, axis=1,  keepdims=True)
        return p

    def _kl_divergence(self, target, pred):
        return tf.reduce_mean((target - pred) ** 2)

    def get_S(self,U, D):
        row = U.shape[0]
        col = U.shape[1]
        sumU = tf.reduce_sum(U, 1, keepdims=True)
        sumDU = tf.sparse.sparse_dense_matmul(D, sumU)
        sumUDK = tf.sparse.sparse_dense_matmul(D, U)
        sumDU = tf.tile(sumDU, [1, col])
        S = sumDU - sumUDK
        return S

