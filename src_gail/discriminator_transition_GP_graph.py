import scipy.sparse as sp
import tensorflow as tf
import numpy as np

from utils.misc_util import RunningMeanStd
from utils import tf_util as U

from utils.gcn_layers import *
from utils.gcn_metrics import *
from utils.gcn_utils import *

from mpi_adam import MpiAdam
from utils.console_util import fmt_row, colorize
from utils.mujoco_dset import Dset_transition
from box import Box
from dp_env_biped_PID_unnorm_variable_speed import DPEnv

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

class GraphDiscriminator(object):
    def __init__(self, ob_length, hidden_size, adj, total_DOF = 9, entcoeff=0.001, lr_rate=1e-3, scope="discriminator"):
        self.entcoeff = entcoeff
        self.scope = scope
        self.ob_length = ob_length
        self.node_num = adj.shape[0] if isinstance(adj, np.ndarray) else tf.shape(adj)[0]
        self.obs_shape = (total_DOF, ob_length,)
        print('Observation Shape: ', self.obs_shape)
        self.hidden_size = hidden_size
        self.support = self._preprocess_adj(adj)
        self.build_ph()
        # Build grpah
        generator_logits_list, norm_generator_obs = self.build_graph(self.generator_obs_ph, reuse=False)
        expert_logits_list, norm_expert_obs = self.build_graph(self.expert_obs_ph, reuse=True)
        # Build accuracy
        # generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits_list[-1]) < 0.5))
        # expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits_list[-1]) > 0.5))

        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits_list[-1]) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits_list[-1])> 0.5))
        node_losses = list()
        self.reward_list = list()
        self.g_acc_list = list()
        self.e_acc_list = list()
        for generator_logits, expert_logits in zip(generator_logits_list, expert_logits_list):            
            node_losses.append(self.build_node_loss(generator_logits, expert_logits, norm_expert_obs))
            self.reward_list.append(tf.reduce_mean(-tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8), axis=0))
            self.g_acc_list.append(tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5), axis=0))
            self.e_acc_list.append(tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5), axis=0))

        node_losses = tf.add_n(node_losses)

        generator_loss, expert_loss, entropy, entropy_loss, regular_loss, gradient_penalty, reward = [tf.squeeze(x) for x in tf.split(node_losses,node_losses.shape[0])]
        all_weights = [weight for weight in self.get_trainable_variables() if "bias" not in weight.name]
        weight_norm = 1e-4 * tf.reduce_sum(tf.stack([tf.nn.l2_loss(weight) for weight in all_weights]))
        # Loss + Accuracy terms
        self.losses = [tf.reduce_mean(generator_logits), tf.reduce_mean(expert_logits), generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc, regular_loss, weight_norm, gradient_penalty]
        self.loss_name = ["g_logits", "e_logits", "generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc", "regular_loss", "weight_norm", "gradient_penalty"]
        self.total_loss = generator_loss + expert_loss + entropy_loss + regular_loss + weight_norm + gradient_penalty
        # Build Reward for policy
        generator_logits_concat = tf.concat(generator_logits_list, axis=0)
        self.reward_op = tf.reduce_mean(-tf.log(1-tf.nn.sigmoid(generator_logits_concat)+1e-8))
        var_list = self.get_trainable_variables()
        summary_list = self.build_summary()
        self.lossandgrad = U.function([self.generator_obs_ph, self.expert_obs_ph],
                                      self.losses + [U.flatgrad(self.total_loss, var_list)])
        self.summary = U.function([self.generator_obs_ph, self.expert_obs_ph], summary_list)
    
    def build_summary(self):
        summary_list = list()
        for layer_id, layer_results in enumerate(zip(self.reward_list, self.g_acc_list, self.e_acc_list)):
            reward, g_acc, e_acc = layer_results
            for node_id in range(reward.shape[0]):
                summary_list.append(tf.summary.scalar("/layer_%d/reward_node_%d" %(layer_id, node_id), tf.squeeze(reward[node_id])))
                summary_list.append(tf.summary.scalar("/layer_%d/g_acc_node_%d" %(layer_id, node_id), tf.squeeze(g_acc[node_id])))
                summary_list.append(tf.summary.scalar("/layer_%d/e_acc_node_%d" %(layer_id, node_id), tf.squeeze(e_acc[node_id])))
        for loss_name, loss in zip(self.loss_name, self.losses):
            summary_list += [tf.summary.scalar(loss_name, loss)]
        summary_list += [tf.summary.scalar("reward_op", self.reward_op)]
        summary_list += [tf.summary.scalar("total_loss", self.total_loss)]

        return summary_list

    def build_node_loss(self, generator_logits, expert_logits, norm_expert_obs):
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        # generator_loss = tf.reduce_sum(tf.square(generator_logits)/2)

        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # expert_loss = tf.reduce_sum(tf.square(expert_logits-1))

        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -self.entcoeff*entropy
        regular_loss = tf.nn.l2_loss(logits)
        regular_loss = 1e-4 * tf.reduce_mean(regular_loss)
        gradient_penalty = 1e-3 * tf.nn.l2_loss(tf.gradients(tf.log(tf.nn.sigmoid(expert_logits)), norm_expert_obs))
        reward = tf.reduce_sum(-tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8))

        return tf.stack([generator_loss, expert_loss, entropy, entropy_loss, regular_loss, gradient_penalty, reward], axis=0)
        
    def build_ph(self):
        self.generator_obs_ph = tf.placeholder(tf.float32, (None, ) + self.obs_shape, name="observations_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None, ) + self.obs_shape, name="expert_observations_ph")

    def build_graph(self, obs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.obs_shape)
            obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std #（N,G,M)
            obs = obs_ph
            layers = list()
            activations = list()
            logits_list = list()
            
            root_DOF = 3
            all_keypoint_DOF = 6
            mlp_root = MLP(root_DOF*self.ob_length, self.hidden_size, 1, name="mlp_root")
            mlp_actutor = MLP(self.ob_length, self.hidden_size, all_keypoint_DOF, name="mlp_actutor")
            
            obs_root = tf.reshape(obs[:,:root_DOF,:],[-1,1,root_DOF*self.ob_length])
            obs_actutor = obs[:,root_DOF:,:]
            embedding_root = mlp_root(obs_root)
            embedding_actutor = mlp_actutor(obs_actutor)
            embedding = tf.concat([embedding_root, embedding_actutor], axis=1)
            layers.append(GraphConvolutionLayerBatch(self.hidden_size,self.hidden_size,self.support,name="graphconvolutionlayer_0"))
            layers.append(GraphConvolutionLayerBatch(self.hidden_size,self.hidden_size,self.support,name="graphconvolutionlayer_1"))
            
            activations.append(embedding)
            for layer in layers:
                hidden = layer(activations[-1])
                activations.append(hidden)
            for hidden in activations:
                logits_list.append(tf.contrib.layers.fully_connected(hidden, 1, activation_fn=tf.identity,reuse=tf.AUTO_REUSE,scope="disc"))
            # p1 = tf.contrib.layers.fully_connected(obs, 16, activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE,scope="disc0")
            # logits_list.append(tf.contrib.layers.fully_connected(p1, 1, activation_fn=tf.identity,reuse=tf.AUTO_REUSE,scope="disc"))
        return logits_list, obs

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs):
        sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = obs.reshape(1,2,9).transpose(0,2,1)
        feed_dict = {self.generator_obs_ph: obs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward

    def _preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        adj_normalized = self._normalize_adj(adj + np.eye(adj.shape[0]))
        return adj_normalized

    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


if __name__=="__main__":
    connections = [(6,5),(5,4),(4,0),(0,1),(1,2),(2,3)]
    adj = np.zeros((7,7))
    for con in connections:
        adj[con] = 1
        adj.T[con] = 1

    d_stepsize = 1e-2
    batch_size = 32
    reward_giver = GraphDiscriminator(2,16,adj)
    d_adam = MpiAdam(reward_giver.get_trainable_variables())
    
    C = Box.from_yaml(filename="config/gail_ppo_PID_unnorm_variable_speed_graph.yaml")
    env = DPEnv(C)
    expert_dataset = Dset_transition(transitions = env.sample_expert_traj())

    with U.make_session(num_cpu=1) as sess:
        writer = tf.summary.FileWriter("./graphs", sess.graph)
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        for i in range(100):
            transition_batch = np.random.randn(batch_size,9,2)
            transition_expert = expert_dataset.get_next_batch(batch_size) #(N,2*9)
            transition_expert = transition_expert.reshape([-1,2,9]).transpose(0,2,1)
            transition_batch = transition_batch.reshape([-1,2,9]).transpose(0,2,1)


            *newlosses, g = reward_giver.lossandgrad(transition_batch, transition_expert)
            summary_list = reward_giver.summary(transition_batch, transition_expert)
            for summary in summary_list:
                writer.add_summary(summary, i)

            d_adam.update(g, d_stepsize)
            print(fmt_row(13, reward_giver.loss_name))
            print(fmt_row(13, newlosses))
            # print(sess.run(reward_giver.get_trainable_variables()))


# # simple logistic regression
#     np.random.seed(101)
#     tf.set_random_seed(101)


#     # Genrating random linear data 
#     # There will be 50 data points ranging from 0 to 50 
#     x = np.linspace(0, 50, 50) 
#     y = np.linspace(0, 50, 50) 
    
#     # Adding noise to the random linear data 
#     x += np.random.uniform(-4, 4, 50) 
#     y += np.random.uniform(-4, 4, 50) 
    
#     n = len(x) # Number of data points 

#     X = tf.placeholder("float") 
#     Y = tf.placeholder("float") 


#     W = tf.Variable(np.random.randn(), name = "W") 
#     b = tf.Variable(np.random.randn(), name = "b") 

#     learning_rate = 0.01
#     training_epochs = 1000

#     # Hypothesis 
#     y_pred = tf.add(tf.multiply(X, W), b) 

#     # Mean Squared Error Cost Function 
#     cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n) 

#     # Gradient Descent Optimizer 
#     # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 
#     var_list = [W,b]
#     lossandgrad = U.function([X, Y],
#                             [cost] + [U.flatgrad(cost, var_list)])
#     d_adam = MpiAdam(var_list)

#     # Global Variables Initializer 
#     init = tf.global_variables_initializer() 

#     # Starting the Tensorflow Session 
#     with tf.Session() as sess: 
        
#         # Initializing the Variables 
#         sess.run(init) 
        
#         # Iterating through all the epochs 
#         for epoch in range(training_epochs): 
            
#             # Feeding each data point into the optimizer using Feed Dictionary 
#             for (_x, _y) in zip(x, y): 
#                 cost, g = lossandgrad(_x, _y)
#                 d_adam.update(g, learning_rate)
#             # Displaying the result after every 50 epochs 
#             if (epoch + 1) % 50 == 0: 
#                 # Calculating the cost a every epoch 
#                 # c = sess.run(cost, feed_dict = {X : x, Y : y}) 
#                 print("Epoch", (epoch + 1), ": cost =", cost, "W =", sess.run(W), "b =", sess.run(b)) 
        
#         # Storing necessary values to be used outside the Session 
#         training_cost = sess.run(cost, feed_dict ={X: x, Y: y}) 
#         weight = sess.run(W) 
#         bias = sess.run(b) 


    # # Settings
    # flags = tf.app.flags
    # FLAGS = flags.FLAGS
    # flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    # flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    # flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    # flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    # flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
    # flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    # flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    # flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
    # flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
    # # Create data
    # features = preprocess_features(sp.csr.csr_matrix(np.ones((6,1))))
    # adj = sp.csr.csr_matrix(np.random.randn(6,6))
    # support = [preprocess_adj(adj)]
    # support_sparse = preprocess_adj(adj).toarray()
    # support_dense = reward_giver._preprocess_adj(adj)


    # y_train = np.zeros((6,2))
    # y_train[:,0] = 1    
    # train_mask = np.array([True for _ in range(6)])
    # num_supports = 1

    # # Define placeholders
    # placeholders = {
    #     'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    #     'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    #     'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    #     'labels_mask': tf.placeholder(tf.int32),
    #     'dropout': tf.placeholder_with_default(0., shape=()),
    #     'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    # }
    # # Create model

    # gcn_model = GCN(placeholders, input_dim=1, logging=True)


    # # Construct feed dictionary
    # feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    # feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # # Initialize Session
    # sess = tf.Session()
    # # Init variables
    # sess.run(tf.global_variables_initializer())
    # # Get output
    # outs = sess.run([gcn_model.opt_op, gcn_model.loss, gcn_model.accuracy], feed_dict=feed_dict)
