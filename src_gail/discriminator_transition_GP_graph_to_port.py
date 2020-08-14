import scipy.sparse as sp
import tensorflow as tf
import numpy as np

from utils.misc_util import RunningMeanStd
from utils import tf_util as U

from utils.gcn_layers_sparse import *
# from utils.gcn_metrics import *
# from utils.gcn_utils import *

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
    def __init__(self, hidden_size, num_node, lookup_table, connections, entcoeff=0.001, lr_rate=1e-3, logits_regular_coeff=1e-4, scope="discriminator"):
        """
        num_node: the total number of nodes in the graph
        lookup_table: provide the correspondance between input and each node
        connections: list of connections between nodes
        """
        
        self.entcoeff = entcoeff
        self.logits_regular_coeff = logits_regular_coeff
        self.scope = scope
        self.obs_shape = (sum([len(x) for x in lookup_table]),)
        self.hidden_size = hidden_size
        self.connections = connections
        self.lookup_table = lookup_table
        self.num_node = num_node
        # Get adjacent matrix
        self.support = self.get_adj_matrix()
        # Build placeholder
        self.build_ph()
        # Build grpah
        generator_logits_list, norm_generator_obs = self.build_graph(self.generator_obs_ph, reuse=False)
        expert_logits_list, norm_expert_obs = self.build_graph(self.expert_obs_ph, reuse=True)
        # Build accuracy
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
        self.reward_op = tf.reduce_mean(tf.clip_by_value(-tf.log(1-tf.nn.sigmoid(generator_logits_concat)+1e-8), clip_value_min=0, clip_value_max=20))
        var_list = self.get_trainable_variables()
        self.summary_list = self.build_summary()

        self.lossandgrad = U.function([self.generator_obs_ph, self.expert_obs_ph],
                                      self.losses + [U.flatgrad(self.total_loss, var_list)])
        self.summary = U.function([self.generator_obs_ph, self.expert_obs_ph], self.summary_list)


    def get_adj_matrix(self):
        adj = np.zeros((self.num_node,self.num_node))
        for con in self.connections:
            adj[con] = 1
            adj.T[con] = 1
        return self._preprocess_adj(adj)

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
        regular_loss = self.logits_regular_coeff * tf.reduce_mean(regular_loss)
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

            obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std #（N, M)
            
            assert len(obs.get_shape()) == 2, "obs rank error! %d != 2" %(len(obs.get_shape()))
            assert self.num_node == len(self.lookup_table), "%d != %d, num_node doesn't match lookup table for node input!" %(self.num_node, len(self.lookup_table))

            embedding_list = list()
            layer_list = list()
            activation_list = list()
            logits_list = list()
            for idx in range(self.num_node):
                index_list = self.lookup_table[idx]
                obs_for_curr_node = tf.gather(obs, index_list, axis=1)
                mlp_for_curr_node = MLP(len(index_list), self.hidden_size, name="mlp_for_node_%d" %idx)
                embedding_list.append(mlp_for_curr_node(obs_for_curr_node))
            embeddings = tf.stack(embedding_list, axis=1) #(N, G, F)
            layer_list.append(GraphConvolutionLayerBatch(self.hidden_size,self.hidden_size,self.support,name="graphconvolutionlayer_0"))
            layer_list.append(GraphConvolutionLayerBatch(self.hidden_size,self.hidden_size,self.support,name="graphconvolutionlayer_1"))
            
            activation_list.append(embeddings)
            for layer in layer_list:
                hidden = layer(activation_list[-1])
                activation_list.append(hidden)
            for hidden in activation_list:
                logits_list.append(tf.contrib.layers.fully_connected(hidden, 1, activation_fn=tf.identity,reuse=tf.AUTO_REUSE,scope="disc"))

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
        adj_normalized = self._normalize_adj(adj + sp.eye(adj.shape[0]))
        return self._sparse_to_tuple(adj_normalized)


    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj,dtype=np.float32)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def _sparse_to_tuple(self, sparse_mx):
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

if __name__=="__main__":
    d_stepsize = 1e-2
    batch_size = 32
    hidden_size = 64
    num_node = 7
    lookup_table=[
        [0,1,2,3,4,5], # 0:root
        [6,7],         # 1:right_hip
        [8,9],         # 2:right_knee
        [10,11],       # 3:right_ankle
        [12,13],       # 4:left_hip
        [14,15],       # 5:left_knee
        [16,17],       # 6:left_ankle
    ]
    connections=[(6,5),(5,4),(4,0),(0,1),(1,2),(2,3)]
    reward_giver = GraphDiscriminator(hidden_size, num_node, lookup_table, connections)
    trainable_variables = reward_giver.get_trainable_variables()
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
            transition_expert = transition_expert.reshape([-1,2,9]).transpose(0,2,1) #(N,9,2)
            transition_batch = transition_batch.reshape([-1,2,9]).transpose(0,2,1) #(N,9,2)

            *newlosses, g = reward_giver.lossandgrad(transition_batch, transition_expert)
            summary_list = reward_giver.summary(transition_batch, transition_expert)
            for summary in summary_list:
                writer.add_summary(summary, i)

            d_adam.update(g, d_stepsize)
            print(fmt_row(13, reward_giver.loss_name))
            print(fmt_row(13, newlosses))
            # print(sess.run(reward_giver.get_trainable_variables()))
