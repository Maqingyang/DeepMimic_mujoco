import tensorflow as tf
import numpy as np

from utils.misc_util import RunningMeanStd
from utils import tf_util as U

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

class Discriminator(object):
    def __init__(self, ob_length, hidden_size, entcoeff=0.001, lr_rate=1e-3, scope="discriminator"):
        print('Observation shape: ', ob_length)
        self.scope = scope
        self.observation_shape = (ob_length,)
        self.input_shape = self.observation_shape
        self.hidden_size = hidden_size
        self.build_ph()
        # Build grpah
        generator_logits = self.build_graph(self.generator_obs_ph, reuse=False)
        expert_logits = self.build_graph(self.expert_obs_ph, reuse=True)
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff*entropy
        regular_loss = tf.nn.l2_loss(logits)
        regular_loss = 1e-4*tf.reduce_mean(regular_loss)
        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc, regular_loss]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc","regular_loss"]
        self.total_loss = generator_loss + expert_loss + entropy_loss + regular_loss
        # Build Reward for policy
        self.reward_op = -tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8)
        var_list = self.get_trainable_variables()
        self.lossandgrad = U.function([self.generator_obs_ph, self.expert_obs_ph],
                                      self.losses + [U.flatgrad(self.total_loss, var_list)])

    def build_ph(self):
        self.generator_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="observations_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="expert_observations_ph")

    def build_graph(self, obs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
            obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            p_h1 = tf.contrib.layers.fully_connected(obs, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs):
        sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        feed_dict = {self.generator_obs_ph: obs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward