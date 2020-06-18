'''
from baselines/ppo1/mlp_policy.py and add simple modification
(1) add reuse argument
(2) cache the `stochastic` placeholder

action are normalized to [-1, 1], use mean to shift and deviation to scale the range
'''
import tensorflow as tf
import gym

import utils.tf_util as U
from distributions import make_pdtype
from utils.misc_util import dense, RunningMeanStd

actutor_seq = ["right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"]
joint_limit = { "right_hip": [-1.2, 2.57],
                "right_knee": [-3.14, 0],
                "right_ankle": [-1.57, 1.57],
                "left_hip": [-1.2, 2.57],
                "left_knee": [-3.14, 0],
                "left_ankle": [-1.57, 1.57]
                }  

class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:, 0]

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i+1), weight_init=U.normc_initializer(1.0)))

        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            # mean is expected in range [-1, 1]
            unnorm_w = []
            unnorm_b = []
            for ac_name in actutor_seq:
                L, H = joint_limit[ac_name]
                unnorm_w.append((H-L)/2.)
                unnorm_b.append((H+L)/2.)
            w = tf.constant(unnorm_w, dtype=mean.dtype)
            b = tf.constant(unnorm_b, dtype=mean.dtype)
            mean = w*mean + b
            # logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            std = tf.constant(0.1, dtype=mean.dtype)
            logstd = tf.log(std)
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            raise ImportError("Action is not continuous!")
            pdparam = dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        # change for BC
        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []


if __name__ == "__main__":
    MlpPolicy("pi", reuse=False, ob_space=gym.spaces.Box, ac_space=6, hid_size=100, num_hid_layers=2)