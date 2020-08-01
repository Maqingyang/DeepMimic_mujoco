import gym
import time
import os
import os.path as osp
import dataset
from dataset import Dataset
import logger
import argparse
import datetime

import tensorflow as tf
import utils.tf_util as U
import numpy as np

from cg import cg
from utils.mujoco_dset import Dset_transition
from utils.misc_util import set_global_seeds, zipsame, boolean_flag
from utils.math_util import explained_variance
from utils.console_util import fmt_row, colorize
from contextlib import contextmanager
from mpi4py import MPI
from collections import deque
from mpi_adam import MpiAdam
from statistics import stats
from mlp_policy import MlpPolicy
from box import Box

from dp_env_biped_PID_unnorm import DPEnv
from discriminator_transition_GP_graph import GraphDiscriminator

pos_len = 9
obs_len_per_node = 2

def traj_segment_generator(pi, env, reward_giver, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    env_rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    cur_ep_env_ret = 0
    cur_ep_env_rets = []
    cur_ep_disc_ret = 0
    cur_ep_disc_rets = []
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    next_obs = obs.copy()
    transitions = np.array([np.zeros(pos_len*2) for _ in range(horizon)])
    env_rews = np.zeros(horizon, 'float32')
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        masked_ob = ob.copy()
        masked_ob[0] = 0 # mask root_x
        ac, vpred = pi.act(stochastic=stochastic, ob=masked_ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "next_ob": next_obs, "transitions":transitions, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "cur_ep_env_rets": cur_ep_env_rets, "cur_ep_disc_rets":cur_ep_disc_rets}
            _, vpred = pi.act(stochastic=stochastic, ob=masked_ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            cur_ep_env_rets = []
            cur_ep_disc_rets = []
            ep_lens = []
        i = t % horizon

        obs[i] = masked_ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        next_ob, env_rew, new, _ = env.step(ac)
        pos_0 = ob[:pos_len].copy()
        pos_1 = next_ob[:pos_len].copy()
        pos_1[0] -= pos_0[0]
        pos_0[0] = 0
        # pos_1[0] = 0 
        # pos_0[0] = 0

        transition = np.concatenate([pos_0, pos_1])
        transitions[i] = transition
        d_rew = 10 * reward_giver.get_reward(transition)
        
        next_obs[i] = next_ob
        ob = next_ob

        rews[i] = d_rew
        env_rews[i] = env_rew

        cur_ep_ret += d_rew
        cur_ep_env_ret += env_rew
        cur_ep_disc_ret += 10 * reward_giver.get_reward(transition) 
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            cur_ep_env_rets.append(cur_ep_env_ret)
            cur_ep_disc_rets.append(cur_ep_disc_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_env_ret = 0
            cur_ep_disc_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, policy_func, reward_giver, expert_dataset, rank, ckpt_dir, task_name,
          g_step=1, save_per_iter=100, g_optim_batchsize=64,
          timesteps_per_batch=4096, clip_param=0.2, entcoeff=0.01, g_optim_epochs=4,
          gamma=0.99, lam=0.95, adam_epsilon=1e-5, lr_schedule='linear',
          g_stepsize=1e-4, d_stepsize=1e-4, 
          max_timesteps=0, writer=None,
          callback=None):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)
    oldpi = policy_func("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])  # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon
                    
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent
    # entbonus = entcoeff * meanent

    

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold

    surr1 = ratio * atarg 
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
    # PPO's pessimistic surrogate (L^CLIP)
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vferr

    losses = [pol_surr, pol_entpen, vferr, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]
    var_list = pi.get_trainable_variables()
    


    g_adam = MpiAdam(var_list, epsilon=adam_epsilon)
    d_adam = MpiAdam(reward_giver.get_trainable_variables())

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)


    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])

    lossandgrad = U.function([ob, ac, atarg, ret, lrmult],
                                        [U.flatgrad(total_loss, var_list)] + losses)

    clipped_ratio = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), clip_param)))
    ac_std = tf.reduce_mean(pi.pd.std)
    compute_clipped_ratio = U.function([ob, ac, atarg, ret, lrmult],[clipped_ratio])
    compute_ac_std = U.function([ob, ac, atarg, ret, lrmult],[ac_std])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult],
                                            losses)

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    g_adam.sync()
    d_adam.sync()
    if rank == 0:
        print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, reward_giver, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards
    true_rewbuffer = deque(maxlen=40)
    disc_rewbuffer = deque(maxlen=40)



    replay_buffer = {'transitions':[]}



    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break


        # Save model
        if rank == 0 and iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            fname = osp.join(ckpt_dir,"%06d/%s" %(iters_so_far, task_name))
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver = tf.train.Saver()
            saver.save(tf.get_default_session(), fname)
        # Increase Env Max Time
        if iters_so_far % 500 == 0 and iters_so_far != 0:
            env.set_max_time(2 * env.get_max_time())

        if lr_schedule == 'constant':
            cur_lrmult = 1.0
        elif lr_schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError


        logger.log("********** Iteration %i ************" % iters_so_far)



        # ------------------ Update G ------------------
        logger.log("Optimizing Policy...")
        for _ in range(g_step):
            with timed("sampling"):
                seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            transitions = seg["transitions"]

            if len(replay_buffer['transitions']) > 100: 
                replay_buffer['transitions'].pop(0)
            replay_buffer['transitions'].append(transitions)

            vpredbefore = seg["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

            g_dataset = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=True)
            # set old parameter values to new parameter values
            assign_old_eq_new()  
            logger.log("Optimizing...")
            logger.log(fmt_row(13, loss_names))

            # Here we do a bunch of optimization epochs over the data
            for k in range(g_optim_epochs):
                # list of tuples, each of which gives the loss for a minibatch
                losses = []
                for i, batch in enumerate(g_dataset.iterate_once(g_optim_batchsize)):
                    grad, *newlosses = lossandgrad(batch["ob"], batch["ac"],
                                                            batch["atarg"], batch["vtarg"], cur_lrmult,
                                                           )
                    g_adam.update(grad, g_stepsize * cur_lrmult)
                    losses.append(newlosses)
                logger.log(fmt_row(13, np.mean(losses, axis=0)))
                all_data = g_dataset.all_data()
            
            logger.record_tabular("MeanClippedRatio", compute_clipped_ratio(all_data["ob"], all_data["ac"], all_data["atarg"], all_data["vtarg"], cur_lrmult)[0])
            logger.record_tabular("MeanAcStd", compute_ac_std(all_data["ob"], all_data["ac"], all_data["atarg"], all_data["vtarg"], cur_lrmult)[0])


        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, reward_giver.loss_name))
        batch_size = 256
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch

        transitions_all = np.concatenate(replay_buffer['transitions'], axis=0)
        # sample_idx = np.random.randint(0, transitions_all.shape[0], timesteps_per_batch*8)
        # transitions_downsample = transitions_all[sample_idx]





        for transition_batch, _ in dataset.iterbatches((transitions_all, transitions_all),
                                                      include_final_partial_batch=False,
                                                      batch_size=batch_size):
            transition_expert = expert_dataset.get_next_batch(len(transition_batch)) #(N,2*9)
            transition_batch = transition_batch.reshape([-1,obs_len_per_node,pos_len]).transpose(0,2,1)
            transition_expert = transition_expert.reshape([-1,obs_len_per_node,pos_len]).transpose(0,2,1)
            # update running mean/std for reward_giver
            if hasattr(reward_giver, "obs_rms"): reward_giver.obs_rms.update(np.concatenate((transition_batch, transition_expert), 0))
            *newlosses, g = reward_giver.lossandgrad(transition_batch, transition_expert)
            d_adam.update(allmean(g), d_stepsize)
            d_losses.append(newlosses)

        if rank == 0:
            transition_batch = replay_buffer['transitions'][-1] #(timesteps_per_batch, 2*9)
            transition_expert = expert_dataset.get_next_batch(len(transition_batch)) #(N,2*9)
            transition_batch = transition_batch.reshape([-1,obs_len_per_node,pos_len]).transpose(0,2,1)
            transition_expert = transition_expert.reshape([-1,obs_len_per_node,pos_len]).transpose(0,2,1)
            summary_list = reward_giver.summary(transition_batch, transition_expert)
            for summary in summary_list:
                writer.add_summary(summary, iters_so_far)


        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))
        # if rank == 0:
        #     summary_list = reward_giver.summary(transition_batch, transition_expert)
        #     for summary in summary_list:
        #         writer.add_summary(summary, iters_so_far)


        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["cur_ep_env_rets"], seg["cur_ep_disc_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples

        def flatten_lists(listoflists):
            return [el for list_ in listoflists for el in list_]

        lens, rews, true_rets, disc_rets = map(flatten_lists, zip(*listoflrpairs))
        true_rewbuffer.extend(true_rets)
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        disc_rewbuffer.extend(disc_rets)

        

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
        logger.record_tabular("EpDiscRewMean", np.mean(disc_rewbuffer))
        # logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens) * g_step * g_optim_epochs
        iters_so_far += 1

        # logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("ItersSoFar", iters_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        # logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank == 0:
            logger.dump_tabular()




def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    U.load_state(load_model_path)

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    from tqdm import tqdm
    for _ in tqdm(range(number_trajs)):
        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret

# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    ob[0] = 0
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    while True:
        ac, vpred = pi.act(stochastic=stochastic, ob=ob)
        obs.append(ob)
        print("root x vel", ob[9])
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        ob[0] = 0
        env.render()
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj



def main(args):
    U.make_session(num_cpu=1).__enter__()
    C = Box.from_yaml(filename=args.config)
    set_global_seeds(C.seed)

    env = DPEnv(C)


    def policy_func(name, ob_space, ac_space, reuse=False):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=C.policy_hidden_size, num_hid_layers=C.num_hid_layers)

    if args.task == 'train':
        import logging
        import os.path as osp
        import bench

        gym.logger.setLevel(logging.INFO)
        save_name = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
        save_name += "-%s" % osp.basename(args.config).split('.')[0]
        checkpoint_dir = osp.join(C.checkpoint_dir, save_name)
        log_dir = osp.join(checkpoint_dir, "log")
        task_name = C.motion_file.split('.')[0]

        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            C.to_yaml(osp.join(checkpoint_dir, "config.yaml"))
            logger.configure(dir=log_dir)
            writer = tf.summary.FileWriter(osp.join(log_dir,"./graphs"), tf.get_default_graph())
            
        if MPI.COMM_WORLD.Get_rank() != 0:
            logger.set_level(logger.DISABLED)
        else:
            logger.set_level(logger.INFO)
        """
        for bipedal, obs is (9,)
        0:root
        1:right_hip
        2:right_knee
        3:right_ankle
        4:left_hip
        5:left_knee
        6:left_ankle
        """
        connections = [(6,5),(5,4),(4,0),(0,1),(1,2),(2,3)]
        adj = np.zeros((7,7))
        for con in connections:
            adj[con] = 1
            adj.T[con] = 1

        env = bench.Monitor(env, logger.get_dir() and
                            osp.join(logger.get_dir(), "monitor.json"))


        expert_dataset = Dset_transition(transitions = env.env.sample_expert_traj())
        reward_giver = GraphDiscriminator(obs_len_per_node, C.adversary_hidden_size, adj, entcoeff=C.adversary_entcoeff)

        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = C.seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        

        learn(env, policy_func, reward_giver, expert_dataset, rank, checkpoint_dir, task_name,
                g_step=C.g_step, save_per_iter=C.save_per_iter, g_optim_batchsize=C.g_optim_batchsize,
                timesteps_per_batch=C.timesteps_per_batch, clip_param=C.clip_param, entcoeff=C.entcoeff, g_optim_epochs=C.g_optim_epochs,
                gamma=C.gamma, lam=C.lam, adam_epsilon=C.adam_epsilon, lr_schedule=C.lr_schedule,
                g_stepsize=C.g_stepsize, d_stepsize=C.d_stepsize, 
                max_timesteps=C.max_timesteps, writer=writer,
                callback=None)


    elif args.task == 'evaluate':
        runner(env,
               policy_func,
               args.load_model_path,
               timesteps_per_batch=1024,
               number_trajs=20,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample,
               )
    env.close()

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--config', help='yaml config file path', type=str)
    # for evaluatation
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = argsparser()
    main(args)
