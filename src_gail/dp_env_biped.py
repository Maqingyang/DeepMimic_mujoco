#!/usr/bin/env python3
import numpy as np
import math
import random
from os import getcwd

from mujoco.mocap_bipedal_2D import MocapDM
from mujoco.mujoco_interface import MujocoInterface
from mujoco.mocap_util import JOINT_WEIGHT
from mujoco_py import load_model_from_xml, MjSim, MjViewer

from gym.envs.mujoco import mujoco_env
from gym import utils

from config_biped import Config
from pyquaternion import Quaternion

from transformations import quaternion_from_euler


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]
def lerp(qpos_0, qpos_1, ratio):
    return qpos_0*ratio + qpos_1*(1.-ratio)


class DPEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_file_path = Config.xml_path

        self.mocap = MocapDM()
        self.interface = MujocoInterface()
        self.load_mocap(Config.mocap_path)

        self.weight_pose = 0.5
        self.weight_vel = 0.05
        self.weight_root = 0.2
        self.weight_end_eff = 0.15
        self.weight_com = 0.1

        self.scale_pose = 2.0
        self.scale_vel = 0.1
        self.scale_end_eff = 40.0
        self.scale_root = 5.0
        self.scale_com = 10.0
        self.scale_err = 1.0

        self.reference_state_init()
        self.idx_curr = -1
        self.idx_tmp_count = -1

        mujoco_env.MujocoEnv.__init__(self, xml_file_path, 1)
        self.viewer = MjViewer(self.sim)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        ## add mocap data as additional observation
        target_config = self.target_config
        target_vel = self.target_vel
        return np.concatenate((position, velocity, target_config, target_vel))

    def reference_state_init(self):
        self.idx_init = random.randint(0, self.mocap_data_len-1)
        # self.idx_init = 0
        self.idx_curr = self.idx_init
        self.idx_tmp_count = 0

    def early_termination(self):
        pass

    def get_joint_configs(self):
        data = self.sim.data
        return data.qpos 

    def load_mocap(self, filepath):
        self.mocap.load_mocap(filepath)
        self.mocap_dt = self.mocap.dt
        self.mocap_data_len = len(self.mocap.data)
        self.mocap_period = self.mocap_data_len*self.mocap_dt

    def calc_config_errs(self, env_config, mocap_config):
        assert len(env_config) == len(mocap_config)
        return np.sum(np.abs(env_config - mocap_config))

    def calc_config_reward(self):
        assert len(self.mocap.data) != 0
        err_configs = 0.0

        curr_config = self.get_joint_configs()
        target_config = self.target_config
        err_configs = self.calc_config_errs(curr_config, target_config)
        # reward_config = math.exp(-self.scale_err * self.scale_pose * err_configs)
        reward_config = math.exp(-err_configs)

        return reward_config

    def update_target_frame(self):
        curr_time = (self.data.time + self.idx_init*self.mocap_dt) % self.mocap_period
        self.idx_curr = int(curr_time // self.mocap_dt)  
        ratio = 1 - (curr_time % self.mocap_dt) / self.mocap_dt
        idx_next = (self.idx_curr + 1) % self.mocap_data_len
        config_next = self.mocap.data_config[idx_next]

        if idx_next == 0:
            config_next[:2] += self.mocap.data_config[self.idx_curr][:2]
    
        target_config = lerp(self.mocap.data_config[self.idx_curr],config_next,ratio)
        target_vel = lerp(self.mocap.data_vel[self.idx_curr],self.mocap.data_vel[idx_next],ratio)

        self.target_config = target_config
        self.target_vel = target_vel

    def step(self, action):
        # self.step_len = int(self.mocap_dt // self.model.opt.timestep)
        self.step_len = 1
        # step_times = int(self.mocap_dt // self.model.opt.timestep)
        step_times = 1
        for i in range(20): # 500 HZ / 20 = 25 HZ
            self.do_simulation(action, step_times)

        self.update_target_frame()
        # reward_alive = 1.0
        reward_obs = 10 * self.calc_config_reward()
        # reward_acs = 0.1 * np.square(self.sim.data.ctrl).sum()
        # reward = reward_obs + reward_alive - reward_acs
        reward = reward_obs

        # info = dict(reward_obs=reward_obs, reward_acs=reward_acs, reward_forward=reward_forward)
        info = dict()

        observation = self._get_obs()
        done = self.is_done()

        return observation, reward, done, info

    def is_done(self):
        mass = np.expand_dims(self.model.body_mass, 1)
        xpos = self.sim.data.xipos
        z_com = (np.sum(mass * xpos, 0) / np.sum(mass))[2] # bipedal mass center at 0.7937
        done = bool((z_com < 0.4) or (z_com > 1.2))
        return done

    def goto(self, pos):
        self.sim.data.qpos[:] = pos[:]
        self.sim.forward()

    def get_time(self):
        return self.sim.data.time

    def reset_model(self):
        env.sim.reset()
        self.reference_state_init()
        qpos = self.mocap.data_config[self.idx_init]
        qvel = self.mocap.data_vel[self.idx_init]
        # qvel = self.init_qvel
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        # self.idx_tmp_count = -self.step_len
        return observation

    def reset_model_init(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        pass
        # self.viewer.cam.trackbodyid = 1
        # self.viewer.cam.distance = self.model.stat.extent * 1.0
        # self.viewer.cam.lookat[2] = 2.0
        # self.viewer.cam.elevation = -20

if __name__ == "__main__":
    env = DPEnv()
    env.reset_model()

    import cv2
    from VideoSaver import VideoSaver
    width = 640
    height = 480

    # vid_save = VideoSaver(width=width, height=height)

    # env.load_mocap("/home/mingfei/Documents/DeepMimic/mujoco/motions/humanoid3d_crawl.txt")
    action_size = env.action_space.shape[0]
    ac = np.zeros(action_size)
    while True:
        # target_config = env.mocap.data_config[env.idx_curr][7:] # to exclude root joint
        # env.sim.data.qpos[7:] = target_config[:]
        # env.sim.forward()

        # qpos = env.mocap.data_config[env.idx_curr]
        # qvel = env.mocap.data_vel[env.idx_curr]
        # qpos = np.zeros_like(env.mocap.data_config[env.idx_curr])
        # qvel = np.zeros_like(env.mocap.data_vel[env.idx_curr])
        # env.set_state(qpos, qvel)
        # env.sim.step()
        observation, reward, done, info = env.step(env.action_space.sample())
        env.goto(env.target_config)
        env.viewer.render()
        if done:
            env.reset_model()
        # env.calc_config_reward()
        # img = env.render(mode = 'rgb_array')[...,::-1]
        # cv2.imwrite("env.png",img)
    # vid_save.close()
