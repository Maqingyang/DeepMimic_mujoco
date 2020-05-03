#!/usr/bin/env python3
import numpy as np
import math
import random
from os import getcwd

from mujoco.mocap_v2 import MocapDM
from mujoco.mujoco_interface import MujocoInterface
from mujoco.mocap_util import JOINT_WEIGHT
from mujoco_py import load_model_from_xml, MjSim, MjViewer

from gym.envs.mujoco import mujoco_env
from gym import utils

from config_modified import Config
from pyquaternion import Quaternion

from transformations import quaternion_from_euler, euler_from_quaternion

BODY_JOINTS = ["chest", "neck", "right_shoulder", "right_elbow", 
            "left_shoulder", "left_elbow", "right_hip", "right_knee", 
            "right_ankle", "left_hip", "left_knee", "left_ankle"]

DOF_DEF = {"root": 3, "chest": 3, "neck": 3, "right_shoulder": 3, 
           "right_elbow": 1, "right_wrist": 0, "left_shoulder": 3, "left_elbow": 1, 
           "left_wrist": 0, "right_hip": 3, "right_knee": 1, "right_ankle": 3, 
           "left_hip": 3, "left_knee": 1, "left_ankle": 3}

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class DPEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_file_path = Config.xml_path
        self.xml_file_path = xml_file_path

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
        utils.EzPickle.__init__(self)

    def _quat2euler(self, quat):
        tmp_quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        euler = euler_from_quaternion(tmp_quat, axes='rxyz')
        return euler

    def _get_obs(self):
        config = self.sim.data.qpos.flat.copy()[2:] # ignore root joint: x, y
        vel = self.sim.data.qvel.flat.copy() # ignore root joint

        return np.concatenate((config, vel))

    def reference_state_init(self):
        self.idx_init = random.randint(0, self.mocap_data_len-1)
        # self.idx_init = 0
        self.idx_curr = self.idx_init
        self.idx_tmp_count = 0

    def early_termination(self):
        target_config = self.mocap.data_config[self.idx_curr][7:] # to exclude root joint
        curr_config = self.get_joint_configs()
        err_configs = self.calc_config_errs(curr_config, target_config)
        if err_configs >= 15.0:
            return True
        return False

    def get_joint_configs(self):
        data = self.sim.data
        return data.qpos[7:] # to exclude root joint

    def get_root_configs(self):
        data = self.sim.data
        return data.qpos[3:7] # to exclude x coord

    def load_mocap(self, filepath):
        self.mocap.load_mocap(filepath)
        self.mocap_dt = self.mocap.dt
        self.mocap_data_len = len(self.mocap.data)

    def calc_config_errs(self, env_config, mocap_config):
        assert len(env_config) == len(mocap_config)
        return np.sum(np.abs(env_config - mocap_config))

    def calc_root_reward(self): # including root joint
        curr_root = self.mocap.data_config[self.idx_curr][3:7]
        target_root = self.get_root_configs()
        assert len(curr_root) == len(target_root)
        assert len(curr_root) == 4

        q_0 = Quaternion(curr_root[0], curr_root[1], curr_root[2], curr_root[3])
        q_1 = Quaternion(target_root[0], target_root[1], target_root[2], target_root[3])

        q_diff =  q_0.conjugate * q_1
        tmp_diff = q_diff.angle

        err_root = abs(tmp_diff)
        reward_root = math.exp(-err_root)
        return reward_root

    def calc_config_reward(self):
        assert len(self.mocap.data) != 0
        err_configs = 0.0

        target_config = self.mocap.data_config[self.idx_curr][7:] # to exclude root joint
        self.curr_frame = target_config
        curr_config = self.get_joint_configs()

        err_configs = self.calc_config_errs(curr_config, target_config)
        # reward_config = math.exp(-self.scale_err * self.scale_pose * err_configs)
        reward_config = math.exp(-err_configs)

        return reward_config, err_configs

    def step(self, action):
        self.step_len = 1
        step_times = 1

        self.do_simulation(action, step_times)

        reward_config,  err_config = self.calc_config_reward()
        reward_root = 10 * self.calc_root_reward()
        reward = reward_config + reward_root

        info = dict()

        self.idx_curr += 1
        self.idx_curr = self.idx_curr % self.mocap_data_len

        observation = self._get_obs()
        done = bool(self.is_done() or err_config >= 10.0)

        return observation, reward, done, info

    def is_done(self):
        mass = np.expand_dims(self.model.body_mass, 1)
        xpos = self.sim.data.xipos
        z_com = (np.sum(mass * xpos, 0) / np.sum(mass))[2]
        # done = bool((z_com < 0.7) or (z_com > 1.2) or self.early_termination())
        done = bool((z_com < 0.7) or (z_com > 1.2))
        # return False
        return done

    def goto(self, pos):
        self.sim.data.qpos[:] = pos[:]
        self.sim.forward()

    def get_time(self):
        return self.sim.data.time

    def reset_model(self):
        self.reference_state_init()
        qpos = self.mocap.data_config[self.idx_init]
        qvel = self.mocap.data_vel[self.idx_init]
        # qvel = self.init_qvel
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        self.idx_tmp_count = -self.step_len
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
        # target_config = env.mocap.data_config[env.idx_curr][:7] # to exclude root joint
        # env.sim.data.qpos[:7] = target_config[:]
        # env.sim.forward()

        qpos = env.mocap.data_config[env.idx_curr]
        qvel = np.zeros_like(env.mocap.data_vel[env.idx_curr])
        # qpos = np.zeros_like(env.mocap.data_config[env.idx_curr])
        # qvel = np.zeros_like(env.mocap.data_vel[env.idx_curr])
        env.set_state(qpos, qvel)
        env.sim.step()
        print("Reward root:", env.calc_config_reward())
        env.idx_curr += 1
        if env.idx_curr == env.mocap_data_len:
            # env.reset_model()
            env.idx_curr = env.idx_curr % env.mocap_data_len

        # print(env._get_obs())
        env.render()

    # vid_save.close()