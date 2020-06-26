#!/usr/bin/env python3
import numpy as np
import math
import random
from os import getcwd
import os.path as osp
from mujoco.mocap_bipedal_2D_multiClip import MocapDM
from mujoco.mujoco_interface import MujocoInterface
from mujoco.mocap_util import JOINT_WEIGHT
from mujoco_py import load_model_from_xml, MjSim, MjViewer, cymj
from mujoco_py.generated import const
from scipy.spatial.transform import Rotation as R

from gym.envs.mujoco import mujoco_env
from gym import utils

# from config_biped_torch import Config
from pyquaternion import Quaternion

from transformations import quaternion_from_euler
from box import Box

actutor_seq = ["right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"]
joint_limit = { "right_hip": [-1.2, 2.57],
                "right_knee": [-3.14, 0],
                "right_ankle": [-1.57, 1.57],
                "left_hip": [-1.2, 2.57],
                "left_knee": [-3.14, 0],
                "left_ankle": [-1.57, 1.57]
                }   

def euler2mat(euler, degrees=True): 
    r = R.from_euler('xyz', euler, degrees=degrees)
    return r.as_matrix()

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

def lerp(qpos_0, qpos_1, ratio):
    return qpos_0*ratio + qpos_1*(1.-ratio)


class DPEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,C):

        xml_file_path = osp.join(getcwd(), C.xml_folder, C.env_file)

        self.mocap = MocapDM()
        self.interface = MujocoInterface()
        motion_file_list = []
        for motion_file in C.motion_file_list:
            motion_file_list.append(osp.join(getcwd(), C.motion_folder, motion_file))
        self.load_mocap(motion_file_list)

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

        self.curr_clip_idx = 0
        self.idx_curr = -1
        self.idx_tmp_count = -1
        self.policy_freq = 25
        self.is_gail = C.is_gail
        self.init_time = 0
        self.max_time = 1
        self.target_root_x_speed_lower_bound = C.target_root_x_speed_lower_bound
        self.target_root_x_speed_higher_bound = C.target_root_x_speed_higher_bound
        self.speed_random_flag = True

        mujoco_env.MujocoEnv.__init__(self, xml_file_path, 1)

        cymj.set_pid_control(self.sim.model, self.sim.data)
        # self.viewer = MjViewer(self.sim)
        utils.EzPickle.__init__(self)


    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        
        if self.is_gail:
            return np.concatenate((position, velocity, [self.target_root_x_speed])) 

        return np.concatenate((position, velocity, target_config, target_vel))


    def early_termination(self):
        pass

    def get_joint_configs(self):
        data = self.sim.data
        return data.qpos 

    def get_joint_vel(self):
        data = self.sim.data
        return data.qvel


    def load_mocap(self, filepath_list):
        self.mocap.load_mocap(filepath_list)
        self.num_clip = len(self.mocap.multi_clip_data)


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


    def calc_speed_reward(self):

        curr_root_x_speed = self.get_joint_vel()[0]
        err_speed = np.square(curr_root_x_speed - self.target_root_x_speed)
        # reward_config = math.exp(-self.scale_err * self.scale_pose * err_configs)
        reward_speed = np.exp(-10*err_speed)

        return reward_speed


    def update_target_speed(self):
        lower_bound = self.target_root_x_speed_lower_bound
        higher_bound = self.target_root_x_speed_higher_bound
        # time_in_period = (self.data.time + float(self.curr_clip_idx == 1) * 7) % 15
        # if time_in_period < 2:
        #     self.target_root_x_speed = lower_bound
        # elif time_in_period < 7:
        #     self.target_root_x_speed = (time_in_period-2)/(7-2)*(higher_bound-lower_bound) + lower_bound
        # elif time_in_period < 10:
        #     self.target_root_x_speed = higher_bound
        # elif time_in_period < 15:
        #     self.target_root_x_speed = higher_bound - (time_in_period-10)/(7-2)*(higher_bound-lower_bound)
        
        if self.data.time < 4:
            self.target_root_x_speed = 5 if self.data.qvel[0] > 3 else 1.5
            self.speed_random_flag = True
        elif int(self.data.time - 4) % 4 == 0 and self.speed_random_flag:
            self.target_root_x_speed = np.random.uniform(self.target_root_x_speed_lower_bound, self.target_root_x_speed_higher_bound)
            self.speed_random_flag = False
        elif int(self.data.time - 4) % 4 == 1 and not self.speed_random_flag:
            self.speed_random_flag = True

    def step(self, action):

            
        self.do_simulation(action, n_frames=int(500/self.policy_freq))
        self.update_target_speed()
        reward_obs = self.calc_speed_reward()

        reward = reward_obs
        info = dict()

        observation = self._get_obs()
        done = self.is_done()

        return observation, reward, done, info

    def imit_reward(self):
        return 10 * self.calc_config_reward()

    def is_done(self):
        mass = np.expand_dims(self.model.body_mass, 1)
        xpos = self.sim.data.xipos
        z_com = (np.sum(mass * xpos, 0) / np.sum(mass))[2] # bipedal mass center at 0.7937
        done = bool((z_com < 0.4) or (z_com > 1.0))
        if self.data.time > self.max_time and self.is_gail:
            done = True
        return done

    def set_max_time(self, t):
        self.max_time = t

    def get_max_time(self):
        return self.max_time

    def goto(self, pos):
        self.sim.data.qpos[:] = pos[:]
        self.sim.forward()

    def get_time(self):
        return self.sim.data.time

    def reference_state_init(self):
        self.curr_clip_idx = np.random.randint(self.num_clip)
        mocap_period = self.mocap.multi_clip_data[self.curr_clip_idx]["period"]
        mocap_dt = self.mocap.multi_clip_data[self.curr_clip_idx]["dt"]
        mocap_config = self.mocap.multi_clip_data[self.curr_clip_idx]["config"]
        mocap_vel = self.mocap.multi_clip_data[self.curr_clip_idx]["vel"]

        curr_time = np.random.uniform(0, mocap_period)
        init_time = curr_time
        idx_curr = int(curr_time // mocap_dt)  
        ratio = 1 - (curr_time % mocap_dt) / mocap_dt
        target_config = lerp(mocap_config[idx_curr], mocap_config[idx_curr + 1], ratio)
        target_vel = lerp(mocap_vel[idx_curr], mocap_vel[idx_curr + 1], ratio)


        qpos = target_config
        qvel = target_vel
        self.set_state(qpos, qvel)

    def reset_model(self):
        self.reference_state_init()
        # self.target_root_x_speed = np.random.uniform(self.target_root_x_speed_lower_bound, self.target_root_x_speed_higher_bound)
        self.update_target_speed()

        observation = self._get_obs()

        return observation

    def reset_model_init(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv)
        )
        return self._get_obs()


    def sample_1_expert_traj(self):
        curr_clip_idx = np.random.randint(self.num_clip)
        mocap_period = self.mocap.multi_clip_data[curr_clip_idx]["period"]
        mocap_dt = self.mocap.multi_clip_data[curr_clip_idx]["dt"]
        mocap_config = self.mocap.multi_clip_data[curr_clip_idx]["config"]
        mocap_vel = self.mocap.multi_clip_data[curr_clip_idx]["vel"]

        interval = np.clip(1+0.3*np.random.randn(), 1e-3, 2) * 1./self.policy_freq

        curr_time = np.random.uniform(0, mocap_period-interval)
        idx_curr = int(curr_time // mocap_dt)  
        ratio = 1 - (curr_time % mocap_dt) / mocap_dt
        pos_0 = lerp(mocap_config[idx_curr],mocap_config[idx_curr + 1],ratio).copy()

        curr_time += interval        
        idx_curr = int(curr_time // mocap_dt)  
        ratio = 1 - (curr_time % mocap_dt) / mocap_dt
        pos_1 = lerp(mocap_config[idx_curr],mocap_config[idx_curr + 1],ratio).copy()

        pos_1[0] = 0
        pos_0[0] = 0

        return np.concatenate([pos_0,pos_1])
        
    def sample_expert_traj(self):
        num_sample = 1024*128
        sample_list = []
        for i in range(num_sample):
            sample = self.sample_1_expert_traj()
            sample_list.append(sample)
        
        return np.array(sample_list)

    def render(self,*arg,**kwarg):
        arrow_pos = np.array([self.get_joint_configs()[0]+1, 0, self.get_joint_configs()[1]-0.3])
        self.viewer.add_marker(pos=arrow_pos, #position of the arrow
                    size=np.array([0.005+0.001*self.target_root_x_speed,0.005+0.001*self.target_root_x_speed,0.2+0.09*self.target_root_x_speed]), #size of the arrow
                    mat=euler2mat([0,90,0]), # orientation as a matrix
                    rgba=np.array([1.,0.85,0,1.]),#color of the arrow
                    type=const.GEOM_ARROW,
                    )

        arrow_label_pos = np.array([self.get_joint_configs()[0]+0.95, 0, self.get_joint_configs()[1]-0.25])
        self.viewer.add_marker(pos=arrow_label_pos, #position of the arrow
                     size=np.array([1,1,1]), #size of the arrow
                    # mat=euler2mat([0,90,0]), # orientation as a matrix
                    # rgba=np.array([1.,1.,1.,1.]),#color of the arrow
                    type=const.GEOM_LABEL,
                    label=str('target speed: %.2f' %self.target_root_x_speed))


        speed_pos = np.array([self.get_joint_configs()[0]-0.25, 0, self.get_joint_configs()[1]+0.3])

        self.viewer.add_marker(pos=speed_pos, #position of the arrow
                     size=np.array([1,1,1]), #size of the arrow
                    # mat=euler2mat([0,90,0]), # orientation as a matrix
                    # rgba=np.array([1.,1.,1.,1.]),#color of the arrow
                    type=const.GEOM_LABEL,
                    label=str('biped speed: %.2f' %(self.get_joint_vel()[0])))

        mujoco_env.MujocoEnv.render(self,*arg,**kwarg)

if __name__ == "__main__":
    C = Box.from_yaml(filename="config/gail_ppo_biped_PID_multiClip.yaml")
    env = DPEnv(C)
    env.reset_model()
    env.sample_expert_traj()
    import cv2
    from VideoSaver import VideoSaver
    width = 640
    height = 480

    # vid_save = VideoSaver(width=width, height=height)

    # env.load_mocap("/home/mingfei/Documents/DeepMimic/mujoco/motions/humanoid3d_crawl.txt")
    action_size = env.action_space.shape[0]
    ac = np.zeros(action_size)
    samples = env.sample_expert_traj()

    while True:
        sample = samples[np.random.randint(0,1024)]
        pos_0 = sample[:9]
        pos_1 = sample[9:]
        env.goto(pos_0)
        env.viewer.render()
        env.goto(pos_1)
        env.viewer.render()



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

        env.reset()
        print(env.data.qvel)
        # env.viewer.render()
        # if done:
        #     env.reset_model()
        # env.calc_config_reward()
        # img = env.render(mode = 'rgb_array')[...,::-1]
        # cv2.imwrite("env.png",img)
    # vid_save.close()
