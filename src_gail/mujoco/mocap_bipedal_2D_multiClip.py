#!/usr/bin/env python3
import os
import json
import math
import copy
import numpy as np
from os import getcwd,chdir
import sys
sys.path.append(getcwd())
sys.path.append("/home/maze/project/DeepMimic_mujoco/src_gail")

from pyquaternion import Quaternion
from mujoco.mocap_util import align_position, align_rotation
from mujoco.mocap_util import BIPEDAL_JOINTS_ORDER, BIPEDAL_JOINTS_DOF, BODY_DEFS

from transformations import euler_from_quaternion, quaternion_from_euler

class MocapDM(object):
    def __init__(self):
        self.num_bodies = len(BODY_DEFS)
        self.pos_dim = 3
        self.rot_dim = 4
        self.raw_data = []
        self.multi_clip_data = []

    def load_mocap(self, filepath_list):
        for filepath in filepath_list:
            self.read_data(filepath)
        

    def calc_rot_vel(self, seg_0, seg_1, dura):
        q_0 = Quaternion(seg_0[0], seg_0[1], seg_0[2], seg_0[3])
        q_1 = Quaternion(seg_1[0], seg_1[1], seg_1[2], seg_1[3])

        q_diff =  q_0.conjugate * q_1
        # q_diff =  q_1 * q_0.conjugate
        axis = q_diff.axis
        angle = q_diff.angle
        
        tmp_diff = angle/dura * axis
        diff_angular = [tmp_diff[0], tmp_diff[1], tmp_diff[2]]

        return diff_angular
        
    def read_data(self, filepath):
        curr_clip_data = {}
        all_states = []
        durations = []
        motions = None

        with open(filepath, 'r') as fin:
            data = json.load(fin)
            motions = np.array(data["Frames"])
            m_shape = np.shape(motions)
            raw_data = np.full(m_shape, np.nan)

            total_time = 0.0
            dt = motions[0][0]
            for each_frame in motions:
                duration = each_frame[0]
                durations.append(duration)
                each_frame[0] = total_time
                total_time += duration

                curr_idx = 1 # jump the duration
                offset_idx = 8
                state = {}
                root_pos = align_position(each_frame[curr_idx:curr_idx+3])
                root_quaternion = align_rotation(each_frame[curr_idx+3:offset_idx])
                state['root_rot'] = np.array(euler_from_quaternion(root_quaternion, axes='rxyz')[1:2]) # (rot_around_y)
                torso_pos = root_pos + Quaternion(root_quaternion).rotate(np.array([0,0,0.19]))
                state['root_pos'] = torso_pos[[0,2]] # (root_x,root_z)

                for each_joint in BIPEDAL_JOINTS_ORDER:
                    curr_idx = offset_idx
                    dof = BIPEDAL_JOINTS_DOF[each_joint]
                    if dof == 1:
                        offset_idx += 1
                        state[each_joint] = each_frame[curr_idx:offset_idx]
                    else:
                        raise NotImplementedError()

                all_states.append(state)

        curr_clip_data['dt'] = dt
        curr_clip_data['durations'] = durations

        data_vel = []
        data_config = []

        for k in range(len(all_states)):
            tmp_vel = []
            tmp_pos = []
            state = all_states[k]
            if k == 0:
                dura = durations[k]
            else:
                dura = durations[k-1]

            # time duration
            init_idx = 0
            offset_idx = 1
            raw_data[k, init_idx:offset_idx] = dura

            # root pos
            init_idx = offset_idx
            offset_idx += 2
            raw_data[k, init_idx:offset_idx] = np.array(state['root_pos'])
            if k == 0:
                tmp_vel += [0,0]
            else:
                tmp_vel += ((raw_data[k, init_idx:offset_idx] - raw_data[k-1, init_idx:offset_idx])*1.0/dura).tolist()
            tmp_pos += state['root_pos'].tolist()

            # root rot
            init_idx = offset_idx
            offset_idx += 1
            raw_data[k, init_idx:offset_idx] = np.array(state['root_rot'])
            if k == 0:
                tmp_vel += [0]
            else:
                # tmp_vel += self.calc_rot_vel(raw_data[k, init_idx:offset_idx], raw_data[k-1, init_idx:offset_idx], dura)
                tmp_vel += list((raw_data[k, init_idx:offset_idx] - raw_data[k-1, init_idx:offset_idx])*1/dura)
            tmp_pos += state['root_rot'].tolist()

            for each_joint in BIPEDAL_JOINTS_ORDER:
                init_idx = offset_idx
                tmp_val = state[each_joint]
                if BIPEDAL_JOINTS_DOF[each_joint] == 1:
                    assert 1 == len(tmp_val)
                    offset_idx += 1
                    raw_data[k, init_idx:offset_idx] = state[each_joint]
                    if k == 0:
                        tmp_vel += [0.0]
                    else:
                        tmp_vel += list(((raw_data[k, init_idx:offset_idx] - raw_data[k-1, init_idx:offset_idx])*1.0/dura).tolist())
                    tmp_pos += state[each_joint].tolist()
                elif BIPEDAL_JOINTS_DOF[each_joint] == 3:
                    raise NotImplementedError()

            data_vel.append(np.array(tmp_vel))
            data_config.append(np.array(tmp_pos))
        data_vel[0] = data_vel[len(all_states)-1] # the first frame vel same as the last
        
        curr_clip_data["config"] = data_config
        curr_clip_data["vel"] = data_vel

        self.multi_clip_data.append(curr_clip_data)
        
    def play(self, mocap_filepath):
        from mujoco_py import load_model_from_xml, MjSim, MjViewer

        curr_path = getcwd()
        xmlpath = '/mujoco/bipedal/envs/asset/bipedal_2d_PID.xml'
        with open(curr_path + xmlpath) as fin:
            MODEL_XML = fin.read()

        model = load_model_from_xml(MODEL_XML)
        sim = MjSim(model)
        viewer = MjViewer(sim)

        self.read_data(mocap_filepath)

        from time import sleep

        phase_offset = np.array([0.0, 0.0, 0.0])

        data_vel = self.multi_clip_data[0]["vel"]
        data_config = self.multi_clip_data[0]["config"]
        
        while True:
            print(data_vel)
            for k in range(len(data_vel)):
                tmp_val = data_config[k]
                tmp_vel = data_vel[k]
                sim_state = sim.get_state()
                sim_state.qpos[:] = tmp_val[:]
                sim_state.qvel[:] = tmp_vel[:]
                print(tmp_vel)
                # sim_state.qpos[:3] +=  phase_offset[:]
                sim.set_state(sim_state)
                sim.forward()
                viewer.render()
                # for i in range(int(durations[0]/0.002)):
                #     sim.step()
                #     viewer.render()
                #     print(sim_state.qpos)

            sim_state = sim.get_state()


if __name__ == "__main__":
    test = MocapDM()
    curr_path = getcwd()
    test.play(curr_path + "/mujoco/motions/biped_walk.txt")
