import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mj_envs.utils.quatmath import quat2euler, euler2quat
from mujoco_py import MjViewer
import os
import random
import sys

ADD_BONUS_REWARDS = True

class PenEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        self.target_obj_bid = 0
        self.S_grasp_sid = 0
        self.eps_ball_sid = 0
        self.obj_bid = 0
        self.obj_t_sid = 0
        self.obj_b_sid = 0
        self.tar_t_sid = 0
        self.tar_b_sid = 0
        self.pen_length = 1.0
        self.tar_length = 1.0
        for arg in kwargs.values():
            if arg in ['PID', 'Torque']:
                self.control_mode = arg  
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        if self.control_mode == 'Torque':
            mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_pen.xml', 5)  # frame_skip = 5, here are define frame_skip.
        elif self.control_mode == 'PID':
            mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_pen.xml', 1)  # frame_skip = 1, here are define frame_skip.
        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        utils.EzPickle.__init__(self)
        self.target_obj_bid = self.sim.model.body_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.eps_ball_sid = self.sim.model.site_name2id('eps_ball')
        self.obj_t_sid = self.sim.model.site_name2id('object_top')
        self.obj_b_sid = self.sim.model.site_name2id('object_bottom')
        self.tar_t_sid = self.sim.model.site_name2id('target_top')
        self.tar_b_sid = self.sim.model.site_name2id('target_bottom')

        self.pen_length = np.linalg.norm(self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])
        self.tar_length = np.linalg.norm(self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])

        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])

    def step(self, a):  # a -> action
        a = np.clip(a, -1.0, 1.0)
        try:
            starting_up = False
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            starting_up = True
            a = a                             # only for the initialization phase
        if self.control_mode == 'Torque':
            self.do_simulation(a, self.frame_skip) # control frequency: 100HZ (= planning frequency) / use this to directly output torque
        elif self.control_mode == 'PID':
            self.do_simulation(a, 1) # control frequency: 500HZ / use this to output PD target

        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        desired_loc = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.pen_length
        desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length

        # pos cost
        dist = np.linalg.norm(obj_pos-desired_loc)
        reward = -dist
        # orien cost (dense rewards)
        orien_similarity = np.dot(obj_orien, desired_orien)
        reward += orien_similarity

        if ADD_BONUS_REWARDS:
            # bonus for being close to desired orientation
            if dist < 0.075 and orien_similarity > 0.9:
                reward += 10
            if dist < 0.075 and orien_similarity > 0.95:  # orien_similarity = 0.95 -> angle diff = 18 deg
                reward += 50

        # penalty for dropping the pen
        done = False
        if obj_pos[2] < 0.075:
            reward -= 5
            done = True if not starting_up else False  # early stop when dropping the pen

        goal_achieved = True if (dist < 0.075 and orien_similarity > 0.90) else False
        # goal_achieved = True if (dist < 0.075 and orien_similarity > 0.95) else False
        return self.get_obs(), reward, done, dict(goal_achieved=goal_achieved)

    def get_obs(self):   # obj represented in the world frame(fixed on the table) by obj_pos, obj_orien, desired_orien
        qp = self.data.qpos.ravel()
        obj_vel = self.data.qvel[-6:].ravel()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel() # represented in the world frame
        # print("obj_pos:", obj_pos)
        desired_pos = self.data.site_xpos[self.eps_ball_sid].ravel() # the desired position is fixed for this task, not in the world frame
        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.pen_length 
        # print("self.data.site_xpos[self.obj_t_sid]:", self.data.site_xpos[self.obj_t_sid])
        # print("self.data.site_xpos[self.obj_b_sid]:", self.data.site_xpos[self.obj_b_sid])
        # print((self.data.site_xpos[self.obj_t_sid] + self.data.site_xpos[self.obj_b_sid]) / 2)
        desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length # keep fixed in this task
        # self.data.site_xpos[self.tar_t_sid] and self.data.site_xpos[self.tar_b_sid] are defined in the world frame, which is fixed on the table
        # same for self.data.site_xpos[self.obj_t_sid] and self.data.site_xpos[self.obj_b_sid] (in world frame)
        # in this case, obj_orien and desired_orien are the unit directional vectors defined in the world frame (two point diff/ pen length)
        # we can consider that the desired orien define a new coordinate framework 
        # print("visual pos:", qp[24:27])
        # print("world pos:", obj_pos)
        # print("velosity itegral:", obj_vel[:3] * 0.01)
        # print("visual ori:", qp[27:])
        # print("world ori:", obj_orien)
        return np.concatenate([qp[:-6], obj_pos, obj_vel, obj_orien, desired_orien,
                               obj_pos-desired_pos, obj_orien-desired_orien])

    def get_hand_vel(self):
        hand_vel = self.data.qvel[:-6].ravel()
        return hand_vel

    def get_full_obs_visualization(self):  # this is needed to visualize the trained the policy in a correct manner (I personal don't know in which frame do the object's position and orientation defined)
        qp = self.data.qpos.ravel()
        qv = self.data.qvel.ravel()
        return np.concatenate([qp, qv])

    def reset_model(self):
        # angle_range_upper = 1
        # angle_range_lower = -1
        # within distribution: [-1, 1]
        angle_range_upper = 1
        angle_range_lower = -1  
        # out of distribution: [-1.2, -1] and [1, 1.2]
        qp = self.init_qpos.copy() # zeros
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        desired_orien = np.zeros(3)  # randomly set the desired orientation
        desired_orien[0] = self.np_random.uniform(low=angle_range_lower, high=angle_range_upper)  # random pitch angle
        desired_orien[1] = self.np_random.uniform(low=angle_range_lower, high=angle_range_upper)  # random yaw angle
        # roll angle does not matter
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.sim.forward()
        return self.get_obs(), desired_orien

    def reset_model4Koopman(self, ori, init_pos, init_vel):
        qp = self.init_qpos.copy()
        qp[:24] = np.random.random(24) * init_pos  # randomize the initial pose of the hands
        # qp[24:] = np.random.random(6) * init_pos * 0.2  # random the initial pose of the pen
        # qp = np.random.random(len(qp)) * init_pos
        qv = self.init_qvel.copy()
        qv[:24] = np.random.random(24) * init_vel
        # qv[24:] = np.random.random(6) * init_vel * 0.2 # random the initial pose of the pen
        self.set_state(qp, qv)
        desired_orien = ori
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.sim.forward()
        return self.get_obs()

    def KoopmanVisualize(self, state_dict):
        qp = self.init_qpos.copy()
        qp = state_dict['qpos']
        qv = self.init_qvel.copy()
        qv = state_dict['qvel']
        self.set_state(qp, qv)
        desired_orien = state_dict['desired_orien']
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.sim.forward()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        desired_orien = self.model.body_quat[self.target_obj_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, desired_orien=desired_orien)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        This function can be used to visualize the policy
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        desired_orien = state_dict['desired_orien']
        self.set_state(qp, qv)
        self.model.body_quat[self.target_obj_bid] = desired_orien
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = -45
        self.sim.forward()
        self.viewer.cam.distance = 1.0

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if pen within 15 degrees of target for 20 steps
        for path in paths:
            # if np.sum(path['env_infos']['goal_achieved']) > 20:
            if np.sum(path['env_infos']['goal_achieved']) > 10:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
