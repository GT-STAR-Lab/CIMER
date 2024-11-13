import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer, MjRenderContextOffscreen

import os

ADD_BONUS_REWARDS = True

class RelocateEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        self.target_obj_sid = 0
        self.S_grasp_sid = 0
        self.obj_bid = 0
        self.offset = 0  # set it to be -0.05 for power_drill, for others, use 0
        for arg in kwargs.values():
            if arg in ['PID', 'Torque']:
                self.control_mode = arg  
            else:
                self.object_name = arg  
        try:
            if len(self.object_name) > 0:
                file_name = '/assets/DAPG_relocate_' + self.object_name + '.xml'
            else:
                file_name = '/assets/DAPG_relocate.xml'
        except:
            file_name = '/assets/DAPG_relocate.xml'
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        if self.control_mode == 'Torque':
            mujoco_env.MujocoEnv.__init__(self, curr_dir+file_name, 5)  # frame_skip = 5, here are define frame_skip.
        elif self.control_mode == 'PID':
            mujoco_env.MujocoEnv.__init__(self, curr_dir+file_name, 1)  # frame_skip = 1, here are define frame_skip.
        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        self.target_obj_sid = self.sim.model.site_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        utils.EzPickle.__init__(self)
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
        if self.control_mode == 'Torque':
            self.do_simulation(a, self.frame_skip) # control frequency: 100HZ (= planning frequency) / use this to directly output torque
        elif self.control_mode == 'PID':
            self.do_simulation(a, 1) # control frequency: 500HZ / use this to output PD target
        ob = self.get_obs()  # get observations data
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        # print("from the relocation environment file ",palm_pos,obj_pos,target_pos)
        # print("()palm_pos: ", palm_pos)
        # print("()obj_pos: ", obj_pos)
        reward = -0.1*np.linalg.norm(palm_pos-obj_pos)              # take hand to object
        if obj_pos[2] > 0.04:                                       # if object off the table
            reward += 1.0                                           # bonus for lifting the object
            reward += -0.5*np.linalg.norm(palm_pos-target_pos)      # make hand go to target
            reward += -0.5*np.linalg.norm(obj_pos-target_pos)       # make object go to target
        # the reward functions are very specific to the task

        if ADD_BONUS_REWARDS: # set to be true
            if np.linalg.norm(obj_pos-target_pos) < 0.1:
                reward += 10.0                                          # bonus for object close to target
            if np.linalg.norm(obj_pos-target_pos) < 0.05:
                reward += 20.0                                          # bonus for object "very" close to target
        
        goal_achieved = True if np.linalg.norm(obj_pos-target_pos) < 0.1 else False
        return ob, reward, False, dict(goal_achieved=goal_achieved)  # should the return be goal_achieved instead of False?

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        # get the observation dat for these environments
        qp = self.data.qpos.ravel()  # (36,)   qpos details {'total': 36, 'hand_Txyz': 3, 'hand_Rxyz': 3, 'hand_joints': 24, 'obj_Txyz': 3, 'obj_Rxyz': 3}
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel() # (3,) 
        obj_pos[1] -= self.offset
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel() # (3,) 
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel() # (3,)
        # original version
        # return np.concatenate([qp[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos])  # qp[:-6] 
        # concatenate the obs with obj/palm/target positions 
        return np.concatenate([qp[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos, obj_pos, palm_pos, target_pos])  # qp[:-6] 
        
    # The original version
    # def reset_model(self):
    #     qp = self.init_qpos.copy()
    #     qv = self.init_qvel.copy()
    #     self.set_state(qp, qv)

    #     # Generalized area
    #     # self.model.body_pos[self.obj_bid,0] = self.np_random.uniform(low=-0.3, high=0.3)
    #     # self.model.body_pos[self.obj_bid,1] = self.np_random.uniform(low=-0.3, high=0.3)
    #     # self.model.site_pos[self.target_obj_sid,0] = self.np_random.uniform(low=-0.3, high=0.3)
    #     # self.model.site_pos[self.target_obj_sid,1] = self.np_random.uniform(low=-0.3, high=0.3)
    #     # self.model.site_pos[self.target_obj_sid,2] = self.np_random.uniform(low=0.15, high=0.35)

    #     # Constrained area (Original)
    #     self.model.body_pos[self.obj_bid,0] = self.np_random.uniform(low=-0.15, high=0.15)
    #     self.model.body_pos[self.obj_bid,1] = self.np_random.uniform(low=-0.15, high=0.3)  # some place on the table (z=0)
    #     self.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=-0.2, high=0.2)
    #     self.model.site_pos[self.target_obj_sid,1] = self.np_random.uniform(low=-0.2, high=0.2)
    #     self.model.site_pos[self.target_obj_sid,2] = self.np_random.uniform(low=0.15, high=0.35)
    #     self.sim.forward()
    #     return self.get_obs(), 

    def reset_model(self):
        # within training distribution
        xy_range_upper = 0.25
        xy_range_lower = -0.25
        z_range_upper = 0.35
        z_range_lower = 0.15
        # outof training distribution
        # xy_range_upper = 0.25
        # xy_range_lower = -0.25
        # z_range_upper = 0.40
        # z_range_lower = 0.35
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        # Generalized area
        # self.model.body_pos[self.obj_bid,0] = self.np_random.uniform(low=-0.3, high=0.3)
        # self.model.body_pos[self.obj_bid,1] = self.np_random.uniform(low=-0.3, high=0.3)
        # self.model.site_pos[self.target_obj_sid,0] = self.np_random.uniform(low=-0.3, high=0.3)
        # self.model.site_pos[self.target_obj_sid,1] = self.np_random.uniform(low=-0.3, high=0.3)
        # self.model.site_pos[self.target_obj_sid,2] = self.np_random.uniform(low=0.15, high=0.35)

        # Constrained area (Original)
        # to generate random initial position of the object
        # self.model.body_pos[self.obj_bid,0] = self.np_random.uniform(low=-0.2, high=0.2)
        # self.model.body_pos[self.obj_bid,1] = self.np_random.uniform(low=-0.2, high=0.2)  # some place on the table (z=0)
        self.model.body_pos[self.obj_bid,0] = 0
        self.model.body_pos[self.obj_bid,1] = self.offset
        # target place can be in the air (z != 0)
        desired_pos = np.zeros(3)
        desired_pos[0] = self.np_random.uniform(low=xy_range_lower, high=xy_range_upper)  
        desired_pos[1] = self.np_random.uniform(low=xy_range_lower, high=xy_range_upper)  
        desired_pos[2] = self.np_random.uniform(low=z_range_lower, high=z_range_upper)  
        # to generate fixed goal position of the object
        # desired_pos[2] = 0.3  
        self.model.site_pos[self.target_obj_sid,0] = desired_pos[0]
        self.model.site_pos[self.target_obj_sid,1] = desired_pos[1]
        self.model.site_pos[self.target_obj_sid,2] = desired_pos[2]
        self.sim.forward()
        return self.get_obs(), desired_pos
        
    def KoopmanVisualize(self, state_dict):
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.set_state(qp, qv)
        obj_pos = state_dict['obj_pos']
        target_pos = state_dict['target_pos']
        self.model.body_pos[self.obj_bid] = obj_pos
        self.model.site_pos[self.target_obj_sid] = target_pos
        self.sim.forward()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, target_pos=target_pos, palm_pos=palm_pos,
            qpos=qp, qvel=qv)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        obj_pos = state_dict['obj_pos']
        target_pos = state_dict['target_pos']
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid] = obj_pos
        self.model.site_pos[self.target_obj_sid] = target_pos
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjRenderContextOffscreen(self.sim, -1)
        # self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if object close to target for 25 steps
        for path in paths:
            # if np.sum(path['env_infos']['goal_achieved']) > 25:
            if np.sum(path['env_infos']['goal_achieved']) > 10:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage

    def get_obs_dict(self, sim):
        # qpos for hand, xpos for obj, xpos for target
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_jnt'] = sim.data.qpos[:-6].copy()
        obs_dict['palm_obj_err'] = sim.data.site_xpos[self.S_grasp_sid] - sim.data.body_xpos[self.obj_bid]
        obs_dict['palm_tar_err'] = sim.data.site_xpos[self.S_grasp_sid] - sim.data.site_xpos[self.target_obj_sid]
        obs_dict['obj_tar_err'] = sim.data.body_xpos[self.obj_bid] - sim.data.site_xpos[self.target_obj_sid]
        # keys missing from DAPG-env but needed for rewards calculations
        obs_dict['obj_pos']  = sim.data.body_xpos[self.obj_bid].copy()
        return obs_dict