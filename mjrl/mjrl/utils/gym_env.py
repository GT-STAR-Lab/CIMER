"""
Wrapper around a gym env that provides convenience functions
"""

import gym
from mjrl.policies.ncp_network import NCPNetwork
from mjrl.policies.rnn_network import RNNNetwork
import numpy as np
import pickle
import copy
from mjrl.KODex_utils.coord_trans import ori_transform, ori_transform_inverse
from mjrl.KODex_utils.fc_network import FCNetwork
import torch
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
import pandas as pd

class EnvSpec(object):
    def __init__(self, obs_dim, act_dim, horizon):
        self.observation_dim = obs_dim
        self.action_dim = act_dim
        self.horizon = horizon


class GymEnv(object):
    def __init__(self, env, control_mode, object_name=None, env_kwargs=None,
                 obs_mask=None, act_repeat=1, 
                 *args, **kwargs):
        # get the correct env behavior
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if type(env) == str:
            env = gym.make(env, control_mode = control_mode, object_name = object_name)
    # def __init__(self, env, env_kwargs=None,
    #              obs_mask=None, act_repeat=1, 
    #              *args, **kwargs):
    
    #     # get the correct env behavior
    #     if type(env) == str:
    #         env = gym.make(env)
        
        elif isinstance(env, gym.Env):
            env = env
        elif callable(env):
            env = env(**env_kwargs)
        else:
            print("Unsupported environment format")
            raise AttributeError
        self.env = env
        self.env_id = env.spec.id
        self.act_repeat = act_repeat

        try:
            self._horizon = env.spec.max_episode_steps  # max_episode_steps is defnied in the __init__.py file (under )
        except AttributeError:
            self._horizon = env.spec._horizon
        assert self._horizon % act_repeat == 0
        self._horizon = self._horizon // self.act_repeat

        try:
            self._action_dim = self.env.env.action_dim
        except AttributeError:
            self._action_dim = self.env.action_space.shape[0]

        try:
            self._observation_dim = self.env.env.obs_dim
        except AttributeError:
            self._observation_dim = self.env.observation_space.shape[0]

        # Specs
        self.spec = EnvSpec(self._observation_dim, self._action_dim, self._horizon)

        # obs mask
        self.obs_mask = np.ones(self._observation_dim) if obs_mask is None else obs_mask

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def horizon(self):
        return self._horizon

    # <touch name="ST_Tch_fftip"  site="Tch_fftip"/>
    # <touch name="ST_Tch_mftip"  site="Tch_mftip"/>
    # <touch name="ST_Tch_rftip"  site="Tch_rftip"/>
    # <touch name="ST_Tch_lftip"  site="Tch_lftip"/>
    # <touch name="ST_Tch_thtip"  site="Tch_thtip"/>
    @property
    def contact_forces(self):
        tips_force = {}
        raw_tip_force = self.env.data.sensordata
        tips_force['ff'] = raw_tip_force[0]
        tips_force['mf'] = raw_tip_force[1]
        tips_force['rf'] = raw_tip_force[2]
        tips_force['lf'] = raw_tip_force[3]
        tips_force['th'] = raw_tip_force[4]
        return tips_force

    def reset(self, seed=None):
        try:
            self.env._elapsed_steps = 0
            return self.env.env.reset_model(seed=seed)
        except:
            if seed is not None:
                self.set_seed(seed)
            return self.env.reset()
    
    def reset4Koopman(self, seed=None, ori=None, init_pos=None, init_vel=None):
        try:
            self.env._elapsed_steps = 0
            return self.env.env.reset_model4Koopman(seed=seed, ori = ori, init_pos = init_pos, init_vel = init_vel)
        except:
            if seed is not None:
                self.set_seed(seed)
            return self.env.reset_model4Koopman(ori = ori, init_pos = init_pos, init_vel = init_vel)

    def reset_model(self, seed=None):
        # overloading for legacy code
        return self.reset(seed)

    def step(self, action):
        action = action.clip(self.action_space.low, self.action_space.high)
        # type(action_space) -> <class 'gym.spaces.box.Box'>
        # self.action_space.low -> numpy.ndarray(lowest boundary)
        # self.action_space.high -> numpy.ndarray(highest boundary)
        if self.act_repeat == 1: 
            obs, cum_reward, done, ifo = self.env.step(action)  # the system dynamics is defined in each separate env python file
            # if(ifo['goal_achieved']):
            #     print("done: ", ifo)    
            # Run one timestep of the environment’s dynamics.
        else:
            cum_reward = 0.0
            for i in range(self.act_repeat):
                obs, reward, done, ifo = self.env.step(action) # the actual operations can be found in the env files
                # seems done is always set to be False
                cum_reward += reward
                if done: break
        return self.obs_mask * obs, cum_reward, done, ifo

    def render(self):
        try:
            self.env.env.mujoco_render_frames = True
            self.env.env.mj_render()
        except:
            self.env.render()

    def set_seed(self, seed=123):
        try:
            self.env.seed(seed)
        except AttributeError:
            self.env._seed(seed)

    def get_obs(self):
        try:
            return self.obs_mask * self.env.env.get_obs()
        except:
            return self.obs_mask * self.env.env._get_obs()

    def get_env_infos(self):
        try:
            return self.env.env.get_env_infos()
        except:
            return {}

    # ===========================================
    # Trajectory optimization related
    # Envs should support these functions in case of trajopt

    def get_env_state(self):
        try:
            return self.env.env.get_env_state()
        except:
            raise NotImplementedError

    def set_env_state(self, state_dict):
        try:
            self.env.env.set_env_state(state_dict)
        except:
            raise NotImplementedError

    def real_env_step(self, bool_val):
        try:
            self.env.env.real_step = bool_val
        except:
            raise NotImplementedError

    # ===========================================

    def visualize_policy(self, policy, policy_name, ori, init_pos, init_vel, horizon=1000, num_episodes=1, mode='exploration', **kwargs):  #test by generating the new observation data
        task = self.env_id.split('-')[0]
        success_threshold = 20 if task == 'pen' else 25
        episodes = []
        total_score = 0.0
        success_count = 0
        for ep in range(num_episodes):
            print("Episode %d" % ep)
            # ori -> desired orientation
            # o = self.reset4Koopman(seed = None, ori = ori[index], init_pos = init_pos, init_vel = init_vel) # codes for generating the demo data with the same goal orientation(defined in ori)
            # print("desired orientation:", o[36:39])
            # print("yaw:", np.arctan(o[36]/o[38]) / np.pi) # yaw
            # print("pitch:", np.arctan(o[37]/o[38]) / np.pi) # pitch]
            o, desired_orien = self.reset(seed = ep)
            hand_vel = self.env.get_hand_vel()
            observations_visualization = self.env.get_full_obs_visualization() # can be used for visualization but objects are defined in a weird frame
            d = False 
            t = 0
            score = 0.0
            episode_data = {
                'init_state_dict': copy.deepcopy(self.get_env_state()),  # set the initial states/the desired orientation is represented in quat
                'actions': [],
                'observations': [],
                'observations_visualization': [],
                'handVelocity': [],
                'rewards': [],
                'goal_achieved': []
            }
            if isinstance(policy.model, NCPNetwork):
                hidden_state = np.zeros((1, policy.model.rnn_cell.state_size))
            if isinstance(policy.model, RNNNetwork):
                hidden_state = (np.zeros((1, 1, policy.model.rnn_cell.hidden_size)), np.zeros((1, 1, policy.model.rnn_cell.hidden_size)))
            episode_data['desired_orien'] = desired_orien  # record the goal orientation angle
            while t < horizon and d is False:
                episode_data['handVelocity'].append(hand_vel)
                episode_data['observations'].append(o)
                episode_data['observations_visualization'].append(observations_visualization)
                if isinstance(policy.model, NCPNetwork):
                    a = policy.get_action(o, hidden_state)
                    hidden_state = a[1]['hidden_state']
                elif isinstance(policy.model, RNNNetwork):
                    a = policy.get_action(o, hidden_state)
                    hidden_state = a[1]['hidden_state']
                else:
                    a = policy.get_action(o)
                a = a[0] if mode == 'exploration' else a[1]['evaluation']
                o, r, d, goal_achieved = self.step(a)  # need the action to simulate the process
                hand_vel = self.env.get_hand_vel()
                observations_visualization = self.env.get_full_obs_visualization()
                episode_data['actions'].append(a)
                episode_data['rewards'].append(r)
                episode_data['goal_achieved'].append(goal_achieved['goal_achieved'])
                score = score + r
                if (not kwargs['record']):
                    self.render()
                t = t+1
            episodes.append(copy.deepcopy(episode_data))
            total_score += score
            print("Episode score = %f, Success = %d" % (score, sum(episode_data['goal_achieved']) > success_threshold))
            if sum(episode_data['goal_achieved']) > success_threshold:
                success_count += 1
            if success_count == 1000:
                break
        print("Average score = %f" % (total_score / num_episodes))
        successful_episodes = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes))
        # if task == 'relocate':
        #     successful_episodes = list(filter(lambda episode: episode['goal_achieved'][-1], successful_episodes))
        print("Success rate = %f" % (len(successful_episodes) / len(episodes)))
        print(len(successful_episodes))
        if (kwargs['record']):
            pickle.dump(successful_episodes, open('/home/yhan389/Desktop/KoopmanManipulation_DataSplit/demo_data/Pen_more_demo/' + policy_name, 'wb'))

    def visualize_policy_on_demos(self, policy, demos, horizon=1000, num_episodes=1, dapg_policy = True, mode='exploration', Visualize = False, object_name = '', **kwargs):
        task = self.env_id.split('-')[0]
        if task == 'door' or task == 'hammer':
            success_threshold = 1
        elif task == 'pen' or task == 'relocate':
            success_threshold = 10
        episodes = []
        total_score = 0.0
        save_path = os.path.join(os.getcwd(), 'Videos', task, object_name)[:-1] + '_DAPG.mp4'
        # save_traj = 50  # used for the default object on each task
        save_traj = 20  # used for the new objects on the relocation task
        for idx in range(len(demos)):
            print("Episode %d" % idx)
            self.reset()
            self.set_env_state(demos[idx]['init_state_dict'])
            o = self.get_obs()
            d = False
            t = 0
            score = 0.0
            episode_data = {
                'goal_achieved': []
            }
            isRNN = False
            if isinstance(policy.model, RNNNetwork):
                # generate the hidden states at time 0
                # hidden_state = (np.zeros((1, 1, policy.model.rnn_cell.hidden_size)), np.zeros((1, 1, policy.model.rnn_cell.hidden_size)))
                hidden_state = (  # h_{0} and c_{0}
                    np.zeros((1, 1, policy.model.rnn_cell.hidden_size)),
                    np.zeros((1, 1, policy.model.rnn_cell.hidden_size))
                )
                isRNN = True
            if isinstance(policy.model, NCPNetwork):
                hidden_state = np.zeros((1, policy.model.rnn_cell.state_size))
                isRNN = True
            while t < horizon and d is False:
                if isRNN:
                    a = policy.get_action(o, hidden_state)
                    a, hidden_state = (a[0], a[1]['hidden_state']) if mode == 'exploration' else (a[1]['evaluation'], a[1]['hidden_state'])
                else:
                    if task == 'relocate' and dapg_policy:
                        a = policy.get_action(o[:-9])[0] if mode == 'exploration' else policy.get_action(o[:-9])[1]['evaluation']
                    else:
                        a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                o, r, d, goal_achieved = self.step(a)
                episode_data['goal_achieved'].append(goal_achieved['goal_achieved'])
                score = score + r
                if Visualize:
                    self.render()
                t = t+1
            episodes.append(copy.deepcopy(episode_data))
            total_score += score
            # print("Episode score = %f, Success = %d" % (score, sum(episode_data['goal_achieved']) > success_threshold and episode_data['goal_achieved'][-1]))
            print("Episode score = %f, Success = %d" % (score, sum(episode_data['goal_achieved']) > success_threshold))
        print("Average score = %f" % (total_score / len(demos)))
        successful_episodes = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes))
        # if task == 'relocate':
        # successful_episodes = list(filter(lambda episode: episode['goal_achieved'][-1], successful_episodes))
        print("Success rate = %f" % (len(successful_episodes) / len(demos)))

    def evaluate_trained_mean_policy_with_torque(self, policy, demos, horizon=1000, dapg_policy=True, num_episodes=1, **kwargs):
        task = self.env_id.split('-')[0]
        if task == 'door' or task == 'hammer':
            success_threshold = 1
        elif task == 'pen' or task == 'relocate':
            success_threshold = 10
        episodes = []
        total_score = 0.0
        for idx in range(len(demos)):
            self.reset()
            self.set_env_state(demos[idx]['init_state_dict'])
            o = self.get_obs()
            d = False
            t = 0
            score = 0.0
            episode_data = {
                'goal_achieved': []
            }
            isRNN = False
            if isinstance(policy.model, RNNNetwork):
                # generate the hidden states at time 0
                # hidden_state = (np.zeros((1, 1, policy.model.rnn_cell.hidden_size)), np.zeros((1, 1, policy.model.rnn_cell.hidden_size)))
                hidden_state = (  # h_{0} and c_{0}
                    np.zeros((1, 1, policy.model.rnn_cell.hidden_size)),
                    np.zeros((1, 1, policy.model.rnn_cell.hidden_size))
                )
                isRNN = True
            if isinstance(policy.model, NCPNetwork):
                hidden_state = np.zeros((1, policy.model.rnn_cell.state_size))
                isRNN = True
            while t < horizon and d is False:
                if isRNN:
                    a = policy.get_action(o, hidden_state)
                    a, hidden_state = (a[1]['evaluation'], a[1]['hidden_state'])
                else:
                    if task == 'relocate' and dapg_policy:
                        a = policy.get_action(o[:-9])[1]['evaluation']
                    else:
                        a = policy.get_action(o)[1]['evaluation']
                o, r, d, goal_achieved = self.step(a)
                episode_data['goal_achieved'].append(goal_achieved['goal_achieved'])
                score = score + r
                t = t+1
            episodes.append(copy.deepcopy(episode_data))
            total_score += score
            # print("Episode score = %f, Success = %d" % (score, sum(episode_data['goal_achieved']) > success_threshold and episode_data['goal_achieved'][-1]))
            # print("Episode score = %f, Success = %d" % (score, sum(episode_data['goal_achieved']) > success_threshold))
        # print("Average score = %f" % (total_score / len(demos)))
        successful_episodes = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes))
        # if task == 'relocate':
        # successful_episodes = list(filter(lambda episode: episode['goal_achieved'][-1], successful_episodes))
        return len(successful_episodes) / len(demos)

    def evaluate_policy(self, resnet_model, 
                        Eval_data,
                        PID_controller,
                        coeffcients,
                        Koopman_obser,
                        KODex, 
                        task_horizon, 
                        future_state,
                        history_state,
                        policy,
                        num_episodes=5,
                        obj_dynamics=True,
                        gamma=1,
                        percentile=[],
                        get_full_dist=False,
                        terminate_at_done=True,
                        seed=123):
        print("Begin evalauting current policy!")
        self.set_seed(seed)
        task = self.env_id.split('-')[0]
        num_future_s = len(future_state)
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)
        tracking_rewards = np.zeros(num_episodes)
        episodes = []
        total_score = 0.0
        if task == 'pen':
            success_threshold = 10
            # success_threshold = 20 if task == 'pen' else 25
            for ep in tqdm(range(num_episodes)):
                episode_data = {
                    'goal_achieved': []
                }
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                init_hand_state = Eval_data[ep]['handpos']
                init_objpos = Eval_data[ep]['objpos']
                init_objvel = Eval_data[ep]['objvel']
                init_objorient_world = Eval_data[ep]['objorient']
                desired_ori = Eval_data[ep]['desired_ori']
                init_objorient = ori_transform(init_objorient_world, desired_ori) 
                hand_OriState = init_hand_state
                obj_OriState = np.append(init_objpos, np.append(init_objorient, init_objvel))  # ori: represented in the transformed frame
                obj_OriState_ = np.append(init_objpos, np.append(init_objorient_world, init_objvel)) # ori: represented in the original frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                hand_states_traj = np.zeros([task_horizon, num_hand])
                object_states_traj = np.zeros([task_horizon, num_obj])
                hand_states_traj[0, :] = hand_OriState
                object_states_traj[0, :] = obj_OriState_
                z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                for t_ in range(task_horizon - 1):
                    
                    z_t_1_computed = np.dot(KODex, z_t)
                    z_t = z_t_1_computed.copy()
                    x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                    hand_OriState = x_t_1_computed[:num_hand]
                    obj_pos = x_t_1_computed[num_hand: num_hand + 3]
                    # x_t_1_computed[num_hand + 3: num_hand + 6] -> ori: represented in the transformed frame
                    obj_ori = ori_transform_inverse(x_t_1_computed[num_hand + 3: num_hand + 6], desired_ori) # ori: represented in the original frame
                    obj_vel = x_t_1_computed[num_hand + 6: num_hand + 12]
                    obj_OriState = np.append(obj_pos, np.append(obj_ori, obj_vel))
                    hand_states_traj[t_ + 1, :] = hand_OriState
                    object_states_traj[t_ + 1, :] = obj_OriState
                    # Visualize the KODex results, to check the given reference motion if good or not. 
                    if self.env.control_mode == 'PID':  # if Torque mode, we skil the visualization of PD controller.
                        PID_controller.set_goal(hand_OriState)
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            next_o, r, done, goal_achieved = self.step(torque_action)             
                # print("desired ori:", desired_ori)
                # print("final Koopman ori:", obj_ori)
                # print("similarity:", np.dot(desired_ori, obj_ori) / np.linalg.norm(obj_ori))  # koopman ori may not be a perfect unit vector.
                
                # re-set the experiments, as we finish visualizing the reference motion of KODex
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                o = self.get_obs()
                t, done, obj_height = 0, False, o[26]
                # while t < task_horizon and (done == False or terminate_at_done == False):
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                for i in range(history_state):
                    if obj_dynamics:
                        prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                    else:
                        prev_states[i] = hand_states_traj[0]
                prev_actions = dict()
                for i in range(history_state + 1):
                    prev_actions[i] = hand_states_traj[0]
                while t < task_horizon - 1 and obj_height > 0.15:  # early-terminate when the object falls off
                    o = self.get_obs()
                    current_hand_state = o[:24]
                    current_objpos = o[24:27]
                    current_objvel = o[27:33]
                    current_objori = o[33:36]
                    hand_OriState = current_hand_state
                    obj_OriState = np.append(current_objpos, np.append(current_objori, current_objvel))  # ori: represented in the transformed frame
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= task_horizon:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                else:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= task_horizon:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                else:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= task_horizon:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                else:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= task_horizon:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                else:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[1]['evaluation'] # mean_action is True -> noise-free actions from controller
                    if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                        a += hand_states_traj[t + 1].copy() 
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        PID_controller.set_goal(a)
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:len(a)], self.get_env_state()['qvel'][:len(a)])
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    obj_height = next_o[26]
                    episode_data['goal_achieved'].append(goal_achieved['goal_achieved'])
                    ep_returns[ep] += (gamma ** t) * r  # only task specific rewards
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t + 1]
                    reference['obj_pos'] = object_states_traj[t + 1][:3]
                    reference['obj_vel'] = object_states_traj[t + 1][6:]
                    reference['obj_ori'] = object_states_traj[t + 1][3:6]
                    obs = dict()
                    obs['hand_state'] = next_o[:24]
                    obs['obj_pos'] = next_o[24:27]
                    obs['obj_vel'] = next_o[27:33]
                    obs['obj_ori'] = next_o[33:36]
                    tracking_reward = Comp_tracking_reward_pen(reference, obs, coeffcients)['total_reward']
                    tracking_rewards[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                    t += 1    
                # print("Task reward = %f, Tracking reward = %f, Success = %d" % (ep_returns[ep], tracking_rewards[ep], sum(episode_data['goal_achieved']) > success_threshold))
                episodes.append(copy.deepcopy(episode_data))
                total_score += ep_returns[ep]
                # asd
            successful_episodes = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes))
            print("Average score = %f" % (total_score / num_episodes))
            print("Success rate = %f" % (len(successful_episodes) / len(episodes)))
        elif task == 'relocate':
            success_threshold = 10 
            success_list_sim = []
            for ep in tqdm(range(num_episodes)):
                episode_data = {
                    'goal_achieved': []
                }
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                
                init_hand_state = Eval_data[ep]['handpos']
                # init_objpos = Eval_data[ep]['objpos'] # converged object position
                # init_objvel = Eval_data[ep]['objvel']
                # init_objori = Eval_data[ep]['objorient']
                desired_pos_main = Eval_data[ep]['desired_pos']
                # init_objpos_world = desired_pos + init_objpos # in the world frame(on the table)
                hand_OriState = init_hand_state
                # rgb, depth = self.env.env.mj_render()
                # rgb = (rgb.astype(np.uint8) - 128.0) / 128
                # depth = depth[...,np.newaxis]
                # rgbd = np.concatenate((rgb,depth),axis=2)
                # rgbd = np.transpose(rgbd, (2, 0, 1))
                # rgbd = rgbd[np.newaxis, ...]
                # rgbd = torch.from_numpy(rgbd).float().to(self.device)
                # # desired_pos = Test_data[k][0]['init']['target_pos']
                # desired_pos = desired_pos_main[np.newaxis, ...]
                # desired_pos = torch.from_numpy(desired_pos).float().to(self.device)
                # implict_objpos = resnet_model(rgbd, desired_pos) 
                # obj_OriState = implict_objpos[0].cpu().detach().numpy()
                obj_OriState = Eval_data[ep]['obj_features']
                # obj_OriState = np.append(init_objpos, np.append(init_objori, init_objvel))  # ori: represented in the transformed frame (converged to desired pos)
                # obj_OriState_ = np.append(init_objpos_world, np.append(init_objori, init_objvel)) # ori: represented in the world frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                hand_states_traj = np.zeros([task_horizon, num_hand])
                object_states_traj = np.zeros([task_horizon, num_obj])
                hand_states_traj[0, :] = hand_OriState
                object_states_traj[0, :] = obj_OriState
                z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                success_count_sim = np.zeros(task_horizon)
                rollout_temp=[]
                for t_ in range(task_horizon - 1):
                    rollout_temp.append(z_t)
                    if t_%10==0:
                        plt.imshow(self.env.mj_render()[0])
                        plt.savefig(f"/home/pratik/Desktop/mjrl_repo/CIMER_KOROL/CIMER/hand_dapg/dapg/controller_training/output{t_}.jpg")
                    z_t_1_computed = np.dot(KODex, z_t)
                    z_t = z_t_1_computed.copy()
                    x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                    hand_OriState = x_t_1_computed[:num_hand]
                    obj_OriState = x_t_1_computed[num_hand:]
                    # obj_pos = x_t_1_computed[num_hand: num_hand + 3] # converged object position
                    # obj_pos_world = desired_pos + obj_pos
                    # obj_ori = x_t_1_computed[num_hand + 3: num_hand + 6]
                    # obj_vel = x_t_1_computed[num_hand + 6: num_hand + 12]
                    # obj_OriState = np.append(obj_pos_world, np.append(obj_ori, obj_vel))
                    # rgb, depth = self.env.env.mj_render()
                    # rgb = (rgb.astype(np.uint8) - 128.0) / 128
                    # depth = depth[...,np.newaxis]
                    # rgbd = np.concatenate((rgb,depth),axis=2)
                    # rgbd = np.transpose(rgbd, (2, 0, 1))
                    # rgbd = rgbd[np.newaxis, ...]
                    # rgbd = torch.from_numpy(rgbd).float().to(self.device)
                    # # desired_pos = Test_data[k][0]['init']['target_pos']
                    # desired_pos = desired_pos_main[np.newaxis, ...]
                    # desired_pos = torch.from_numpy(desired_pos).float().to(self.device)
                    # implict_objpos = resnet_model(rgbd, desired_pos) 
                    # obj_OriState = implict_objpos[0].cpu().detach().numpy()
                    hand_states_traj[t_ + 1, :] = hand_OriState
                    object_states_traj[t_ + 1, :] = obj_OriState
                    if self.env.control_mode == 'PID':  # if Torque mode, we skil the visualization of PD controller.
                        PID_controller.set_goal(hand_OriState)
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            # for relocation task, it we set a higher control frequency, we can expect a much better PD performance
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            torque_action[1] -= 0.95  # hand_Txyz[1] -> hand_T_y
                            # print('1', hand_OriState)
                            next_o, r, done, goal_achieved = self.step(torque_action)    
                    err = self.env.get_obs_dict(self.env.sim)['obj_tar_err']

                # if (k % 10 == 0 and t % 5 ==0):
                #     print(f"desired_pos {desired_pos}, obj_pos {obj_pos}, err {err}")
                    if np.linalg.norm(err) < 0.1:
                        success_count_sim[t_] = 1
                print(sum(success_count_sim))
                if sum(success_count_sim) > success_threshold:
                    print(f"success in {ep}")
                    success_list_sim.append(1)
                df=pd.DataFrame(rollout_temp)
                df.to_csv("/home/pratik/Desktop/mjrl_repo/CIMER_KOROL/CIMER/hand_dapg/dapg/controller_training/output.csv")
                print('----------------------------------------------------------------------------------------------------------')
                # re-set the experiments, as we finish visualizing the reference motion of KODex
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                o = self.get_obs()
                t, done, obj_height = 0, False, o[41]
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                if not policy.freeze_base:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0]
                else:
                    for i in range(history_state):
                        if obj_dynamics:
                            if policy.m == 24:
                                prev_states[i] = np.append(hand_states_traj[0][6:30], object_states_traj[0]) 
                            elif policy.m == 27:
                                prev_states[i] = np.append(hand_states_traj[0][3:30], object_states_traj[0]) 
                        else:
                            if policy.m == 24:
                                prev_states[i] = hand_states_traj[0][6:30]
                            elif policy.m == 27:
                                prev_states[i] = hand_states_traj[0][3:30]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        if policy.m == 24:
                            prev_actions[i] = hand_states_traj[0][6:30]
                        elif policy.m == 27:
                            prev_actions[i] = hand_states_traj[0][3:30]
                while t < task_horizon - 1 and obj_height > -0.05:  # what would be early-termination for relocation task?
                    o = self.get_obs()
                    if not policy.freeze_base:
                        current_hand_state = o[:30]
                    else:
                        if policy.m == 24:
                            current_hand_state = o[6:30]
                            num_hand = 24
                        elif policy.m == 27:
                            current_hand_state = o[3:30]
                            num_hand = 27
                    # current_objpos = o[39:42]  # in world frame
                    # current_objvel = self.get_env_state()['qvel'][30:36]
                    # current_objori = self.get_env_state()['qpos'][33:36]
                    hand_OriState = current_hand_state
                    rgb, depth = self.env.env.mj_render()
                    rgb = (rgb.astype(np.uint8) - 128.0) / 128
                    depth = depth[...,np.newaxis]
                    rgbd = np.concatenate((rgb,depth),axis=2)
                    rgbd = np.transpose(rgbd, (2, 0, 1))
                    rgbd = rgbd[np.newaxis, ...]
                    rgbd = torch.from_numpy(rgbd).float().to(self.device)
                    # desired_pos = Test_data[k][0]['init']['target_pos']
                    desired_pos = desired_pos_main[np.newaxis, ...]
                    desired_pos = torch.from_numpy(desired_pos).float().to(self.device)
                    implict_objpos = resnet_model(rgbd, desired_pos) 
                    obj_OriState = implict_objpos[0].cpu().detach().numpy()
                    # obj_OriState = np.append(current_objpos, np.append(current_objori, current_objvel))  # ori: represented in the transformed frame
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][6:30], object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:30], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][6:30], object_states_traj[t + future_state[t_]])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][3:30], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][6:30]
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:30]
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][6:30]
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][3:30]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][6:30], object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:30], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][6:30], object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][3:30], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][6:30]
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:30]
                                    else:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][6:30]
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][3:30]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[1]['evaluation'] # mean_action is True -> noise-free actions from controller
                    if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                        if not policy.freeze_base:
                            a += hand_states_traj[t + 1].copy() 
                        else:
                            if policy.m == 24:
                                a += hand_states_traj[t + 1][6:30].copy() 
                            elif policy.m == 27:
                                a += hand_states_traj[t + 1][3:30].copy() 
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        if not policy.freeze_base:
                            PID_controller.set_goal(a)
                        else:
                            if policy.m == 24:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:6], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:6], a))
                            elif policy.m == 27:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:3], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:3], a))
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            torque_action[1] -= 0.95  # hand_Txyz[1] -> hand_T_y
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    obj_height = next_o[41]
                    episode_data['goal_achieved'].append(goal_achieved['goal_achieved'])
                    ep_returns[ep] += (gamma ** t) * r  # only task specific rewards
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t + 1]
                    reference['obj_feature'] = object_states_traj[t + 1]
                    # reference['obj_vel'] = object_states_traj[t + 1][6:]
                    # reference['obj_ori'] = object_states_traj[t + 1][3:6]
                    obs = dict()
                    obs['hand_state'] = next_o[:30]
                    rgb, depth = self.env.env.mj_render()
                    rgb = (rgb.astype(np.uint8) - 128.0) / 128
                    depth = depth[...,np.newaxis]
                    rgbd = np.concatenate((rgb,depth),axis=2)
                    rgbd = np.transpose(rgbd, (2, 0, 1))
                    rgbd = rgbd[np.newaxis, ...]
                    rgbd = torch.from_numpy(rgbd).float().to(self.device)
                    # desired_pos = Test_data[k][0]['init']['target_pos']
                    desired_pos = desired_pos_main[np.newaxis, ...]
                    desired_pos = torch.from_numpy(desired_pos).float().to(self.device)
                    implict_objpos = resnet_model(rgbd, desired_pos) 
                    # obj_OriState = implict_objpos[0].cpu().detach().numpy()
                    obs['obj_feature']= implict_objpos[0].cpu().detach().numpy()
                    # obs['obj_pos'] = next_o[39:42]
                    # obs['obj_vel'] = self.get_env_state()['qvel'][30:36]
                    # obs['obj_ori'] = self.get_env_state()['qpos'][33:36]
                    # if not policy.freeze_base:
                    #     tracking_reward = Comp_tracking_reward_relocate(reference, obs, coeffcients)['total_reward']
                    # else:
                    #     if not policy.include_Rots: # only on fingers
                    #         tracking_reward = Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['total_reward']
                    #     else: # include rotation angles  
                    #         tracking_reward = Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['total_reward']
                    tracking_reward = Comp_tracking_reward_relocate(reference, obs, coeffcients)['total_reward']
                    tracking_rewards[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                    t += 1   
                episodes.append(copy.deepcopy(episode_data))  
                total_score += ep_returns[ep]
            successful_episodes = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes))
            print("Average score = %f" % (total_score / num_episodes))
            print("Success rate = %f" % (len(successful_episodes) / len(episodes))) 
            print("Success rate (sim) = %f" % (len(success_list_sim) / num_episodes)) 
        elif task == 'door':
            success_list_sim = []
            for ep in tqdm(range(num_episodes)):
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                init_hand_state = Eval_data[ep]['handpos']
                init_objpos = Eval_data[ep]['objpos']
                init_objvel = Eval_data[ep]['objvel']
                init_handle = Eval_data[ep]['handle_init']
                hand_OriState = init_hand_state
                obj_OriState = np.append(init_objpos, np.append(init_objvel, init_handle))  # ori: represented in the transformed frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                hand_states_traj = np.zeros([task_horizon, num_hand])
                object_states_traj = np.zeros([task_horizon, num_obj])
                hand_states_traj[0, :] = hand_OriState
                object_states_traj[0, :] = obj_OriState
                z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                for t_ in range(task_horizon - 1):
                    z_t_1_computed = np.dot(KODex, z_t)
                    z_t = z_t_1_computed.copy()
                    x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                    hand_OriState = x_t_1_computed[:num_hand]
                    obj_pos = x_t_1_computed[num_hand: num_hand + 3] # converged object position
                    obj_vel = x_t_1_computed[num_hand + 3: num_hand + 4]
                    init_handle = x_t_1_computed[num_hand + 4: num_hand + 7]
                    obj_OriState = np.append(obj_pos, np.append(obj_vel, init_handle))
                    hand_states_traj[t_ + 1, :] = hand_OriState
                    object_states_traj[t_ + 1, :] = obj_OriState
                    if self.env.control_mode == 'PID':  # if Torque mode, we skil the visualization of PD controller.
                        PID_controller.set_goal(hand_OriState)
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            # for relocation task, it we set a higher control frequency, we can expect a much better PD performance
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            next_o, r, done, goal_achieved = self.step(torque_action)    
                # re-set the experiments, as we finish visualizing the reference motion of KODex
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                o = self.get_obs()
                t, done = 0, False
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                if not policy.freeze_base:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0]
                else:
                    for i in range(history_state):
                        if obj_dynamics:
                            if policy.m == 24:
                                prev_states[i] = np.append(hand_states_traj[0][4:28], object_states_traj[0]) 
                            elif policy.m == 25:
                                prev_states[i] = np.append(hand_states_traj[0][3:28], object_states_traj[0]) 
                            elif policy.m == 26:
                                prev_states[i] = np.append(np.append(hand_states_traj[0][0], hand_states_traj[0][3:28]), object_states_traj[0]) 
                            elif policy.m == 27:
                                prev_states[i] = np.append(hand_states_traj[0][1:28], object_states_traj[0]) 
                        else:
                            if policy.m == 24:
                                prev_states[i] = hand_states_traj[0][4:28]
                            elif policy.m == 25:
                                prev_states[i] = hand_states_traj[0][3:28]
                            elif policy.m == 26:
                                prev_states[i] = np.append(hand_states_traj[0][0], hand_states_traj[0][3:28])
                            elif policy.m == 27:
                                prev_states[i] = hand_states_traj[0][1:28]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        if policy.m == 24:
                            prev_actions[i] = hand_states_traj[0][4:28]
                        elif policy.m == 25:
                            prev_actions[i] = hand_states_traj[0][3:28]
                        elif policy.m == 26:
                            prev_actions[i] = np.append(hand_states_traj[0][0], hand_states_traj[0][3:28])
                        elif policy.m == 27:
                            prev_actions[i] = hand_states_traj[0][1:28]
                while t < task_horizon - 1:  # what would be early-termination for door task?
                    o = self.get_obs()
                    if not policy.freeze_base:
                        current_hand_state = self.get_env_state()['qpos'][:num_hand]
                    else:
                        if policy.m == 24:
                            current_hand_state = self.get_env_state()['qpos'][4:28]
                            num_hand = 24
                        elif policy.m == 25:
                            current_hand_state = self.get_env_state()['qpos'][3:28]
                            num_hand = 25
                        elif policy.m == 26:
                            current_hand_state = np.append(self.get_env_state()['qpos'][0], self.get_env_state()['qpos'][3:28])
                            num_hand = 26
                        elif policy.m == 27:  # include all rotations
                            current_hand_state = self.get_env_state()['qpos'][1:28]
                            num_hand = 27
                    current_objpos = o[32:35]  # in world frame
                    current_objvel = self.get_env_state()['qvel'][28:29]
                    init_hand_state = self.get_env_state()['door_body_pos']
                    hand_OriState = current_hand_state
                    obj_OriState = np.append(current_objpos, np.append(current_objvel, init_hand_state)) 
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][4:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 25:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 26:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28]), object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][1:28], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][4:28], object_states_traj[t + future_state[t_]])
                                        elif policy.m == 25:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][3:28], object_states_traj[t + future_state[t_]])
                                        elif policy.m == 26:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[t + future_state[t_]][0], hand_states_traj[t + future_state[t_]][3:28]), object_states_traj[t + future_state[t_]])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][1:28], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][4:28]
                                        elif policy.m == 25:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:28]
                                        elif policy.m == 26:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28])
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][1:28]
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][4:28]
                                        elif policy.m == 25:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][3:28]
                                        elif policy.m == 26:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][0], hand_states_traj[t + future_state[t_]][3:28])
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][1:28]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][4:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 25:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 26:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28]), object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][1:28], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][4:28], object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 25:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][3:28], object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 26:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[t + future_state[t_ - 1]][0], hand_states_traj[t + future_state[t_ - 1]][3:28]), object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][1:28], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][4:28]
                                        elif policy.m == 25:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:28]
                                        elif policy.m == 26:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28])
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][1:28]
                                    else:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][4:28]
                                        elif policy.m == 25:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][3:28]
                                        elif policy.m == 26:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][0], hand_states_traj[t + future_state[t_ - 1]][3:28])
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][1:28]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[1]['evaluation'] # mean_action is True -> noise-free actions from controller
                    if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                        if not policy.freeze_base:
                            a += hand_states_traj[t + 1].copy() 
                        else:
                            if policy.m == 24:
                                a += hand_states_traj[t + 1][4:28].copy() 
                            elif policy.m == 25:
                                a += hand_states_traj[t + 1][3:28].copy() 
                            elif policy.m == 26:
                                a += np.append(hand_states_traj[t + 1][0].copy(), hand_states_traj[t + 1][3:28].copy())
                            elif policy.m == 27:
                                a += hand_states_traj[t + 1][1:28].copy() 
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        if not policy.freeze_base:
                            PID_controller.set_goal(a)
                        else:
                            if policy.m == 24:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:4], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:4], a))
                            elif policy.m == 25:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:3], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:3], a))
                            elif policy.m == 26:
                                PID_controller.set_goal(np.append(np.append(a[0], hand_states_traj[t + 1][1:3]), a[1:]))
                                num_hand = len(np.append(np.append(a[0], hand_states_traj[t + 1][1:3]), a[1:]))
                            elif policy.m == 27:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:1], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:1], a))
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    ep_returns[ep] += (gamma ** t) * r  # only task specific rewards
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t + 1]
                    reference['obj_pos'] = object_states_traj[t + 1][:3]
                    reference['obj_vel'] = object_states_traj[t + 1][3:4]
                    reference['init_handle'] = object_states_traj[t + 1][4:]
                    obs = dict()
                    obs['hand_state'] = self.get_env_state()['qpos'][:num_hand]
                    obs['obj_pos'] = next_o[32:35]
                    obs['obj_vel'] = self.get_env_state()['qvel'][28:29]
                    obs['init_handle'] = self.get_env_state()['door_body_pos']
                    if not policy.freeze_base:
                        tracking_reward = Comp_tracking_reward_door(reference, obs, coeffcients)['total_reward']  
                    else:
                        if not policy.include_Rots:  # only on fingers
                            tracking_reward = Comp_tracking_reward_door_freeze_base(reference, obs, coeffcients)['total_reward']  
                        else: # include rotation angles
                            tracking_reward = Comp_tracking_reward_door_include_Rots(reference, obs, coeffcients)['total_reward']  
                    tracking_rewards[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                    t += 1   
                    current_hinge_pos = next_o[28:29] # door opening angle
                if current_hinge_pos > 1.35:
                    success_list_sim.append(1)
                total_score += ep_returns[ep]
            print("Average score = %f" % (total_score / num_episodes))
            print("Success rate = %f" % (len(success_list_sim) / num_episodes)) 
        elif task == 'hammer':
            success_list_sim = []
            for ep in tqdm(range(num_episodes)):
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                init_hand_state = Eval_data[ep]['handpos']
                init_objpos = Eval_data[ep]['objpos']
                init_objori = Eval_data[ep]['objorient']
                init_objvel = Eval_data[ep]['objvel']
                goal_nail = Eval_data[ep]['nail_goal']
                hand_OriState = init_hand_state
                obj_OriState = np.append(init_objpos, np.append(init_objori, np.append(init_objvel, goal_nail)))  # ori: represented in the transformed frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                hand_states_traj = np.zeros([task_horizon, num_hand])
                object_states_traj = np.zeros([task_horizon, num_obj])
                hand_states_traj[0, :] = hand_OriState
                object_states_traj[0, :] = obj_OriState
                z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                for t_ in range(task_horizon - 1):
                    z_t_1_computed = np.dot(KODex, z_t)
                    z_t = z_t_1_computed.copy()
                    x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                    hand_OriState = x_t_1_computed[:num_hand]
                    obj_pos = x_t_1_computed[num_hand: num_hand + 3] # tool pos
                    obj_ori = x_t_1_computed[num_hand + 3: num_hand + 6] # tool ori
                    obj_vel = x_t_1_computed[num_hand + 6: num_hand + 12]
                    nail_pos = x_t_1_computed[num_hand + 12:]
                    obj_OriState = np.append(obj_pos, np.append(obj_ori, np.append(obj_vel, nail_pos)))
                    hand_states_traj[t_ + 1, :] = hand_OriState
                    object_states_traj[t_ + 1, :] = obj_OriState
                    if self.env.control_mode == 'PID':  # if Torque mode, we skil the visualization of PD controller.
                        PID_controller.set_goal(hand_OriState)
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            # for relocation task, it we set a higher control frequency, we can expect a much better PD performance
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            next_o, r, done, goal_achieved = self.step(torque_action)    
                # re-set the experiments, as we finish visualizing the reference motion of KODex
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                o = self.get_obs()
                t, done = 0, False
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                if not policy.freeze_base:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0]
                else:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0][2:26], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0][2:26]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0][2:26]
                while t < task_horizon - 1:  # what would be early-termination for door task?
                    o = self.get_obs()
                    if not policy.freeze_base:
                        current_hand_state = self.get_env_state()['qpos'][:num_hand]
                    else:
                        current_hand_state = self.get_env_state()['qpos'][2:26]
                        num_hand = 24
                    current_objpos = o[49:52] + o[42:45]
                    current_objvel = o[27:33]
                    current_objori = o[39:42]
                    nail_goal = o[46:49]
                    hand_OriState = current_hand_state
                    obj_OriState = np.append(current_objpos, np.append(current_objori, np.append(current_objvel, nail_goal))) 
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][2:26], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][2:26], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][2:26]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][2:26]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][2:26], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][2:26], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][2:26]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][2:26]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[1]['evaluation'] # mean_action is True -> noise-free actions from controller
                    if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                        if not policy.freeze_base:
                            a += hand_states_traj[t + 1].copy() 
                        else:
                            a += hand_states_traj[t + 1][2:26].copy() 
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        if not policy.freeze_base:
                            PID_controller.set_goal(a)
                        else:
                            PID_controller.set_goal(np.append(hand_states_traj[t + 1][:2], a))
                            num_hand = len(np.append(hand_states_traj[t + 1][:2], a))
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    ep_returns[ep] += (gamma ** t) * r  # only task specific rewards
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t + 1]
                    reference['obj_pos'] = object_states_traj[t + 1][:3]
                    reference['obj_ori'] = object_states_traj[t + 1][3:6]
                    reference['obj_vel'] = object_states_traj[t + 1][6:12]
                    reference['nail_goal'] = object_states_traj[t + 1][12:]
                    obs = dict()
                    obs['hand_state'] = self.get_env_state()['qpos'][:num_hand]
                    obs['obj_pos'] = next_o[49:52] + next_o[42:45]
                    obs['obj_ori'] = next_o[39:42]
                    obs['obj_vel'] = next_o[27:33]
                    obs['nail_goal'] = next_o[46:49]
                    if not policy.freeze_base:
                        tracking_reward = Comp_tracking_reward_hammer(reference, obs, coeffcients)['total_reward']  
                    else:
                        tracking_reward = Comp_tracking_reward_hammer_freeze_base(reference, obs, coeffcients)['total_reward']  
                    tracking_rewards[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                    t += 1   
                    dist = np.linalg.norm(next_o[42:45] - next_o[46:49])
                if dist < 0.01:
                    success_list_sim.append(1)
                total_score += ep_returns[ep]
            print("Average score = %f" % (total_score / num_episodes))
            print("Success rate = %f" % (len(success_list_sim) / num_episodes))
        # ep_returns and tracking_rewards are both discounted rewards
        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)  # mean eval -> rewards
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        mean_tracking_eval, tracking_std = np.mean(tracking_rewards), np.std(tracking_rewards)  # mean eval -> rewards
        min_tracking_eval, max_tracking_eval = np.amin(tracking_rewards), np.amax(tracking_rewards)
        base_stats = [mean_eval, std, min_eval, max_eval, mean_tracking_eval, tracking_std, min_tracking_eval, max_tracking_eval]
        percentile_stats = []
        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))
        full_dist = ep_returns if get_full_dist is True else None
        return [base_stats, percentile_stats, full_dist]
    
    def evaluate_trained_mean_policy(self, 
                        Eval_data,
                        PID_controller,
                        coeffcients,
                        Koopman_obser,
                        KODex, 
                        task_horizon, 
                        future_state,
                        history_state,
                        policy,
                        num_episodes=5,
                        obj_dynamics=True,
                        gamma=1,
                        seed=123):
        self.set_seed(seed)
        task = self.env_id.split('-')[0]
        # success_threshold = 20 if task == 'pen' else 25
        ep_returns_PD = np.zeros(num_episodes)
        ep_returns_mean_act = np.zeros(num_episodes)
        # tracking_rewards -> hand tracking rewards + object tracking rewards
        tracking_rewards_PD = np.zeros(num_episodes)
        tracking_rewards_mean_act = np.zeros(num_episodes)
        # hand tracking rewards
        hand_rewards_PD = np.zeros(num_episodes)
        hand_rewards_mean_act = np.zeros(num_episodes)
        # object tracking rewards
        object_rewards_PD = np.zeros(num_episodes)
        object_rewards_mean_act = np.zeros(num_episodes)
        episodes = []
        episodes_PD = []
        total_score_mean_act = 0.0
        num_future_s = len(future_state)
        if task == 'relocate':
            success_threshold = 10
            PD_tips_force = {'ff': [], 'mf': [], 'rf': [], 'lf': [], 'th': [], 'sum': []}
            MA_PD_tips_force = {'ff': [], 'mf': [], 'rf': [], 'lf': [], 'th': [], 'sum': []}
            Joint_adaptations = []
            for ep in tqdm(range(num_episodes)):
                ff_force = []
                mf_force = []
                rf_force = []
                lf_force = []
                th_force = []
                sum_force = []
                episode_data_PD = {
                    'goal_achieved': []
                }
                episode_data_mean_act = {
                    'goal_achieved': []
                }
                # initialze the experiments, for PD controller
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                init_hand_state = Eval_data[ep]['handpos']
                init_objpos = Eval_data[ep]['objpos'] # converged object position
                init_objvel = Eval_data[ep]['objvel']
                init_objori = Eval_data[ep]['objorient']
                desired_pos = Eval_data[ep]['desired_pos']
                init_objpos_world = desired_pos + init_objpos # in the world frame(on the table)
                hand_OriState = init_hand_state
                obj_OriState = np.append(init_objpos, np.append(init_objori, init_objvel))  # ori: represented in the transformed frame (converged to desired pos)
                obj_OriState_ = np.append(init_objpos_world, np.append(init_objori, init_objvel)) # ori: represented in the world frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                hand_states_traj = np.zeros([task_horizon, num_hand])
                object_states_traj = np.zeros([task_horizon, num_obj])
                joint_modifications = []
                hand_states_traj[0, :] = hand_OriState
                object_states_traj[0, :] = obj_OriState_
                z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                for t_ in range(task_horizon - 1):
                    tips_force = self.contact_forces
                    ff_force.append(tips_force['ff'])
                    mf_force.append(tips_force['mf'])
                    rf_force.append(tips_force['rf'])
                    lf_force.append(tips_force['lf'])
                    th_force.append(tips_force['th'])
                    sum_f = 0
                    for value in tips_force.values():
                        sum_f += value
                    sum_force.append(sum_f)
                    z_t_1_computed = np.dot(KODex, z_t)
                    z_t = z_t_1_computed.copy()
                    x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                    hand_OriState = x_t_1_computed[:num_hand]
                    obj_pos = x_t_1_computed[num_hand: num_hand + 3] # converged object position
                    obj_pos_world = desired_pos + obj_pos
                    obj_ori = x_t_1_computed[num_hand + 3: num_hand + 6]
                    obj_vel = x_t_1_computed[num_hand + 6: num_hand + 12]
                    obj_OriState = np.append(obj_pos_world, np.append(obj_ori, obj_vel))
                    hand_states_traj[t_ + 1, :] = hand_OriState
                    object_states_traj[t_ + 1, :] = obj_OriState
                    if self.env.control_mode == 'PID':  # if Torque mode, we skip the visualization of PD controller.
                        PID_controller.set_goal(hand_OriState)
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            # for relocation task, it we set a higher control frequency, we can expect a much better PD performance
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            torque_action[1] -= 0.95  # hand_Txyz[1] -> hand_T_y
                            next_o, r, done, goal_achieved = self.step(torque_action)      
                    ep_returns_PD[ep] += (gamma ** t_) * r  # only task specific rewards
                    episode_data_PD['goal_achieved'].append(goal_achieved['goal_achieved'])
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t_ + 1]
                    reference['obj_pos'] = object_states_traj[t_ + 1][:3]
                    reference['obj_vel'] = object_states_traj[t_ + 1][6:]
                    reference['obj_ori'] = object_states_traj[t_ + 1][3:6]
                    obs = dict()
                    obs['hand_state'] = next_o[:30]
                    # if t_ == 24:
                    #     print(hand_OriState)
                    #     time.sleep(5)
                    obs['obj_pos'] = next_o[39:42]
                    obs['obj_vel'] = self.get_env_state()['qvel'][30:36]
                    obs['obj_ori'] = self.get_env_state()['qpos'][33:36]     
                    if not policy.freeze_base:    
                        tracking_reward = Comp_tracking_reward_relocate(reference, obs, coeffcients)['total_reward'] 
                        tracking_rewards_PD[ep] += (gamma ** t_) * tracking_reward  # only tracking rewards
                        hand_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_relocate(reference, obs, coeffcients)['hand_reward'] 
                        object_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_relocate(reference, obs, coeffcients)['object_reward']
                    else:
                        if not policy.include_Rots:
                            tracking_reward = Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['total_reward'] 
                            tracking_rewards_PD[ep] += (gamma ** t_) * tracking_reward  # only tracking rewards
                            hand_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['hand_reward'] 
                            object_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['object_reward']
                        else:
                            tracking_reward = Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['total_reward'] 
                            tracking_rewards_PD[ep] += (gamma ** t_) * tracking_reward  # only tracking rewards
                            hand_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['hand_reward'] 
                            object_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['object_reward']
                PD_tips_force['ff'].append(ff_force)
                PD_tips_force['mf'].append(mf_force)
                PD_tips_force['rf'].append(rf_force)
                PD_tips_force['lf'].append(lf_force)
                PD_tips_force['th'].append(th_force)
                PD_tips_force['sum'].append(sum_force)
                # print("tracking_rewards_PD[ep]:", tracking_rewards_PD[ep])
                episodes_PD.append(copy.deepcopy(episode_data_PD))
                # re-set the experiments, for Motion adapter with mean action
                self.reset()
                ff_force = []
                mf_force = []
                rf_force = []
                lf_force = []
                th_force = []
                sum_force = []
                self.set_env_state(Eval_data[ep]['init_states'])
                o = self.get_obs()
                t, done, obj_height = 0, False, o[41]
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                if not policy.freeze_base:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0]
                else:
                    for i in range(history_state):
                        if obj_dynamics:
                            if policy.m == 24:
                                prev_states[i] = np.append(hand_states_traj[0][6:30], object_states_traj[0]) 
                            elif policy.m == 27:
                                prev_states[i] = np.append(hand_states_traj[0][3:30], object_states_traj[0]) 
                        else:
                            if policy.m == 24:
                                prev_states[i] = hand_states_traj[0][6:30]
                            elif policy.m == 27:
                                prev_states[i] = hand_states_traj[0][3:30]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        if policy.m == 24:
                            prev_actions[i] = hand_states_traj[0][6:30]
                        elif policy.m == 27:
                            prev_actions[i] = hand_states_traj[0][3:30]
                while t < task_horizon - 1 and obj_height > -0.05 :  # what would be early-termination for relocation task?
                    # print(t)
                    tips_force = self.contact_forces
                    ff_force.append(tips_force['ff'])
                    mf_force.append(tips_force['mf'])
                    rf_force.append(tips_force['rf'])
                    lf_force.append(tips_force['lf'])
                    th_force.append(tips_force['th'])
                    sum_f = 0
                    for value in tips_force.values():
                        sum_f += value
                    sum_force.append(sum_f)
                    o = self.get_obs()
                    if not policy.freeze_base:
                        current_hand_state = o[:30]
                    else:
                        if policy.m == 24:
                            current_hand_state = o[6:30]
                            num_hand = 24
                        elif policy.m == 27:
                            current_hand_state = o[3:30]
                            num_hand = 27
                    current_objpos = o[39:42]  # in world frame
                    current_objvel = self.get_env_state()['qvel'][30:36]
                    current_objori = self.get_env_state()['qpos'][33:36]
                    hand_OriState = current_hand_state
                    obj_OriState = np.append(current_objpos, np.append(current_objori, current_objvel))  # ori: represented in the transformed frame
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][6:30], object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:30], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][6:30], object_states_traj[t + future_state[t_]])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][3:30], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][6:30]
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:30]
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][6:30]
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][3:30]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][6:30], object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:30], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][6:30], object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][3:30], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][6:30]
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:30]
                                    else:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][6:30]
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][3:30]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[1]['evaluation'] # mean_action is True -> noise-free actions from controller
                    try:
                        if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                            if not policy.freeze_base:
                                a += hand_states_traj[t + 1].copy() 
                            else:
                                if policy.m == 24:
                                    a += hand_states_traj[t + 1][6:30].copy() 
                                elif policy.m == 27:
                                    a += hand_states_traj[t + 1][3:30].copy() 
                    except:
                        pass
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        if not policy.freeze_base:
                            joint_modifications.append(np.abs(a - hand_states_traj[t + 1, :])) # absolute diff between adapted joints and original joints
                            PID_controller.set_goal(a)  # the changes (np.abs(a - hand_states_traj[t + 1, :])) is much smaller than those in Hammer task.
                        else:
                            if policy.m == 24:
                                tmp = np.append((hand_states_traj[t + 1, :][:6] - hand_states_traj[t + 1, :][:6]), np.abs(a - hand_states_traj[t + 1, :][6:30]))
                                joint_modifications.append(tmp) # absolute diff between adapted joints and original joints
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:6], a))  # the changes (np.abs(a - hand_states_traj[t + 1, :])) is much smaller than those in Hammer task.
                                num_hand = len(np.append(hand_states_traj[t + 1][:6], a))
                            elif policy.m == 27:
                                tmp = np.append((hand_states_traj[t + 1, :][:3] - hand_states_traj[t + 1, :][:3]), np.abs(a - hand_states_traj[t + 1, :][3:30]))
                                joint_modifications.append(tmp) # absolute diff between adapted joints and original joints
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:3], a))  # the changes (np.abs(a - hand_states_traj[t + 1, :])) is much smaller than those in Hammer task.
                                num_hand = len(np.append(hand_states_traj[t + 1][:3], a))
                        # print("reference value:", hand_states_traj[t + 1, :])
                        # print("a value:", a)                        
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            torque_action[1] -= 0.95  # hand_Txyz[1] -> hand_T_y
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    obj_height = next_o[41]
                    episode_data_mean_act['goal_achieved'].append(goal_achieved['goal_achieved'])
                    ep_returns_mean_act[ep] += (gamma ** t) * r  # only task specific rewards
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t + 1]
                    reference['obj_pos'] = object_states_traj[t + 1][:3]
                    reference['obj_vel'] = object_states_traj[t + 1][6:]
                    reference['obj_ori'] = object_states_traj[t + 1][3:6]
                    obs = dict()
                    obs['hand_state'] = next_o[:30]
                    # if t == 24:
                    #     print(a)
                    #     time.sleep(5)
                    obs['obj_pos'] = next_o[39:42]
                    obs['obj_vel'] = self.get_env_state()['qvel'][30:36]
                    obs['obj_ori'] = self.get_env_state()['qpos'][33:36]
                    if not policy.freeze_base:
                        tracking_reward = Comp_tracking_reward_relocate(reference, obs, coeffcients)['total_reward']
                        tracking_rewards_mean_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                        hand_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate(reference, obs, coeffcients)['hand_reward']
                        object_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate(reference, obs, coeffcients)['object_reward']
                    else:
                        if not policy.include_Rots:
                            tracking_reward = Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['total_reward']
                            tracking_rewards_mean_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                            hand_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['hand_reward']
                            object_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['object_reward']
                        else:
                            tracking_reward = Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['total_reward']
                            tracking_rewards_mean_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                            hand_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['hand_reward']
                            object_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['object_reward']
                    t += 1   
                Joint_adaptations.append(joint_modifications)
                MA_PD_tips_force['ff'].append(ff_force)
                MA_PD_tips_force['mf'].append(mf_force)
                MA_PD_tips_force['rf'].append(rf_force)
                MA_PD_tips_force['lf'].append(lf_force)
                MA_PD_tips_force['th'].append(th_force)
                MA_PD_tips_force['sum'].append(sum_force)
                # print("tracking_rewards_mean_act[ep]", tracking_rewards_mean_act[ep])
                episodes.append(copy.deepcopy(episode_data_mean_act))
                total_score_mean_act += ep_returns_mean_act[ep]
            successful_episodes = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes))            
            # successful_episodes = list(filter(lambda episode: episode['goal_achieved'][-1], successful_episodes))
            successful_episodes_PD = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes_PD))
            # successful_episodes_PD = list(filter(lambda episode: episode['goal_achieved'][-1], successful_episodes_PD))
            # find the index
            force_compare_index = list()
            for index_ in range(num_episodes):
                # PD - MA_PD (successful rollouts on PD while unsuccessful on MA_PD)
                # if sum(episodes[index_]['goal_achieved']) < success_threshold and sum(episodes_PD[index_]['goal_achieved']) > success_threshold:
                #     force_compare_index.append(index_)
                # MA_PD - PD
                if sum(episodes[index_]['goal_achieved']) > success_threshold and sum(episodes_PD[index_]['goal_achieved']) < success_threshold:
                    force_compare_index.append(index_)
            return len(successful_episodes_PD) / len(episodes), len(successful_episodes) / len(episodes)

    def evaluate_trained_policy(self, 
                        Eval_data,
                        PID_controller,
                        coeffcients,
                        Koopman_obser,
                        KODex, 
                        task_horizon, 
                        future_state,
                        history_state,
                        policy,
                        num_episodes=5,
                        obj_dynamics=True,
                        gamma=1,
                        visual=False,
                        terminate_at_done=True,
                        seed=123):
        self.set_seed(seed)
        task = self.env_id.split('-')[0]
        # success_threshold = 20 if task == 'pen' else 25
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns_PD = np.zeros(num_episodes)
        ep_returns_mean_act = np.zeros(num_episodes)
        ep_returns_noisy_act = np.zeros(num_episodes)
        # tracking_rewards -> hand tracking rewards + object tracking rewards
        tracking_rewards_PD = np.zeros(num_episodes)
        tracking_rewards_mean_act = np.zeros(num_episodes)
        tracking_rewards_noisy_act = np.zeros(num_episodes)
        # hand tracking rewards
        hand_rewards_PD = np.zeros(num_episodes)
        hand_rewards_mean_act = np.zeros(num_episodes)
        hand_rewards_noisy_act = np.zeros(num_episodes)
        # object tracking rewards
        object_rewards_PD = np.zeros(num_episodes)
        object_rewards_mean_act = np.zeros(num_episodes)
        object_rewards_noisy_act = np.zeros(num_episodes)
        episodes = []
        episodes_PD = []
        episodes_noisy = []
        total_score_mean_act = 0.0
        num_future_s = len(future_state)
        if task == 'pen':
            success_threshold = 10
            for ep in tqdm(range(num_episodes)):
                episode_data_PD = {
                    'goal_achieved': []
                }
                episode_data_mean_act = {
                    'goal_achieved': []
                }
                episode_data_noisy_act = {
                    'goal_achieved': []
                }
                # initialze the experiments, for PD controller
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                init_hand_state = Eval_data[ep]['handpos']
                init_objpos = Eval_data[ep]['objpos']
                init_objvel = Eval_data[ep]['objvel']
                init_objorient_world = Eval_data[ep]['objorient']
                desired_ori = Eval_data[ep]['desired_ori']
                init_objorient = ori_transform(init_objorient_world, desired_ori) 
                hand_OriState = init_hand_state
                obj_OriState = np.append(init_objpos, np.append(init_objorient, init_objvel))  # ori: represented in the transformed frame
                obj_OriState_ = np.append(init_objpos, np.append(init_objorient_world, init_objvel)) # ori: represented in the original frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                hand_states_traj = np.zeros([task_horizon, num_hand])
                object_states_traj = np.zeros([task_horizon, num_obj])
                hand_states_traj[0, :] = hand_OriState
                object_states_traj[0, :] = obj_OriState_
                z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                if visual:
                    print("Episode %d, KOdex with PD." %(ep + 1))
                for t_ in range(task_horizon - 1):
                    self.render() if visual is True else None
                    z_t_1_computed = np.dot(KODex, z_t)
                    z_t = z_t_1_computed.copy()
                    x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                    hand_OriState = x_t_1_computed[:num_hand]
                    obj_pos = x_t_1_computed[num_hand: num_hand + 3]
                    # x_t_1_computed[num_hand + 3: num_hand + 6] -> ori: represented in the transformed frame
                    obj_ori = ori_transform_inverse(x_t_1_computed[num_hand + 3: num_hand + 6], desired_ori) # ori: represented in the original frame
                    obj_vel = x_t_1_computed[num_hand + 6: num_hand + 12]
                    obj_OriState = np.append(obj_pos, np.append(obj_ori, obj_vel))
                    hand_states_traj[t_ + 1, :] = hand_OriState
                    object_states_traj[t_ + 1, :] = obj_OriState
                    # Visualize the KODex results, to check the given reference motion if good or not. 
                    if self.env.control_mode == 'PID':  # if Torque mode, we skil the visualization of PD controller.
                        PID_controller.set_goal(hand_OriState)
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            next_o, r, done, goal_achieved = self.step(torque_action)     
                    ep_returns_PD[ep] += (gamma ** t_) * r  # only task specific rewards
                    episode_data_PD['goal_achieved'].append(goal_achieved['goal_achieved'])
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t_ + 1]
                    reference['obj_pos'] = object_states_traj[t_ + 1][:3]
                    reference['obj_vel'] = object_states_traj[t_ + 1][6:]
                    reference['obj_ori'] = object_states_traj[t_ + 1][3:6]
                    obs = dict()
                    obs['hand_state'] = next_o[:24]
                    obs['obj_pos'] = next_o[24:27]
                    obs['obj_vel'] = next_o[27:33]
                    obs['obj_ori'] = next_o[33:36]            
                    tracking_reward = Comp_tracking_reward_pen(reference, obs, coeffcients)['total_reward']  
                    tracking_rewards_PD[ep] += (gamma ** t_) * tracking_reward  # only tracking rewards
                    hand_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_pen(reference, obs, coeffcients)['hand_reward']  
                    object_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_pen(reference, obs, coeffcients)['object_reward'] 
                episodes_PD.append(copy.deepcopy(episode_data_PD))
                # print("PD: discounted task reward %f"%(ep_returns_PD[ep]))
                # print("PD: discounted tracking reward %f"%(tracking_rewards_PD[ep]))
                # print("desired ori:", desired_ori)
                # print("final Koopman ori:", obj_ori)
                # print("similarity:", np.dot(desired_ori, obj_ori) / np.linalg.norm(obj_ori))  # koopman ori may not be a perfect unit vector.
                
                # re-set the experiments, for Motion adapter with mean action
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                o = self.get_obs()
                t, done, obj_height = 0, False, o[26]
                # while t < task_horizon - 1 and (done == False or terminate_at_done == False):
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                for i in range(history_state):
                    if obj_dynamics:
                        prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                    else:
                        prev_states[i] = hand_states_traj[0]
                prev_actions = dict()
                for i in range(history_state + 1):
                    prev_actions[i] = hand_states_traj[0]
                if visual:
                    print("Episode %d, KOdex with motion adapter (mean action)." %(ep + 1))
                while t < task_horizon - 1 and obj_height > 0.15:  # early-terminate when the object falls off (reorientation task)
                    self.render() if visual is True else None
                    o = self.get_obs()
                    current_hand_state = o[:24]
                    current_objpos = o[24:27]
                    current_objvel = o[27:33]
                    current_objori = o[33:36]
                    hand_OriState = current_hand_state
                    obj_OriState = np.append(current_objpos, np.append(current_objori, current_objvel))  # ori: represented in the transformed frame
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= task_horizon:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                else:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= task_horizon:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                else:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= task_horizon:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                else:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= task_horizon:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                else:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[1]['evaluation'] # noise-free mean actions from controller
                    try:
                        if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                            a += hand_states_traj[t + 1].copy() 
                    except:
                        pass
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        PID_controller.set_goal(a)
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:len(a)], self.get_env_state()['qvel'][:len(a)])
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    obj_height = next_o[26]
                    episode_data_mean_act['goal_achieved'].append(goal_achieved['goal_achieved'])
                    ep_returns_mean_act[ep] += (gamma ** t) * r  # only task specific rewards
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t + 1]
                    reference['obj_pos'] = object_states_traj[t + 1][:3]
                    reference['obj_vel'] = object_states_traj[t + 1][6:]
                    reference['obj_ori'] = object_states_traj[t + 1][3:6]
                    obs = dict()
                    obs['hand_state'] = next_o[:24]
                    obs['obj_pos'] = next_o[24:27]
                    obs['obj_vel'] = next_o[27:33]
                    obs['obj_ori'] = next_o[33:36]
                    tracking_reward = Comp_tracking_reward_pen(reference, obs, coeffcients)['total_reward']  
                    tracking_rewards_mean_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                    hand_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_pen(reference, obs, coeffcients)['hand_reward']
                    object_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_pen(reference, obs, coeffcients)['object_reward']
                    t += 1    
                # print("Task reward = %f, Tracking reward = %f, Success = %d" % (ep_returns[ep], tracking_rewards[ep], sum(episode_data['goal_achieved']) > success_threshold))
                episodes.append(copy.deepcopy(episode_data_mean_act))
                total_score_mean_act += ep_returns_mean_act[ep]
                # print("Motion Adapter with mean action: discounted task reward %f"%(ep_returns_mean_act[ep]))
                # print("Motion Adapter with mean action: discounted tracking reward %f"%(tracking_rewards_mean_act[ep]))
                # re-set the experiments, for Motion adapter with noisy action
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                o = self.get_obs()
                t, done, obj_height = 0, False, o[26]
                # while t < control_horizon and (done == False or terminate_at_done == False):
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                for i in range(history_state):
                    if obj_dynamics:
                        prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                    else:
                        prev_states[i] = hand_states_traj[0]
                prev_actions = dict()
                for i in range(history_state + 1):
                    prev_actions[i] = hand_states_traj[0]
                if visual:
                    print("Episode %d, KOdex with motion adapter (noisy action)." %(ep + 1))
                while t < task_horizon - 1 and obj_height > 0.15:  # early-terminate when the object falls off
                    self.render() if visual is True else None
                    o = self.get_obs()
                    current_hand_state = o[:24]
                    current_objpos = o[24:27]
                    current_objvel = o[27:33]
                    current_objori = o[33:36]
                    hand_OriState = current_hand_state
                    obj_OriState = np.append(current_objpos, np.append(current_objori, current_objvel))  # ori: represented in the transformed frame
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= task_horizon:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                else:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= task_horizon:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                else:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= task_horizon:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                else:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= task_horizon:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                else:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[0] # noisy actions from controller
                    try:
                        if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                            a += hand_states_traj[t + 1].copy() 
                    except:
                        pass
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        PID_controller.set_goal(a)
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:len(a)], self.get_env_state()['qvel'][:len(a)])
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    obj_height = next_o[26]
                    episode_data_noisy_act['goal_achieved'].append(goal_achieved['goal_achieved'])
                    ep_returns_noisy_act[ep] += (gamma ** t) * r  # only task specific rewards
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t + 1]
                    reference['obj_pos'] = object_states_traj[t + 1][:3]
                    reference['obj_vel'] = object_states_traj[t + 1][6:]
                    reference['obj_ori'] = object_states_traj[t + 1][3:6]
                    obs = dict()
                    obs['hand_state'] = next_o[:24]
                    obs['obj_pos'] = next_o[24:27]
                    obs['obj_vel'] = next_o[27:33]
                    obs['obj_ori'] = next_o[33:36]
                    tracking_reward = Comp_tracking_reward_pen(reference, obs, coeffcients)['total_reward']  
                    tracking_rewards_noisy_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                    hand_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_pen(reference, obs, coeffcients)['hand_reward']   
                    object_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_pen(reference, obs, coeffcients)['object_reward']   
                    t += 1    
                episodes_noisy.append(copy.deepcopy(episode_data_noisy_act))
                # print("Task reward = %f, Tracking reward = %f, Success = %d" % (ep_returns[ep], tracking_rewards[ep], sum(episode_data['goal_achieved']) > success_threshold))
                # print("Motion Adapter with noisy action: discounted task reward %f"%(ep_returns_noisy_act[ep]))
                # print("Motion Adapter with noisy action: discounted tracking reward %f"%(tracking_rewards_noisy_act[ep]))
            successful_episodes = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes))
            successful_episodes_PD = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes_PD))
            successful_episodes_noisy = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes_noisy))
            print("(Motion Adapter with mean action) Average score (task reward) = %f" % (total_score_mean_act / num_episodes))
            print("(Motion Adapter with mean action) Success rate = %f" % (len(successful_episodes) / len(episodes)))
            print("(Motion Adapter with noisy action) Success rate = %f" % (len(successful_episodes_noisy) / len(episodes)))
            print("(PD) Success rate = %f" % (len(successful_episodes_PD) / len(episodes)))
        elif task == 'relocate':
            success_threshold = 10
            PD_tips_force = {'ff': [], 'mf': [], 'rf': [], 'lf': [], 'th': [], 'sum': []}
            MA_PD_tips_force = {'ff': [], 'mf': [], 'rf': [], 'lf': [], 'th': [], 'sum': []}
            Joint_adaptations = []
            for ep in tqdm(range(num_episodes)):
                ff_force = []
                mf_force = []
                rf_force = []
                lf_force = []
                th_force = []
                sum_force = []
                episode_data_PD = {
                    'goal_achieved': []
                }
                episode_data_mean_act = {
                    'goal_achieved': []
                }
                episode_data_noisy_act = {
                    'goal_achieved': []
                }
                # initialze the experiments, for PD controller
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                init_hand_state = Eval_data[ep]['handpos']
                init_objpos = Eval_data[ep]['objpos'] # converged object position
                init_objvel = Eval_data[ep]['objvel']
                init_objori = Eval_data[ep]['objorient']
                desired_pos = Eval_data[ep]['desired_pos']
                init_objpos_world = desired_pos + init_objpos # in the world frame(on the table)
                hand_OriState = init_hand_state
                obj_OriState = np.append(init_objpos, np.append(init_objori, init_objvel))  # ori: represented in the transformed frame (converged to desired pos)
                obj_OriState_ = np.append(init_objpos_world, np.append(init_objori, init_objvel)) # ori: represented in the world frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                hand_states_traj = np.zeros([task_horizon, num_hand])
                object_states_traj = np.zeros([task_horizon, num_obj])
                joint_modifications = []
                hand_states_traj[0, :] = hand_OriState
                object_states_traj[0, :] = obj_OriState_
                z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                if visual:
                    print("Episode %d, KOdex with PD." %(ep + 1))
                for t_ in range(task_horizon - 1):
                    tips_force = self.contact_forces
                    ff_force.append(tips_force['ff'])
                    mf_force.append(tips_force['mf'])
                    rf_force.append(tips_force['rf'])
                    lf_force.append(tips_force['lf'])
                    th_force.append(tips_force['th'])
                    sum_f = 0
                    for value in tips_force.values():
                        sum_f += value
                    sum_force.append(sum_f)
                    self.render() if visual is True else None
                    z_t_1_computed = np.dot(KODex, z_t)
                    z_t = z_t_1_computed.copy()
                    x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                    hand_OriState = x_t_1_computed[:num_hand]
                    obj_pos = x_t_1_computed[num_hand: num_hand + 3] # converged object position
                    obj_pos_world = desired_pos + obj_pos
                    obj_ori = x_t_1_computed[num_hand + 3: num_hand + 6]
                    obj_vel = x_t_1_computed[num_hand + 6: num_hand + 12]
                    obj_OriState = np.append(obj_pos_world, np.append(obj_ori, obj_vel))
                    hand_states_traj[t_ + 1, :] = hand_OriState
                    object_states_traj[t_ + 1, :] = obj_OriState
                    if self.env.control_mode == 'PID':  # if Torque mode, we skip the visualization of PD controller.
                        PID_controller.set_goal(hand_OriState)
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            # for relocation task, it we set a higher control frequency, we can expect a much better PD performance
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            torque_action[1] -= 0.95  # hand_Txyz[1] -> hand_T_y
                            next_o, r, done, goal_achieved = self.step(torque_action)      
                    ep_returns_PD[ep] += (gamma ** t_) * r  # only task specific rewards
                    episode_data_PD['goal_achieved'].append(goal_achieved['goal_achieved'])
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t_ + 1]
                    reference['obj_pos'] = object_states_traj[t_ + 1][:3]
                    reference['obj_vel'] = object_states_traj[t_ + 1][6:]
                    reference['obj_ori'] = object_states_traj[t_ + 1][3:6]
                    obs = dict()
                    obs['hand_state'] = next_o[:30]
                    # if t_ == 24:
                    #     print(hand_OriState)
                    #     time.sleep(5)
                    obs['obj_pos'] = next_o[39:42]
                    obs['obj_vel'] = self.get_env_state()['qvel'][30:36]
                    obs['obj_ori'] = self.get_env_state()['qpos'][33:36]     
                    if not policy.freeze_base:    
                        tracking_reward = Comp_tracking_reward_relocate(reference, obs, coeffcients)['total_reward'] 
                        tracking_rewards_PD[ep] += (gamma ** t_) * tracking_reward  # only tracking rewards
                        hand_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_relocate(reference, obs, coeffcients)['hand_reward'] 
                        object_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_relocate(reference, obs, coeffcients)['object_reward']
                    else:
                        if not policy.include_Rots:
                            tracking_reward = Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['total_reward'] 
                            tracking_rewards_PD[ep] += (gamma ** t_) * tracking_reward  # only tracking rewards
                            hand_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['hand_reward'] 
                            object_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['object_reward']
                        else:
                            tracking_reward = Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['total_reward'] 
                            tracking_rewards_PD[ep] += (gamma ** t_) * tracking_reward  # only tracking rewards
                            hand_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['hand_reward'] 
                            object_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['object_reward']
                PD_tips_force['ff'].append(ff_force)
                PD_tips_force['mf'].append(mf_force)
                PD_tips_force['rf'].append(rf_force)
                PD_tips_force['lf'].append(lf_force)
                PD_tips_force['th'].append(th_force)
                PD_tips_force['sum'].append(sum_force)
                # print("tracking_rewards_PD[ep]:", tracking_rewards_PD[ep])
                episodes_PD.append(copy.deepcopy(episode_data_PD))
                # re-set the experiments, for Motion adapter with mean action
                self.reset()
                ff_force = []
                mf_force = []
                rf_force = []
                lf_force = []
                th_force = []
                sum_force = []
                self.set_env_state(Eval_data[ep]['init_states'])
                o = self.get_obs()
                t, done, obj_height = 0, False, o[41]
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                if not policy.freeze_base:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0]
                else:
                    for i in range(history_state):
                        if obj_dynamics:
                            if policy.m == 24:
                                prev_states[i] = np.append(hand_states_traj[0][6:30], object_states_traj[0]) 
                            elif policy.m == 27:
                                prev_states[i] = np.append(hand_states_traj[0][3:30], object_states_traj[0]) 
                        else:
                            if policy.m == 24:
                                prev_states[i] = hand_states_traj[0][6:30]
                            elif policy.m == 27:
                                prev_states[i] = hand_states_traj[0][3:30]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        if policy.m == 24:
                            prev_actions[i] = hand_states_traj[0][6:30]
                        elif policy.m == 27:
                            prev_actions[i] = hand_states_traj[0][3:30]
                if visual:
                    print("Episode %d, KOdex with motion adapter (mean action)." %(ep + 1))
                while t < task_horizon - 1 and obj_height > -0.05 :  # what would be early-termination for relocation task?
                    # print(t)
                    tips_force = self.contact_forces
                    ff_force.append(tips_force['ff'])
                    mf_force.append(tips_force['mf'])
                    rf_force.append(tips_force['rf'])
                    lf_force.append(tips_force['lf'])
                    th_force.append(tips_force['th'])
                    sum_f = 0
                    for value in tips_force.values():
                        sum_f += value
                    sum_force.append(sum_f)
                    self.render() if visual is True else None
                    o = self.get_obs()
                    if not policy.freeze_base:
                        current_hand_state = o[:30]
                    else:
                        if policy.m == 24:
                            current_hand_state = o[6:30]
                            num_hand = 24
                        elif policy.m == 27:
                            current_hand_state = o[3:30]
                            num_hand = 27
                    current_objpos = o[39:42]  # in world frame
                    current_objvel = self.get_env_state()['qvel'][30:36]
                    current_objori = self.get_env_state()['qpos'][33:36]
                    hand_OriState = current_hand_state
                    obj_OriState = np.append(current_objpos, np.append(current_objori, current_objvel))  # ori: represented in the transformed frame
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][6:30], object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:30], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][6:30], object_states_traj[t + future_state[t_]])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][3:30], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][6:30]
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:30]
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][6:30]
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][3:30]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][6:30], object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:30], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][6:30], object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][3:30], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][6:30]
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:30]
                                    else:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][6:30]
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][3:30]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[1]['evaluation'] # mean_action is True -> noise-free actions from controller
                    try:
                        if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                            if not policy.freeze_base:
                                a += hand_states_traj[t + 1].copy() 
                            else:
                                if policy.m == 24:
                                    a += hand_states_traj[t + 1][6:30].copy() 
                                elif policy.m == 27:
                                    a += hand_states_traj[t + 1][3:30].copy() 
                    except:
                        pass
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        if not policy.freeze_base:
                            joint_modifications.append(np.abs(a - hand_states_traj[t + 1, :])) # absolute diff between adapted joints and original joints
                            PID_controller.set_goal(a)  # the changes (np.abs(a - hand_states_traj[t + 1, :])) is much smaller than those in Hammer task.
                        else:
                            if policy.m == 24:
                                tmp = np.append((hand_states_traj[t + 1, :][:6] - hand_states_traj[t + 1, :][:6]), np.abs(a - hand_states_traj[t + 1, :][6:30]))
                                joint_modifications.append(tmp) # absolute diff between adapted joints and original joints
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:6], a))  # the changes (np.abs(a - hand_states_traj[t + 1, :])) is much smaller than those in Hammer task.
                                num_hand = len(np.append(hand_states_traj[t + 1][:6], a))
                            elif policy.m == 27:
                                tmp = np.append((hand_states_traj[t + 1, :][:3] - hand_states_traj[t + 1, :][:3]), np.abs(a - hand_states_traj[t + 1, :][3:30]))
                                joint_modifications.append(tmp) # absolute diff between adapted joints and original joints
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:3], a))  # the changes (np.abs(a - hand_states_traj[t + 1, :])) is much smaller than those in Hammer task.
                                num_hand = len(np.append(hand_states_traj[t + 1][:3], a))
                        # print("reference value:", hand_states_traj[t + 1, :])
                        # print("a value:", a)                        
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            torque_action[1] -= 0.95  # hand_Txyz[1] -> hand_T_y
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    obj_height = next_o[41]
                    episode_data_mean_act['goal_achieved'].append(goal_achieved['goal_achieved'])
                    ep_returns_mean_act[ep] += (gamma ** t) * r  # only task specific rewards
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t + 1]
                    reference['obj_pos'] = object_states_traj[t + 1][:3]
                    reference['obj_vel'] = object_states_traj[t + 1][6:]
                    reference['obj_ori'] = object_states_traj[t + 1][3:6]
                    obs = dict()
                    obs['hand_state'] = next_o[:30]
                    # if t == 24:
                    #     print(a)
                    #     time.sleep(5)
                    obs['obj_pos'] = next_o[39:42]
                    obs['obj_vel'] = self.get_env_state()['qvel'][30:36]
                    obs['obj_ori'] = self.get_env_state()['qpos'][33:36]
                    if not policy.freeze_base:
                        tracking_reward = Comp_tracking_reward_relocate(reference, obs, coeffcients)['total_reward']
                        tracking_rewards_mean_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                        hand_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate(reference, obs, coeffcients)['hand_reward']
                        object_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate(reference, obs, coeffcients)['object_reward']
                    else:
                        if not policy.include_Rots:
                            tracking_reward = Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['total_reward']
                            tracking_rewards_mean_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                            hand_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['hand_reward']
                            object_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['object_reward']
                        else:
                            tracking_reward = Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['total_reward']
                            tracking_rewards_mean_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                            hand_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['hand_reward']
                            object_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['object_reward']
                    t += 1   
                Joint_adaptations.append(joint_modifications)
                MA_PD_tips_force['ff'].append(ff_force)
                MA_PD_tips_force['mf'].append(mf_force)
                MA_PD_tips_force['rf'].append(rf_force)
                MA_PD_tips_force['lf'].append(lf_force)
                MA_PD_tips_force['th'].append(th_force)
                MA_PD_tips_force['sum'].append(sum_force)
                # print("tracking_rewards_mean_act[ep]", tracking_rewards_mean_act[ep])
                episodes.append(copy.deepcopy(episode_data_mean_act))
                total_score_mean_act += ep_returns_mean_act[ep]
                # re-set the experiments, for Motion adapter with noisy action
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                o = self.get_obs()
                t, done, obj_height = 0, False, o[41]
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                if not policy.freeze_base:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0]
                else:
                    for i in range(history_state):
                        if obj_dynamics:
                            if policy.m == 24:
                                prev_states[i] = np.append(hand_states_traj[0][6:30], object_states_traj[0]) 
                            elif policy.m == 27:
                                prev_states[i] = np.append(hand_states_traj[0][3:30], object_states_traj[0]) 
                        else:
                            if policy.m == 24:
                                prev_states[i] = hand_states_traj[0][6:30]
                            elif policy.m == 27:
                                prev_states[i] = hand_states_traj[0][3:30]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        if policy.m == 24:
                            prev_actions[i] = hand_states_traj[0][6:30]
                        elif policy.m == 27:
                            prev_actions[i] = hand_states_traj[0][3:30]
                if visual:
                    print("Episode %d, KOdex with motion adapter (noisy action)." %(ep + 1))
                while t < task_horizon - 1 and obj_height > -0.05:  # what would be early-termination for relocation task?
                    self.render() if visual is True else None
                    o = self.get_obs()
                    if not policy.freeze_base:
                        current_hand_state = o[:30]
                    else:
                        if policy.m == 24:
                            current_hand_state = o[6:30]
                            num_hand = 24
                        elif policy.m == 27:
                            current_hand_state = o[3:30]
                            num_hand = 27
                    current_objpos = o[39:42]  # in world frame
                    current_objvel = self.get_env_state()['qvel'][30:36]
                    current_objori = self.get_env_state()['qpos'][33:36]
                    hand_OriState = current_hand_state
                    obj_OriState = np.append(current_objpos, np.append(current_objori, current_objvel))  # ori: represented in the transformed frame
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][6:30], object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:30], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][6:30], object_states_traj[t + future_state[t_]])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][3:30], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][6:30]
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:30]
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][6:30]
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][3:30]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][6:30], object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:30], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][6:30], object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][3:30], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][6:30]
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:30]
                                    else:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][6:30]
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][3:30]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[0] # mean_action is True -> noise-free actions from controller
                    try:
                        if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                            if not policy.freeze_base:
                                a += hand_states_traj[t + 1].copy() 
                            else:
                                if policy.m == 24:
                                    a += hand_states_traj[t + 1][6:30].copy() 
                                elif policy.m == 27:
                                    a += hand_states_traj[t + 1][3:30].copy() 
                    except:
                        pass
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        if not policy.freeze_base:
                            PID_controller.set_goal(a)
                        else:
                            if policy.m == 24:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:6], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:6], a))
                            elif policy.m == 27:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:3], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:3], a))
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            torque_action[1] -= 0.95  # hand_Txyz[1] -> hand_T_y
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    obj_height = next_o[41]
                    episode_data_noisy_act['goal_achieved'].append(goal_achieved['goal_achieved'])
                    ep_returns_noisy_act[ep] += (gamma ** t) * r  # only task specific rewards
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t + 1]
                    reference['obj_pos'] = object_states_traj[t + 1][:3]
                    reference['obj_vel'] = object_states_traj[t + 1][6:]
                    reference['obj_ori'] = object_states_traj[t + 1][3:6]
                    obs = dict()
                    obs['hand_state'] = next_o[:30]
                    obs['obj_pos'] = next_o[39:42]
                    obs['obj_vel'] = self.get_env_state()['qvel'][30:36]
                    obs['obj_ori'] = self.get_env_state()['qpos'][33:36]
                    if not policy.freeze_base:
                        tracking_reward = Comp_tracking_reward_relocate(reference, obs, coeffcients)['total_reward']
                        tracking_rewards_noisy_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                        hand_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate(reference, obs, coeffcients)['hand_reward']
                        object_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate(reference, obs, coeffcients)['object_reward']
                    else:
                        if not policy.include_Rots:
                            tracking_reward = Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['total_reward']
                            tracking_rewards_noisy_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                            hand_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['hand_reward']
                            object_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['object_reward']
                        else:
                            tracking_reward = Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['total_reward']
                            tracking_rewards_noisy_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                            hand_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['hand_reward']
                            object_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['object_reward']
                    t += 1   
                episodes_noisy.append(copy.deepcopy(episode_data_noisy_act))
                # print("tracking_rewards_noisy_act[ep]", tracking_rewards_noisy_act[ep])
            successful_episodes = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes))            
            # successful_episodes = list(filter(lambda episode: episode['goal_achieved'][-1], successful_episodes))
            successful_episodes_PD = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes_PD))
            # successful_episodes_PD = list(filter(lambda episode: episode['goal_achieved'][-1], successful_episodes_PD))
            successful_episodes_noisy = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes_noisy))
            # successful_episodes_noisy = list(filter(lambda episode: episode['goal_achieved'][-1], successful_episodes_noisy))
            # find the index
            force_compare_index = list()
            for index_ in range(num_episodes):
                # PD - MA_PD (successful rollouts on PD while unsuccessful on MA_PD)
                # if sum(episodes[index_]['goal_achieved']) < success_threshold and sum(episodes_PD[index_]['goal_achieved']) > success_threshold:
                #     force_compare_index.append(index_)
                # MA_PD - PD
                if sum(episodes[index_]['goal_achieved']) > success_threshold and sum(episodes_PD[index_]['goal_achieved']) < success_threshold:
                    force_compare_index.append(index_)
            print("(Motion Adapter with mean action) Average score (task reward) = %f" % (total_score_mean_act / num_episodes))
            print("(Motion Adapter with mean action) Success rate = %f" % (len(successful_episodes) / len(episodes)))
            print("(Motion Adapter with noisy action) Success rate = %f" % (len(successful_episodes_noisy) / len(episodes)))
            print("(PD) Success rate = %f" % (len(successful_episodes_PD) / len(episodes)))
        elif task == 'door':
            success_list_sim = []
            success_list_PD = []
            success_list_sim_noisy = []
            R_z_motion_PD = []
            R_z_motion_MA = []
            for ep in tqdm(range(num_episodes)):
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                init_hand_state = Eval_data[ep]['handpos']
                init_objpos = Eval_data[ep]['objpos']
                init_objvel = Eval_data[ep]['objvel']
                init_handle = Eval_data[ep]['handle_init']
                hand_OriState = init_hand_state
                obj_OriState = np.append(init_objpos, np.append(init_objvel, init_handle))  # ori: represented in the transformed frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                hand_states_traj = np.zeros([task_horizon, num_hand])
                object_states_traj = np.zeros([task_horizon, num_obj])
                hand_states_traj[0, :] = hand_OriState
                r_z_motion = np.zeros(task_horizon)
                r_z_motion[0] = hand_OriState[3].copy()
                object_states_traj[0, :] = obj_OriState
                z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                if visual:
                    print("Episode %d, KOdex with PD." %(ep + 1))
                for t_ in range(task_horizon - 1):
                    self.render() if visual is True else None
                    z_t_1_computed = np.dot(KODex, z_t)
                    z_t = z_t_1_computed.copy()
                    x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                    hand_OriState = x_t_1_computed[:num_hand]
                    obj_pos = x_t_1_computed[num_hand: num_hand + 3] # converged object position
                    obj_vel = x_t_1_computed[num_hand + 3: num_hand + 4]
                    init_handle = x_t_1_computed[num_hand + 4: num_hand + 7]
                    obj_OriState = np.append(obj_pos, np.append(obj_vel, init_handle))
                    hand_states_traj[t_ + 1, :] = hand_OriState
                    object_states_traj[t_ + 1, :] = obj_OriState
                    if self.env.control_mode == 'PID':  # if Torque mode, we skil the visualization of PD controller.
                        PID_controller.set_goal(hand_OriState)
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            # for relocation task, it we set a higher control frequency, we can expect a much better PD performance
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            next_o, r, done, goal_achieved = self.step(torque_action)    
                            # re-set the experiments, as we finish visualizing the reference motion of KODex
                    r_z_motion[t_ + 1] = self.get_env_state()['qpos'][3].copy()
                    ep_returns_PD[ep] += (gamma ** t_) * r  # only task specific rewards
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t_ + 1]
                    reference['obj_pos'] = object_states_traj[t_ + 1][:3]
                    reference['obj_vel'] = object_states_traj[t_ + 1][3:4]
                    reference['init_handle'] = object_states_traj[t_ + 1][4:]
                    obs = dict()
                    obs['hand_state'] = self.get_env_state()['qpos'][:num_hand]
                    obs['obj_pos'] = next_o[32:35]
                    obs['obj_vel'] = self.get_env_state()['qvel'][28:29]
                    obs['init_handle'] = self.get_env_state()['door_body_pos']
                    if not policy.freeze_base:
                        tracking_reward = Comp_tracking_reward_door(reference, obs, coeffcients)['total_reward'] 
                        tracking_rewards_PD[ep] += (gamma ** t_) * tracking_reward  # only tracking rewards
                        hand_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_door(reference, obs, coeffcients)['hand_reward'] 
                        object_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_door(reference, obs, coeffcients)['object_reward'] 
                    else:
                        if not policy.include_Rots:
                            tracking_reward = Comp_tracking_reward_door_freeze_base(reference, obs, coeffcients)['total_reward'] 
                            tracking_rewards_PD[ep] += (gamma ** t_) * tracking_reward  # only tracking rewards
                            hand_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_door_freeze_base(reference, obs, coeffcients)['hand_reward'] 
                            object_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_door_freeze_base(reference, obs, coeffcients)['object_reward'] 
                        else:
                            tracking_reward = Comp_tracking_reward_door_include_Rots(reference, obs, coeffcients)['total_reward'] 
                            tracking_rewards_PD[ep] += (gamma ** t_) * tracking_reward  # only tracking rewards
                            hand_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_door_include_Rots(reference, obs, coeffcients)['hand_reward'] 
                            object_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_door_include_Rots(reference, obs, coeffcients)['object_reward'] 
                    current_hinge_pos = next_o[28:29] # door opening angle
                R_z_motion_PD.append(r_z_motion)
                if current_hinge_pos > 1.35:
                    success_list_PD.append(1)
                # re-set the experiments, for Motion adapter with mean action
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                o = self.get_obs()
                t, done = 0, False
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                if not policy.freeze_base:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0]
                else:
                    for i in range(history_state):
                        if obj_dynamics:
                            if policy.m == 24:
                                prev_states[i] = np.append(hand_states_traj[0][4:28], object_states_traj[0]) 
                            elif policy.m == 25:
                                prev_states[i] = np.append(hand_states_traj[0][3:28], object_states_traj[0]) 
                            elif policy.m == 26:
                                prev_states[i] = np.append(np.append(hand_states_traj[0][0], hand_states_traj[0][3:28]), object_states_traj[0]) 
                            elif policy.m == 27:
                                prev_states[i] = np.append(hand_states_traj[0][1:28], object_states_traj[0]) 
                        else:
                            if policy.m == 24:
                                prev_states[i] = hand_states_traj[0][4:28]
                            elif policy.m == 25:
                                prev_states[i] = hand_states_traj[0][3:28]
                            elif policy.m == 26:
                                prev_states[i] = np.append(hand_states_traj[0][0], hand_states_traj[0][3:28])
                            elif policy.m == 27:
                                prev_states[i] = hand_states_traj[0][1:28]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        if policy.m == 24:
                            prev_actions[i] = hand_states_traj[0][4:28]
                        elif policy.m == 25:
                            prev_actions[i] = hand_states_traj[0][3:28]
                        elif policy.m == 26:    
                            prev_actions[i] = np.append(hand_states_traj[0][0], hand_states_traj[0][3:28])
                        elif policy.m == 27:
                            prev_actions[i] = hand_states_traj[0][1:28]
                if visual:
                    print("Episode %d, KOdex with motion adapter (mean action)." %(ep + 1))
                r_z_motion = np.zeros(task_horizon)
                r_z_motion[0] = hand_states_traj[0][3].copy()
                while t < task_horizon - 1:  # what would be early-termination for door task?
                    self.render() if visual is True else None
                    o = self.get_obs()
                    if not policy.freeze_base:
                        current_hand_state = self.get_env_state()['qpos'][:num_hand]
                    else:
                        if policy.m == 24:  # 24 DoFs in door opening task
                            num_hand = 24
                            current_hand_state = self.get_env_state()['qpos'][4:28]
                        elif policy.m == 25:
                            num_hand = 25
                            current_hand_state = self.get_env_state()['qpos'][3:28]
                        elif policy.m == 26:
                            num_hand = 26
                            current_hand_state = np.append(self.get_env_state()['qpos'][0], self.get_env_state()['qpos'][3:28])
                        elif policy.m == 27:
                            num_hand = 27
                            current_hand_state = self.get_env_state()['qpos'][1:28]
                    current_objpos = o[32:35]  # in world frame
                    current_objvel = self.get_env_state()['qvel'][28:29]
                    init_hand_state = self.get_env_state()['door_body_pos']
                    hand_OriState = current_hand_state
                    obj_OriState = np.append(current_objpos, np.append(current_objvel, init_hand_state))  # ori: represented in the transformed frame
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][4:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 25:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 26:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28]), object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][1:28], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][4:28], object_states_traj[t + future_state[t_]])
                                        elif policy.m == 25:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][3:28], object_states_traj[t + future_state[t_]])
                                        elif policy.m == 26:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[t + future_state[t_]][0], hand_states_traj[t + future_state[t_]][3:28]), object_states_traj[t + future_state[t_]])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][1:28], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][4:28]
                                        elif policy.m == 25:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:28]
                                        elif policy.m == 26:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28])
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][1:28]
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][4:28]
                                        elif policy.m == 25:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][3:28]
                                        elif policy.m == 26:    
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][0], hand_states_traj[t + future_state[t_]][3:28])
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][1:28]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][4:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 25:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 26:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28]), object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][1:28], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][4:28], object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 25:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][3:28], object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 26:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[t + future_state[t_ - 1]][0], hand_states_traj[t + future_state[t_ - 1]][3:28]), object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][1:28], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][4:28]
                                        elif policy.m == 25:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:28]
                                        elif policy.m == 26:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28])
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][1:28]
                                    else:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][4:28]
                                        elif policy.m == 25:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][3:28]
                                        elif policy.m == 26:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][0], hand_states_traj[t + future_state[t_ - 1]][3:28])
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][1:28]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[1]['evaluation'] # mean_action is True -> noise-free actions from controller
                    try:
                        if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                            if not policy.freeze_base:
                                a += hand_states_traj[t + 1].copy() 
                            else:
                                if policy.m == 24:
                                    a += hand_states_traj[t + 1][4:28].copy() 
                                elif policy.m == 25:
                                    a += hand_states_traj[t + 1][3:28].copy() 
                                elif policy.m == 26:
                                    a += np.append(hand_states_traj[t + 1][0].copy(), hand_states_traj[t + 1][3:28].copy())
                                elif policy.m == 27:
                                    a += hand_states_traj[t + 1][1:28].copy() 
                    except:
                        pass
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        if not policy.freeze_base:
                            PID_controller.set_goal(a)
                        else:
                            if policy.m == 24:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:4], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:4], a))
                            elif policy.m == 25:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:3], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:3], a))
                            elif policy.m == 26:
                                PID_controller.set_goal(np.append(np.append(a[0], hand_states_traj[t + 1][1:3]), a[1:]))
                                num_hand = len(np.append(np.append(a[0], hand_states_traj[t + 1][1:3]), a[1:]))
                            elif policy.m == 27:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:1], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:1], a))
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    ep_returns_mean_act[ep] += (gamma ** t) * r  # only task specific rewards
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t + 1]
                    reference['obj_pos'] = object_states_traj[t + 1][:3]
                    reference['obj_vel'] = object_states_traj[t + 1][3:4]
                    reference['init_handle'] = object_states_traj[t + 1][4:]
                    obs = dict()
                    obs['hand_state'] = self.get_env_state()['qpos'][:num_hand]
                    obs['obj_pos'] = next_o[32:35]
                    obs['obj_vel'] = self.get_env_state()['qvel'][28:29]
                    obs['init_handle'] = self.get_env_state()['door_body_pos']
                    r_z_motion[t + 1] = obs['hand_state'][3].copy()
                    if not policy.freeze_base:
                        tracking_reward = Comp_tracking_reward_door(reference, obs, coeffcients)['total_reward']  
                        tracking_rewards_mean_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                        hand_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_door(reference, obs, coeffcients)['hand_reward']  
                        object_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_door(reference, obs, coeffcients)['object_reward']  
                    else:
                        if not policy.include_Rots:
                            tracking_reward = Comp_tracking_reward_door_freeze_base(reference, obs, coeffcients)['total_reward']  
                            tracking_rewards_mean_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                            hand_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_door_freeze_base(reference, obs, coeffcients)['hand_reward']  
                            object_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_door_freeze_base(reference, obs, coeffcients)['object_reward']  
                        else:
                            tracking_reward = Comp_tracking_reward_door_include_Rots(reference, obs, coeffcients)['total_reward']  
                            tracking_rewards_mean_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                            hand_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_door_include_Rots(reference, obs, coeffcients)['hand_reward']  
                            object_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_door_include_Rots(reference, obs, coeffcients)['object_reward']  
                    t += 1   
                    current_hinge_pos = next_o[28:29] # door opening angle
                R_z_motion_MA.append(r_z_motion)
                if current_hinge_pos > 1.35:
                    success_list_sim.append(1)
                total_score_mean_act += ep_returns_mean_act[ep]
                # re-set the experiments, for Motion adapter with noisy action
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                o = self.get_obs()
                t, done = 0, False
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                if not policy.freeze_base:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0]
                else:
                    for i in range(history_state):
                        if obj_dynamics:
                            if policy.m == 24:
                                prev_states[i] = np.append(hand_states_traj[0][4:28], object_states_traj[0]) 
                            elif policy.m == 25:
                                prev_states[i] = np.append(hand_states_traj[0][3:28], object_states_traj[0]) 
                            elif policy.m == 26:
                                prev_states[i] = np.append(np.append(hand_states_traj[0][0], hand_states_traj[0][3:28]), object_states_traj[0]) 
                            elif policy.m == 27:
                                prev_states[i] = np.append(hand_states_traj[0][1:28], object_states_traj[0]) 
                        else:
                            if policy.m == 24:
                                prev_states[i] = hand_states_traj[0][4:28]
                            elif policy.m == 25:
                                prev_states[i] = hand_states_traj[0][3:28]
                            elif policy.m == 26:
                                prev_states[i] = np.append(hand_states_traj[0][0], hand_states_traj[0][3:28])
                            elif policy.m == 27:
                                prev_states[i] = hand_states_traj[0][1:28]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        if policy.m == 24:
                            prev_actions[i] = hand_states_traj[0][4:28]           
                        elif policy.m == 25:
                            prev_actions[i] = hand_states_traj[0][3:28]           
                        elif policy.m == 26:
                            prev_actions[i] = np.append(hand_states_traj[0][0], hand_states_traj[0][3:28])
                        elif policy.m == 27:
                            prev_actions[i] = hand_states_traj[0][1:28]
                if visual:
                    print("Episode %d, KOdex with motion adapter (noisy action)." %(ep + 1))
                while t < task_horizon - 1:  # what would be early-termination for door task?
                    self.render() if visual is True else None
                    o = self.get_obs()
                    if not policy.freeze_base:
                        current_hand_state = self.get_env_state()['qpos'][:num_hand]
                    else:
                        if policy.m == 24:  # 24 DoFs in door opening task
                            num_hand = 24
                            current_hand_state = self.get_env_state()['qpos'][4:28]
                        elif policy.m == 25:
                            num_hand = 25
                            current_hand_state = self.get_env_state()['qpos'][3:28]
                        elif policy.m == 26:
                            num_hand = 26
                            current_hand_state = np.append(self.get_env_state()['qpos'][0], self.get_env_state()['qpos'][3:28])
                        elif policy.m == 27:
                            num_hand = 27
                            current_hand_state = self.get_env_state()['qpos'][1:28]
                    current_objpos = o[32:35]  # in world frame
                    current_objvel = self.get_env_state()['qvel'][28:29]
                    init_hand_state = self.get_env_state()['door_body_pos']
                    hand_OriState = current_hand_state
                    obj_OriState = np.append(current_objpos, np.append(current_objvel, init_hand_state))  # ori: represented in the transformed frame
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][4:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 25:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 26:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28]), object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][1:28], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][4:28], object_states_traj[t + future_state[t_]])
                                        elif policy.m == 25:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][3:28], object_states_traj[t + future_state[t_]])
                                        elif policy.m == 26:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[t + future_state[t_]][0], hand_states_traj[t + future_state[t_]][3:28]), object_states_traj[t + future_state[t_]])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][1:28], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][4:28]
                                        elif policy.m == 25:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:28]
                                        elif policy.m == 26:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28])
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][1:28]
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][4:28]
                                        elif policy.m == 25:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][3:28]
                                        elif policy.m == 26:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][0], hand_states_traj[t + future_state[t_]][3:28])
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][1:28]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][4:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 25:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 26:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28]), object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][1:28], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][4:28], object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 25:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][3:28], object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 26:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[t + future_state[t_ - 1]][0], hand_states_traj[t + future_state[t_ - 1]][3:28]), object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][1:28], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][4:28]
                                        elif policy.m == 25:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:28]
                                        elif policy.m == 26:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28])
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][1:28]
                                    else:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][4:28]
                                        elif policy.m == 25:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][3:28]
                                        elif policy.m == 26:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][0], hand_states_traj[t + future_state[t_ - 1]][3:28])
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][1:28]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[0] # mean_action is True -> noise-free actions from controller
                    try:
                        if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                            if not policy.freeze_base:
                                a += hand_states_traj[t + 1].copy() 
                            else:
                                if policy.m == 24:
                                    a += hand_states_traj[t + 1][4:28].copy() 
                                elif policy.m == 25:
                                    a += hand_states_traj[t + 1][3:28].copy() 
                                elif policy.m == 26:
                                    a += np.append(hand_states_traj[t + 1][0].copy(), hand_states_traj[t + 1][3:28].copy())
                                elif policy.m == 27:
                                    a += hand_states_traj[t + 1][1:28].copy() 
                    except:
                        pass
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        if not policy.freeze_base:
                            PID_controller.set_goal(a)
                        else:
                            if policy.m == 24:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:4], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:4], a))
                            elif policy.m == 25:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:3], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:3], a))
                            elif policy.m == 26:
                                PID_controller.set_goal(np.append(np.append(a[0], hand_states_traj[t + 1][1:3]), a[1:]))
                                num_hand = len(np.append(np.append(a[0], hand_states_traj[t + 1][1:3]), a[1:]))
                            elif policy.m == 27:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:1], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:1], a))
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    ep_returns_noisy_act[ep] += (gamma ** t) * r  # only task specific rewards
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t + 1]
                    reference['obj_pos'] = object_states_traj[t + 1][:3]
                    reference['obj_vel'] = object_states_traj[t + 1][3:4]
                    reference['init_handle'] = object_states_traj[t + 1][4:]
                    obs = dict()
                    obs['hand_state'] = self.get_env_state()['qpos'][:num_hand]
                    obs['obj_pos'] = next_o[32:35]
                    obs['obj_vel'] = self.get_env_state()['qvel'][28:29]
                    obs['init_handle'] = self.get_env_state()['door_body_pos']
                    if not policy.freeze_base:
                        tracking_reward = Comp_tracking_reward_door(reference, obs, coeffcients)['total_reward']  
                        tracking_rewards_noisy_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                        hand_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_door(reference, obs, coeffcients)['hand_reward']  
                        object_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_door(reference, obs, coeffcients)['object_reward']  
                    else:
                        if not policy.include_Rots:
                            tracking_reward = Comp_tracking_reward_door_freeze_base(reference, obs, coeffcients)['total_reward']  
                            tracking_rewards_noisy_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                            hand_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_door_freeze_base(reference, obs, coeffcients)['hand_reward']  
                            object_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_door_freeze_base(reference, obs, coeffcients)['object_reward']  
                        else:
                            tracking_reward = Comp_tracking_reward_door_include_Rots(reference, obs, coeffcients)['total_reward']  
                            tracking_rewards_noisy_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                            hand_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_door_include_Rots(reference, obs, coeffcients)['hand_reward']  
                            object_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_door_include_Rots(reference, obs, coeffcients)['object_reward']  
                    t += 1   
                    current_hinge_pos = next_o[28:29] # door opening angle
                if current_hinge_pos > 1.35:
                    success_list_sim_noisy.append(1)
            print("(Motion Adapter with mean action) Average score (task reward) = %f" % (total_score_mean_act / num_episodes))
            print("(Motion Adapter with mean action) Success rate = %f" % (len(success_list_sim) / num_episodes))
            print("(Motion Adapter with noisy action) Success rate = %f" % (len(success_list_sim_noisy) / num_episodes))
            print("(PD) Success rate = %f" % (len(success_list_PD) / num_episodes))
        elif task == 'hammer':
            success_list_sim = []
            success_list_PD = []
            success_list_sim_noisy = []
            success_index_PD = [0] * num_episodes
            success_index_MA_PD = [0] * num_episodes
            success_index_MA_PD_noisy = [0] * num_episodes
            MA_PD_plamPos = []
            PD_plamPos = []
            Joint_adaptations = []
            PD_hit_pos = []
            MA_hit_pos = []
            PD_hit_force = []
            MA_hit_force = []
            for ep in tqdm(range(num_episodes)):
                plamPos = []
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                init_hand_state = Eval_data[ep]['handpos']
                init_objpos = Eval_data[ep]['objpos']
                init_objori = Eval_data[ep]['objorient']
                init_objvel = Eval_data[ep]['objvel']
                goal_nail = Eval_data[ep]['nail_goal']
                already_hit = False
                hand_OriState = init_hand_state
                obj_OriState = np.append(init_objpos, np.append(init_objori, np.append(init_objvel, goal_nail)))  # ori: represented in the transformed frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                hand_states_traj = np.zeros([task_horizon, num_hand])
                object_states_traj = np.zeros([task_horizon, num_obj])
                joint_modifications = []
                hand_states_traj[0, :] = hand_OriState
                object_states_traj[0, :] = obj_OriState
                z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                if visual:
                    print("Episode %d, KOdex with PD." %(ep + 1))
                for t_ in range(task_horizon - 1):
                    self.render() if visual is True else None  
                    z_t_1_computed = np.dot(KODex, z_t)
                    z_t = z_t_1_computed.copy()
                    x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                    hand_OriState = x_t_1_computed[:num_hand]
                    obj_pos = x_t_1_computed[num_hand: num_hand + 3] # tool pos
                    obj_ori = x_t_1_computed[num_hand + 3: num_hand + 6] # tool ori
                    obj_vel = x_t_1_computed[num_hand + 6: num_hand + 12]
                    nail_pos = x_t_1_computed[num_hand + 12:]
                    obj_OriState = np.append(obj_pos, np.append(obj_ori, np.append(obj_vel, nail_pos)))
                    hand_states_traj[t_ + 1, :] = hand_OriState
                    object_states_traj[t_ + 1, :] = obj_OriState
                    if self.env.control_mode == 'PID':  # if Torque mode, we skil the visualization of PD controller.
                        PID_controller.set_goal(hand_OriState)
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            # for relocation task, it we set a higher control frequency, we can expect a much better PD performance
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            next_o, r, done, goal_achieved = self.step(torque_action)    
                    nail_impact = next_o[45]  # 0 -> force, 1 -> True
                    nail_position = next_o[42:45].copy()
                    nail_position[2] += 0.035 / 2
                    minimum_distance = 100000
                    contact_index = 1000
                    for i in range(self.env.sim.data.ncon):
                        con = self.env.sim.data.contact[i]
                        if np.linalg.norm(con.pos - nail_position) < minimum_distance:
                            contact_index = i
                            minimum_distance = np.linalg.norm(con.pos - nail_position)
                    if nail_impact and self.env.sim.data.ncon > 0 and not already_hit:
                        contact_pos = self.env.sim.data.contact[contact_index].pos
                        # print("contact pos:", contact_pos)
                        tips_force = self.contact_forces
                        PD_hit_force.append(self.env.data.sensordata[0])
                        # print("nail_position:", nail_position)
                        PD_hit_pos.append(contact_pos - nail_position)
                        already_hit = True
                    ep_returns_PD[ep] += (gamma ** t_) * r  # only task specific rewards
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t_ + 1]
                    reference['obj_pos'] = object_states_traj[t_ + 1][:3]
                    reference['obj_ori'] = object_states_traj[t_ + 1][3:6]
                    reference['obj_vel'] = object_states_traj[t_ + 1][6:12]
                    reference['nail_goal'] = object_states_traj[t_ + 1][12:]
                    obs = dict()
                    obs['hand_state'] = self.get_env_state()['qpos'][:num_hand]
                    obs['obj_pos'] = next_o[49:52] + next_o[42:45]
                    obs['obj_ori'] = next_o[39:42]
                    obs['obj_vel'] = next_o[27:33]
                    obs['nail_goal'] = next_o[46:49]
                    plamPos.append(next_o[35] - next_o[44]) # plam height - nail height
                    if not policy.freeze_base:
                        tracking_reward = Comp_tracking_reward_hammer(reference, obs, coeffcients)['total_reward'] 
                        tracking_rewards_PD[ep] += (gamma ** t_) * tracking_reward  # only tracking rewards
                        hand_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_hammer(reference, obs, coeffcients)['hand_reward'] 
                        object_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_hammer(reference, obs, coeffcients)['object_reward'] 
                    else:
                        tracking_reward = Comp_tracking_reward_hammer_freeze_base(reference, obs, coeffcients)['total_reward'] 
                        tracking_rewards_PD[ep] += (gamma ** t_) * tracking_reward  # only tracking rewards
                        hand_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_hammer_freeze_base(reference, obs, coeffcients)['hand_reward'] 
                        object_rewards_PD[ep] += (gamma ** t_) * Comp_tracking_reward_hammer_freeze_base(reference, obs, coeffcients)['object_reward'] 
                    dist = np.linalg.norm(next_o[42:45] - next_o[46:49])
                PD_plamPos.append(plamPos.copy())
                if dist < 0.01:
                    success_list_PD.append(1)
                    success_index_PD[ep] = 1
                # re-set the experiments, for Motion adapter with mean action
                plamPos = []
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                already_hit = False
                o = self.get_obs()
                t, done = 0, False
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                if not policy.freeze_base:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0]
                else:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0][2:26], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0][2:26]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0][2:26]
                if visual:
                    print("Episode %d, KOdex with motion adapter (mean action)." %(ep + 1))
                while t < task_horizon - 1:  # what would be early-termination for hammer task?
                    self.render() if visual is True else None
                    o = self.get_obs()
                    if not policy.freeze_base:
                        current_hand_state = self.get_env_state()['qpos'][:num_hand]
                    else:
                        current_hand_state = self.get_env_state()['qpos'][2:26]
                        num_hand = 24
                    current_objpos = o[49:52] + o[42:45]
                    current_objvel = o[27:33]
                    current_objori = o[39:42]
                    nail_goal = o[46:49]
                    hand_OriState = current_hand_state
                    obj_OriState = np.append(current_objpos, np.append(current_objori, np.append(current_objvel, nail_goal))) 
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][2:26], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][2:26], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][2:26]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][2:26]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][2:26], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][2:26], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][2:26]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][2:26]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[1]['evaluation'] # mean_action is True -> noise-free actions from controller
                    try:
                        if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                            if not policy.freeze_base:
                                a += hand_states_traj[t + 1].copy() 
                            else:
                                a += hand_states_traj[t + 1][2:26].copy() 
                    except:
                        pass
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        if not policy.freeze_base:
                            joint_modifications.append(a - hand_states_traj[t + 1, :]) # absolute diff between adapted joints and original joints
                            PID_controller.set_goal(a) # examize the a values (we could see that the motion adaper makes very large changes)
                        else:
                            tmp = np.append((hand_states_traj[t + 1, :][:2] - hand_states_traj[t + 1, :][:2]), np.abs(a - hand_states_traj[t + 1, :][2:26]))
                            joint_modifications.append(tmp) # absolute diff between adapted joints and original joints
                            PID_controller.set_goal(np.append(hand_states_traj[t + 1][:2], a))
                            num_hand = len(np.append(hand_states_traj[t + 1][:2], a))
                        # I think this issue could also be fixed when we encourage the adapted trajecotry to stay close to the original ones.
                        # print("reference value:", hand_states_traj[t + 1, :])
                        # print("a value:", a)
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    # return np.concatenate([qp[:-6], qv[-6:], palm_pos, obj_pos, obj_rot, target_pos, np.array([nail_impact]), goal_pos, tool_pos - target_pos])
                    nail_impact = next_o[45]  # 0 -> force, 1 -> True
                    nail_position = next_o[42:45].copy()
                    nail_position[2] += 0.035 / 2
                    minimum_distance = 100000
                    contact_index = 1000
                    for i in range(self.env.sim.data.ncon):
                        con = self.env.sim.data.contact[i]
                        if np.linalg.norm(con.pos - nail_position) < minimum_distance:
                            contact_index = i
                            minimum_distance = np.linalg.norm(con.pos - nail_position)
                    if nail_impact and self.env.sim.data.ncon > 0 and not already_hit:
                        contact_pos = self.env.sim.data.contact[contact_index].pos
                        MA_hit_force.append(self.env.data.sensordata[0])
                        # print("contact pos:", contact_pos)
                        # print("nail_position:", nail_position)
                        MA_hit_pos.append(contact_pos - nail_position)
                        already_hit = True
                    ep_returns_mean_act[ep] += (gamma ** t) * r  # only task specific rewards
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t + 1]
                    reference['obj_pos'] = object_states_traj[t + 1][:3]
                    reference['obj_ori'] = object_states_traj[t + 1][3:6]
                    reference['obj_vel'] = object_states_traj[t + 1][6:12]
                    reference['nail_goal'] = object_states_traj[t + 1][12:]
                    obs = dict()
                    obs['hand_state'] = self.get_env_state()['qpos'][:num_hand]
                    obs['obj_pos'] = next_o[49:52] + next_o[42:45]
                    obs['obj_ori'] = next_o[39:42]
                    obs['obj_vel'] = next_o[27:33]
                    obs['nail_goal'] = next_o[46:49]
                    plamPos.append(next_o[35] - next_o[44]) # plam height - nail height
                    if not policy.freeze_base:
                        tracking_reward = Comp_tracking_reward_hammer(reference, obs, coeffcients)['total_reward']  
                        tracking_rewards_mean_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                        hand_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_hammer(reference, obs, coeffcients)['hand_reward']  
                        object_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_hammer(reference, obs, coeffcients)['object_reward']  
                    else:
                        tracking_reward = Comp_tracking_reward_hammer_freeze_base(reference, obs, coeffcients)['total_reward']  
                        tracking_rewards_mean_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                        hand_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_hammer_freeze_base(reference, obs, coeffcients)['hand_reward']  
                        object_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_hammer_freeze_base(reference, obs, coeffcients)['object_reward']  
                    t += 1   
                    dist = np.linalg.norm(next_o[42:45] - next_o[46:49])
                Joint_adaptations.append(joint_modifications)
                MA_PD_plamPos.append(plamPos)
                if dist < 0.01:
                    success_list_sim.append(1)
                    success_index_MA_PD[ep] = 1
                total_score_mean_act += ep_returns_mean_act[ep]

                # re-set the experiments, for Motion adapter with noisy action
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                o = self.get_obs()
                t, done = 0, False
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                if not policy.freeze_base:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0]
                else:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0][2:26], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0][2:26]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0][2:26]
                if visual:
                    print("Episode %d, KOdex with motion adapter (noisy action)." %(ep + 1))
                while t < task_horizon - 1:  # what would be early-termination for door task?
                    self.render() if visual is True else None
                    o = self.get_obs()
                    if not policy.freeze_base:
                        current_hand_state = self.get_env_state()['qpos'][:num_hand]
                    else:
                        current_hand_state = self.get_env_state()['qpos'][2:26]
                        num_hand = 24
                    current_objpos = o[49:52] + o[42:45]
                    current_objvel = o[27:33]
                    current_objori = o[39:42]
                    nail_goal = o[46:49]
                    hand_OriState = current_hand_state
                    obj_OriState = np.append(current_objpos, np.append(current_objori, np.append(current_objvel, nail_goal))) 
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][2:26], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][2:26], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][2:26]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][2:26]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][2:26], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][2:26], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][2:26]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][2:26]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[0] # mean_action is True -> noise-free actions from controller
                    try:
                        if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                            if not policy.freeze_base:
                                a += hand_states_traj[t + 1].copy() 
                            else:
                                a += hand_states_traj[t + 1][2:26].copy() 
                    except:
                        pass
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        if not policy.freeze_base:
                            PID_controller.set_goal(a)
                        else:
                            PID_controller.set_goal(np.append(hand_states_traj[t + 1][:2], a))
                            num_hand = len(np.append(hand_states_traj[t + 1][:2], a))
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    ep_returns_noisy_act[ep] += (gamma ** t) * r  # only task specific rewards
                    reference = dict()
                    reference['hand_state'] = hand_states_traj[t + 1]
                    reference['obj_pos'] = object_states_traj[t + 1][:3]
                    reference['obj_ori'] = object_states_traj[t + 1][3:6]
                    reference['obj_vel'] = object_states_traj[t + 1][6:12]
                    reference['nail_goal'] = object_states_traj[t + 1][12:]
                    obs = dict()
                    obs['hand_state'] = self.get_env_state()['qpos'][:num_hand]
                    obs['obj_pos'] = next_o[49:52] + next_o[42:45]
                    obs['obj_ori'] = next_o[39:42]
                    obs['obj_vel'] = next_o[27:33]
                    obs['nail_goal'] = next_o[46:49]
                    if not policy.freeze_base:
                        tracking_reward = Comp_tracking_reward_hammer(reference, obs, coeffcients)['total_reward']  
                        tracking_rewards_noisy_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                        hand_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_hammer(reference, obs, coeffcients)['hand_reward']  
                        object_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_hammer(reference, obs, coeffcients)['object_reward']  
                    else:
                        tracking_reward = Comp_tracking_reward_hammer_freeze_base(reference, obs, coeffcients)['total_reward']  
                        tracking_rewards_noisy_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                        hand_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_hammer_freeze_base(reference, obs, coeffcients)['hand_reward']  
                        object_rewards_noisy_act[ep] += (gamma ** t) * Comp_tracking_reward_hammer_freeze_base(reference, obs, coeffcients)['object_reward']  
                    t += 1   
                    dist = np.linalg.norm(next_o[42:45] - next_o[46:49])
                if dist < 0.01:
                    success_list_sim_noisy.append(1)
                    success_index_MA_PD_noisy[ep] = 1
            # find the index
            hammer_compare_index = list()
            for index_ in range(num_episodes):
                # PD - MA_PD (successful rollouts on PD while unsuccessful on MA_PD)
                # if success_index_MA_PD[index_] == 0 and success_index_PD[index_] == 1:
                #     hammer_compare_index.append(index_)
                #     print(index_)
                # MA_PD - PD
                if success_index_MA_PD[index_] == 1 and success_index_PD[index_] == 0:
                    hammer_compare_index.append(index_)
            print("(Motion Adapter with mean action) Average score (task rewards) = %f" % (total_score_mean_act / num_episodes))
            print("(Motion Adapter with mean action) Success rate = %f" % (len(success_list_sim) / num_episodes))
            print("(Motion Adapter with noisy action) Success rate = %f" % (len(success_list_sim_noisy) / num_episodes))
            print("(PD) Success rate = %f" % (len(success_list_PD) / num_episodes))
        # ep_returns and tracking_rewards are both discounted rewards
        total_stats = [[ep_returns_PD, tracking_rewards_PD, hand_rewards_PD, object_rewards_PD], [ep_returns_mean_act, tracking_rewards_mean_act, hand_rewards_mean_act, object_rewards_mean_act], [ep_returns_noisy_act, tracking_rewards_noisy_act, hand_rewards_noisy_act, object_rewards_noisy_act]]
        base_stats = []
        for i in range(len(total_stats)):
            mean_eval, std = np.mean(total_stats[i][0]), np.std(total_stats[i][0])  # mean eval -> rewards
            min_eval, max_eval = np.amin(total_stats[i][0]), np.amax(total_stats[i][0])
            mean_tracking_eval, tracking_std = np.mean(total_stats[i][1]), np.std(total_stats[i][1])  # mean eval -> rewards
            min_tracking_eval, max_tracking_eval = np.amin(total_stats[i][1]), np.amax(total_stats[i][1])
            mean_hand_tracking_eval, hand_tracking_std = np.mean(total_stats[i][2]), np.std(total_stats[i][2]) 
            min_hand_tracking_eval, max_hand_tracking_eval = np.amin(total_stats[i][2]), np.amax(total_stats[i][2])
            mean_object_tracking_eval, object_tracking_std = np.mean(total_stats[i][3]), np.std(total_stats[i][3]) 
            min_object_tracking_eval, max_object_tracking_eval = np.amin(total_stats[i][3]), np.amax(total_stats[i][3])
            base_stats.append([mean_eval, std, min_eval, max_eval, mean_tracking_eval, tracking_std, min_tracking_eval, max_tracking_eval, mean_hand_tracking_eval, mean_object_tracking_eval])
        if task == 'relocate':
            return base_stats, [force_compare_index, MA_PD_tips_force, PD_tips_force, Joint_adaptations] 
        elif task == 'hammer':
            return base_stats, [hammer_compare_index, MA_PD_plamPos, PD_plamPos, Joint_adaptations, PD_hit_pos, MA_hit_pos, PD_hit_force, MA_hit_force] 
        elif task == 'door':
            return base_stats, [R_z_motion_MA, R_z_motion_PD]
        else:
            return base_stats

    def demo_collection(self, 
                        Eval_data,
                        PID_controller,
                        coeffcients,
                        Koopman_obser,
                        KODex, 
                        task_horizon, 
                        future_state,
                        history_state,
                        policy,
                        num_episodes=5,
                        obj_dynamics=True,
                        gamma=1,
                        visual=False,
                        terminate_at_done=True,
                        seed=123):
        self.set_seed(seed)
        task = self.env_id.split('-')[0]
        # success_threshold = 20 if task == 'pen' else 25
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns_mean_act = np.zeros(num_episodes)
        # tracking_rewards -> hand tracking rewards + object tracking rewards
        tracking_rewards_mean_act = np.zeros(num_episodes)
        # hand tracking rewards
        hand_rewards_mean_act = np.zeros(num_episodes)
        # object tracking rewards
        object_rewards_mean_act = np.zeros(num_episodes)
        episodes = []
        total_score_mean_act = 0.0
        num_future_s = len(future_state)
        success_threshold = 10
        for ep in tqdm(range(num_episodes)):
            init_hand_state = Eval_data[ep]['handpos']
            init_objpos = Eval_data[ep]['objpos'] # converged object position
            init_objvel = Eval_data[ep]['objvel']
            init_objori = Eval_data[ep]['objorient']
            desired_pos = Eval_data[ep]['desired_pos']
            init_objpos_world = desired_pos + init_objpos # in the world frame(on the table)
            hand_OriState = init_hand_state
            obj_OriState = np.append(init_objpos, np.append(init_objori, init_objvel))  # ori: represented in the transformed frame (converged to desired pos)
            obj_OriState_ = np.append(init_objpos_world, np.append(init_objori, init_objvel)) # ori: represented in the world frame
            num_hand = len(hand_OriState)
            num_obj = len(obj_OriState)
            hand_states_traj = np.zeros([task_horizon, num_hand])
            object_states_traj = np.zeros([task_horizon, num_obj])
            joint_modifications = []
            hand_states_traj[0, :] = hand_OriState
            object_states_traj[0, :] = obj_OriState_
            z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
            for t_ in range(task_horizon - 1):
                z_t_1_computed = np.dot(KODex, z_t)
                z_t = z_t_1_computed.copy()
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                hand_OriState = x_t_1_computed[:num_hand]
                obj_pos = x_t_1_computed[num_hand: num_hand + 3] # converged object position
                obj_pos_world = desired_pos + obj_pos
                obj_ori = x_t_1_computed[num_hand + 3: num_hand + 6]
                obj_vel = x_t_1_computed[num_hand + 6: num_hand + 12]
                obj_OriState = np.append(obj_pos_world, np.append(obj_ori, obj_vel))
                hand_states_traj[t_ + 1, :] = hand_OriState
                object_states_traj[t_ + 1, :] = obj_OriState
            # initialze the experiments, for PD controller
            self.reset()
            self.set_env_state(Eval_data[ep]['init_states'])
            episode_data = {
                'init_state_dict': copy.deepcopy(Eval_data[ep]['init_states']),  # set the initial states/the desired orientation is represented in quat
                'index': ep,
                'PD_targets': [],  
                'hand_state': [],  # hand state
                'obj_pos': [],
                'obj_vel': [],
                'obj_ori': [],
                'goal_achieved': []
            }
            o = self.get_obs()
            t, done, obj_height = 0, False, o[41]
            # for the default history states, set them to be initial hand states
            prev_states = dict()
            if not policy.freeze_base:
                for i in range(history_state):
                    if obj_dynamics:
                        prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                    else:
                        prev_states[i] = hand_states_traj[0]
                prev_actions = dict()
                for i in range(history_state + 1):
                    prev_actions[i] = hand_states_traj[0]
            else:
                for i in range(history_state):
                    if obj_dynamics:
                        prev_states[i] = np.append(hand_states_traj[0][6:30], object_states_traj[0]) 
                    else:
                        prev_states[i] = hand_states_traj[0][6:30]
                prev_actions = dict()
                for i in range(history_state + 1):
                    prev_actions[i] = hand_states_traj[0][6:30]
            if visual:
                print("Episode %d, KOdex with motion adapter (mean action)." %(ep + 1))
            while t < task_horizon - 1 and obj_height > -0.05 :  # what would be early-termination for relocation task?
                self.render() if visual is True else None
                o = self.get_obs()
                episode_data['hand_state'].append(o[:30])
                episode_data['obj_pos'].append(o[39:42])
                episode_data['obj_vel'].append(self.get_env_state()['qvel'][30:36])
                episode_data['obj_ori'].append(self.get_env_state()['qpos'][33:36])
                if not policy.freeze_base:
                    current_hand_state = o[:30]
                else:
                    current_hand_state = o[6:30]
                    num_hand = 24
                current_objpos = o[39:42]  # in world frame
                current_objvel = self.get_env_state()['qvel'][30:36]
                current_objori = self.get_env_state()['qpos'][33:36]
                hand_OriState = current_hand_state
                obj_OriState = np.append(current_objpos, np.append(current_objori, current_objvel))  # ori: represented in the transformed frame
                if history_state >= 0: # we add the history information into policy inputs
                    if obj_dynamics:  # if we use the object dynamics as part of policy input
                        policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                        prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                        for i in range(history_state + 1):
                            policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                        future_index = (history_state + 1) * (2 * num_hand + num_obj)
                        if not policy.freeze_base:
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= task_horizon:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                else:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                        else:
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= task_horizon:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][6:30], object_states_traj[task_horizon - 1])
                                else:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][6:30], object_states_traj[t + future_state[t_]])
                    else:
                        policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                        prev_states[history_state] = hand_OriState # add the current states 
                        for i in range(history_state + 1):
                            policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                        future_index = (history_state + 1) * (2 * num_hand)
                        if not policy.freeze_base:
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= task_horizon:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                else:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                        else:
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= task_horizon:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][6:30]
                                else:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][6:30]
                else: # no history information into policy inputs
                    if obj_dynamics:  # if we use the object dynamics as part of policy input
                        policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                        policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                        if not policy.freeze_base:
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= task_horizon:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                else:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= task_horizon:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][6:30], object_states_traj[task_horizon - 1])
                                else:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][6:30], object_states_traj[t + future_state[t_ - 1]])
                    else:
                        policy_input = np.zeros((num_future_s + 1) * num_hand)
                        policy_input[:num_hand] = hand_OriState
                        if not policy.freeze_base:
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= task_horizon:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                else:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                        else:
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= task_horizon:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][6:30]
                                else:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][6:30]
                a = policy.get_action(policy_input)  # current states and future goal states
                a = a[1]['evaluation'] # mean_action is True -> noise-free actions from controller
                try:
                    if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                        if not policy.freeze_base:
                            a += hand_states_traj[t + 1].copy() 
                        else:
                            a += hand_states_traj[t + 1][6:30].copy() 
                except:
                    pass
                # update the history information
                if history_state >= 0:
                    for i in range(history_state):
                        prev_actions[i] = prev_actions[i + 1]
                        prev_states[i] = prev_states[i + 1]
                    prev_actions[history_state] = a
                if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                    next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                elif self.env.control_mode == 'PID': # a represents the target joint position 
                    if not policy.freeze_base:
                        PID_controller.set_goal(a)  # the changes (np.abs(a - hand_states_traj[t + 1, :])) is much smaller than those in Hammer task.
                        episode_data['PD_targets'].append(a.copy())
                    else:
                        PID_controller.set_goal(np.append(hand_states_traj[t + 1][:6], a))  # the changes (np.abs(a - hand_states_traj[t + 1, :])) is much smaller than those in Hammer task.
                        episode_data['PD_targets'].append(np.append(hand_states_traj[t + 1][:6], a).copy())
                        num_hand = len(np.append(hand_states_traj[t + 1][:6], a))
                    # print("reference value:", hand_states_traj[t + 1, :])
                    # print("a value:", a)                        
                    for _z in range(5): # control frequency: 500HZ / use this to output PD target
                        torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                        torque_action[1] -= 0.95  # hand_Txyz[1] -> hand_T_y
                        next_o, r, done, goal_achieved = self.step(torque_action)  
                obj_height = next_o[41]
                episode_data['goal_achieved'].append(goal_achieved['goal_achieved'])
                ep_returns_mean_act[ep] += (gamma ** t) * r  # only task specific rewards
                reference = dict()
                reference['hand_state'] = hand_states_traj[t + 1]
                reference['obj_pos'] = object_states_traj[t + 1][:3]
                reference['obj_vel'] = object_states_traj[t + 1][6:]
                reference['obj_ori'] = object_states_traj[t + 1][3:6]
                obs = dict()
                obs['hand_state'] = next_o[:30]
                obs['obj_pos'] = next_o[39:42]
                obs['obj_vel'] = self.get_env_state()['qvel'][30:36]
                obs['obj_ori'] = self.get_env_state()['qpos'][33:36]
                if not policy.freeze_base:
                    tracking_reward = Comp_tracking_reward_relocate(reference, obs, coeffcients)['total_reward']
                    tracking_rewards_mean_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                    hand_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate(reference, obs, coeffcients)['hand_reward']
                    object_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate(reference, obs, coeffcients)['object_reward']
                else:
                    tracking_reward = Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['total_reward']
                    tracking_rewards_mean_act[ep] += (gamma ** t) * tracking_reward  # only tracking rewards
                    hand_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['hand_reward']
                    object_rewards_mean_act[ep] += (gamma ** t) * Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['object_reward']
                t += 1   
            o = self.get_obs()
            episode_data['hand_state'].append(o[:30])
            episode_data['obj_pos'].append(o[39:42])
            episode_data['obj_vel'].append(self.get_env_state()['qvel'][30:36])
            episode_data['obj_ori'].append(self.get_env_state()['qpos'][33:36])
            # print("tracking_rewards_mean_act[ep]", tracking_rewards_mean_act[ep])
            episodes.append(copy.deepcopy(episode_data))
            total_score_mean_act += ep_returns_mean_act[ep]
        # episode['goal_achieved'][-1] == True -> ensure the stable grasp at end
        successful_episodes = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold and episode['goal_achieved'][-1] == True, episodes))       
        print("(Motion Adapter with mean action) Average score (task reward) = %f" % (total_score_mean_act / num_episodes))
        print("(Motion Adapter with mean action) Success rate = %f" % (len(successful_episodes) / len(episodes)))
        print("Number of success rollouts = %d." % len(successful_episodes))
        if (not visual):
            pickle.dump(successful_episodes, open('/home/yhan389/Desktop/MultiObj_KODex/Dataset/New_Grasping_Relocation/Cylinder/demo.pickle', 'wb'))
    
    def Visualze_CIMER_policy(self, 
                        Eval_data,
                        PID_controller,
                        coeffcients,
                        Koopman_obser,
                        KODex, 
                        task_horizon, 
                        future_state,
                        history_state,
                        policy,
                        num_episodes=5,
                        obj_dynamics=True,
                        gamma=1,
                        visual=False,
                        object_name='',
                        terminate_at_done=True,
                        seed=123):
        self.set_seed(seed)
        task = self.env_id.split('-')[0]
        num_future_s = len(future_state)
        save_path = os.path.join(os.getcwd(), 'Videos', task, object_name)[:-1] + '_CIMER.mp4'
        # save_traj = 50  # used for the default object on each task
        save_traj = 20  # used for the new objects on the relocation task
        if task == 'relocate':
            success_list_sim = []
            success_threshold = 10
            for ep in tqdm(range(num_episodes)):
                Episodes = []
                init_hand_state = Eval_data[ep]['handpos']
                init_objpos = Eval_data[ep]['objpos'] # converged object position
                init_objvel = Eval_data[ep]['objvel']
                init_objori = Eval_data[ep]['objorient']
                desired_pos = Eval_data[ep]['desired_pos']
                init_objpos_world = desired_pos + init_objpos # in the world frame(on the table)
                hand_OriState = init_hand_state
                obj_OriState = np.append(init_objpos, np.append(init_objori, init_objvel))  # ori: represented in the transformed frame (converged to desired pos)
                obj_OriState_ = np.append(init_objpos_world, np.append(init_objori, init_objvel)) # ori: represented in the world frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                hand_states_traj = np.zeros([task_horizon, num_hand])
                object_states_traj = np.zeros([task_horizon, num_obj])
                joint_modifications = []
                hand_states_traj[0, :] = hand_OriState
                object_states_traj[0, :] = obj_OriState_
                z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                for t_ in range(task_horizon - 1):
                    z_t_1_computed = np.dot(KODex, z_t)
                    z_t = z_t_1_computed.copy()
                    x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                    hand_OriState = x_t_1_computed[:num_hand]
                    obj_pos = x_t_1_computed[num_hand: num_hand + 3] # converged object position
                    obj_pos_world = desired_pos + obj_pos
                    obj_ori = x_t_1_computed[num_hand + 3: num_hand + 6]
                    obj_vel = x_t_1_computed[num_hand + 6: num_hand + 12]
                    obj_OriState = np.append(obj_pos_world, np.append(obj_ori, obj_vel))
                    hand_states_traj[t_ + 1, :] = hand_OriState
                    object_states_traj[t_ + 1, :] = obj_OriState 
                # re-set the experiments, for Motion adapter with mean action
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                o = self.get_obs()
                t, done, obj_height = 0, False, o[41]
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                if not policy.freeze_base:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0]
                else:
                    for i in range(history_state):
                        if obj_dynamics:
                            if policy.m == 24:
                                prev_states[i] = np.append(hand_states_traj[0][6:30], object_states_traj[0]) 
                            elif policy.m == 27:
                                prev_states[i] = np.append(hand_states_traj[0][3:30], object_states_traj[0]) 
                        else:
                            if policy.m == 24:
                                prev_states[i] = hand_states_traj[0][6:30]
                            elif policy.m == 27:
                                prev_states[i] = hand_states_traj[0][3:30]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        if policy.m == 24:
                            prev_actions[i] = hand_states_traj[0][6:30]
                        elif policy.m == 27:
                            prev_actions[i] = hand_states_traj[0][3:30]
                if visual:
                    print("Episode %d, KOdex with motion adapter (mean action)." %(ep + 1))
                while t < task_horizon - 1 and obj_height > -0.05 :  # what would be early-termination for relocation task?
                    self.render() if visual is True else None
                    o = self.get_obs()
                    if not policy.freeze_base:
                        current_hand_state = o[:30]
                    else:
                        if policy.m == 24:
                            current_hand_state = o[6:30]
                            num_hand = 24
                        elif policy.m == 27:
                            current_hand_state = o[3:30]
                            num_hand = 27
                    current_objpos = o[39:42]  # in world frame
                    current_objvel = self.get_env_state()['qvel'][30:36]
                    current_objori = self.get_env_state()['qpos'][33:36]
                    hand_OriState = current_hand_state
                    obj_OriState = np.append(current_objpos, np.append(current_objori, current_objvel))  # ori: represented in the transformed frame
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][6:30], object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:30], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][6:30], object_states_traj[t + future_state[t_]])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][3:30], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][6:30]
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:30]
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][6:30]
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][3:30]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][6:30], object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:30], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][6:30], object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][3:30], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][6:30]
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:30]
                                    else:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][6:30]
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][3:30]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[1]['evaluation'] # mean_action is True -> noise-free actions from controller
                    try:
                        if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                            if not policy.freeze_base:
                                a += hand_states_traj[t + 1].copy() 
                            else:
                                if policy.m == 24:
                                    a += hand_states_traj[t + 1][6:30].copy() 
                                elif policy.m == 27:
                                    a += hand_states_traj[t + 1][3:30].copy() 
                    except:
                        pass
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        if not policy.freeze_base:
                            joint_modifications.append(np.abs(a - hand_states_traj[t + 1, :])) # absolute diff between adapted joints and original joints
                            PID_controller.set_goal(a)  # the changes (np.abs(a - hand_states_traj[t + 1, :])) is much smaller than those in Hammer task.
                        else:
                            if policy.m == 24:
                                tmp = np.append((hand_states_traj[t + 1, :][:6] - hand_states_traj[t + 1, :][:6]), np.abs(a - hand_states_traj[t + 1, :][6:30]))
                                joint_modifications.append(tmp) # absolute diff between adapted joints and original joints
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:6], a))  # the changes (np.abs(a - hand_states_traj[t + 1, :])) is much smaller than those in Hammer task.
                                num_hand = len(np.append(hand_states_traj[t + 1][:6], a))
                            elif policy.m == 27:
                                tmp = np.append((hand_states_traj[t + 1, :][:3] - hand_states_traj[t + 1, :][:3]), np.abs(a - hand_states_traj[t + 1, :][3:30]))
                                joint_modifications.append(tmp) # absolute diff between adapted joints and original joints
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:3], a))  # the changes (np.abs(a - hand_states_traj[t + 1, :])) is much smaller than those in Hammer task.
                                num_hand = len(np.append(hand_states_traj[t + 1][:3], a))
                        # print("reference value:", hand_states_traj[t + 1, :])
                        # print("a value:", a)                        
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            torque_action[1] -= 0.95  # hand_Txyz[1] -> hand_T_y
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    Episodes.append(goal_achieved['goal_achieved'])
                    obj_height = next_o[41]
                    t += 1   
                if sum(Episodes) > success_threshold:
                    success_list_sim.append(1)
            print("(CIMER) Success rate = %f" % (len(success_list_sim) / num_episodes))
        elif task == 'door':
            success_list_sim = []
            for ep in tqdm(range(num_episodes)):
                init_hand_state = Eval_data[ep]['handpos']
                init_objpos = Eval_data[ep]['objpos']
                init_objvel = Eval_data[ep]['objvel']
                init_handle = Eval_data[ep]['handle_init']
                hand_OriState = init_hand_state
                obj_OriState = np.append(init_objpos, np.append(init_objvel, init_handle))  # ori: represented in the transformed frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                hand_states_traj = np.zeros([task_horizon, num_hand])
                object_states_traj = np.zeros([task_horizon, num_obj])
                hand_states_traj[0, :] = hand_OriState
                object_states_traj[0, :] = obj_OriState
                z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                if visual:
                    print("Episode %d, KOdex with PD." %(ep + 1))
                for t_ in range(task_horizon - 1):
                    z_t_1_computed = np.dot(KODex, z_t)
                    z_t = z_t_1_computed.copy()
                    x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                    hand_OriState = x_t_1_computed[:num_hand]
                    obj_pos = x_t_1_computed[num_hand: num_hand + 3] # converged object position
                    obj_vel = x_t_1_computed[num_hand + 3: num_hand + 4]
                    init_handle = x_t_1_computed[num_hand + 4: num_hand + 7]
                    obj_OriState = np.append(obj_pos, np.append(obj_vel, init_handle))
                    hand_states_traj[t_ + 1, :] = hand_OriState
                    object_states_traj[t_ + 1, :] = obj_OriState
                # re-set the experiments, for Motion adapter with mean action
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                o = self.get_obs()
                t, done = 0, False
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                if not policy.freeze_base:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0]
                else:
                    for i in range(history_state):
                        if obj_dynamics:
                            if policy.m == 24:
                                prev_states[i] = np.append(hand_states_traj[0][4:28], object_states_traj[0]) 
                            elif policy.m == 25:
                                prev_states[i] = np.append(hand_states_traj[0][3:28], object_states_traj[0]) 
                            elif policy.m == 26:
                                prev_states[i] = np.append(np.append(hand_states_traj[0][0], hand_states_traj[0][3:28]), object_states_traj[0]) 
                            elif policy.m == 27:
                                prev_states[i] = np.append(hand_states_traj[0][1:28], object_states_traj[0]) 
                        else:
                            if policy.m == 24:
                                prev_states[i] = hand_states_traj[0][4:28]
                            elif policy.m == 25:
                                prev_states[i] = hand_states_traj[0][3:28]
                            elif policy.m == 26:
                                prev_states[i] = np.append(hand_states_traj[0][0], hand_states_traj[0][3:28])
                            elif policy.m == 27:
                                prev_states[i] = hand_states_traj[0][1:28]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        if policy.m == 24:
                            prev_actions[i] = hand_states_traj[0][4:28]
                        elif policy.m == 25:
                            prev_actions[i] = hand_states_traj[0][3:28]
                        elif policy.m == 26:    
                            prev_actions[i] = np.append(hand_states_traj[0][0], hand_states_traj[0][3:28])
                        elif policy.m == 27:
                            prev_actions[i] = hand_states_traj[0][1:28]
                if visual:
                    print("Episode %d, KOdex with motion adapter (mean action)." %(ep + 1))
                while t < task_horizon - 1:  # what would be early-termination for door task?
                    self.render() if visual is True else None
                    o = self.get_obs()
                    if not policy.freeze_base:
                        current_hand_state = self.get_env_state()['qpos'][:num_hand]
                    else:
                        if policy.m == 24:  # 24 DoFs in door opening task
                            num_hand = 24
                            current_hand_state = self.get_env_state()['qpos'][4:28]
                        elif policy.m == 25:
                            num_hand = 25
                            current_hand_state = self.get_env_state()['qpos'][3:28]
                        elif policy.m == 26:
                            num_hand = 26
                            current_hand_state = np.append(self.get_env_state()['qpos'][0], self.get_env_state()['qpos'][3:28])
                        elif policy.m == 27:
                            num_hand = 27
                            current_hand_state = self.get_env_state()['qpos'][1:28]
                    current_objpos = o[32:35]  # in world frame
                    current_objvel = self.get_env_state()['qvel'][28:29]
                    init_hand_state = self.get_env_state()['door_body_pos']
                    hand_OriState = current_hand_state
                    obj_OriState = np.append(current_objpos, np.append(current_objvel, init_hand_state))  # ori: represented in the transformed frame
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][4:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 25:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 26:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28]), object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][1:28], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][4:28], object_states_traj[t + future_state[t_]])
                                        elif policy.m == 25:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][3:28], object_states_traj[t + future_state[t_]])
                                        elif policy.m == 26:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[t + future_state[t_]][0], hand_states_traj[t + future_state[t_]][3:28]), object_states_traj[t + future_state[t_]])
                                        elif policy.m == 27:
                                            policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][1:28], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][4:28]
                                        elif policy.m == 25:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:28]
                                        elif policy.m == 26:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28])
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][1:28]
                                    else:
                                        if policy.m == 24:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][4:28]
                                        elif policy.m == 25:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][3:28]
                                        elif policy.m == 26:    
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][0], hand_states_traj[t + future_state[t_]][3:28])
                                        elif policy.m == 27:
                                            policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][1:28]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][4:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 25:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][3:28], object_states_traj[task_horizon - 1])
                                        elif policy.m == 26:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28]), object_states_traj[task_horizon - 1])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][1:28], object_states_traj[task_horizon - 1])
                                    else:
                                        if policy.m == 24:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][4:28], object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 25:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][3:28], object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 26:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[t + future_state[t_ - 1]][0], hand_states_traj[t + future_state[t_ - 1]][3:28]), object_states_traj[t + future_state[t_ - 1]])
                                        elif policy.m == 27:
                                            policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][1:28], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][4:28]
                                        elif policy.m == 25:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][3:28]
                                        elif policy.m == 26:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][0], hand_states_traj[task_horizon - 1][3:28])
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][1:28]
                                    else:
                                        if policy.m == 24:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][4:28]
                                        elif policy.m == 25:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][3:28]
                                        elif policy.m == 26:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][0], hand_states_traj[t + future_state[t_ - 1]][3:28])
                                        elif policy.m == 27:
                                            policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][1:28]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[1]['evaluation'] # mean_action is True -> noise-free actions from controller
                    try:
                        if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                            if not policy.freeze_base:
                                a += hand_states_traj[t + 1].copy() 
                            else:
                                if policy.m == 24:
                                    a += hand_states_traj[t + 1][4:28].copy() 
                                elif policy.m == 25:
                                    a += hand_states_traj[t + 1][3:28].copy() 
                                elif policy.m == 26:
                                    a += np.append(hand_states_traj[t + 1][0].copy(), hand_states_traj[t + 1][3:28].copy())
                                elif policy.m == 27:
                                    a += hand_states_traj[t + 1][1:28].copy() 
                    except:
                        pass
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        if not policy.freeze_base:
                            PID_controller.set_goal(a)
                        else:
                            if policy.m == 24:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:4], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:4], a))
                            elif policy.m == 25:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:3], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:3], a))
                            elif policy.m == 26:
                                PID_controller.set_goal(np.append(np.append(a[0], hand_states_traj[t + 1][1:3]), a[1:]))
                                num_hand = len(np.append(np.append(a[0], hand_states_traj[t + 1][1:3]), a[1:]))
                            elif policy.m == 27:
                                PID_controller.set_goal(np.append(hand_states_traj[t + 1][:1], a))
                                num_hand = len(np.append(hand_states_traj[t + 1][:1], a))
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    t += 1   
                    current_hinge_pos = next_o[28:29] # door opening angle
                if current_hinge_pos > 1.35:
                    success_list_sim.append(1)
            print("(CIMER) Success rate = %f" % (len(success_list_sim) / num_episodes))
        elif task == 'hammer':
            success_list_sim = []
            for ep in tqdm(range(num_episodes)):
                init_hand_state = Eval_data[ep]['handpos']
                init_objpos = Eval_data[ep]['objpos']
                init_objori = Eval_data[ep]['objorient']
                init_objvel = Eval_data[ep]['objvel']
                goal_nail = Eval_data[ep]['nail_goal']
                hand_OriState = init_hand_state
                obj_OriState = np.append(init_objpos, np.append(init_objori, np.append(init_objvel, goal_nail)))  # ori: represented in the transformed frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                hand_states_traj = np.zeros([task_horizon, num_hand])
                object_states_traj = np.zeros([task_horizon, num_obj])
                hand_states_traj[0, :] = hand_OriState
                object_states_traj[0, :] = obj_OriState
                z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                for t_ in range(task_horizon - 1):
                    z_t_1_computed = np.dot(KODex, z_t)
                    z_t = z_t_1_computed.copy()
                    x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                    hand_OriState = x_t_1_computed[:num_hand]
                    obj_pos = x_t_1_computed[num_hand: num_hand + 3] # tool pos
                    obj_ori = x_t_1_computed[num_hand + 3: num_hand + 6] # tool ori
                    obj_vel = x_t_1_computed[num_hand + 6: num_hand + 12]
                    nail_pos = x_t_1_computed[num_hand + 12:]
                    obj_OriState = np.append(obj_pos, np.append(obj_ori, np.append(obj_vel, nail_pos)))
                    hand_states_traj[t_ + 1, :] = hand_OriState
                    object_states_traj[t_ + 1, :] = obj_OriState
                # re-set the experiments, for Motion adapter with mean action
                self.reset()
                self.set_env_state(Eval_data[ep]['init_states'])
                o = self.get_obs()
                t, done = 0, False
                # for the default history states, set them to be initial hand states
                prev_states = dict()
                if not policy.freeze_base:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0]
                else:
                    for i in range(history_state):
                        if obj_dynamics:
                            prev_states[i] = np.append(hand_states_traj[0][2:26], object_states_traj[0]) 
                        else:
                            prev_states[i] = hand_states_traj[0][2:26]
                    prev_actions = dict()
                    for i in range(history_state + 1):
                        prev_actions[i] = hand_states_traj[0][2:26]
                if visual:
                    print("Episode %d, CIEMR policy." %(ep + 1))
                while t < task_horizon - 1:  # what would be early-termination for hammer task?
                    self.render() if visual is True else None  
                    o = self.get_obs()
                    if not policy.freeze_base:
                        current_hand_state = self.get_env_state()['qpos'][:num_hand]
                    else:
                        current_hand_state = self.get_env_state()['qpos'][2:26]
                        num_hand = 24
                    current_objpos = o[49:52] + o[42:45]
                    current_objvel = o[27:33]
                    current_objori = o[39:42]
                    nail_goal = o[46:49]
                    hand_OriState = current_hand_state
                    obj_OriState = np.append(current_objpos, np.append(current_objori, np.append(current_objvel, nail_goal))) 
                    if history_state >= 0: # we add the history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                            prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand + num_obj)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][2:26], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][2:26], object_states_traj[t + future_state[t_]])
                        else:
                            policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                            prev_states[history_state] = hand_OriState # add the current states 
                            for i in range(history_state + 1):
                                policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                            future_index = (history_state + 1) * (2 * num_hand)
                            if not policy.freeze_base:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                            else:
                                for t_ in range(num_future_s):
                                    if t + future_state[t_] >= task_horizon:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][2:26]
                                    else:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][2:26]
                    else: # no history information into policy inputs
                        if obj_dynamics:  # if we use the object dynamics as part of policy input
                            policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                            policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[task_horizon - 1][2:26], object_states_traj[task_horizon - 1])
                                    else:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][2:26], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            policy_input = np.zeros((num_future_s + 1) * num_hand)
                            policy_input[:num_hand] = hand_OriState
                            if not policy.freeze_base:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                            else:
                                for t_ in range(1, num_future_s + 1):
                                    if t + future_state[t_ - 1] >= task_horizon:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[task_horizon - 1][2:26]
                                    else:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][2:26]
                    a = policy.get_action(policy_input)  # current states and future goal states
                    a = a[1]['evaluation'] # mean_action is True -> noise-free actions from controller
                    try:
                        if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                            if not policy.freeze_base:
                                a += hand_states_traj[t + 1].copy() 
                            else:
                                a += hand_states_traj[t + 1][2:26].copy() 
                    except:
                        pass
                    # update the history information
                    if history_state >= 0:
                        for i in range(history_state):
                            prev_actions[i] = prev_actions[i + 1]
                            prev_states[i] = prev_states[i + 1]
                        prev_actions[history_state] = a
                    if self.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                        next_o, r, done, goal_achieved = self.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                    elif self.env.control_mode == 'PID': # a represents the target joint position 
                        if not policy.freeze_base:
                            PID_controller.set_goal(a) # examize the a values (we could see that the motion adaper makes very large changes)
                        else:
                            PID_controller.set_goal(np.append(hand_states_traj[t + 1][:2], a))
                            num_hand = len(np.append(hand_states_traj[t + 1][:2], a))
                        # I think this issue could also be fixed when we encourage the adapted trajecotry to stay close to the original ones.
                        # print("reference value:", hand_states_traj[t + 1, :])
                        # print("a value:", a)
                        for _z in range(5): # control frequency: 500HZ / use this to output PD target
                            torque_action = PID_controller(self.get_env_state()['qpos'][:num_hand], self.get_env_state()['qvel'][:num_hand])
                            next_o, r, done, goal_achieved = self.step(torque_action)  
                    t += 1   
                    dist = np.linalg.norm(next_o[42:45] - next_o[46:49])
                if dist < 0.01:
                    success_list_sim.append(1)
            print("(CIMER) Success rate = %f" % (len(success_list_sim) / num_episodes))
        
def Comp_tracking_reward_pen(reference, states, coeffcients):
    # we need to think about how to set up the object tracking rewards. (this is very important to us if we want to remove the task-specific rewards)
    # object tracking reward should be very similar to task-specific rewards
    rewards = dict()
    r_hand = coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'] - states['hand_state']) ** 2 / len(reference['hand_state']))
    r_obj_ori = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_ori'] - states['obj_ori']) ** 2 / len(reference['obj_ori']))
    if coeffcients['ADD_BONUS_REWARDS'] == 1:    
        # bonus of object tracking
        if np.linalg.norm(reference['obj_ori'] - states['obj_ori']) < 0.1:
            r_obj_ori += 50
        elif np.linalg.norm(reference['obj_ori'] - states['obj_ori']) < 0.2:
            r_obj_ori += 20
        elif np.linalg.norm(reference['obj_ori'] - states['obj_ori']) < 0.3:
            r_obj_ori += 10
    r_obj_pos = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_pos'] - states['obj_pos']) ** 2 / len(reference['obj_pos']))
    r_obj_vel = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_vel'] - states['obj_vel']) ** 2 / len(reference['obj_vel']))
    rewards['total_reward'] = r_hand + r_obj_ori
    rewards['hand_reward'] = r_hand
    rewards['object_reward'] = r_obj_ori
    return rewards

def Comp_tracking_reward_relocate(reference, states, coeffcients):
    # we need to think about how to set up the object tracking rewards. (this is very important to us if we want to remove the task-specific rewards)
    # object tracking reward should be very similar to task-specific rewards
    rewards = dict()
    # the hand tracking parameters are different for fingers and base
    r_hand = coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'][:6] - states['hand_state'][:6]) ** 2 / len(reference['hand_state'][:6]))
    r_hand += coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'][6:] - states['hand_state'][6:]) ** 2 / len(reference['hand_state'][6:]))
    # r_obj_ori = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_ori'] - states['obj_ori']) ** 2 / len(reference['obj_ori']))
    # r_obj_pos = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_pos'] - states['obj_pos']) ** 2 / len(reference['obj_pos']))
    r_obj_feature = coeffcients['object_track'] * np.sum((np.square(reference['obj_feature'] - states['obj_feature'])))/len(states['obj_feature'])

    if coeffcients['ADD_BONUS_REWARDS'] == 1:    
        # bonus of object tracking
        if np.linalg.norm(reference['obj_feature'] - states['obj_feature']) < 0.2:
            r_obj_feature += 25
        elif np.linalg.norm(reference['obj_feature'] - states['obj_feature']) < 0.5:
            r_obj_feature += 10
        elif np.linalg.norm(reference['obj_feature'] - states['obj_feature']) < 1:
            r_obj_feature += 5
    # r_obj_vel = coeffcients['object_track'] * np.exp(-400 * np.linalg.norm(reference['obj_vel'] - states['obj_vel']) ** 2 / len(reference['obj_vel']))
    rewards['total_reward'] = r_hand + r_obj_feature
    rewards['hand_reward'] = r_hand
    rewards['object_reward'] = r_obj_feature
    return rewards

def Comp_tracking_reward_relocate_freeze_base(reference, states, coeffcients):
    # we need to think about how to set up the object tracking rewards. (this is very important to us if we want to remove the task-specific rewards)
    # object tracking reward should be very similar to task-specific rewards
    rewards = dict()
    # the hand tracking parameters are different for fingers and base
    r_hand = coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'][6:] - states['hand_state'][6:]) ** 2 / len(reference['hand_state'][6:]))
    r_obj_ori = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_ori'] - states['obj_ori']) ** 2 / len(reference['obj_ori']))
    r_obj_pos = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_pos'] - states['obj_pos']) ** 2 / len(reference['obj_pos']))
    if coeffcients['ADD_BONUS_REWARDS'] == 1:    
        # bonus of object tracking
        if np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.02:
            r_obj_pos += 25
        elif np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.05:
            r_obj_pos += 10
        elif np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.1:
            r_obj_pos += 5
    r_obj_vel = coeffcients['object_track'] * np.exp(-400 * np.linalg.norm(reference['obj_vel'] - states['obj_vel']) ** 2 / len(reference['obj_vel']))
    rewards['total_reward'] = r_hand + r_obj_pos 
    rewards['hand_reward'] = r_hand
    rewards['object_reward'] = r_obj_pos 
    return rewards

def Comp_tracking_reward_relocate_include_Rots(reference, states, coeffcients):
    # we need to think about how to set up the object tracking rewards. (this is very important to us if we want to remove the task-specific rewards)
    # object tracking reward should be very similar to task-specific rewards
    rewards = dict()
    # the hand tracking parameters are different for fingers and base
    r_hand = coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'][3:6] - states['hand_state'][3:6]) ** 2 / len(reference['hand_state'][3:6]))
    r_hand += coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'][6:] - states['hand_state'][6:]) ** 2 / len(reference['hand_state'][6:]))
    r_obj_ori = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_ori'] - states['obj_ori']) ** 2 / len(reference['obj_ori']))
    r_obj_pos = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_pos'] - states['obj_pos']) ** 2 / len(reference['obj_pos']))
    if coeffcients['ADD_BONUS_REWARDS'] == 1:    
        # bonus of object tracking
        if np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.02:
            r_obj_pos += 25
        elif np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.05:
            r_obj_pos += 10
        elif np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.1:
            r_obj_pos += 5
    r_obj_vel = coeffcients['object_track'] * np.exp(-400 * np.linalg.norm(reference['obj_vel'] - states['obj_vel']) ** 2 / len(reference['obj_vel']))
    rewards['total_reward'] = r_hand + r_obj_pos 
    rewards['hand_reward'] = r_hand
    rewards['object_reward'] = r_obj_pos 
    return rewards

def Comp_tracking_reward_door(reference, states, coeffcients):
    rewards = dict()
    # the hand tracking parameters are different for fingers and base
    r_hand = coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'][:4] - states['hand_state'][:4]) ** 2 / len(reference['hand_state'][:4]))
    # print("base:", np.linalg.norm(reference['hand_state'][:4] - states['hand_state'][:4]))
    if coeffcients['ADD_BONUS_PENALTY'] == 1:
        if np.linalg.norm(reference['hand_state'][:4] - states['hand_state'][:4]) > 1:
            r_hand -= 10
        elif np.linalg.norm(reference['hand_state'][:4] - states['hand_state'][:4]) > 0.7:
            r_hand -= 5
        elif np.linalg.norm(reference['hand_state'][:4] - states['hand_state'][:4]) > 0.4:
            r_hand -= 1
    r_hand += coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]) ** 2 / len(reference['hand_state'][4:]))
    # print("finger:", np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]))
    if coeffcients['ADD_BONUS_PENALTY'] == 1:
        if np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]) > 3:
            r_hand -= 3
        elif np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]) > 2:
            r_hand -= 2
        elif np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]) > 1:    
            r_hand -= 1
    r_obj_pos = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_pos'] - states['obj_pos']) ** 2 / len(reference['obj_pos']))
    if coeffcients['ADD_BONUS_REWARDS'] == 1:    
        # bonus of object tracking
        if np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.1:
            r_obj_pos += 25
        elif np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.15:
            r_obj_pos += 10
        elif np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.25:
            r_obj_pos += 5
    r_obj_vel = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_vel'] - states['obj_vel']) ** 2 / len(reference['obj_vel']))
    rewards['total_reward'] = r_hand + r_obj_pos
    rewards['hand_reward'] = r_hand
    rewards['object_reward'] = r_obj_pos
    return rewards

def Comp_tracking_reward_door_freeze_base(reference, states, coeffcients):
    rewards = dict()
    # the hand tracking parameters are different for fingers and base
    r_hand = coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]) ** 2 / len(reference['hand_state'][4:]))
    # print("finger:", np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]))
    if coeffcients['ADD_BONUS_PENALTY'] == 1:
        if np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]) > 3:
            r_hand -= 3
        elif np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]) > 2:
            r_hand -= 2
        elif np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]) > 1:    
            r_hand -= 1
    r_obj_pos = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_pos'] - states['obj_pos']) ** 2 / len(reference['obj_pos']))
    if coeffcients['ADD_BONUS_REWARDS'] == 1:    
        # bonus of object tracking
        if np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.1:
            r_obj_pos += 25
        elif np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.15:
            r_obj_pos += 10
        elif np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.25:
            r_obj_pos += 5
    r_obj_vel = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_vel'] - states['obj_vel']) ** 2 / len(reference['obj_vel']))
    rewards['total_reward'] = r_hand + r_obj_pos
    rewards['hand_reward'] = r_hand
    rewards['object_reward'] = r_obj_pos
    return rewards

def Comp_tracking_reward_door_include_Rots(reference, states, coeffcients):
    rewards = dict()
    # the hand tracking parameters are different for fingers and base
    r_hand = coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'][1:4] - states['hand_state'][1:4]) ** 2 / len(reference['hand_state'][1:4]))
    # print("base:", np.linalg.norm(reference['hand_state'][:4] - states['hand_state'][:4]))
    if coeffcients['ADD_BONUS_PENALTY'] == 1:
        if np.linalg.norm(reference['hand_state'][1:4] - states['hand_state'][1:4]) > 1:
            r_hand -= 10
        elif np.linalg.norm(reference['hand_state'][1:4] - states['hand_state'][1:4]) > 0.7:
            r_hand -= 5
        elif np.linalg.norm(reference['hand_state'][1:4] - states['hand_state'][1:4]) > 0.4:
            r_hand -= 1
    r_hand += coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]) ** 2 / len(reference['hand_state'][4:]))
    # print("finger:", np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]))
    if coeffcients['ADD_BONUS_PENALTY'] == 1:
        if np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]) > 3:
            r_hand -= 3
        elif np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]) > 2:
            r_hand -= 2
        elif np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]) > 1:    
            r_hand -= 1
    r_obj_pos = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_pos'] - states['obj_pos']) ** 2 / len(reference['obj_pos']))
    if coeffcients['ADD_BONUS_REWARDS'] == 1:    
        # bonus of object tracking
        if np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.1:
            r_obj_pos += 25
        elif np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.15:
            r_obj_pos += 10
        elif np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.25:
            r_obj_pos += 5
    r_obj_vel = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_vel'] - states['obj_vel']) ** 2 / len(reference['obj_vel']))
    rewards['total_reward'] = r_hand + r_obj_pos
    rewards['hand_reward'] = r_hand
    rewards['object_reward'] = r_obj_pos
    return rewards

def Comp_tracking_reward_hammer(reference, states, coeffcients):
    rewards = dict()
    # the hand tracking parameters are different for fingers and base
    r_hand = coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'][:2] - states['hand_state'][:2]) ** 2 / len(reference['hand_state'][:2]))
    r_hand += coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'][2:] - states['hand_state'][2:]) ** 2 / len(reference['hand_state'][2:]))
    r_obj_ori = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_ori'] - states['obj_ori']) ** 2 / len(reference['obj_ori']))
    r_obj_pos = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_pos'] - states['obj_pos']) ** 2 / len(reference['obj_pos']))
    if coeffcients['ADD_BONUS_REWARDS'] == 1:    
        # bonus of object tracking
        if np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.01:
            r_obj_pos += 25
        elif np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.04:
            r_obj_pos += 15
        if np.linalg.norm(reference['obj_ori'] - states['obj_ori']) < 0.05:
            r_obj_ori += 25
        elif np.linalg.norm(reference['obj_ori'] - states['obj_ori']) < 0.15:            
            r_obj_ori += 15
    r_obj_vel = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_vel'] - states['obj_vel']) ** 2 / len(reference['obj_vel']))
    rewards['total_reward'] = r_hand + r_obj_pos + r_obj_ori
    rewards['hand_reward'] = r_hand
    rewards['object_reward'] = r_obj_pos + r_obj_ori
    return rewards

def Comp_tracking_reward_hammer_freeze_base(reference, states, coeffcients):
    rewards = dict()
    # the hand tracking parameters are different for fingers and base
    r_hand = coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'][2:] - states['hand_state'][2:]) ** 2 / len(reference['hand_state'][2:]))
    r_obj_ori = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_ori'] - states['obj_ori']) ** 2 / len(reference['obj_ori']))
    r_obj_pos = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_pos'] - states['obj_pos']) ** 2 / len(reference['obj_pos']))
    if coeffcients['ADD_BONUS_REWARDS'] == 1:    
        # bonus of object tracking
        if np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.01:
            r_obj_pos += 25
        elif np.linalg.norm(reference['obj_pos'] - states['obj_pos']) < 0.04:
            r_obj_pos += 15
        if np.linalg.norm(reference['obj_ori'] - states['obj_ori']) < 0.05:
            r_obj_ori += 25
        elif np.linalg.norm(reference['obj_ori'] - states['obj_ori']) < 0.15:            
            r_obj_ori += 15
    r_obj_vel = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_vel'] - states['obj_vel']) ** 2 / len(reference['obj_vel']))
    rewards['total_reward'] = r_hand + r_obj_pos + r_obj_ori
    rewards['hand_reward'] = r_hand
    rewards['object_reward'] = r_obj_pos + r_obj_ori
    return rewards