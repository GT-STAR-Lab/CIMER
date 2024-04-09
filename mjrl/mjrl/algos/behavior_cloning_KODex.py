"""
Minimize bc loss (MLE, MSE, RWR etc.) with pytorch optimizers
"""

import logging
from re import A
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
from torch.autograd import Variable
from mjrl.utils.logger import DataLog
from mjrl.KODex_utils.coord_trans import ori_transform, ori_transform_inverse
from tqdm import tqdm
import pickle
import os

class BC:
    def __init__(self, job_dir,
                 eval_data,
                 policy,
                 refer_motion,
                 Koopman_obser,
                 task_id,
                 object_dynamics,
                 robot_dim,
                 obj_dim,
                 epochs = 5,
                 batch_size = 64,
                 lr = 1e-3,
                 optimizer = None,
                 loss_type = 'MSE',  # can be 'MLE' or 'MSE'
                 **kwargs,
                 ):
        self.job_dir = job_dir
        self.eval_data = eval_data
        self.policy = policy
        self.epochs = epochs
        self.mb_size = batch_size
        self.loss_type = loss_type
        self.KODex = refer_motion
        self.Koopman_obser = Koopman_obser
        self.task_id = task_id
        self.object_dynamics = object_dynamics
        self.robot_dim = robot_dim
        self.obj_dim = obj_dim
        # construct optimizer
        self.optimizer = torch.optim.Adam(self.policy.trainable_params, lr=lr) if optimizer is None else optimizer

        # Loss criterion if required
        if loss_type == 'MSE':
            self.loss_criterion = torch.nn.MSELoss()

    def compute_transformations(self, **kwargs):
        # get transformations
        observations, actions = self.generate_data(**kwargs)
        in_shift, in_scale = np.mean(observations, axis=0), np.std(observations, axis=0)  # compute the data shifting, this value will also be set for the RL
        if self.policy.policy_output == 'jp':   # 'jp' -> the outputs are directly joint positions, initialized to be reference_motion_{t+1}
            out_shift, out_scale = np.mean(actions, axis=0), np.std(actions, axis=0)
        else:   # 'djp' -> the outputs are residual joint positions, initialized to be np.zeros()
            out_shift_shape = np.mean(actions, axis=0).shape
            out_scale_shape = np.std(actions, axis=0).shape
            out_shift, out_scale = np.zeros(out_shift_shape), np.ones(out_scale_shape)
        return observations, actions, in_shift, in_scale, out_shift, out_scale

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # set scalings in the target policy
        self.policy.model.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.policy.old_model.set_transformations(in_shift, in_scale, out_shift, out_scale)

    def set_variance_with_data(self, out_scale):
        # set the variance of gaussian policy based on out_scale
        params = self.policy.get_param_values()
        # print("before update, the log_std is:", params[-self.policy.m:])
        params[-self.policy.m:] = np.log(out_scale + 1e-12)
        # print("after update, the log_std is:", params[-self.policy.m:])
        self.policy.set_param_values(params)

    def loss(self, data, idx=None):
        if self.loss_type == 'MLE':
            return self.mle_loss(data, idx)
        elif self.loss_type == 'MSE':
            return self.mse_loss(data, idx)
        else:
            print("Please use valid loss type")
            return None

    def mle_loss(self, data, idx):
        # use indices if provided (e.g. for mini-batching)
        # otherwise, use all the data
        idx = range(data['observations'].shape[0]) if idx is None else idx
        if type(data['observations']) == torch.Tensor:
            idx = torch.LongTensor(idx)
        obs = data['observations'][idx]
        act = data['expert_actions'][idx]
        LL, mu, log_std = self.policy.new_dist_info(obs, act)
        # minimize negative log likelihood
        return -torch.mean(LL)

    def mse_loss(self, data, idx=None):  # supervised learning, set labels as actions in the data data, conditioned on the observation data
        idx = range(data['observations'].shape[0]) if idx is None else idx
        if type(data['observations']) is torch.Tensor:
            idx = torch.LongTensor(idx)
        obs = data['observations'][idx]
        act_expert = data['expert_actions'][idx]
        if type(data['observations']) is not torch.Tensor:
            obs = Variable(torch.from_numpy(obs).float(), requires_grad=False)
            act_expert = Variable(torch.from_numpy(act_expert).float(), requires_grad=False)
        act_pi = self.policy.model(obs)  # feed-forward computation of the policy networks
        return self.loss_criterion(act_pi, act_expert.detach()) # action experts extracted from the demo data

    def fit(self, data, suppress_fit_tqdm=False, **kwargs):
        # data is a dict
        # keys should have "observations" and "expert_actions"
        if os.path.isdir(self.job_dir) == False:
            os.mkdir(self.job_dir)
        previous_dir = os.getcwd()
        os.chdir(self.job_dir) # important! we are now in the directory to save data
        validate_keys = all([k in data.keys() for k in ["observations", "expert_actions"]])
        assert validate_keys is True
        num_samples = data["observations"].shape[0]  # number of total observations and subsequent actions.
        # log stats before
        print("loss_before:", self.loss(data, idx=range(num_samples)).data.numpy().ravel()[0])
        # train loop
        for ep in config_tqdm(range(self.epochs), suppress_fit_tqdm):
            for mb in range(int(num_samples / self.mb_size)):
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                self.optimizer.zero_grad()
                loss = self.loss(data, idx=rand_idx)
                loss.backward()
                self.optimizer.step()
        params_after_opt = self.policy.get_param_values()
        self.policy.set_param_values(params_after_opt, set_new=True, set_old=True)  # pre-trained the policy parameters, as pointed out in the paper
        print("loss_after:", self.loss(data, idx=range(num_samples)).data.numpy().ravel()[0])
        if (kwargs['save_policy']):
            pickle.dump(self.policy, open('BC_policy.pickle', 'wb'))
        os.chdir(previous_dir)

    def generate_data(self, **kwargs):
        self.robot_dim = self.robot_dim
        self.obj_dim =self.obj_dim
        if kwargs['history_s'] >= 0:
            if self.object_dynamics:
                observations = np.zeros([kwargs['num_traj'] * (kwargs['task_horizon'] - 1), (len(kwargs['future_s']) + kwargs['history_s'] + 1) * (self.robot_dim + self.obj_dim) + (kwargs['history_s'] + 1) * self.robot_dim])
            else:
                observations = np.zeros([kwargs['num_traj'] * (kwargs['task_horizon'] - 1), (len(kwargs['future_s']) + kwargs['history_s'] + 1) * self.robot_dim + (kwargs['history_s'] + 1) * self.robot_dim])
        else:
            if self.object_dynamics:
                observations = np.zeros([kwargs['num_traj'] * (kwargs['task_horizon'] - 1), (len(kwargs['future_s']) + 1) * (self.robot_dim + self.obj_dim)])
            else:
                observations = np.zeros([kwargs['num_traj'] * (kwargs['task_horizon'] - 1), (len(kwargs['future_s']) + 1) * self.robot_dim])
        actions = np.zeros([kwargs['num_traj'] * (kwargs['task_horizon'] - 1), self.robot_dim])
        index = 0
        for ep in tqdm(range(kwargs['num_traj'])):
            if self.task_id == 'pen':
                init_hand_state = self.eval_data[ep]['handpos']
                init_objpos = self.eval_data[ep]['objpos']
                init_objvel = self.eval_data[ep]['objvel']
                init_objorient_world = self.eval_data[ep]['objorient']
                desired_ori = self.eval_data[ep]['desired_ori']
                init_objorient = ori_transform(init_objorient_world, desired_ori) 
                hand_OriState = init_hand_state
                obj_OriState = np.append(init_objpos, np.append(init_objorient, init_objvel))  # ori: represented in the transformed frame
                obj_OriState_ = np.append(init_objpos, np.append(init_objorient_world, init_objvel)) # ori: represented in the original frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                hand_states_traj = np.zeros([kwargs['task_horizon'], num_hand])
                object_states_traj = np.zeros([kwargs['task_horizon'], num_obj])
                hand_states_traj[0, :] = hand_OriState
                object_states_traj[0, :] = obj_OriState_
                z_t = self.Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                for t_ in range(kwargs['task_horizon'] - 1):
                    z_t_1_computed = np.dot(self.KODex, z_t)
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
            elif self.task_id == 'relocate':
                init_hand_state = self.eval_data[ep]['handpos']
                init_objpos = self.eval_data[ep]['objpos'] # converged object position
                init_objvel = self.eval_data[ep]['objvel']
                init_objori = self.eval_data[ep]['objorient']
                desired_pos = self.eval_data[ep]['desired_pos']
                init_objpos_world = desired_pos + init_objpos # in the world frame(on the table)
                hand_OriState = init_hand_state
                obj_OriState = np.append(init_objpos, np.append(init_objori, init_objvel))  # ori: represented in the transformed frame (converged to desired pos)
                obj_OriState_ = np.append(init_objpos_world, np.append(init_objori, init_objvel)) # ori: represented in the world frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                if not self.policy.freeze_base:
                    hand_states_traj = np.zeros([kwargs['task_horizon'], num_hand])
                else:
                    hand_states_traj = np.zeros([kwargs['task_horizon'], self.robot_dim])
                object_states_traj = np.zeros([kwargs['task_horizon'], num_obj])
                if not self.policy.freeze_base:
                    hand_states_traj[0, :] = hand_OriState
                else:
                    if self.robot_dim == 24:
                        hand_states_traj[0, :] = hand_OriState[6:30]
                    elif self.robot_dim == 27:
                        hand_states_traj[0, :] = hand_OriState[3:30]
                object_states_traj[0, :] = obj_OriState_
                z_t = self.Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                for t_ in range(kwargs['task_horizon'] - 1):
                    z_t_1_computed = np.dot(self.KODex, z_t)
                    z_t = z_t_1_computed.copy()
                    x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                    hand_OriState = x_t_1_computed[:num_hand]
                    obj_pos = x_t_1_computed[num_hand: num_hand + 3] # converged object position
                    obj_pos_world = desired_pos + obj_pos
                    obj_ori = x_t_1_computed[num_hand + 3: num_hand + 6]
                    obj_vel = x_t_1_computed[num_hand + 6: num_hand + 12]
                    obj_OriState = np.append(obj_pos_world, np.append(obj_ori, obj_vel))
                    if not self.policy.freeze_base:
                        hand_states_traj[t_ + 1, :] = hand_OriState
                    else:
                        if self.robot_dim == 24:
                            hand_states_traj[t_ + 1, :] = hand_OriState[6:30]
                        elif self.robot_dim == 27:
                            hand_states_traj[t_ + 1, :] = hand_OriState[3:30]
                    object_states_traj[t_ + 1, :] = obj_OriState
            elif self.task_id == 'door':
                init_hand_state = self.eval_data[ep]['handpos']
                init_objpos = self.eval_data[ep]['objpos']
                init_objvel = self.eval_data[ep]['objvel']
                init_handle = self.eval_data[ep]['handle_init']
                hand_OriState = init_hand_state
                obj_OriState = np.append(init_objpos, np.append(init_objvel, init_handle))  # ori: represented in the transformed frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                if not self.policy.freeze_base:
                    hand_states_traj = np.zeros([kwargs['task_horizon'], num_hand])
                else:
                    hand_states_traj = np.zeros([kwargs['task_horizon'], self.robot_dim])
                object_states_traj = np.zeros([kwargs['task_horizon'], num_obj])
                if not self.policy.freeze_base:
                    hand_states_traj[0, :] = hand_OriState
                else:
                    if self.robot_dim == 24:
                        hand_states_traj[0, :] = hand_OriState[4:28]
                    elif self.robot_dim == 25:
                        hand_states_traj[0, :] = hand_OriState[3:28]
                    elif self.robot_dim == 26:
                        hand_states_traj[0, :] = np.append(hand_OriState[0], hand_OriState[3:28])
                    elif self.robot_dim == 27:
                        hand_states_traj[0, :] = hand_OriState[1:28]
                object_states_traj[0, :] = obj_OriState
                z_t = self.Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                for t_ in range(kwargs['task_horizon'] - 1):
                    z_t_1_computed = np.dot(self.KODex, z_t)
                    z_t = z_t_1_computed.copy()
                    x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                    hand_OriState = x_t_1_computed[:num_hand]
                    obj_pos = x_t_1_computed[num_hand: num_hand + 3] # converged object position
                    obj_vel = x_t_1_computed[num_hand + 3: num_hand + 4]
                    init_handle = x_t_1_computed[num_hand + 4: num_hand + 7]
                    obj_OriState = np.append(obj_pos, np.append(obj_vel, init_handle))
                    if not self.policy.freeze_base:
                        hand_states_traj[t_ + 1, :] = hand_OriState
                    else:
                        if self.robot_dim == 24:
                            hand_states_traj[t_ + 1, :] = hand_OriState[4:28]
                        elif self.robot_dim == 25:
                            hand_states_traj[t_ + 1, :] = hand_OriState[3:28]
                        elif self.robot_dim == 26:
                            hand_states_traj[t_ + 1, :] = np.append(hand_OriState[0], hand_OriState[3:28])
                        elif self.robot_dim == 27:
                            hand_states_traj[t_ + 1, :] = hand_OriState[1:28]
                    object_states_traj[t_ + 1, :] = obj_OriState
            elif self.task_id == 'hammer':
                init_hand_state = self.eval_data[ep]['handpos']
                init_objpos = self.eval_data[ep]['objpos']
                init_objori = self.eval_data[ep]['objorient']
                init_objvel = self.eval_data[ep]['objvel']
                goal_nail = self.eval_data[ep]['nail_goal']
                hand_OriState = init_hand_state
                obj_OriState = np.append(init_objpos, np.append(init_objori, np.append(init_objvel, goal_nail)))  # ori: represented in the transformed frame
                num_hand = len(hand_OriState)
                num_obj = len(obj_OriState)
                if not self.policy.freeze_base:
                    hand_states_traj = np.zeros([kwargs['task_horizon'], num_hand])
                else:
                    hand_states_traj = np.zeros([kwargs['task_horizon'], 24])
                object_states_traj = np.zeros([kwargs['task_horizon'], num_obj])
                if not self.policy.freeze_base:
                    hand_states_traj[0, :] = hand_OriState
                else:
                    hand_states_traj[0, :] = hand_OriState[2:26]
                object_states_traj[0, :] = obj_OriState
                z_t = self.Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
                for t_ in range(kwargs['task_horizon'] - 1):
                    z_t_1_computed = np.dot(self.KODex, z_t)
                    z_t = z_t_1_computed.copy()
                    x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                    hand_OriState = x_t_1_computed[:num_hand]
                    obj_pos = x_t_1_computed[num_hand: num_hand + 3] # tool pos
                    obj_ori = x_t_1_computed[num_hand + 3: num_hand + 6] # tool ori
                    obj_vel = x_t_1_computed[num_hand + 6: num_hand + 12]
                    nail_pos = x_t_1_computed[num_hand + 12:]
                    obj_OriState = np.append(obj_pos, np.append(obj_ori, np.append(obj_vel, nail_pos)))
                    if not self.policy.freeze_base:
                        hand_states_traj[t_ + 1, :] = hand_OriState
                    else:
                        hand_states_traj[t_ + 1, :] = hand_OriState[2:26]
                    object_states_traj[t_ + 1, :] = obj_OriState
            # for the default history states, set them to be initial hand states
            if self.policy.freeze_base:
                num_hand = self.robot_dim 
                # if self.task_id == 'door':
                #     num_hand = self.robot_dim   
                # else:
                #     num_hand = 24
            prev_states = dict()
            for ii in range(kwargs['history_s']):
                if self.object_dynamics:
                    prev_states[ii] = np.append(hand_states_traj[0], object_states_traj[0]) 
                else:
                    prev_states[ii] = hand_states_traj[0]
            prev_actions = dict()
            for ii in range(kwargs['history_s'] + 1):
                prev_actions[ii] = hand_states_traj[0]
            for i in range(kwargs['task_horizon'] - 1):
                if kwargs['history_s'] >= 0: # we add the history information into policy inputs
                    if self.object_dynamics:  # if we use the object dynamics as part of policy input
                        obser_ = np.zeros((len(kwargs['future_s']) + kwargs['history_s'] + 1) * (num_hand + num_obj) + (kwargs['history_s'] + 1) * num_hand)
                        prev_states[kwargs['history_s']] = np.append(hand_states_traj[i, :], object_states_traj[i, :]) # add the current states 
                        for ii in range(kwargs['history_s'] + 1):
                            obser_[ii * (2 * num_hand + num_obj): (ii + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[ii], prev_states[ii])
                        future_index = (kwargs['history_s'] + 1) * (2 * num_hand + num_obj)
                        for t_ in range(len(kwargs['future_s'])):
                            if i + kwargs['future_s'][t_] >= kwargs['task_horizon']:
                                obser_[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[kwargs['task_horizon'] - 1], object_states_traj[kwargs['task_horizon'] - 1])
                            else:
                                obser_[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[i + kwargs['future_s'][t_]], object_states_traj[i + kwargs['future_s'][t_]])
                    else:
                        obser_ = np.zeros((len(kwargs['future_s']) + kwargs['history_s'] + 1) * num_hand + (kwargs['history_s'] + 1) * num_hand)
                        prev_states[kwargs['history_s']] = hand_states_traj[i, :]
                        for ii in range(kwargs['history_s'] + 1):
                            obser_[ii * (2 * num_hand): (ii + 1) * (2 * num_hand)] = np.append(prev_actions[ii], prev_states[ii])
                        future_index = (kwargs['history_s'] + 1) * (2 * num_hand) # add the current states 
                        for t_ in range(len(kwargs['future_s'])):
                            if i + kwargs['future_s'][t_] >= kwargs['task_horizon']:
                                obser_[future_index + num_hand * t_: future_index + num_hand * (t_ + 1)] = hand_states_traj[kwargs['task_horizon'] - 1]
                            else:
                                obser_[future_index + num_hand * t_: future_index + num_hand * (t_ + 1)] = hand_states_traj[i + kwargs['future_s'][t_]]
                else:
                    if self.object_dynamics:  # if we use the object dynamics as part of policy input
                        obser_ = np.zeros((len(kwargs['future_s']) + 1) * (num_hand + num_obj))
                        obser_[:num_hand + num_obj] = np.append(hand_states_traj[i], object_states_traj[i])
                        for t_ in range(1, len(kwargs['future_s']) + 1):
                            if i + kwargs['future_s'][t_ - 1] >= kwargs['task_horizon']:
                                obser_[(num_hand + num_obj) * t_:(num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[kwargs['task_horizon'] - 1], object_states_traj[kwargs['task_horizon'] - 1])
                            else:
                                obser_[(num_hand + num_obj) * t_:(num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[i + kwargs['future_s'][t_ - 1]], object_states_traj[i + kwargs['future_s'][t_ - 1]])
                    else:
                        obser_ = np.zeros((len(kwargs['future_s']) + 1) * num_hand)
                        obser_[:num_hand] = hand_states_traj[i]
                        for t_ in range(1, len(kwargs['future_s']) + 1):
                            if i + kwargs['future_s'][t_ - 1] >= kwargs['task_horizon']:
                                obser_[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[kwargs['task_horizon'] - 1]
                            else:
                                obser_[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[i + kwargs['future_s'][t_ - 1]]
                if self.policy.policy_output == 'jp':  
                    act_ = hand_states_traj[i + 1, :] # Initialization to be target joint positions at next step.
                elif self.policy.policy_output == 'djp':
                    act_ = hand_states_traj[i + 1, :] - hand_states_traj[i + 1, :]  # Residual policy -> Initialization to be zeros.
                observations[index, :] = obser_
                actions[index, :] = act_
                # update the history information
                if kwargs['history_s'] >= 0:
                    for ii in range(kwargs['history_s']):
                        prev_actions[ii] = prev_actions[ii + 1]
                        prev_states[ii] = prev_states[ii + 1]
                    prev_actions[kwargs['history_s']] = hand_states_traj[i + 1, :]
                index += 1
        return observations, actions

    def data_augment(self, observations, actions, kwargs):  # do we need to add noise to future states?
        print("original_shape:", observations.shape)
        print("original_shape:", actions.shape)
        bc_hand_noise = kwargs['data_augment_params']['bc_hand_noise']
        bc_obj_noise = kwargs['data_augment_params']['bc_obj_noise']
        bc_aug_size = kwargs['data_augment_params']['bc_aug_size']
        corrective_actions = np.tile(actions, (bc_aug_size, 1))
        for i in range(bc_aug_size):
            temp_noisy_observations = observations.copy()
            if self.object_dynamics:  # with object dynamics
                if kwargs['history_s'] >= 0: # with history states & actions
                    prev_action_index = [(self.robot_dim + self.obj_dim + self.robot_dim) * t for t in range(kwargs['history_s'] + 1)]
                    prev_state_index = [(self.robot_dim + self.obj_dim + self.robot_dim) * t + self.robot_dim for t in range(kwargs['history_s'] + 1)]
                    for index in prev_action_index:
                        temp_noisy_observations[:, index: index + self.robot_dim] += np.random.normal(0, bc_hand_noise, temp_noisy_observations[:, index: index + self.robot_dim].shape)
                    for index in prev_state_index:
                        temp_noisy_observations[:, index: index + self.robot_dim] += np.random.normal(0, bc_hand_noise, temp_noisy_observations[:, index: index + self.robot_dim].shape)
                        if self.task_id not in {'door', 'hammer'}:  
                            temp_noisy_observations[:, index + self.robot_dim: index + self.robot_dim + self.obj_dim] += np.random.normal(0, bc_obj_noise, temp_noisy_observations[:, index + self.robot_dim: index + self.robot_dim + self.obj_dim].shape)
                        else:  # for the door and hammer tasks, the last three elements are fixed values, (e.g., desired goal nail position, or initial handle position)
                            temp_noisy_observations[:, index + self.robot_dim: index + self.robot_dim + self.obj_dim - 3] += np.random.normal(0, bc_obj_noise, temp_noisy_observations[:, index + self.robot_dim: index + self.robot_dim + self.obj_dim - 3].shape)
                else:  # only current states
                    temp_noisy_observations[:, 0: self.robot_dim] += np.random.normal(0, bc_hand_noise, temp_noisy_observations[:, 0: self.robot_dim].shape)
                    if self.task_id not in {'door', 'hammer'}:  
                        temp_noisy_observations[:, self.robot_dim: self.robot_dim + self.obj_dim] += np.random.normal(0, bc_obj_noise, temp_noisy_observations[:, self.robot_dim: self.robot_dim + self.obj_dim].shape)
                    else:  # for the door and hammer tasks, the last three elements are fixed values, (e.g., desired goal nail position, or initial handle position)
                        temp_noisy_observations[:, self.robot_dim: self.robot_dim + self.obj_dim - 3] += np.random.normal(0, bc_obj_noise, temp_noisy_observations[:, self.robot_dim: self.robot_dim + self.obj_dim - 3].shape)
            else:
                if kwargs['history_s'] >= 0:
                    temp_noisy_observations[:, 0: 2 * (kwargs['history_s'] + 1) * self.robot_dim] += np.random.normal(0, bc_hand_noise, temp_noisy_observations[:, 0: 2 * (kwargs['history_s'] + 1) * self.robot_dim].shape)
                else:  # only current states
                    temp_noisy_observations[:, 0: self.robot_dim] += np.random.normal(0, bc_hand_noise, temp_noisy_observations[:, 0: self.robot_dim].shape)
            if i == 0:
                noisy_observations = temp_noisy_observations
            else:
                noisy_observations = np.append(noisy_observations, temp_noisy_observations, axis = 0)
        print("noisy_obs_shape:", noisy_observations.shape)
        print("noisy_act_shape:", corrective_actions.shape)
        # we need to quantify the effects of the added noise (directly from the data, instead of running the simulator?)
        return noisy_observations, corrective_actions

    def train(self, **kwargs):  
        observations, actions = self.generate_data(**kwargs)
        if kwargs['data_augment_params']['bc_augment'] == 1:
            noisy_observations, corrective_actions = self.data_augment(observations, actions, kwargs)  # corrective actions are the original actions
            observations = np.concatenate((observations, noisy_observations), axis = 0)
            actions = np.concatenate((actions, corrective_actions), axis = 0)
        data = dict(observations=observations, expert_actions=actions)
        self.fit(data, **kwargs)


def config_tqdm(range_inp, suppress_tqdm=False):
    if suppress_tqdm:
        return range_inp
    else:
        return tqdm(range_inp)
        
