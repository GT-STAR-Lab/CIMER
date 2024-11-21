import logging
import numpy as np
from mjrl.policies.ncp_network import NCPNetwork
from mjrl.policies.rnn_network import RNNNetwork
from mjrl.utils.gym_env import GymEnv
from mjrl.utils import tensor_utils
from mjrl.KODex_utils.coord_trans import ori_transform, ori_transform_inverse
logging.disable(logging.CRITICAL)
import multiprocessing as mp
import time as timer
logging.disable(logging.CRITICAL)
import torch
import pandas as pd
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Single core rollout to sample trajectories
# =======================================================
def do_rollout(
        num_traj,
        env,
        task_id,
        policy,
        eval_mode = False,
        horizon = 1e6,
        future_state = (1,2,3),
        history_state=0,
        base_seed = None,
        env_kwargs=None,
        Koopman_obser=None, 
        KODex=None,
        coeffcients=None,
        obj_dynamics=True,
        control_mode='Torque',
        PD_controller=None,
        resnet_model=None
):
    """
    :param num_traj:    number of trajectories (int)
    :param env:         environment (env class, str with env_name, or factory function)
    :param policy:      policy to use for action selection
    :param eval_mode:   use evaluation mode for action computation (bool)
    :param horizon:     max horizon length for rollout (<= env.horizon)
    :param base_seed:   base seed for rollouts (int)
    :param env_kwargs:  dictionary with parameters, will be passed to env generator
    :return:
    """

    # get the correct env behavior
    if type(env) == str:
        env = GymEnv(env, control_mode)
    elif isinstance(env, GymEnv):
        env = env
    elif callable(env):
        env = env(**env_kwargs)
    else:
        print("Unsupported environment format")
        raise AttributeError
    if base_seed is not None:
        env.set_seed(base_seed)
        np.random.seed(base_seed)
    else:
        np.random.seed()
    paths = []
    num_future_s = len(future_state)
    temp=[]
    for ep in range(num_traj):
        # seeding
        if base_seed is not None:
            seed = base_seed + ep # base_seed 123
            env.set_seed(seed)
            np.random.seed(seed)
        observations=[]
        actions=[]
        rewards=[]
        task_rewards=[]
        tracking_rewards=[]
        tracking_hand_rewards=[]
        tracking_object_rewards=[]
        agent_infos = []
        env_infos = []
        if task_id == 'pen':
            o, desired_orien = env.reset()
            init_hand_state = o[:24]
            init_objpos = o[24:27]
            init_objvel = o[27:33]
            desired_ori = o[36:39]
            init_objorient = ori_transform(o[33:36], desired_ori) 
            hand_OriState = init_hand_state
            obj_OriState = np.append(init_objpos, np.append(init_objorient, init_objvel))  # ori: represented in the transformed frame
            obj_OriState_ = np.append(init_objpos, np.append(o[33:36], init_objvel)) # ori: represented in the original frame
            num_hand = len(hand_OriState)
            num_obj = len(obj_OriState)
            hand_states_traj = np.zeros([horizon, num_hand])
            object_states_traj = np.zeros([horizon, num_obj])
            hand_states_traj[0, :] = hand_OriState
            object_states_traj[0, :] = obj_OriState_
            z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
            for t_ in range(horizon - 1):
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
            done = False
            t = 0
            obj_height = o[26]
            # while t < horizon - 1 and done != True:  # generate a single traj
            # for the default history states, set them to be initial hand statesimport time
            for i in range(history_state):
                if obj_dynamics:
                    prev_states[i] = np.append(hand_states_traj[0], object_states_traj[0]) 
                else:
                    prev_states[i] = hand_states_traj[0]
            prev_actions = dict()
            for i in range(history_state + 1):
                prev_actions[i] = hand_states_traj[0]
            while t < horizon - 1 and obj_height > 0.15:  # generate a single traj, and early-terminate it when the object falls off  (reorientation task)
                current_hand_state = o[:24]
                current_objpos = o[24:27]
                current_objvel = o[27:33]
                current_objori = o[33:36]
                hand_OriState = current_hand_state
                obj_OriState = np.append(current_objpos, np.append(current_objori, current_objvel))  # ori: represented in the original frame
                if history_state >= 0: # we add the history information into policy inputs
                    if obj_dynamics:  # if we use the object dynamics as part of policy input
                        policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                        prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                        for i in range(history_state + 1):
                            policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                        future_index = (history_state + 1) * (2 * num_hand + num_obj)
                        for t_ in range(num_future_s):
                            if t + future_state[t_] >= horizon:
                                policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1], object_states_traj[horizon - 1])
                            else:
                                policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                    else:
                        policy_input = np.zeros((num_future_s + history_state + 1) * num_hand + (history_state + 1) * num_hand)
                        prev_states[history_state] = hand_OriState # add the current states 
                        for i in range(history_state + 1):
                            policy_input[i * (2 * num_hand): (i + 1) * (2 * num_hand)] = np.append(prev_actions[i], prev_states[i])
                        future_index = (history_state + 1) * (2 * num_hand)
                        for t_ in range(num_future_s):
                            if t + future_state[t_] >= horizon:
                                policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[horizon - 1]
                            else:
                                policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                else:
                    if obj_dynamics:  # if we use the object dynamics as part of policy input
                        policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                        policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                        for t_ in range(1, num_future_s + 1):
                            if t + future_state[t_ - 1] >= horizon:
                                policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1], object_states_traj[horizon - 1])
                            else:
                                policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                    else:
                        policy_input = np.zeros((num_future_s + 1) * num_hand)
                        policy_input[:num_hand] = hand_OriState
                        for t_ in range(1, num_future_s + 1):
                            if t + future_state[t_ - 1] >= horizon:
                                policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[horizon - 1]
                            else:
                                policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                a, agent_info = policy.get_action(policy_input) # FC network when using gaussian_mlp as policy
                # a.shape = 30 -> for the relocation task
                # return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]
                # a -> action = mean + noise
                # agent_info -> {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}
                if eval_mode: # eval_mode -> no noise is added to mean (evaluate current policy)
                    a = agent_info['evaluation']
                if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                    a += hand_states_traj[t + 1].copy() 
                # update the history information
                if history_state >= 0:
                    for i in range(history_state):
                        prev_actions[i] = prev_actions[i + 1]
                        prev_states[i] = prev_states[i + 1]
                    prev_actions[history_state] = a
                # when generaing samples, eval_mode is False
                env_info_base = env.get_env_infos()  # seems always to be {}
                if env.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                    next_o, r, done, env_info_step = env.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                elif env.env.control_mode == 'PID': # a represents the target joint position 
                    PD_controller.set_goal(a)
                    for _ in range(5): # control frequency: 500HZ / use this to output PD target
                        torque_action = PD_controller(env.get_env_state()['qpos'][:len(a)], env.get_env_state()['qvel'][:len(a)])
                        next_o, r, done, env_info_step = env.step(torque_action)  
                # apply the action to the environment and obtained the observation at next step and the reward value after the execution
                obj_height = next_o[26]
                # r is task-specific rewards 
                task_rewards.append(r)
                # add tracking rewards
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
                tracking_hand_reward = Comp_tracking_reward_pen(reference, obs, coeffcients)['hand_reward']   
                tracking_object_reward = Comp_tracking_reward_pen(reference, obs, coeffcients)['object_reward']   
                tracking_rewards.append(tracking_reward) 
                tracking_hand_rewards.append(tracking_hand_reward)
                tracking_object_rewards.append(tracking_object_reward)
                r = r * coeffcients['task_ratio'] + tracking_reward * coeffcients['tracking_ratio']  # total reward
                # below is important to ensure correct env_infos for the timestep
                env_info = env_info_step if env_info_base == {} else env_info_base
                # observations.append(o)  
                observations.append(policy_input) 
                actions.append(a)
                rewards.append(r)
                agent_infos.append(agent_info)  # include the control actions
                env_infos.append(env_info)  # include the indicator if the task is finished
                o = next_o
                t += 1
        elif task_id == 'relocate':
            o, desired_pos = env.reset()
            temp_resnet_features=[]
            desired_pos_main=desired_pos
            init_hand_state = o[:30]
            init_objpos_world = o[39:42] # object position in the world frame
            init_objpos_new = init_objpos_world - desired_pos # converged object position
            init_objvel = env.get_env_state()['qvel'][30:36]
            init_objorient = env.get_env_state()['qpos'][33:36]
            hand_OriState = init_hand_state
            rgb, depth = env.env.mj_render()
            rgb = (rgb.astype(np.uint8) - 128.0) / 128
            depth = depth[...,np.newaxis]
            rgbd = np.concatenate((rgb,depth),axis=2)
            rgbd = np.transpose(rgbd, (2, 0, 1))
            rgbd = rgbd[np.newaxis, ...]
            rgbd = torch.from_numpy(rgbd).float().to(device)
            # desired_pos = Test_data[k][0]['init']['target_pos']
            desired_pos = desired_pos_main[np.newaxis, ...]
            desired_pos = torch.from_numpy(desired_pos).float().to(device)
            if ep==0:
                start_time = time.perf_counter()
            implict_objpos = resnet_model(rgbd, desired_pos) 
            if ep==0:
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Elapsed time for calling resnet: {elapsed_time} seconds")

            obj_OriState = implict_objpos[0].cpu().detach().numpy()
            # obj_OriState = np.append(init_objpos_new, np.append(init_objorient, init_objvel))  # ori: represented in the transformed frame
            # obj_OriState_ = np.append(init_objpos_world, np.append(init_objorient, init_objvel)) # ori: represented in the original frame
            num_hand = len(hand_OriState)
            num_obj = len(obj_OriState)
            hand_states_traj = np.zeros([horizon, num_hand])
            object_states_traj = np.zeros([horizon, num_obj])
            hand_states_traj[0, :] = hand_OriState
            object_states_traj[0, :] = obj_OriState
            z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
            if ep==0:
                start_time = time.perf_counter()
            for t_ in range(horizon - 1):
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
                hand_states_traj[t_ + 1, :] = hand_OriState
                object_states_traj[t_ + 1, :] = obj_OriState
            if ep==0:
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Elapsed time for open loop rollout: {elapsed_time} seconds")
            done = False
            t = 0
            obj_height = o[41]
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
            if ep==0:
                start_time = time.perf_counter()
            while t < horizon - 1  and obj_height > -0.05:  # what would be early-termination for relocation task?
                if not policy.freeze_base:
                    current_hand_state = o[:30]
                else:
                    if policy.m == 24:
                        current_hand_state = o[6:30]
                        num_hand = 24
                    elif policy.m == 27:
                        current_hand_state = o[3:30]
                        num_hand = 27
                # current_objpos = o[39:42] # object position in the world frame
                # current_objvel = env.get_env_state()['qvel'][30:36]
                # current_objori = env.get_env_state()['qpos'][33:36]
                hand_OriState = current_hand_state
                # obj_OriState = np.append(current_objpos, np.append(current_objori, current_objvel))  # ori: represented in the original frame
                rgb, depth = env.env.mj_render()
                rgb = (rgb.astype(np.uint8) - 128.0) / 128
                depth = depth[...,np.newaxis]
                rgbd = np.concatenate((rgb,depth),axis=2)
                rgbd = np.transpose(rgbd, (2, 0, 1))
                rgbd = rgbd[np.newaxis, ...]
                rgbd = torch.from_numpy(rgbd).float().to(device)
                # desired_pos = Test_data[k][0]['init']['target_pos']
                desired_pos = desired_pos_main[np.newaxis, ...]
                desired_pos = torch.from_numpy(desired_pos).float().to(device)
                implict_objpos = resnet_model(rgbd, desired_pos) 
                
                obj_OriState = implict_objpos[0].cpu().detach().numpy()
                if ep<10:
                    temp_resnet_features.append(obj_OriState)
                if history_state >= 0: # we add the history information into policy inputs
                    if obj_dynamics:  # if we use the object dynamics as part of policy input
                        policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                        prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                        for i in range(history_state + 1):
                            policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                        future_index = (history_state + 1) * (2 * num_hand + num_obj)
                        if not policy.freeze_base:
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= horizon:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1], object_states_traj[horizon - 1])
                                else:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                        else:
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= horizon:
                                    if policy.m == 24:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1][6:30], object_states_traj[horizon - 1])
                                    elif policy.m == 27:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1][3:30], object_states_traj[horizon - 1])
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
                                if t + future_state[t_] >= horizon:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[horizon - 1]
                                else:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                        else:
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= horizon:
                                    if policy.m == 24:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[horizon - 1][6:30]
                                    elif policy.m == 27:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[horizon - 1][3:30]
                                else:
                                    if policy.m == 24:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][6:30]
                                    elif policy.m == 27:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][3:30]
                else:
                    if obj_dynamics:  # if we use the object dynamics as part of policy input
                        policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                        policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                        if not policy.freeze_base:
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= horizon:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1], object_states_traj[horizon - 1])
                                else:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= horizon:
                                    if policy.m == 24:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1][6:30], object_states_traj[horizon - 1])
                                    elif policy.m == 27:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1][3:30], object_states_traj[horizon - 1])
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
                                if t + future_state[t_ - 1] >= horizon:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[horizon - 1]
                                else:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                        else:
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= horizon:
                                    if policy.m == 24:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[horizon - 1][6:30]
                                    elif policy.m == 27:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[horizon - 1][3:30]
                                else:
                                    if policy.m == 24:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][6:30]
                                    elif policy.m == 27:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][3:30]
                a, agent_info = policy.get_action(policy_input) # FC network when using gaussian_mlp as policy
                # a.shape = 30 -> for the relocation task
                # return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]
                # a -> action = mean + noise
                # agent_info -> {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}
                if eval_mode: # eval_mode -> no noise is added to mean (evaluate current policy)
                    a = agent_info['evaluation']
                if policy.policy_output == 'djp':  # delta joint position, so we have to add the current hand joints
                    if not policy.freeze_base:
                        a += hand_states_traj[t + 1].copy() 
                    else:
                        if policy.m == 24:
                            a += hand_states_traj[t + 1][6:30].copy() 
                        elif policy.m ==27:
                            a += hand_states_traj[t + 1][3:30].copy() 
                # update the history information
                if history_state >= 0:
                    for i in range(history_state):
                        prev_actions[i] = prev_actions[i + 1]
                        prev_states[i] = prev_states[i + 1]
                    prev_actions[history_state] = a
                # when generaing samples, eval_mode is False
                env_info_base = env.get_env_infos()  # seems always to be {}
                if env.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                    next_o, r, done, env_info_step = env.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                elif env.env.control_mode == 'PID': # a represents the target joint position 
                    if not policy.freeze_base:
                        PD_controller.set_goal(a)
                    else:
                        if policy.m == 24:
                            PD_controller.set_goal(np.append(hand_states_traj[t + 1][:6], a))
                            num_hand = len(np.append(hand_states_traj[t + 1][:6], a))
                        elif policy.m ==27:
                            PD_controller.set_goal(np.append(hand_states_traj[t + 1][:3], a))
                            num_hand = len(np.append(hand_states_traj[t + 1][:3], a))
                    for _ in range(5): # control frequency: 500HZ / use this to output PD target
                        torque_action = PD_controller(env.get_env_state()['qpos'][:num_hand], env.get_env_state()['qvel'][:num_hand])
                        torque_action[1] -= 0.95  # hand_Txyz[1] -> hand_T_y
                        next_o, r, done, env_info_step = env.step(torque_action)  
                # apply the action to the environment and obtained the observation at next step and the reward value after the execution
                obj_height = next_o[41]
                # r is task-specific rewards 
                task_rewards.append(r)
                # add tracking rewards
                reference = dict()
                reference['hand_state'] = hand_states_traj[t + 1]
                reference['obj_feature'] = object_states_traj[t + 1]
                # reference['obj_pos'] = object_states_traj[t + 1][:3]
                # reference['obj_vel'] = object_states_traj[t + 1][6:]
                # reference['obj_ori'] = object_states_traj[t + 1][3:6]
                obs = dict()
                obs['hand_state'] = next_o[:30]
                rgb, depth = env.env.mj_render()
                rgb = (rgb.astype(np.uint8) - 128.0) / 128
                depth = depth[...,np.newaxis]
                rgbd = np.concatenate((rgb,depth),axis=2)
                rgbd = np.transpose(rgbd, (2, 0, 1))
                rgbd = rgbd[np.newaxis, ...]
                rgbd = torch.from_numpy(rgbd).float().to(device)
                # desired_pos = Test_data[k][0]['init']['target_pos']
                desired_pos = desired_pos_main[np.newaxis, ...]
                desired_pos = torch.from_numpy(desired_pos).float().to(device)
                implict_objpos = resnet_model(rgbd, desired_pos) 
                # obj_OriState = implict_objpos[0].cpu().detach().numpy()
                obs['obj_feature']= implict_objpos[0].cpu().detach().numpy()
                # obs['obj_pos'] = next_o[39:42]
                # obs['obj_vel'] = env.get_env_state()['qvel'][30:36]
                # obs['obj_ori'] = env.get_env_state()['qpos'][33:36]    
                # if not policy.freeze_base: 
                #     tracking_reward = Comp_tracking_reward_relocate(reference, obs, coeffcients)['total_reward']   
                #     tracking_hand_reward = Comp_tracking_reward_relocate(reference, obs, coeffcients)['hand_reward']   
                #     tracking_object_reward = Comp_tracking_reward_relocate(reference, obs, coeffcients)['object_reward']  
                # else:
                #     if not policy.include_Rots: # only on fingers
                #         tracking_reward = Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['total_reward']   
                #         tracking_hand_reward = Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['hand_reward']   
                #         tracking_object_reward = Comp_tracking_reward_relocate_freeze_base(reference, obs, coeffcients)['object_reward'] 
                #     else:
                #         tracking_reward = Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['total_reward']   
                #         tracking_hand_reward = Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['hand_reward']   
                #         tracking_object_reward = Comp_tracking_reward_relocate_include_Rots(reference, obs, coeffcients)['object_reward'] 
                tracking_reward = Comp_tracking_reward_relocate(reference, obs, coeffcients)['total_reward']
                tracking_rewards.append(tracking_reward) 
                # tracking_hand_rewards.append(tracking_hand_reward)
                # tracking_object_rewards.append(tracking_object_reward)
                r = r * coeffcients['task_ratio'] + tracking_reward * coeffcients['tracking_ratio']  # total reward
                # below is important to ensure correct env_infos for the timestep
                env_info = env_info_step if env_info_base == {} else env_info_base
                # observations.append(o)  
                observations.append(policy_input) 
                actions.append(a)
                rewards.append(r)
                agent_infos.append(agent_info)  # include the control actions
                env_infos.append(env_info)  # include the indicator if the task is finished
                o = next_o
                t += 1
            if ep==0:
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Elapsed time for closed loop rollout: {elapsed_time} seconds")
            if ep<10:
                temp.append(temp_resnet_features)
            if ep==10:
                temp=np.array(temp)
                np.save('rl_resnet_features.npy', np.array(temp))
        elif task_id == 'door':
            o, desired_pos = env.reset()
            init_hand_state = env.get_env_state()['qpos'][:28]
            init_objpos = o[32:35] # current handle position
            init_objvel = env.get_env_state()['qvel'][28:29]
            init_handle = desired_pos
            hand_OriState = init_hand_state
            obj_OriState = np.append(init_objpos, np.append(init_objvel, init_handle))  # ori: represented in the transformed frame
            num_hand = len(hand_OriState)
            num_obj = len(obj_OriState)
            hand_states_traj = np.zeros([horizon, num_hand])
            object_states_traj = np.zeros([horizon, num_obj])
            hand_states_traj[0, :] = hand_OriState
            object_states_traj[0, :] = obj_OriState
            z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
            for t_ in range(horizon - 1):
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
            done = False
            t = 0
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
            while t < horizon - 1:
                # env.render()
                if not policy.freeze_base:
                    current_hand_state = env.get_env_state()['qpos'][:num_hand]
                else:
                    if policy.m == 24:
                        current_hand_state = env.get_env_state()['qpos'][4:28]
                        num_hand = 24
                    elif policy.m == 25:
                        current_hand_state = env.get_env_state()['qpos'][3:28]
                        num_hand = 25
                    elif policy.m == 26:
                        current_hand_state = np.append(env.get_env_state()['qpos'][0], env.get_env_state()['qpos'][3:28])
                        num_hand = 26
                    elif policy.m == 27:
                        current_hand_state = env.get_env_state()['qpos'][1:28]
                        num_hand = 27
                current_objpos = o[32:35]
                current_objvel = env.get_env_state()['qvel'][28:29]
                init_handle = env.get_env_state()['door_body_pos']
                hand_OriState = current_hand_state
                obj_OriState = np.append(current_objpos, np.append(current_objvel, init_handle))
                if history_state >= 0: # we add the history information into policy inputs
                    if obj_dynamics:  # if we use the object dynamics as part of policy input
                        policy_input = np.zeros((num_future_s + history_state + 1) * (num_hand + num_obj) + (history_state + 1) * num_hand)
                        prev_states[history_state] = np.append(hand_OriState, obj_OriState) # add the current states 
                        for i in range(history_state + 1):
                            policy_input[i * (2 * num_hand + num_obj): (i + 1) * (2 * num_hand + num_obj)] = np.append(prev_actions[i], prev_states[i])
                        future_index = (history_state + 1) * (2 * num_hand + num_obj)
                        if not policy.freeze_base:
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= horizon:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1], object_states_traj[horizon - 1])
                                else:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                        else:
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= horizon:
                                    if policy.m == 24:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1][4:28], object_states_traj[horizon - 1])
                                    elif policy.m == 25:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1][3:28], object_states_traj[horizon - 1])
                                    elif policy.m == 26:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[horizon - 1][0], hand_states_traj[horizon - 1][3:28]), object_states_traj[horizon - 1])
                                    elif policy.m == 27:
                                        policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1][1:28], object_states_traj[horizon - 1])
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
                                if t + future_state[t_] >= horizon:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[horizon - 1]
                                else:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                        else:
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= horizon:
                                    if policy.m == 24:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[horizon - 1][4:28]
                                    elif policy.m == 25:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[horizon - 1][3:28]
                                    elif policy.m == 26:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = np.append(hand_states_traj[horizon - 1][0], hand_states_traj[horizon - 1][3:28])
                                    elif policy.m == 27:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[horizon - 1][1:28]
                                else:
                                    if policy.m == 24:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][4:28]
                                    elif policy.m == 25:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][3:28]
                                    elif policy.m == 26:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]][0], hand_states_traj[t + future_state[t_]][3:28])
                                    elif policy.m == 27:
                                        policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][1:28]
                else:
                    if obj_dynamics:  # if we use the object dynamics as part of policy input
                        policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                        policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                        if not policy.freeze_base:
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= horizon:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1], object_states_traj[horizon - 1])
                                else:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= horizon:
                                    if policy.m == 24:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1][4:28], object_states_traj[horizon - 1])
                                    elif policy.m == 25:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1][3:28], object_states_traj[horizon - 1])
                                    elif policy.m == 26:    
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(np.append(hand_states_traj[horizon - 1][0], hand_states_traj[horizon - 1][3:28]), object_states_traj[horizon - 1])
                                    elif policy.m == 27:
                                        policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1][1:28], object_states_traj[horizon - 1])
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
                                if t + future_state[t_ - 1] >= horizon:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[horizon - 1]
                                else:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                        else:
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= horizon:
                                    if policy.m == 24:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[horizon - 1][4:28]
                                    elif policy.m == 25:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[horizon - 1][3:28]
                                    elif policy.m == 26:    
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = np.append(hand_states_traj[horizon - 1][0], hand_states_traj[horizon - 1][3:28])
                                    elif policy.m == 27:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[horizon - 1][1:28]
                                else:
                                    if policy.m == 24:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][4:28]
                                    elif policy.m == 25:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][3:28]
                                    elif policy.m == 26:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][0], hand_states_traj[t + future_state[t_ - 1]][3:28])
                                    elif policy.m == 27:
                                        policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][1:28]
                a, agent_info = policy.get_action(policy_input) # FC network when using gaussian_mlp as policy
                # a.shape = 30 -> for the relocation task
                # return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]
                # a -> action = mean + noise
                # agent_info -> {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}
                if eval_mode: # eval_mode -> no noise is added to mean (evaluate current policy)
                    a = agent_info['evaluation']
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
                # when generaing samples, eval_mode is False
                env_info_base = env.get_env_infos()  # seems always to be {}
                if env.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                    next_o, r, done, env_info_step = env.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                elif env.env.control_mode == 'PID': # a represents the target joint position 
                    if not policy.freeze_base:
                        PD_controller.set_goal(a)
                    else:
                        if policy.m == 24:
                            PD_controller.set_goal(np.append(hand_states_traj[t + 1][:4], a))
                            num_hand = len(np.append(hand_states_traj[t + 1][:4], a))
                        elif policy.m == 25:
                            PD_controller.set_goal(np.append(hand_states_traj[t + 1][:3], a))
                            num_hand = len(np.append(hand_states_traj[t + 1][:3], a))
                        elif policy.m == 26:
                            PD_controller.set_goal(np.append(np.append(a[0], hand_states_traj[t + 1][1:3]), a[1:]))
                            num_hand = len(np.append(np.append(a[0], hand_states_traj[t + 1][1:3]), a[1:]))
                        elif policy.m == 27:
                            PD_controller.set_goal(np.append(hand_states_traj[t + 1][:1], a))
                            num_hand = len(np.append(hand_states_traj[t + 1][:1], a))
                    for _ in range(5): # control frequency: 500HZ / use this to output PD target
                        torque_action = PD_controller(env.get_env_state()['qpos'][:num_hand], env.get_env_state()['qvel'][:num_hand])
                        next_o, r, done, env_info_step = env.step(torque_action)  
                # apply the action to the environment and obtained the observation at next step and the reward value after the execution
                # r is task-specific rewards 
                task_rewards.append(r)
                # add tracking rewards
                reference = dict()
                reference['hand_state'] = hand_states_traj[t + 1]
                reference['obj_pos'] = object_states_traj[t + 1][:3]
                reference['obj_vel'] = object_states_traj[t + 1][3:4]
                reference['init_handle'] = object_states_traj[t + 1][4:]
                obs = dict()
                obs['hand_state'] = env.get_env_state()['qpos'][:num_hand]
                obs['obj_pos'] = next_o[32:35]
                obs['obj_vel'] = env.get_env_state()['qvel'][28:29]
                obs['init_handle'] = env.get_env_state()['door_body_pos']
                if not policy.freeze_base:
                    tracking_reward = Comp_tracking_reward_door(reference, obs, coeffcients)['total_reward']   
                    tracking_hand_reward = Comp_tracking_reward_door(reference, obs, coeffcients)['hand_reward']   
                    tracking_object_reward = Comp_tracking_reward_door(reference, obs, coeffcients)['object_reward']   
                else:
                    if not policy.include_Rots: # only on fingers
                        tracking_reward = Comp_tracking_reward_door_freeze_base(reference, obs, coeffcients)['total_reward']   
                        tracking_hand_reward = Comp_tracking_reward_door_freeze_base(reference, obs, coeffcients)['hand_reward']   
                        tracking_object_reward = Comp_tracking_reward_door_freeze_base(reference, obs, coeffcients)['object_reward']   
                    else:
                        tracking_reward = Comp_tracking_reward_door_include_Rots(reference, obs, coeffcients)['total_reward']   
                        tracking_hand_reward = Comp_tracking_reward_door_include_Rots(reference, obs, coeffcients)['hand_reward']   
                        tracking_object_reward = Comp_tracking_reward_door_include_Rots(reference, obs, coeffcients)['object_reward']   
                tracking_rewards.append(tracking_reward) 
                tracking_hand_rewards.append(tracking_hand_reward)
                tracking_object_rewards.append(tracking_object_reward)
                r = r * coeffcients['task_ratio'] + tracking_reward * coeffcients['tracking_ratio']  # total reward
                # below is important to ensure correct env_infos for the timestep
                env_info = env_info_step if env_info_base == {} else env_info_base
                # observations.append(o)  
                observations.append(policy_input) 
                actions.append(a)
                rewards.append(r)
                agent_infos.append(agent_info)  # include the control actions
                env_infos.append(env_info)  # include the indicator if the task is finished
                o = next_o
                t += 1
        elif task_id == 'hammer':
            o, height = env.reset()
            init_hand_state = o[:26]
            init_objpos = o[49:52] + o[42:45]  # current tool position
            init_objvel = o[27:33]
            init_objorient = o[39:42] 
            nail_goal = o[46:49]
            hand_OriState = init_hand_state
            obj_OriState = np.append(init_objpos, np.append(init_objorient, np.append(init_objvel, nail_goal)))
            num_hand = len(hand_OriState)
            num_obj = len(obj_OriState)
            hand_states_traj = np.zeros([horizon, num_hand])
            object_states_traj = np.zeros([horizon, num_obj])
            hand_states_traj[0, :] = hand_OriState
            object_states_traj[0, :] = obj_OriState
            z_t = Koopman_obser.z(hand_OriState, obj_OriState)  # initial states in lifted space
            for t_ in range(horizon - 1):
                z_t_1_computed = np.dot(KODex, z_t)
                z_t = z_t_1_computed.copy()
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  # retrieved robot & object states
                hand_OriState = x_t_1_computed[:num_hand]
                obj_pos = x_t_1_computed[num_hand: num_hand + 3] # tool pos
                obj_ori = x_t_1_computed[num_hand + 3: num_hand + 6] # tool ori
                obj_vel = x_t_1_computed[num_hand + 6: num_hand + 12]
                nail_goal = x_t_1_computed[num_hand + 12:]
                obj_OriState = np.append(obj_pos, np.append(obj_ori, np.append(obj_vel, nail_goal)))
                hand_states_traj[t_ + 1, :] = hand_OriState
                object_states_traj[t_ + 1, :] = obj_OriState
            done = False
            t = 0
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
            while t < horizon - 1:
                # env.render()
                if not policy.freeze_base:
                    current_hand_state = o[:26]
                else:
                    current_hand_state = o[2:26]
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
                                if t + future_state[t_] >= horizon:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1], object_states_traj[horizon - 1])
                                else:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_]], object_states_traj[t + future_state[t_]])
                        else:
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= horizon:
                                    policy_input[future_index + (num_hand + num_obj) * t_: future_index + (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1][2:26], object_states_traj[horizon - 1])
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
                                if t + future_state[t_] >= horizon:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[horizon - 1]
                                else:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]]
                        else:
                            for t_ in range(num_future_s):
                                if t + future_state[t_] >= horizon:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[horizon - 1][2:26]
                                else:
                                    policy_input[future_index + num_hand * t_:future_index + num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_]][2:26]
                else:
                    if obj_dynamics:  # if we use the object dynamics as part of policy input
                        policy_input = np.zeros((num_future_s + 1) * (num_hand + num_obj))
                        policy_input[:num_hand + num_obj] = np.append(hand_OriState, obj_OriState)
                        if not policy.freeze_base:
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= horizon:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1], object_states_traj[horizon - 1])
                                else:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]], object_states_traj[t + future_state[t_ - 1]])
                        else:
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= horizon:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[horizon - 1][2:26], object_states_traj[horizon - 1])
                                else:
                                    policy_input[(num_hand + num_obj) * t_: (num_hand + num_obj) * (t_ + 1)] = np.append(hand_states_traj[t + future_state[t_ - 1]][2:26], object_states_traj[t + future_state[t_ - 1]])
                    else:
                        policy_input = np.zeros((num_future_s + 1) * num_hand)
                        policy_input[:num_hand] = hand_OriState
                        if not policy.freeze_base:
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= horizon:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[horizon - 1]
                                else:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]]
                        else:
                            for t_ in range(1, num_future_s + 1):
                                if t + future_state[t_ - 1] >= horizon:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[horizon - 1][2:26]
                                else:
                                    policy_input[num_hand * t_: num_hand * (t_ + 1)] = hand_states_traj[t + future_state[t_ - 1]][2:26]
                a, agent_info = policy.get_action(policy_input) # FC network when using gaussian_mlp as policy
                # a.shape = 30 -> for the relocation task
                # return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]
                # a -> action = mean + noise
                # agent_info -> {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}
                if eval_mode: # eval_mode -> no noise is added to mean (evaluate current policy)
                    a = agent_info['evaluation']
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
                # when generaing samples, eval_mode is False
                env_info_base = env.get_env_infos()  # seems always to be {}
                if env.env.control_mode == 'Torque':  # a directly represents the torque values, no need to use PID
                    next_o, r, done, env_info_step = env.step(a)   # control frequency: 100HZ (= planning frequency) / use this to directly output torque
                elif env.env.control_mode == 'PID': # a represents the target joint position 
                    if not policy.freeze_base:
                        PD_controller.set_goal(a)
                    else:
                        PD_controller.set_goal(np.append(hand_states_traj[t + 1][:2], a))
                        num_hand = len(np.append(hand_states_traj[t + 1][:2], a))
                    for _ in range(5): # control frequency: 500HZ / use this to output PD target
                        torque_action = PD_controller(env.get_env_state()['qpos'][:num_hand], env.get_env_state()['qvel'][:num_hand])
                        next_o, r, done, env_info_step = env.step(torque_action)  
                # apply the action to the environment and obtained the observation at next step and the reward value after the execution
                # r is task-specific rewards 
                task_rewards.append(r)
                # add tracking rewards
                reference = dict()
                reference['hand_state'] = hand_states_traj[t + 1]
                reference['obj_pos'] = object_states_traj[t + 1][:3]
                reference['obj_ori'] = object_states_traj[t + 1][3:6]
                reference['obj_vel'] = object_states_traj[t + 1][6:12]
                reference['nail_goal'] = object_states_traj[t + 1][12:]
                obs = dict()
                obs['hand_state'] = next_o[:26]
                obs['obj_pos'] = next_o[49:52] + next_o[42:45]
                obs['obj_ori'] = next_o[39:42]
                obs['obj_vel'] = next_o[27:33]
                obs['nail_goal'] = next_o[46:49]
                if not policy.freeze_base:
                    tracking_reward = Comp_tracking_reward_hammer(reference, obs, coeffcients)['total_reward']   
                    tracking_hand_reward = Comp_tracking_reward_hammer(reference, obs, coeffcients)['hand_reward']   
                    tracking_object_reward = Comp_tracking_reward_hammer(reference, obs, coeffcients)['object_reward']   
                else:
                    tracking_reward = Comp_tracking_reward_hammer_freeze_base(reference, obs, coeffcients)['total_reward']   
                    tracking_hand_reward = Comp_tracking_reward_hammer_freeze_base(reference, obs, coeffcients)['hand_reward']   
                    tracking_object_reward = Comp_tracking_reward_hammer_freeze_base(reference, obs, coeffcients)['object_reward'] 
                tracking_rewards.append(tracking_reward) 
                tracking_hand_rewards.append(tracking_hand_reward)
                tracking_object_rewards.append(tracking_object_reward)
                r = r * coeffcients['task_ratio'] + tracking_reward * coeffcients['tracking_ratio']  # total reward
                # below is important to ensure correct env_infos for the timestep
                env_info = env_info_step if env_info_base == {} else env_info_base
                # observations.append(o)  
                observations.append(policy_input) 
                actions.append(a)
                rewards.append(r)
                agent_infos.append(agent_info)  # include the control actions
                env_infos.append(env_info)  # include the indicator if the task is finished
                o = next_o
                t += 1
        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            task_rewards=np.array(task_rewards),
            tracking_rewards=np.array(tracking_rewards),
            tracking_hand_rewards=np.array(tracking_hand_rewards),
            tracking_object_rewards=np.array(tracking_object_rewards),
            terminated=done  # if the task is completed at last
        )
        paths.append(path)
    del(env)
    return paths


def sample_paths(
        num_traj,
        env,
        task_id,
        policy,
        eval_mode = False,
        horizon = 1e6,
        future_state=(1,2,3),
        history_state=0,
        base_seed = None,
        num_cpu = 1,
        max_process_time=300,
        max_timeouts=4,
        suppress_print=False,
        env_kwargs=None,
        Koopman_obser=None, 
        KODex=None,
        coeffcients=None,
        obj_dynamics=True,
        control_mode='Torque',
        PD_controller=None,
        resnet_model=None
        ):
    num_cpu = 1 if num_cpu is None else num_cpu
    num_cpu = mp.cpu_count() if num_cpu == 'max' else num_cpu
    assert type(num_cpu) == int
    if num_cpu == 1:
        input_dict = dict(num_traj=num_traj, env=env, task_id = task_id, policy=policy,
                          eval_mode=eval_mode, horizon=horizon, future_state=future_state, history_state=history_state, base_seed=base_seed,
                          env_kwargs=env_kwargs, Koopman_obser=Koopman_obser, KODex=KODex,coeffcients=coeffcients,obj_dynamics=obj_dynamics,
                          control_mode=control_mode,PD_controller=PD_controller,resnet_model=resnet_model)
        # dont invoke multiprocessing if not necessary
        return do_rollout(**input_dict)

    # do multiprocessing otherwise
    paths_per_cpu = int(np.ceil(num_traj/num_cpu))
    input_dict_list= []
    for i in range(num_cpu):
        input_dict = dict(num_traj=paths_per_cpu, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon, future_state=future_state,
                          history_state=history_state, base_seed=base_seed + i * paths_per_cpu,
                          env_kwargs=env_kwargs, Koopman_obser=Koopman_obser, KODex=KODex,coeffcients=coeffcients,obj_dynamics=obj_dynamics,
                          control_mode=control_mode,PD_controller=PD_controller, resnet_model=resnet_model)
        input_dict_list.append(input_dict)
    if suppress_print is False:
        start_time = timer.time()
        print("####### Gathering Samples #######")

    results = _try_multiprocess(do_rollout, input_dict_list,
                                num_cpu, max_process_time, max_timeouts)
    paths = []
    # result is a paths type and results is list of paths
    for result in results:
        for path in result:
            paths.append(path)  

    if suppress_print is False:
        print("======= Samples Gathered  ======= | >>>> Time taken = %f " %(timer.time()-start_time) )

    return paths


def sample_data_batch(
        num_samples,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        future_state=(1,2,3),
        history_state=0,
        base_seed = None,
        num_cpu = 1,
        paths_per_call = 1,
        env_kwargs=None,
        Koopman_obser=None, 
        KODex=None,
        coeffcients=None,
        obj_dynamics=True,
        control_mode='Torque',
        PD_controller=None,
        ):

    num_cpu = 1 if num_cpu is None else num_cpu
    num_cpu = mp.cpu_count() if num_cpu == 'max' else num_cpu
    assert type(num_cpu) == int

    start_time = timer.time()
    print("####### Gathering Samples #######")
    sampled_so_far = 0
    paths_so_far = 0
    paths = []
    base_seed = 123 if base_seed is None else base_seed
    while sampled_so_far < num_samples:
        base_seed = base_seed + 12345
        new_paths = sample_paths(paths_per_call * num_cpu, env, policy,
                                 eval_mode, horizon, base_seed, num_cpu,
                                 suppress_print=True, env_kwargs=env_kwargs)
        for path in new_paths:
            paths.append(path)
        paths_so_far += len(new_paths)
        new_samples = np.sum([len(p['rewards']) for p in new_paths])
        sampled_so_far += new_samples
    print("======= Samples Gathered  ======= | >>>> Time taken = %f " % (timer.time() - start_time))
    print("................................. | >>>> # samples = %i # trajectories = %i " % (
    sampled_so_far, paths_so_far))
    return paths


def _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts):
    
    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=None)
    parallel_runs = [pool.apply_async(func, kwds=input_dict) for input_dict in input_dict_list]
    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()
        return _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts-1)

    pool.close()
    pool.terminate()
    pool.join()  
    return results

# def Comp_tracking_reward(reference, states, coeffcients):
#     r_hand = coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'] - states['hand_state']) ** 2 / len(reference['hand_state']))
#     r_obj_ori = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_ori'] - states['obj_ori']) ** 2 / len(reference['obj_ori']))
#     r_obj_pos = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_pos'] - states['obj_pos']) ** 2 / len(reference['obj_pos']))
#     r_obj_vel = coeffcients['object_track'] * np.exp(-5 * np.linalg.norm(reference['obj_vel'] - states['obj_vel']) ** 2 / len(reference['obj_vel']))
#     return r_hand + r_obj_ori + r_obj_vel

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
    if coeffcients['ADD_BONUS_PENALTY'] == 1:
        if np.linalg.norm(reference['hand_state'][:4] - states['hand_state'][:4]) > 1:
            r_hand -= 10
        elif np.linalg.norm(reference['hand_state'][:4] - states['hand_state'][:4]) > 0.7:
            r_hand -= 5
        elif np.linalg.norm(reference['hand_state'][:4] - states['hand_state'][:4]) > 0.4:
            r_hand -= 1
    r_hand += coeffcients['hand_track'] * np.exp(-5 * np.linalg.norm(reference['hand_state'][4:] - states['hand_state'][4:]) ** 2 / len(reference['hand_state'][4:]))
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
