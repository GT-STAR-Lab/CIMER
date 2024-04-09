"""
Basic reinforce algorithm using on-policy rollouts
Also has function to perform linesearch on KL (improves stability)
"""

import logging
from re import A
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
from torch.autograd import Variable

# samplers
import mjrl.samplers.core as trajectory_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog


class BatchREINFORCE:
    def __init__(self, env, policy, baseline,
                 learn_rate=0.01,
                 seed=123,
                 desired_kl=None,
                 save_logs=False,
                 **kwargs
                 ):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.alpha = learn_rate
        self.seed = seed
        self.save_logs = save_logs
        self.running_score = None
        self.desired_kl = desired_kl
        if save_logs: self.logger = DataLog()

    def CPI_surrogate(self, observations, actions, advantages):
        # CPI: Conservative Policy Iteration
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        old_dist_info = self.policy.old_dist_info(observations, actions)  # the new and old policy indeed are exactly the same -> 
        # LR is always ones tensor for the feedforward computation, but for the autograd(backward), we still get the gradient wrt the trainable parameters of the new model.
        # I think this strategy is about find the !local! change of the policy parameters to make the surrograte values better
        new_dist_info = self.policy.new_dist_info(observations, actions)
        # importance sampling
        LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info) # the direction of increasing the policy values for enlarging the probability of taking current action
        surr = torch.mean(LR*adv_var)  # surr is the conservative policy iteration (CPI) loss
        # corrsponds to equation (2) in the paper(likelihood * advantage). the advantage value measures the goodness of taking this action
        # this one of computing the likelihood ratio (the so-called “surrogate” objective) is from TRPO paper, by defining new and old policy networks
        return surr

    def kl_old_new(self, observations, actions):
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        return mean_kl

    def flat_vpg(self, observations, actions, advantages):
        cpi_surr = self.CPI_surrogate(observations, actions, advantages)
        vpg_grad = torch.autograd.grad(cpi_surr, self.policy.trainable_params)  # pytorch autograd to obtain the gradient of the trainable parameters.
        # pay attention: only the parameters of new model dist are trainable, which means we are looking for some local change to the new model parameters
        # print(len(vpg_grad))  # vpg_grad -> the gradient of cpi_surr wrt each network weights (linear + bias) and log_std
        vpg_grad = np.concatenate([g.contiguous().view(-1).data.numpy() for g in vpg_grad])  # reduce to 1 dimension
        return vpg_grad

    # ----------------------------------------------------------
    def train_step(self, N,
                   env=None,
                   sample_mode='trajectories',
                   horizon=1e6,
                   future_state=(1,2,3),
                   history_state=0,
                   gamma=0.995,
                   gae_lambda=0.97,
                   num_cpu='max',
                   env_kwargs=None,
                   Koopman_obser=None, 
                   KODex=None,
                   coeffcients=None,
                   obj_dynamics=True,
                   control_mode='Torque',
                   PD_controller=None,
                   ):
        print("enter this function -- batch_reinforce")
        # Clean up input arguments
        env = self.env if env is None else env
        if sample_mode != 'trajectories' and sample_mode != 'samples':
            print("sample_mode in NPG must be either 'trajectories' or 'samples'")
            quit()

        ts = timer.time()
        if sample_mode == 'trajectories':
            input_dict = dict(num_traj=N, env=env, task_id = self.env.env_id.split('-')[0], policy=self.policy, horizon=horizon,future_state=future_state,history_state=history_state,
                              base_seed=self.seed, num_cpu=num_cpu, env_kwargs=env_kwargs, Koopman_obser=Koopman_obser, KODex=KODex, coeffcients=coeffcients,obj_dynamics=obj_dynamics,control_mode=control_mode,PD_controller=PD_controller)
            paths = trajectory_sampler.sample_paths(**input_dict)  # obtained the generated paths using the current policy
        elif sample_mode == 'samples':
            input_dict = dict(num_samples=N, env=env, policy=self.policy, horizon=horizon,future_state=future_state,history_state=history_state,
                              base_seed=self.seed, num_cpu=num_cpu, env_kwargs=env_kwargs, Koopman_obser=Koopman_obser, KODex=KODex, coeffcients=coeffcients,obj_dynamics=obj_dynamics,control_mode=control_mode,PD_controller=PD_controller)
            paths = trajectory_sampler.sample_data_batch(**input_dict)
        if self.save_logs:
            self.logger.log_kv('time_sampling', timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed # N -> number of trajectory
        # The below are for the RL
        # compute returns
        process_samples.compute_returns(paths, gamma)  # each dict in paths is added with returns / returns are not advantage values
        # compute advantages
        process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)  # advantage function is defined in the paper (eqn. (1))
        # each dict in paths is again added with advantages and baseline

        # print("action: ", paths[0]['actions'].shape)  # returns ->Q value computed by rollout method
        # print("return: ", paths[0]['returns'].shape)  # returns ->Q value computed by rollout method
        # print("rewards: ", paths[0]['rewards'].shape)
        # print("advantages: ", paths[0]['advantages'].shape)
        # print("agent_infos: ", paths[0]['agent_infos']['log_std'])

        # the baseline model used here has not been updated yet (has not been applied with fit function), that's why in the paper, they mentioned that the baseline model at iteration k
        # that is used to compute the advantages and gradients (in train_from_paths) is indeed fitted using the trajectoris at iteration k-1.
        # Answer: Baseline model is being considered as an approximate value function V^{\pi}
        
        # compute the task & tracking rewards
        task_rewards = [sum(p["task_rewards"]) for p in paths]  # (not discounted) mean combined rewards
        tracking_rewards = [sum(p["tracking_rewards"]) for p in paths] # (not discounted) mean combined rewards
        tracking_hand_rewards = [sum(p["tracking_hand_rewards"]) for p in paths] # (not discounted) mean combined rewards
        tracking_object_rewards = [sum(p["tracking_object_rewards"]) for p in paths] # (not discounted) mean combined rewards
        mean_task_rewards = np.mean(task_rewards)
        mean_tracking_rewards = np.mean(tracking_rewards)
        mean_tracking_hand_rewards = np.mean(tracking_hand_rewards)
        mean_tracking_object_rewards = np.mean(tracking_object_rewards)
        min_task_rewards = np.amin(task_rewards)
        max_task_rewards = np.amax(task_rewards)
        min_tracking_rewards = np.amin(tracking_rewards)
        max_tracking_rewards = np.amax(tracking_rewards)
        min_tracking_hand_rewards = np.amin(tracking_hand_rewards)
        max_tracking_hand_rewards = np.amax(tracking_hand_rewards)
        min_tracking_object_rewards = np.amin(tracking_object_rewards)
        max_tracking_object_rewards = np.amax(tracking_object_rewards)

        weighted_rewards = np.mean([p["returns"][0] for p in paths])
        if self.save_logs:
            self.logger.log_kv('mean_log_std', np.mean(self.policy.log_std_val))
            self.logger.log_kv('mean_std', np.mean(np.exp(self.policy.log_std_val)))
            self.logger.log_kv('rollout_mean_task_rewards', mean_task_rewards)
            self.logger.log_kv('rollout_min_task_rewards', min_task_rewards)
            self.logger.log_kv('rollout_max_task_rewards', max_task_rewards)
            self.logger.log_kv('rollout_mean_tracking_rewards', mean_tracking_rewards)
            self.logger.log_kv('rollout_min_tracking_rewards', min_tracking_rewards)
            self.logger.log_kv('rollout_max_tracking_rewards', max_tracking_rewards)
            self.logger.log_kv('rollout_mean_tracking_hand_rewards', mean_tracking_hand_rewards)
            self.logger.log_kv('rollout_min_tracking_hand_rewards', min_tracking_hand_rewards)
            self.logger.log_kv('rollout_max_tracking_hand_rewards', max_tracking_hand_rewards)
            self.logger.log_kv('rollout_mean_tracking_object_rewards', mean_tracking_object_rewards)
            self.logger.log_kv('rollout_min_tracking_object_rewards', min_tracking_object_rewards)
            self.logger.log_kv('rollout_max_tracking_object_rewards', max_tracking_object_rewards)
            self.logger.log_kv('discounted_traj_rewards', weighted_rewards)
            
        # train from paths
        eval_statistics = self.train_from_paths(paths)  # function (train_from_paths) is re-defied in npg_cg.py and dapg.py
        # in the job_script.py, the object belongs to the class of dapg, so here, (train_from_paths) is loaded from dapg.py
        eval_statistics.append(N)
        # log number of samples
        if self.save_logs:
            num_samples = np.sum([p["rewards"].shape[0] for p in paths])
            self.logger.log_kv('num_samples', num_samples)
        # fit baseline 
        if self.save_logs:
            ts = timer.time()
            error_before, error_after = self.baseline.fit(paths, return_errors=True)  ## baseline model is used to approximate the value function
            self.logger.log_kv('time_VF', timer.time()-ts)
            self.logger.log_kv('VF_error_before', error_before)
            self.logger.log_kv('VF_error_after', error_after)
        else:
            self.baseline.fit(paths)  # fit the baseline model (value function approximation)
        return eval_statistics

    # ----------------------------------------------------------
    def train_from_paths(self, paths): # update polocy (the basic vanilla reinforce gradient policy)
        observations, actions, advantages, base_stats, self.running_score = self.process_paths(paths)
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]

        # VPG
        ts = timer.time()
        # the vpg also uses the likelihood ratio(old and new policy) as the surrogate objective instead of a single likelihood(new policy).
        # In pieter's code, for vpg training, they only used the single likelihood(new policy) as the surrogate objective
        vpg_grad = self.flat_vpg(observations, actions, advantages)  # vpg -> vanilla policy gradient, just involve actions, observations and advantages
        # vpg_gras is like the determined steepest direction, so the next step is to step forward along this direction, using line search or trust region for policy optimization.
        t_gLL += timer.time() - ts

        # Policy update with linesearch (in Pieter;s code, they used ADAM for update)
        # because we just want to make a small chaneg to the policy paramaetrs, so they used the line search to guarantee the kl divergence 
        # between the updated and old polocy upon the same set of observation data is within the threshold (self.desired_kl)
        # ------------------------------
        if self.desired_kl is not None:  
            max_ctr = 100
            alpha = self.alpha  # npg will compute an adaptive learning rate during each iteration
            curr_params = self.policy.get_param_values()
            for ctr in range(max_ctr):
                new_params = curr_params + alpha * vpg_grad
                self.policy.set_param_values(new_params, set_new=True, set_old=False) # update the current policy values at each step
                # while fixing the old one
                kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
                if kl_dist <= self.desired_kl:  # constraint on the size of the policy update, so half the step size if KL divergence is too large
                    break
                else:
                    print("backtracking")
                    alpha = alpha / 2.0
        else:
            curr_params = self.policy.get_param_values()
            new_params = curr_params + self.alpha * vpg_grad

        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', self.alpha)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                except:
                    pass

        return base_stats


    def process_paths(self, paths):
        # Concatenate from all the trajectories
        observations = np.array([path["observations"] for path in paths])
        actions = np.array([path["actions"] for path in paths])
        advantages = np.array([path["advantages"] for path in paths])

        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        running_score = mean_return if self.running_score is None else \
                        0.9 * self.running_score + 0.1 * mean_return

        return observations, actions, advantages, base_stats, running_score


    def log_rollout_statistics(self, paths):
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        self.logger.log_kv('stoc_pol_mean', mean_return)  # no discount factor
        self.logger.log_kv('stoc_pol_std', std_return)
        self.logger.log_kv('stoc_pol_max', max_return)
        self.logger.log_kv('stoc_pol_min', min_return)
        try:
            success_rate = self.env.env.env.evaluate_success(paths)
            self.logger.log_kv('rollout_success', success_rate)
        except:
            pass
