import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

# samplers
import mjrl.samplers.core as trajectory_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve
from mjrl.algos.batch_reinforce import BatchREINFORCE


class PPO(BatchREINFORCE):
    def __init__(self, env, policy, baseline,
                 demo_paths=None,
                 clip_coef = 0.2, # same as used in the animal imitation paper
                 epochs = 10, # same as used in the animal imitation paper
                 mb_size = 64,
                 learn_rate = 3e-4,  # 3e-4 is the value used in PPO paper.
                 seed = 123,
                 save_logs = False,
                 kl_target_diver = 0.,
                 downscale = 1.,
                 **kwargs
                 ):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.learn_rate = learn_rate
        self.seed = seed
        self.save_logs = save_logs
        self.clip_coef = clip_coef
        self.epochs = epochs
        self.mb_size = mb_size
        self.demo_paths = demo_paths
        self.running_score = None
        self.desired_kl = kl_target_diver
        self.downscale = downscale
        self.iter_count = 0.0
        if save_logs: self.logger = DataLog()

        self.optimizer = torch.optim.Adam(self.policy.trainable_params, lr=learn_rate) 
        # how about we tune the lr separately for NN and std?
        # TODO: try adaptive learning rate: to learn from rl_pytorch

    def PPO_surrogate(self, observations, actions, advantages):
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
        LR_clip = torch.clamp(LR, min=1-self.clip_coef, max=1+self.clip_coef) # autodiff can be used to compute the gradient
        #LR_clip serves a similar function to the KL-divergence constraint in TRPO: policy changes can not be too large.
        ppo_surr = torch.mean(torch.min(LR*adv_var,LR_clip*adv_var))
        return ppo_surr

    # ----------------------------------------------------------
    def train_from_paths(self, paths):
        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths]) # traj * time_steps
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)  # z-score normalized the advantages for training.
        advantages = self.downscale * advantages
        # Z-score normalization refers to the process of normalizing every value in a dataset such that 
        # the mean of all of the values is 0 and the standard deviation is 1.
        # The advantage is that the any outliers would not have influential effects
        # NOTE : advantage should be zero mean in expectation
        # normalized step size invariant to advantage scaling,
        # but scaling can help with least squares
        if self.demo_paths is not None:  # regulazier type == 'dapg' -> to use augmented prior data
            demo_obs = self.demo_paths['observations']
            demo_act = self.demo_paths['actions']
            demo_adv = self.demo_paths['lam_0'] * (self.demo_paths['lam_1'] ** self.iter_count) * np.ones((demo_obs.shape[0]))  # mismatch from the paper
            # TODO: Based on Watch and Match paper, try an adaptive weighting function
            self.iter_count += 1
            # concatenate all
            all_obs = np.concatenate([observations, demo_obs])
            all_act = np.concatenate([actions, demo_act])
            all_adv = 1e-2*np.concatenate([advantages/(np.std(advantages) + 1e-8), demo_adv]) # why the adv is timed with 1e-2?
        else:
            all_obs = observations
            all_act = actions
            all_adv = advantages
        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
                             0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        params_before_opt = self.policy.get_param_values()

        ts = timer.time()
        num_samples = all_obs.shape[0]  # observations.shape[0] -> horizon * num_traj
        batch_iter = int(num_samples / self.mb_size)
        shuffle_index = np.arange(0, batch_iter * self.mb_size)
        for ep in range(self.epochs):
            np.random.seed(self.seed + ep)  # fixed the initial seed value for reproducible results
            np.random.shuffle(shuffle_index)
            for mb in range(batch_iter):
                rand_idx = shuffle_index[mb * self.mb_size: (mb + 1)* self.mb_size]
                obs = all_obs[rand_idx]
                act = all_act[rand_idx]
                if self.desired_kl != 0.:  # adjust the learning rate 
                    kl_dist = self.kl_old_new(obs, act).data.numpy().ravel()[0]  # the old policy has not been updated
                    if kl_dist > self.desired_kl * 2.0: # decrease the learning rate (min to be 1e-5), as the policy change is too large
                        self.learn_rate = max(1e-5, self.learn_rate / 1.5)
                    elif kl_dist < self.desired_kl / 2.0 and kl_dist > 0.0: # increase the learning rate (max to be 1e-3), as the policy change is too small
                        self.learn_rate = min(1e-4, self.learn_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learn_rate
                adv = all_adv[rand_idx]
                self.optimizer.zero_grad()
                loss = - self.PPO_surrogate(obs, act, adv)  # maximize the loss, so minimize the inverse
                loss.backward()
                self.optimizer.step()
                kl_dist = self.kl_old_new(obs, act).data.numpy().ravel()[0]
        params_after_opt = self.policy.get_param_values()
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(params_after_opt, set_new=True, set_old=True)
        t_opt = timer.time() - ts

        # Log information
        if self.save_logs:
            self.logger.log_kv('t_opt', t_opt)
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
