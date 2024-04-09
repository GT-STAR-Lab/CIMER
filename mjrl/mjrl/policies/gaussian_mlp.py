import numpy as np
from mjrl.utils.fc_network import FCNetwork
import torch
from torch.autograd import Variable


class MLP:
    def __init__(self, observation_dim,
                 action_dim,
                 policy_output='jp', # jp->joint position; djp->delta joint position (residual policy)
                 hidden_sizes=(64,64),
                 min_log_std=-3,  # minimum of log std, pay attention that std value should be nonnegative.
                 init_log_std=0,  # np.exp(log_std) = 1, if init_log_std = 0
                 freeze_base=False,
                 include_Rots=False,
                 seed=None):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below, so there always exist some noise for exploration.
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        self.n = observation_dim  # number of states
        self.m = action_dim  # number of actions
        self.min_log_std = min_log_std
        self.policy_output = policy_output
        self.freeze_base = freeze_base
        self.include_Rots = include_Rots
        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        # Policy network
        # ------------------------
        self.model = FCNetwork(self.n, self.m, hidden_sizes, nonlinearity = 'relu')
    
        # make weights small
        for param in list(self.model.parameters())[-2:]:  # only last layer (weights and bias)
           param.data = 1e-2 * param.data
        self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)  # another trainable parameter -> action noise
        # self.log_std -> logged standard deviation
        # for each joint i, in 1,..,m, the standard deviation is different. But we consider the covariance matrix to be a diagonal matrix.
        self.trainable_params = list(self.model.parameters()) + [self.log_std]  # only parameters of self.model are trainable.
        # Old Policy network 
        # ------------------------
        self.old_model = FCNetwork(self.n, self.m, hidden_sizes, nonlinearity = 'relu')
        self.old_log_std = Variable(torch.ones(self.m) * init_log_std)
        self.old_params = list(self.old_model.parameters()) + [self.old_log_std]
        for idx, param in enumerate(self.old_params):
            param.data = self.trainable_params[idx].data.clone()  # share the same weights as self.model
        # Easy access variables
        # -------------------------
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)

    # Utility functions
    # ============================================
    def get_param_values(self):
        params = np.concatenate([p.contiguous().view(-1).data.numpy()
                                 for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params, set_new=True, set_old=True):
        if set_new:
            current_idx = 0
            for idx, param in enumerate(self.trainable_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value (only apply the lower bound)
            # [-1] -> self.log_std
            self.trainable_params[-1].data = \
                torch.clamp(self.trainable_params[-1], self.min_log_std).data
            # update log_std_val for sampling (used to generate noise for the control actions)
            # log_sdt can be vary small, say
            self.log_std_val = np.float64(self.log_std.data.numpy().ravel()) 
        if set_old:
            current_idx = 0
            for idx, param in enumerate(self.old_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.old_params[-1].data = \
                torch.clamp(self.old_params[-1], self.min_log_std).data

    # Main functions
    # ============================================
    def get_action(self, observation):
        o = np.float32(observation.reshape(1, -1))
        self.obs_var.data = torch.from_numpy(o)
        mean = self.model(self.obs_var).data.numpy().ravel() # FC network when using MLP as policy (self.model = FCNetwork(self.n, self.m, hidden_sizes))
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)  # np.exp(self.log_std_val) -> exp(log(x)) is x, which is the standard deviation
        # print("self.log_std_val:", self.log_std_val)
        # print("np.exp(self.log_std_val):", np.exp(self.log_std_val))
        action = mean + noise  # the noise is very important, especially at the early stage -> exploration to the new state-action pairs
        # the noise magnitude depends on the log_std_val. It decides how much we trust our policy for the good actions. This parameter is algo trainable.
        # I think during training, this value should be smaller which means that we are confident that the current policy would result in better results.
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]

    def mean_LL(self, observations, actions, model=None, log_std=None):  # old model and new model 
        model = self.model if model is None else model
        log_std = self.log_std if log_std is None else log_std
        if type(observations) is not torch.Tensor:
            obs_var = Variable(torch.from_numpy(observations).float(), requires_grad=False)
        else:
            obs_var = observations
        if type(actions) is not torch.Tensor:
            act_var = Variable(torch.from_numpy(actions).float(), requires_grad=False)
        else:
            act_var = actions
        mean = model(obs_var)  # re-run the policy to get the actions wrt to each traj at each time step, should be very similar to the input actions? 
        #  No!, this is noise-free computation!
        # we are considering the act_var as the label data for a supervised learning, seems to be
        # this value would be timed with advantage value: if the noisy action(input:actions) leads to the good results with higher rewards, the advantage value is larger
        # which would make the policy learn to generate the similar outputs as this set of noisy actions (exploration).
        zs = (act_var - mean) / torch.exp(log_std) # np.exp(self.log_std_val) -> exp(log(x)) is x, which is the standard deviation
        # LL is the probablity of the taking the noisy action with the noise-free action being mean and log_std(a learnable vector parameter) being the logged standard deviation.
        # we assume it to be a Multivariate guassian distribution, so we have  - 0.5 * self.m (m-dimensional)
        # also torch.sum(log_std) is the determinant(covariance), as covariance matrix is a diagonal matrix, so it is the sum of diagonal elements
        LL = - 0.5 * torch.sum(zs ** 2, dim=1) + \
             - torch.sum(log_std) + \
             - 0.5 * self.m * np.log(2 * np.pi)  # LL is the log_likelihood
        return mean, LL

    def log_likelihood(self, observations, actions, model=None, log_std=None):  # i believe this function was used in the first version,
        # then they changed to likelihood_ratio function even for the batch_reinforce (vpg) training.
        # in Pieter's paper, for vpg training, they used only log_likelihood (LL).
        # In TRPO, they use the surrogate objective (likelihood_ratio), but for the previous methods, they can simply use log_likelihood
        mean, LL = self.mean_LL(observations, actions, model, log_std)  
        return LL.data.numpy()

    def old_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.old_model, self.old_log_std)  # mean -> exact policy outputs (without noise), LL -> some kind of error
        return [LL, mean, self.old_log_std] 

    def new_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std)
        return [LL, mean, self.log_std]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0] # log_likelihood (LL), using the old policy's mean and log_std
        # -> LR = tensor([1., 1., 1.,  ..., 1., 1., 1.], grad_fn=<ExpBackward0>)
        LL_new = new_dist_info[0]  # log_likelihood (LL), using the old policy's mean and log_std
        LR = torch.exp(LL_new - LL_old) # torch.exp -> log_likelihood -> likelihood
        # LR is eqn. (3) in the paper "Proximal Policy Optimization Algorithms"
        return LR

    def mean_kl(self, new_dist_info, old_dist_info):   # same as in Pieter's code
        old_log_std = old_dist_info[2]  # old_log_std
        new_log_std = new_dist_info[2]  # log_std
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_mean = old_dist_info[1]  # actions generated by old policy, these are noise-free policy outputs, considered as Guassian mean values
        new_mean = new_dist_info[1]  # actions generated by new policy
        """
        Compute the KL divergence of two multivariate Gaussian distribution (new and old policies) with
        diagonal covariance matrices (self.log_std are the diagonal elements)
        """
        # means: (N*A)
        # std: (N*A)
        # The formula to compute KL divergence (same as post online):
        # {(\mu_1 - \mu_2)^2 + \sigma_1^2} / (2\sigma_2^2) + log(\sigma_2/\sigma_1) - 0.5
        # Note that log(\sigma_2/\sigma_1) = log(\sigma_2) (new_log_std) - log(\sigma_1) (old_log_std)
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2  # numerator
        # In Pieter's codes:
        # kl = (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5 + log_std1 - log_std0 
        Dr = 2 * new_std ** 2 + 1e-8  # denominator
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std - 0.5, dim=1)
        return torch.mean(sample_kl)