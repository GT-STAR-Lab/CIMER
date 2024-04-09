from mjrl.policies.rnn_network import RNNNetwork
import numpy as np
import torch
from torch.autograd import Variable
# import kerasncp as kncp
# from kerasncp.torch import LTCCell

class RNN:
    def __init__(self, env,
                 input_sizes=64,
                 hidden_state = 64,
                 LSTM_layer = 1,
                 min_log_std=-3,
                 init_log_std=0,
                 seed=None):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        self.n = env.spec.observation_dim  # number of states
        self.m = env.spec.action_dim  # number of actions
        self.min_log_std = min_log_std

        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network
        # ------------------------
        
        # ltc_cells = LTCCell(ncp_wiring, self.n)
        #  If batch_first is True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). 
        rnn_cell = torch.nn.LSTM(input_sizes, hidden_state, batch_first=True, num_layers = LSTM_layer)  #
        # print(ltc_cells.state_size, ltc_cells.synapse_count, len(ltc_cells._params))
        # self.model = NCPNetwork(ltc_cells, env)
        self.model = RNNNetwork(rnn_cell, env)
        # make weights small
        for param in list(self.model.parameters())[-2:]:  # only last layer
           param.data = 1e-2 * param.data
        self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)
        self.trainable_params = list(self.model.parameters()) + [self.log_std]

        # Old Policy network
        # ------------------------
        # self.old_model = NCPNetwork(ltc_cells, env)
        self.old_model = RNNNetwork(rnn_cell, env)
        self.old_log_std = Variable(torch.ones(self.m) * init_log_std)
        self.old_params = list(self.old_model.parameters()) + [self.old_log_std]
        for idx, param in enumerate(self.old_params):
            param.data = self.trainable_params[idx].data.clone()

        # Easy access variables
        # -------------------------
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)
        self.hidden_state_var = Variable(torch.randn(
            (LSTM_layer, 1, self.model.rnn_cell.hidden_size)), requires_grad=False)
        self.cell_state_var = Variable(torch.randn(
            (LSTM_layer, 1, self.model.rnn_cell.hidden_size)), requires_grad=False)
        print(len(self.get_param_values()))

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
            # clip std at minimum value
            self.trainable_params[-1].data = \
                torch.clamp(self.trainable_params[-1], self.min_log_std).data
            # update log_std_val for sampling
            self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
            # self.model.rnn_cell.apply_weight_constraints()
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
            # self.old_model.rnn_cell.apply_weight_constraints()

    # Main functions
    # ============================================
    # this one is not useful for RNN model
    def get_action_without_hidden(self, observation):
        o = np.float32(observation.reshape(1, 1, -1))
        self.obs_var.data = torch.from_numpy(o)
        mean = self.model(self.obs_var).data.numpy().ravel()
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)
        action = mean + noise
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]
    
    def get_action(self, observation, hidden_state):
        o = np.float32(observation.reshape(1, 1, -1))
        self.obs_var.data = torch.from_numpy(o)
        # print(hidden_state[0].shape)
        self.hidden_state_var.data = torch.from_numpy(np.float32(hidden_state[0]))
        self.cell_state_var.data = torch.from_numpy(np.float32(hidden_state[1]))
        # print(self.hidden_state_var.shape)
        mean, hidden_state = self.model.predict(self.obs_var, (self.hidden_state_var, self.cell_state_var))
        mean = mean.data.numpy().ravel()
        new_hidden_state = hidden_state[0].data.numpy()
        new_cell_state = hidden_state[1].data.numpy()
        hidden_state = (new_hidden_state, new_cell_state)  # these are needed for the next time step.
        #Since the input is the obs at each time step, we have to manually pass the  hidden state and the cell state
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)
        action = mean + noise  # action is added with noise for the exploration purpose
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean, 'hidden_state': hidden_state}]

    def mean_LL(self, observations, actions, model=None, log_std=None):
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
        mean = model(obs_var)  # hidden state values are provided in model.forward() function
        mean = mean.reshape([mean.shape[0] * mean.shape[1], mean.shape[-1]])
        obs_var = obs_var.reshape([obs_var.shape[0] * obs_var.shape[1], obs_var.shape[-1]])
        act_var = act_var.reshape([act_var.shape[0] * act_var.shape[1], act_var.shape[-1]])
        zs = (act_var - mean) / torch.exp(log_std)
        LL = - 0.5 * torch.sum(zs ** 2, dim=1) + \
             - torch.sum(log_std) + \
             - 0.5 * self.m * np.log(2 * np.pi)
        return mean, LL

    def log_likelihood(self, observations, actions, model=None, log_std=None):
        mean, LL = self.mean_LL(observations, actions, model, log_std)
        return LL.data.numpy()

    def old_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.old_model, self.old_log_std)
        return [LL, mean, self.old_log_std]

    def new_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std)
        return [LL, mean, self.log_std]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0]
        LL_new = new_dist_info[0]
        LR = torch.exp(LL_new - LL_old)
        return LR

    def mean_kl(self, new_dist_info, old_dist_info):
        old_log_std = old_dist_info[2]
        new_log_std = new_dist_info[2]
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_mean = old_dist_info[1]
        new_mean = new_dist_info[1]
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
        Dr = 2 * new_std ** 2 + 1e-8
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl)
