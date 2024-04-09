import numpy as np
from mjrl.utils.rnn_model import RNN_model
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_lightning as pl

class LTC_sequence(nn.Module):
    def __init__(self, rnn_cell):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        # nn.Module that unfolds a RNN cell into a sequence
        super(LTC_sequence, self).__init__()
        self.rnn_cell = rnn_cell
        fc_i_hidden_sizes = (500,500)
        assert type(fc_i_hidden_sizes) == tuple
        self.i_layer_sizes = (39, ) + fc_i_hidden_sizes + (20, )
        self.fc_i_layers = nn.ModuleList([nn.Linear(self.i_layer_sizes[i], self.i_layer_sizes[i+1]) \
                         for i in range(len(self.i_layer_sizes) -1)])
        fc_o_hidden_sizes = (500,500)
        assert type(fc_o_hidden_sizes) == tuple
        self.o_layer_sizes = (4, ) + fc_o_hidden_sizes + (30, )
        self.fc_o_layers = nn.ModuleList([nn.Linear(self.o_layer_sizes[i], self.o_layer_sizes[i+1]) \
                         for i in range(len(self.o_layer_sizes) -1)])
        self.nonlinearity = torch.tanh

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_state = torch.zeros(
            (batch_size, self.rnn_cell.state_size), device=device
        )
        out = x.float()
        for i in range(len(self.fc_i_layers)-1):
            out = self.fc_i_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_i_layers[-1](out)
        outputs = []
        for t in range(seq_len):
            inputs = out[:, t] #x[:, t]
            new_output, hidden_state = self.rnn_cell.forward(inputs, hidden_state)
            outputs.append(new_output)
        outputs = torch.stack(outputs, dim=1)  # return entire sequence
        out = outputs.float() #[:,-1]
        for i in range(len(self.fc_o_layers)-1):
            out = self.fc_o_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_o_layers[-1](out)
        return out.float()

    # Main functions
    # ============================================

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
        mean = model(obs_var)
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
