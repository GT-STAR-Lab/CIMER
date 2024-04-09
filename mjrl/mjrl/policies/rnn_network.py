import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNNetwork(nn.Module):
    def __init__(self, rnn_cell, env,
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None):
        """
        :param rnn_cell: RNN (NCP) module
        """
        # nn.Module that unfolds a RNN cell into a sequence
        super(RNNNetwork, self).__init__()
        self.rnn_cell = rnn_cell
        self.env = env
        self.obs_dim = env.spec.observation_dim
        self.act_dim = env.spec.action_dim
        # follow by a fc_layer
        self.fc_input_layer = nn.Linear(in_features=self.obs_dim, out_features=rnn_cell.input_size)  # map observation_dim to RNN_input dim
        self.fc_output_layer = nn.Linear(in_features=rnn_cell.hidden_size, out_features=self.act_dim)
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(in_shift=in_shift,
                           in_scale=in_scale,
                           out_shift=out_shift,
                           out_scale=out_scale
                          )
        self.in_shift  = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale  = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        # x.shape = (1, 200, 39)  (batch_size, seq_len, obs_dim)
        # print(batch_size, seq_len)
        # hidden states were set to be zeros
        hidden_state = (  # h_{0} and c_{0}
            torch.zeros((self.rnn_cell.num_layers, batch_size, self.rnn_cell.hidden_size), device=device),
            torch.zeros((self.rnn_cell.num_layers, batch_size, self.rnn_cell.hidden_size), device=device)
        )
        # print(len(hidden_state))
        x = (x - self.in_shift)/(self.in_scale + 1e-8)
        x = self.fc_input_layer(x)  # (batch_size, seq_len, self.rnn_cell.input_size)
        outputs, hidden_state = self.rnn_cell(x, hidden_state)   # output -> hidden_state at each time step (the top layer)
        # hidden_state = (h_{T}(hidden_state[0]) and c_{T}(hidden_state[1])) at last time step, if there are multiple layers, output are the h_{t} for the top layers
        # output: hidden states h_{t} at each time step, if there are multiple layers, output are the h_{t} for the top layers
        outputs = self.fc_output_layer(outputs)  #(batch_size, seq_len, act_dim)                           
        outputs = outputs * self.out_scale + self.out_shift
        # for t in range(seq_len):
        #     inputs = x[:, t]
        #     inputs = (inputs - self.in_shift)/(self.in_scale + 1e-8)
        #     # print(inputs.shape, hidden_state.shape)
        #     # input()
        #     hidden_state, cell_state = self.rnn_cell(inputs, (hidden_state, cell_state))
        #     new_output = hidden_state * self.out_scale + self.out_shift
            
        #     outputs.append(new_output)
        # outputs = torch.stack(outputs, dim=1)  # return entire sequence
        # print(outputs.shape)
        # input()
        return outputs

    def predict(self, observation, hidden_state):
        observation = (observation - self.in_shift)/(self.in_scale + 1e-8)
        observation = self.fc_input_layer(observation)
        output, hidden_state = self.rnn_cell(observation.view(1, 1, -1), hidden_state)
        output = self.fc_output_layer(output)
        output = output * self.out_scale + self.out_shift
        return output, hidden_state