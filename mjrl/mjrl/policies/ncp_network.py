import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class NCPNetwork(nn.Module):
    def __init__(self, rnn_cell, env,
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None):
        """
        :param rnn_cell: RNN (NCP) module
        """
        # nn.Module that unfolds a RNN cell into a sequence
        super(NCPNetwork, self).__init__()
        self.rnn_cell = rnn_cell
        self.env = env
        self.obs_dim = env.spec.observation_dim
        self.act_dim = env.spec.action_dim
        self.fc_input_layer = nn.Linear(in_features=self.obs_dim, out_features=rnn_cell.sensory_size)
        self.fc_output_layer = nn.Linear(in_features=rnn_cell.output_size, out_features=self.act_dim)
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
        # print(batch_size, seq_len)
        hidden_state = torch.zeros(
            (batch_size, self.rnn_cell.state_size), device=device
        )
        outputs = []
        for t in range(seq_len):
            inputs = x[:, t]
            inputs = (inputs - self.in_shift)/(self.in_scale + 1e-8)
            inputs = self.fc_input_layer(inputs)
            # print(inputs.shape, hidden_state.shape)
            # input()
            new_output, hidden_state = self.rnn_cell.forward(inputs, hidden_state)
            new_output = self.fc_output_layer(new_output)
            new_output = new_output * self.out_scale + self.out_shift
            outputs.append(new_output)
        outputs = torch.stack(outputs, dim=1)  # return entire sequence
        # print(outputs.shape)
        # input()
        return outputs

    def predict(self, observation, hidden_state):
        observation = (observation - self.in_shift)/(self.in_scale + 1e-8)
        observation = self.fc_input_layer(observation)
        new_output, hidden_state = self.rnn_cell.forward(observation, hidden_state)
        new_output = self.fc_output_layer(new_output)
        new_output = new_output * self.out_scale + self.out_shift
        return new_output, hidden_state

    # def predict(self, x, init_data=None):
    #     device = x.device
    #     batch_size = x.size(0)
    #     seq_len = x.size(1)
    #     hidden_state = torch.zeros(
    #         (batch_size, self.rnn_cell.state_size), device=device
    #     )
    #     outputs = []
    #     inputs = x[:, 0]
    #     obs = []
    #     if init_data is not None:
    #         self.env.reset()
    #         self.env.set_env_state(init_data)
    #     for t in range(seq_len):
    #         new_output, hidden_state = self.rnn_cell.forward(inputs, hidden_state)
    #         outputs.append(new_output)
    #         # print(new_output.shape)
    #         # print(new_output)
    #         action = new_output.clone().detach().numpy()
    #         # print(action.shape)
    #         # print(new_output)
    #         # print(action)
    #         # print(new_output.numpy())
    #         if init_data is not None:
    #             obs.append(inputs)
    #             ob, _, _, _ = self.env.step(action.reshape([action.shape[-1]]))
    #             # print(ob.shape)
    #             # print(ob)
    #             inputs = torch.from_numpy(ob).view_as(inputs)
    #             # print(inputs.shape)
    #             # print(inputs)
    #             # input()
    #     outputs = torch.stack(outputs, dim=1)  # return entire sequence
    #     if init_data is not None:
    #         obs = torch.stack(obs, dim=1)
    #     return outputs
