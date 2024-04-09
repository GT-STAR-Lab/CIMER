import numpy as np
import torch
import torch.nn as nn


class FCNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64,64),
                 nonlinearity='tanh',   # either 'tanh' or 'relu'
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None):
        super(FCNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (obs_dim, ) + hidden_sizes + (act_dim, ) # obs_dim: 39, act_dim: 30 (for the relocation task)
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)  # they all None -> shift and scales are all zeros or ones.

        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                         for i in range(len(self.layer_sizes) -1)])  # stack severeal layers together.
        # The weights are initialzied in default by:
#        stdv = 1. / math.sqrt(self.weight.size(1))
#        self.weight.data.uniform_(-stdv, stdv)
#        if self.bias is not None:
#           self.bias.data.uniform_(-stdv, stdv)
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh   

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
        # TODO(Aravind): Remove clamping to CPU
        # This is a temp change that should be fixed shortly
        # forward only in cpu
        if x.is_cuda:
            out = x.to('cpu')
        else:
            out = x
        out = (out - self.in_shift)/(self.in_scale + 1e-8)  # observation data removes the effect of shifting and scaling.
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)  # the last layer does not have an activation function, this may lift the NN outputs
        out = out * self.out_scale + self.out_shift
        return out
