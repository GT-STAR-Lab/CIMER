import numpy as np
import torch
import torch.nn as nn


class RNN_model(nn.Module):
    def __init__(self, obs_dim, act_dim, 
                 rnn_hidden_size=64,
                 n_layers=2,
                 fc_hidden_sizes = (64,64),
                 nonlinearity='tanh',   # either 'tanh' or 'relu'
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None):
        super(RNN_model, self).__init__()

        self.rnn = nn.RNN(obs_dim, rnn_hidden_size, n_layers, nonlinearity=nonlinearity, batch_first=True)
        self.hx = torch.zeros(n_layers, 1, act_dim) #here 1 is the batch size since we are training on one demonstration at a time
        assert type(fc_hidden_sizes) == tuple
        self.layer_sizes = (rnn_hidden_size, ) + fc_hidden_sizes + (act_dim, )
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                         for i in range(len(self.layer_sizes) -1)])

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        #assert type(hidden_sizes) == tuple
        #self.layer_sizes = (obs_dim, ) + hidden_sizes + (act_dim, )
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)

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
        if x.is_cuda:
            out = x.to('cpu')
        else:
            out = x
        #print("Input type: ", type(out))
        #print("Input len: ", len(x))
        #print("rnn_model.py forward input shape: ", x.shape)
        out, hn = self.rnn(out)
        self.hx = hn
        assert out.ndim == 3 # [B, T, D]
        #print(out.shape)
        out = out[:,-1]
        #print(out.shape)
        for i in range(len(self.fc_layers)-1):
            #print(i)
            out = self.fc_layers[i](out)
            #print(out.shape)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        #print(out.shape)
        #out = (out - self.in_shift)/(self.in_scale + 1e-8)
        #out = out * self.out_scale + self.out_shift
        return out, self.hx
