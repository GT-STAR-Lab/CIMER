#!/usr/bin/env python3

"""Inverse dynamics models: f(s_t, s_t1) = a_t"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvDynMLP(nn.Module):
    """MLP inverse dynamics model."""

    def __init__(self, env_spec, mlp_w=64, seed=None):
        super(InvDynMLP, self).__init__()
        # Set the seed (DAPG style)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        # Compute obs dim in a hacky way
        self.obs_dim = env_spec.observation_dim
        self.act_dim = env_spec.action_dim
        # Build the model
        self.fc0 = nn.Linear(self.obs_dim * 2, mlp_w)
        self.fc1 = nn.Linear(mlp_w, mlp_w)
        self.fc2 = nn.Linear(mlp_w, self.act_dim)
        # Make params of the last layer small (following DAPG)
        self.fc2.weight.data *= 1e-2
        self.fc2.bias.data *= 1e-2

    def forward(self, x):
        x = torch.tanh(self.fc0(x))
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
