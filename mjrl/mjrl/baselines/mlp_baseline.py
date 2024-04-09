import numpy as np
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
from mjrl.utils.optimize_model import fit_data

import pickle

# any insights on the baseline model selection 
# baseline model is referred as Critic
class MLPBaseline:
    def __init__(self, inp_dim=None, inp='obs', learn_rate=1e-3, reg_coef=0.0,
                 batch_size=64, epochs=1, use_gpu=False, hidden_sizes=(128, 128)):
        self.n = inp_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg_coef = reg_coef
        self.use_gpu = use_gpu
        self.inp = inp
        self.hidden_sizes = hidden_sizes

        self.model = nn.Sequential()
        layer_sizes = (self.n + 4, ) + hidden_sizes + (1, )  # hidden size = (128, 128), for task relo: 43, 128, 128, 1
        print("baseline layer size: ", layer_sizes)
        for i in range(len(layer_sizes) - 1):
            layer_id = 'fc_' + str(i)
            relu_id = 'relu_' + str(i)
            self.model.add_module(layer_id, nn.Linear(layer_sizes[i], layer_sizes[i+1]))  # the last layer has the output as 1-dimension
            if i != len(layer_sizes) - 2:
                self.model.add_module(relu_id, nn.ReLU())
        if self.use_gpu:
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate, weight_decay=reg_coef) 
        self.loss_function = torch.nn.MSELoss()

    def _features(self, paths):
        if self.inp == 'env_features':
            o = np.concatenate([path["env_infos"]["env_features"][0] for path in paths])
        else:  # value function input
            o = np.concatenate([path["observations"] for path in paths])  # concatenate the observation data
        o = np.clip(o, -10, 10)/10.0  # normalize
        if o.ndim > 2:
            o = o.reshape(o.shape[0], -1)
        N, n = o.shape  # o -> observation data
        num_feat = int( n + 4 )            # linear + time till pow 4
        feat_mat =  np.ones((N, num_feat)) # memory allocation

        # linear features
        feat_mat[:,:n] = o

        k = 0  # start from this row
        for i in range(len(paths)):
            l = len(paths[i]["rewards"])
            al = np.arange(l)/1000.0
            for j in range(4):
                feat_mat[k:k+l, -4+j] = al**(j+1)  # -4 + j -> the last 4 elements along the second axis
            k += l
        return feat_mat

    def fit(self, paths, return_errors=False):
        featmat = self._features(paths)  # (traj_horizon, obs_dim + 4), for example, for relocation, (200, 39+4)
        returns = np.concatenate([path["returns"] for path in paths]).reshape(-1, 1)  # sum discounted rewards (target values)
        featmat = featmat.astype('float32') 
        returns = returns.astype('float32')
        num_samples = returns.shape[0]

        # Make variables with the above data
        if self.use_gpu:
            featmat_var = Variable(torch.from_numpy(featmat).cuda(), requires_grad=False)
            returns_var = Variable(torch.from_numpy(returns).cuda(), requires_grad=False)
        else:
            featmat_var = Variable(torch.from_numpy(featmat), requires_grad=False)
            returns_var = Variable(torch.from_numpy(returns), requires_grad=False)

        if return_errors:
            if self.use_gpu:
                predictions = self.model(featmat_var).cpu().data.numpy().ravel()  # mapping from the observation to the values (baseline model)
                # expect to be the same as the return values (true V values at current step)
            else:
                predictions = self.model(featmat_var).data.numpy().ravel()
            errors = returns.ravel() - predictions  # minimize the error
            error_before = np.sum(errors**2)/(np.sum(returns**2) + 1e-8)
        # featmat_var -> x (raw observation data)
        # returns_var -> y (labeled data) sum of discounted rewards
        # self.epochs -> number of iterations for this inner-loop optimization
        # fit_data uses a simple nonlinear regression to optimize the baseline model for the value function approximation, not the TR method in Pieter Abbeel's paper
        epoch_losses = fit_data(self.model, featmat_var, returns_var, self.optimizer,
                                self.loss_function, self.batch_size, self.epochs)  # trust region method to fit the model? No, the basic version

        # baseline model has been updated and then compared the errors
        if return_errors:
            if self.use_gpu:
                predictions = self.model(featmat_var).cpu().data.numpy().ravel()
            else:
                predictions = self.model(featmat_var).data.numpy().ravel()
            errors = returns.ravel() - predictions
            error_after = np.sum(errors**2)/(np.sum(returns**2) + 1e-8)
            return error_before, error_after

    def predict(self, path):
        featmat = self._features([path]).astype('float32')  # (traj_horizon, obs_dim + 4), for example, for relocation, (200, 39+4)
        if self.use_gpu:
            feat_var = Variable(torch.from_numpy(featmat).float().cuda(), requires_grad=False)
            prediction = self.model(feat_var).cpu().data.numpy().ravel() # baseline model layer size:  (43, 128, 128, 1), final output: (200, 1)
        else:
            feat_var = Variable(torch.from_numpy(featmat).float(), requires_grad=False)
            prediction = self.model(feat_var).data.numpy().ravel()
        return prediction
