from torch.autograd import Variable
import kerasncp as kncp
from kerasncp.torch import LTCCell
from mjrl.utils.gym_env import GymEnv
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import mj_envs

# LightningModule for training a RNNSequence module
class SequenceLearner(pl.LightningModule):
    def __init__(self, model, env=None, demo_paths=None, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr
        self.env = env
        self.alpha = 0.0
        # self.save_hyperparameters(ignore=["model",])
        
        self.train_init_data = np.array([path["init_state_dict"] for path in demo_paths[:50]])
        self.val_init_data = np.array([path["init_state_dict"] for path in demo_paths[50:60]])
        self.test_init_data = np.array([path["init_state_dict"] for path in demo_paths[60:100]])

    def training_step(self, batch, batch_idx):
        # x, y = batch    UserWarning: WARN: Box bound precision lowered by casting to float32, gym/gym/spaces/box.py:78
        x = batch[0].double()
        y = batch[1].double()
        y_hat, x_hat = self.model.forward(x, self.train_init_data[batch_idx])
        y_hat = y_hat.view_as(y)
        x_hat = x_hat.view_as(x)
        loss_y = nn.MSELoss()(y_hat, y)
        loss_x = nn.MSELoss()(x_hat, x)
        loss = self.alpha * loss_x + (1 - self.alpha) * loss_y

        self.log("train_loss", loss, prog_bar=True)
        self.logger.experiment.add_scalar("Train Loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # x, y = batch    UserWarning: WARN: Box bound precision lowered by casting to float32, gym/gym/spaces/box.py:78
        x = batch[0].double()
        y = batch[1].double()
        y_hat, x_hat = self.model.forward(x, self.train_init_data[batch_idx])
        y_hat = y_hat.view_as(y)
        x_hat = x_hat.view_as(x)
        loss_y = nn.MSELoss()(y_hat, y)
        loss_x = nn.MSELoss()(x_hat, x)
        loss = self.alpha * loss_x + (1 - self.alpha) * loss_y

        self.log("val_loss", loss, prog_bar=True)
        self.logger.experiment.add_scalar("Validation Loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch[0].double()
        y = batch[1].double()
        y_hat, x_hat = self.model.forward(x, self.train_init_data[batch_idx])
        y_hat = y_hat.view_as(y)
        x_hat = x_hat.view_as(x)
        loss_y = nn.MSELoss()(y_hat, y)
        loss_x = nn.MSELoss()(x_hat, x)
        loss = self.alpha * loss_x + (1 - self.alpha) * loss_y

        self.log("val_loss", loss, prog_bar=True)
        self.logger.experiment.add_scalar("Validation Loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        optimizer_idx,
        closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        optimizer.optimizer.step(closure=closure)
        # Apply weight constraints
        self.model.rnn_cell.apply_weight_constraints()
