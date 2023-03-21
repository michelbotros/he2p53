import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import numpy as np
from cbr import CBR5


def accumulate_outputs(outputs):
    y_true = np.hstack([output['y_true'] for output in outputs]).flatten()
    y_pred = np.hstack([output['y_pred'] for output in outputs]).flatten()
    return y_true, y_pred


class StainRegressionNet(pl.LightningModule):

    def __init__(self, lr=1e-4, wd=1e-5):
        super().__init__()
        self.model = CBR5()
        self.loss = nn.MSELoss()
        self.lr = lr
        self.wd = wd

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # forward and predict
        x, _, y = train_batch
        y_pred = self.model(x).squeeze()
        loss = self.loss(y_pred, y)

        # log loss
        self.log_dict({'train loss': loss.item()})
        return loss

    def validation_epoch_end(self, outputs):
        y_true, y_prob = accumulate_outputs(outputs)
        # compute PCC
        # scatter plot correlation

    def test_epoch_end(self, outputs):
        y_true, y_pred = accumulate_outputs(outputs)

    def validation_step(self, val_batch, batch_idx):
        # forward and predict
        x, _, y = val_batch
        y_pred = self.model(x).squeeze()
        loss = self.loss(y_pred, y)

        # to numpy
        y_true = y.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        # log loss
        self.log_dict({'val loss': loss.item()})
        return {'y_true': y_true, 'y_pred': y_pred}

    def test_step(self, test_batch, batch_idx):
        # forward and predict
        x, _, y = test_batch
        y_pred = self.model(x).squeeze()
        loss = self.loss(y_pred, y)

        # to numpy
        y_true = y.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        # log loss
        self.log_dict({'test loss': loss.item()})
        return {'y_true': y_true, 'y_pred': y_pred}