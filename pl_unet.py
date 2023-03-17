import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
import torch
from unet import UNet
import torch.nn.functional as F


class PytorchLightningUNet(pl.LightningModule):
    """"
    """
    def __init__(self, init_filters):
        super(PytorchLightningUNet, self).__init__()
        self.net = UNet(n_channels=3, n_classes=3, init_filters=init_filters)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = F.cross_entropy(y_hat, y)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
