import torch
from torch import nn

import lightning as L

from torchmetrics import MeanSquaredError


class LitCrypto(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        loss_module: nn.Module,
        # metric_module: nn.Module,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net", "loss_module", "metric_module"])

        self.net = net

        self.loss_module = loss_module
        self.metric_module = MeanSquaredError()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        loss = self.loss_module(pred, y)
        self.log("train/loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        loss = self.loss_module(pred, y)
        self.log("val/loss", loss.item(), on_epoch=True, on_step=False)

        mse = self.metric_module(pred, y)
        self.log("val/mse", mse, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        mse = self.metric_module(pred, y)
        self.log("test/mse", mse)
