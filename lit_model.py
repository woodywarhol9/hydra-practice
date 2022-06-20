import torchvision
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from hydra.utils import instantiate


class Resnet(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # model
        self.model = instantiate(cfg.model.model)
        self.model.conv1 = instantiate(cfg.model.conv)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.softmax = nn.Softmax(dim=-1)
        # optimizer, scheduler
        self.optimizer = cfg.optimizer
        self.scheduler = cfg.scheduler
        self.monitor = cfg.monitor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return self.softmax(out)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss

    def _evaluate(self, batch: torch.Tensor, stage=None) -> None:
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=-1)
        acc = self.accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_acc", acc, on_step=False,
                     on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self._evaluate(batch, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self._evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer, self.parameters())
        scheduler = instantiate(self.scheduler, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self.monitor}
