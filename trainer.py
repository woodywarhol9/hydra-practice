import hydra
from hydra.utils import get_original_cwd

from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pl_bolts.datamodules import MNISTDataModule

from lit_model import Resnet


@hydra.main(config_path='configs', config_name="config")
def main_hydra(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)
    data = MNISTDataModule(data_dir=get_original_cwd(), val_split=cfg.data.val_split, num_workers=cfg.data.num_workers, batch_size=cfg.data.batch_size,
                           normalize=True, shuffle=True, pin_memory=True)
    lit_model = Resnet(cfg)
    # logger
    logger = TensorBoardLogger(**cfg.logger)
    # callbacks
    early_stopping = EarlyStopping(**cfg.early_stopping)
    lr_monitor = LearningRateMonitor(**cfg.lr_monitor)
    check_point = ModelCheckpoint(**cfg.checkpoint)
    # Trainer
    trainer = Trainer(**cfg.trainer, logger=logger,
                      callbacks=[early_stopping, lr_monitor, check_point])
    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main_hydra()
