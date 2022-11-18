import os

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_memlab import profile, MemReporter
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig

from src.mclip import MClipModelModule
from src.coco2014 import (
    PREPROCESSED_DATA, TRAIN_PREFIX,
    DataCreator, DataModule
)


def seed_everything(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@hydra.main(config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:
    seed_everything(cfg.run.seed)

    logger = WandbLogger(project=cfg.wandb.project,
                         name=cfg.wandb.name,
                         log_model=True)

    size = suffix = cfg.preprocess.sample_size
    h5_path = PREPROCESSED_DATA / (TRAIN_PREFIX + '_' + suffix + '.h5')
    if not h5_path.is_file():
        DataCreator(size, train=True).create_hdf5(suffix)
        DataCreator(size, train=False).create_hdf5(suffix)

    data = DataModule(cfg)
    logger.experiment.config.update({
        'train_size': len(data.train_dataset),
        'validation_size': len(data.val_dataset),
    })

    model = MClipModelModule(cfg, data.logit_scale)
    logger.watch(model)
    # reporter = MemReporter(model)
    # reporter.report()

    trainer = pl.Trainer(max_epochs=cfg.train.max_epochs,
                         accelerator=cfg.train.accelerator,
                         devices=cfg.train.devices,
                         logger=logger)
    trainer.fit(model, data)
    # reporter.report()


if __name__ == '__main__':
    main()
