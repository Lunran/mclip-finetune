import os

import numpy as np
import transformers
import torch
import pytorch_lightning as pl
from pytorch_memlab import profile, MemReporter
from pytorch_lightning.loggers import WandbLogger
import wandb
import hydra
from omegaconf import DictConfig

import mclip
import coco2014


def seed_everything(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@hydra.main(config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:
    import coco2014

    seed_everything(cfg.run.seed)

    logger = WandbLogger(project=cfg.wandb.project,
                         name=cfg.wandb.name,
                         log_model=True)

    if cfg.preprocess.h5_create:
        size = cfg.preprocess.sample_size
        coco2014.DataCreator(train=True, sample_size=size).create_hdf5()
        coco2014.DataCreator(train=False, sample_size=size).create_hdf5()
    data = coco2014.DataModule(cfg)
    logger.experiment.config.update({
        'train_size': len(data.train_dataset),
        'validation_size': len(data.val_dataset),
    })

    model = mclip.MClipModelModule(cfg, data.logit_scale)
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
