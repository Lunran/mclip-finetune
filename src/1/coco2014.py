import os
import json
import pathlib

import h5py
import pandas as pd
from PIL import Image
import open_clip
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig


ORG_DATA = pathlib.Path('org_data')
PREPROCESSED_DATA = pathlib.Path('preprocessed_data')
TRAIN_PREFIX = 'coco2014_train'
VAL_PREFIX = 'coco2014_val'
CAPTION_DIR_NAME = 'captions'
CAPTION_FILE_NAME = 'jp.json'
IMAGE_DIR_NAME = 'images'
LOGIT_SCALE_NAME = 'logit_scale'

IMAGE_MODEL_NAME, IMAGE_PRETRAINED_NAME = 'ViT-B-16-plus-240', 'laion400m_e32'
# IMAGE_MODEL_NAME, IMAGE_PRETRAINED_NAME = 'ViT-L-14', 'laion400m_e32'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CREATE_BATCH_SIZE = 800


class DataCreator():

    def __init__(self, sample_size, train):
        self.prefix = TRAIN_PREFIX if train else VAL_PREFIX

        caption_path = ORG_DATA / self.prefix / CAPTION_DIR_NAME / CAPTION_FILE_NAME
        with caption_path.open('r') as f:
            captions_dict = json.loads(f.read())['annotations']
        df = pd.DataFrame(captions_dict)[['image_id', 'caption']]
        df['caption_concat'] = df.groupby(['image_id'])['caption']\
            .transform(lambda x: ' '.join(x))
        df = df[['image_id','caption_concat']].drop_duplicates().reset_index()
        image_prefix = 'COCO_train2014_' if train else 'COCO_val2014_'
        df['image_file'] = df['image_id'].apply((image_prefix + '{:012}.jpg').format)
        self.df = df[['image_file', 'caption_concat']]

        if sample_size != 'full' and eval(sample_size) < len(self.df):
            self.df = self.df[:eval(sample_size)]

    def create_hdf5(self, suffix):
        clip_model, _, preprocess = \
            open_clip.create_model_and_transforms(
                model_name=IMAGE_MODEL_NAME,
                pretrained=IMAGE_PRETRAINED_NAME,
                device=DEVICE
            )
        for _, param in clip_model.named_parameters():
            param.requires_grad = False
        image_model = clip_model.visual
        logit_scale = clip_model.logit_scale.exp()

        h5_path = PREPROCESSED_DATA / (self.prefix + '_' + suffix + '.h5')
        with h5py.File(h5_path, 'w') as h5:
            captions_group = h5.create_group(CAPTION_DIR_NAME)
            captions_group.create_dataset(name=CAPTION_FILE_NAME, data=self.df.values)

            images_group = h5.create_group(IMAGE_DIR_NAME)
            images_group.create_dataset(name=LOGIT_SCALE_NAME, data=logit_scale.cpu())
            pil_images, paths = [], []
            for idx in range(len(self.df)):
                image_path = ORG_DATA / self.prefix / IMAGE_DIR_NAME\
                                      / self.df.at[idx, "image_file"]
                pil_images.append(preprocess(Image.open(image_path)))
                paths.append(image_path)
                if len(pil_images) >= CREATE_BATCH_SIZE:
                    self.create_dataset(images_group, pil_images, paths, image_model)
                    pil_images, paths = [], []
                    print("processed!")
            if len(pil_images) != 0:
                self.create_dataset(images_group, pil_images, paths, image_model)

    def create_dataset(self, images_group, pil_images, paths, image_model):
        tensor_images = torch.stack(pil_images).to(DEVICE)
        with torch.no_grad():
            image_embs = image_model(tensor_images)
        for i in range(image_embs.shape[0]):
            name = os.path.basename(paths[i])
            image_emb = image_embs[i].cpu()
            images_group.create_dataset(name=name, data=image_emb)


class HDF5dataset(Dataset):

    def __init__(self, hdf5_path):
        self.h5 = h5py.File(hdf5_path, 'r')

        numpy_array = self.h5[CAPTION_DIR_NAME + '/' + CAPTION_FILE_NAME]
        self.df = pd.DataFrame(numpy_array)
        self.df.columns = ['image_file', 'caption_concat']
        for col in self.df:
            self.df[col] = self.df[col].str.decode('utf-8')

        self.logit_scale = self.h5[IMAGE_DIR_NAME + '/' + LOGIT_SCALE_NAME][()]

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.at[idx, "image_file"]
        image_emb = self.h5[IMAGE_DIR_NAME + '/' + file_name][()]
        caption = self.df.at[idx, "caption_concat"]

        return image_emb, caption


class DataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        suffix = cfg.preprocess.sample_size
        h5_path = PREPROCESSED_DATA / (TRAIN_PREFIX + '_' + suffix + '.h5')
        self.train_dataset = HDF5dataset(h5_path)
        h5_path = PREPROCESSED_DATA / (VAL_PREFIX + '_' + suffix + '.h5')
        self.val_dataset = HDF5dataset(h5_path)
        self.logit_scale = self.train_dataset.logit_scale
        self.batch_size = cfg.train.batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=0)   # hdf5 allows 0 only!

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=0)   # hdf5 allows 0 only!


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg : DictConfig) -> None:
    datamodule = DataModule(cfg)
    print(datamodule.logit_scale[()])
    dataloader = datamodule.train_dataloader()
    image_embs, caption_batch = next(iter(dataloader))
    print(image_embs.shape, len(caption_batch))


if __name__ == '__main__':
    main()
