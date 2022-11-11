import glob
import os
import json

import h5py
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


HDF5_PATH = 'coco2014.h5'
CAPTIONS_DIR = 'captions'
CAPTIONS_TRAIN_EN = 'captions_train2014.json'
CAPTIONS_VAL_EN = 'captions_val2014.json'
CAPTIONS_TRAIN_JP = 'stair_captions_v1.2_train.json'
CAPTIONS_VAL_JP = 'stair_captions_v1.2_val.json'
IMAGES_TRAIN_DIR = 'train2014'
IMAGES_VAL_DIR = 'val2014'


def create_hdf5(hdf5_path):
    with h5py.File(hdf5_path, 'w') as h5:
        captions_group = h5.create_group(CAPTIONS_DIR)

        with open(CAPTIONS_DIR + '/' + CAPTIONS_VAL_EN, 'r') as val_en:
            readme_dataset = captions_group.create_dataset(
                name=CAPTIONS_VAL_EN, shape=(1,), dtype=h5py.string_dtype()
            )
            readme_dataset[0] = val_en.read()

        images_group = h5.create_group(IMAGES_VAL_DIR)

        count = 0
        for p in sorted(glob.glob(IMAGES_VAL_DIR + '/*.jpg')):
            image = np.array(Image.open(p)).astype(np.uint8)
            image_dataset = images_group.create_dataset(
                name=os.path.basename(p), data=image, compression='gzip'
            )
            count += 1
            if count%1000 == 0:
                print(count)


class HDF5dataset(Dataset):

    def __init__(self, sample_size):
        self.h5 = h5py.File(HDF5_PATH, 'r')

        captions_str = self.h5[CAPTIONS_DIR + '/' + CAPTIONS_VAL_EN][0]
        captions_dict = json.loads(captions_str)['annotations']
        df = pd.DataFrame(captions_dict)[['image_id', 'caption']]
        df['caption_concat'] = df.groupby(['image_id'])['caption']\
            .transform(lambda x: ' '.join(x))
        df = df[['image_id','caption_concat']].drop_duplicates().reset_index()
        df['image_file'] = df['image_id'].apply('COCO_val2014_{:012}.jpg'.format)
        self.df = df[['image_file', 'caption_concat']]

        if sample_size < len(df):
            self.df = self.df[:sample_size]

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.at[idx, "image_file"]
        numpy_image = self.h5[IMAGES_VAL_DIR + '/' + file_name][()]   # or [:, :, :]
        pil_image = Image.fromarray(numpy_image).resize((224, 224))
        numpy_image = np.array(pil_image).transpose(2, 0, 1)

        caption = self.df.at[idx, "caption_concat"]

        return numpy_image, caption


class DataModule(pl.LightningDataModule):

    def __init__(self, sample_size, batch_size):
        super().__init__()
        self.sample_size = sample_size
        self.batch_size = batch_size

    def setup(self, stage):
        self.dataset = HDF5dataset(self.sample_size)

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=0)   # hdf5 allows 0 only!

    def val_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=0)   # hdf5 allows 0 only!


if __name__ == '__main__':
    #create_hdf5(HDF5_PATH)
    datamodule = DataModule(100, 16)
    datamodule.setup('fit')
    dataloader = datamodule.train_dataloader()
    sample = next(iter(dataloader))
    print(sample[0].shape, len(sample[1]))
