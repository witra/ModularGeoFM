from typing import List, Literal

import numpy as np
import torch
import lightning as L
import rasterio as rio
import yaml
from box import Box
from pyproj import Transformer
from rasterio.transform import xy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from pathlib import Path

class S2Dataset(Dataset):
    def __init__(
            self,
            data_paths : List,
            metadata : Box

    ):
        self.metadata = metadata
        mean = list(metadata.bands.mean.values()) # take a not to calculate the mean and std later.
        std = list(metadata.bands.std.values())
        self.gsd = self.metadata.scale
        # self.waves = list(self.metadata.bands.wavelength.values())
        self.filenames = data_paths
        self.transform = self.create_transforms(mean, std)
        self.bands = self.metadata.bands.index

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        src = rio.open(self.filenames[idx])
        pixels = src.read()[self.bands, :, :]
        pixels =  pixels.astype(np.float16)

        width = src.width
        height = src.height

        # Get center pixel coordinates
        center_row = height // 2
        center_col = width // 2

        # Get projected coordinates (e.g., in UTM)
        x, y = xy(src.transform, center_row, center_col)

        # Transform to lon/lat if needed
        if src.crs.to_string() != "EPSG:4326":
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x, y)
        else:
            assert False, "EPSG is not 4326"

        sample = {
            "pixels" : self.transform(torch.from_numpy(pixels)),
            # "gsd" : torch.tensor(self.gsd),
            "time" : torch.zeros(4),
            "latlon" : torch.zeros(4) #torch.tensor([lat, lon]), # later consider the take the real one
            # "waves": torch.tensor(self.waves)
        }
        return sample

    def create_transforms(self, mean, std):
        """
        Create normalization transforms.

        Args:
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.

        Returns:
            torchvision.transforms.Compose: A composition of transforms.
        """
        return v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.Normalize(mean=mean, std=std),
            ],
        )


class ClayDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            size: int = 256,
            metadata_path:str = "configs/metadata.yaml",
            batch_size: int = 32,
            num_workers: int = 4,
            split_ratio = 0.8
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.size = size
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio

    def setup(self, stage: Literal["fit", "predict"]):
        if stage == "fit":
            paths = [filepath for filepath in self.data_dir.glob('train/*.tif')]
            train_paths, eval_paths = train_test_split(
                paths,
                test_size=(1 - self.split_ratio),
                shuffle=True
            )
            self.train_ds = S2Dataset(train_paths, metadata = self.metadata)
            self.eval_ds = S2Dataset(eval_paths, metadata = self.metadata)

        elif stage == "predict":
            paths = [filepath for filepath in self.data_dir.glob('predict/*.tif')]
            self.test_ds = S2Dataset(paths, metadata = self.metadata)

    def train_dataloader(self):
        return DataLoader(
            dataset= self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
    def val_dataloader(self):
        return DataLoader(
            dataset= self.eval_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
    def predict_dataloader(self):
        return DataLoader(
            dataset= self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,

        )
