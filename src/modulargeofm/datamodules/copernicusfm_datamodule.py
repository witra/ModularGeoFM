import glob
import math
from functools import partial
from pathlib import Path
from typing import Literal

import lightning as L
import numpy as np
import torch
import xarray as xr
import yaml
from box import Box
from kornia.augmentation import Normalize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from xbatcher import BatchGenerator


class CopernicusFMDataset(IterableDataset):
    """
    Args:

        yield:
            [x: x,
            meta_info: [lons, lats, times, areas],
            wave_list: wave_list,
            bandwidth: bandwidth,
            language_embed: language_embed,
            input_mode: spectral,
            kernel_size: kernel_size

    """
    def __init__(self,
                 samples,
                 input_dims,
                 input_overlap,
                 mode: Literal["train", "val", "test", "predict"],
                 augmentation=None,
                 verify_fn ='basic',
                 batch_size_gen=1,
                 time_dim='time',
                 filter_thres=0.05
                 ) -> None:

        self.samples = samples
        self.input_dims = input_dims
        self.input_overlap = input_overlap or {}
        self.augmentation = augmentation
        self.batch_size_gen = batch_size_gen
        self.time_dim = time_dim
        self.mode = mode
        self.filter_thres = filter_thres
        if verify_fn == 'basic':
            self.verify_fn = partial(self.basic_filter, threshold=self.filter_thres)
        elif callable(verify_fn):
            self.verify_fn = verify_fn
        else:
            raise ValueError(f"Invalid verify_fn: {verify_fn}")


    def create_batch_generator(self, subset):
        return BatchGenerator(
            subset,
            input_dims=self.input_dims,
            input_overlap=self.input_overlap)

    def basic_filter(self, patch, threshold=0.05):

        """
            Returns True if the fraction of invalid values (zeros or NaNs) in `patch` is below threshold.
        """
        total_elements = patch.numel()  # total number of elements
        # guard against empty patches
        if total_elements == 0:
            # treat an empty patch as invalid (reject)
            return False
        num_zeros = (patch == 0).sum().item()
        num_nans = torch.isnan(patch).sum().item()
        invalid_fraction = (num_zeros + num_nans) / total_elements
        return invalid_fraction < threshold

    def __iter__(self):
        # handling splitting logic for workers cleverly in the case of different num of samples and time dimension.

        # get num of dataset modalities
        keys = list(self.samples.keys())

        # get time dimension of each
        sample_time_list = []

        for key in keys:
            modality = self.samples.get(key)
            xr_dataset = xr.open_zarr(modality.get('pixels_path'))
            if self.time_dim in xr_dataset.dims:
                num_times = len(xr_dataset[self.time_dim])
            else:
                num_times = 1
            sample_time_list.extend((key, t) for t in range(num_times))
        total_jobs = len(sample_time_list)

        # get num of avail workers
        worker_info = get_worker_info()
        if worker_info is None: # single worker
            start, end = 0, total_jobs
        else:
            job_per_worker = int(math.ceil(total_jobs/worker_info.num_workers))
            start = worker_info.id * job_per_worker
            end = min(start + job_per_worker, total_jobs)


        for global_index in range(start, end):
            sample_key, time = sample_time_list[global_index]
            modality = self.samples.get(sample_key)
            xr_dataset = xr.open_zarr(modality.get('pixels_path'))
            wavelist = modality.get('wavelist', None)
            bandwidth = modality.get('bandwidth', None)
            mean = modality.get('mean', None)
            std = modality.get('std', None)
            bandnames = modality.get('bandnames', None)
            language_embed = modality.get('language_embed', None)
            input_mode = modality.get('input_mode')
            kernel_size = torch.tensor(modality['kernel_size']) if modality.get('kernel_size') is not None else None
            self.normalise = Normalize(mean=torch.tensor(mean), std=torch.tensor(std))

            if self.mode == 'predict':
                bandnames = bandnames[1:]
            if self.time_dim in xr_dataset.dims:
                subset = xr_dataset[bandnames].isel({self.time_dim: time})
            else:
                if time > 0: raise Exception('num of times must be 0')
                subset = xr_dataset.expand_dims(dim={self.time_dim:1}, axis=-1)
                subset = subset[bandnames]

            resolution = abs(subset.rio.resolution()[0]) # in meters
            area = self.input_dims['x'] * self.input_dims['y'] * resolution/(1000**2) # in km2
            area = torch.tensor(area)

            # batch generator
            batch_gen = iter(self.create_batch_generator(subset))

            # Collect and yield batches
            batch_x, batch_y, meta_infos = [], [], []

            for patch_ds in batch_gen:
                try:
                    x_c = torch.tensor(patch_ds['x'].values.mean()).squeeze()
                    y_c = torch.tensor(patch_ds['y'].values.mean()).squeeze()
                    time = torch.tensor(np.nan).squeeze()
                    meta_info = x_c, y_c, time, area,

                    patch = patch_ds.to_array(dim='pixels').values.squeeze()
                    patch_tensor = torch.tensor(patch, dtype=torch.float32)
                    
                    # verify per patch
                    status = False
                    if self.mode != 'predict':
                        patch_test = patch_tensor[1:, :, :] # take only the actual data
                        patch_test = self.normalise(patch_test).squeeze()
                        status = self.verify_fn(patch_test)  # T/F
                    if self.mode == 'predict':
                        status = True
                    if status:
                        # perform augmentation and normalization
                        if self.augmentation and self.mode != 'predict':
                            patch_tensor = self.augmentation(patch_tensor)
                        if self.mode=='predict':
                            # setup the coordinate x, y, and spatial ref
                            coords_y = patch_ds.coords['y'].values
                            coords_x = patch_ds.coords['x'].values
                            spatial_ref = patch_ds.rio.crs.to_epsg()
                            y = [coords_y, coords_x, spatial_ref]
                            pixels_x = patch_tensor
                        else:
                            y = patch_tensor[0,  :, :]
                            pixels_x = patch_tensor[1:, :, :]
                        pixels_x = self.normalise(pixels_x).clamp(min=-1.0, max=1.0).squeeze()
                        pixels_x = torch.nan_to_num(pixels_x, nan=-1.0)
                        batch_y.append(y)
                        batch_x.append(pixels_x)
                        meta_infos.append(meta_info)

                    if len(batch_x) == self.batch_size_gen:
                        # copy to submit to model
                        batch_y_submit = batch_y.copy()
                        batch_x_submit = batch_x.copy()
                        meta_infos_submit = meta_infos.copy()

                        # reset for the next batch
                        batch_y, batch_x, meta_infos = [], [], []

                        batch_x_submit = torch.stack(batch_x_submit, dim=0)
                        meta_infos_submit = torch.stack([torch.stack(metas) for metas in meta_infos_submit])
                        yield dict(x=batch_x_submit,
                                    y=batch_y_submit,
                                    meta_info=meta_infos_submit,
                                    wave_list=wavelist,
                                    bandwidth=bandwidth,
                                    language_embed=language_embed,
                                    input_mode=input_mode,
                                    kernel_size=kernel_size)

                except Exception as e:
                    print(f'skipping problematic patch: {e}')
                    continue

            # Yield any remaining partial batch if the batchgen has been exhausted
            if len(batch_x)>0:
                batch_x = torch.stack(batch_x, dim=0)
                meta_infos = torch.stack([torch.stack(metas) for metas in meta_infos])
                yield dict(x=batch_x,
                           y=batch_y,
                           meta_info=meta_infos,
                           wave_list=wavelist,
                           bandwidth=bandwidth,
                           language_embed=language_embed,
                           input_mode=input_mode,
                           kernel_size=kernel_size)

class CopernicusFMDataModule(L.LightningDataModule):
    """

    """
    def __init__(self,
                 zarr_dirs: [],
                 data_kinds: [],
                 metadata_path: str,
                 input_dims: int,
                 input_overlap={'x': 0, 'y': 0},
                 verify_fn='basic',
                 augmentation=None,
                 batch_size_gen=1,
                 num_workers=4,
                 split_ratio=0.8,
                 filter_thres=0.05,
                 random_state=46):
        super().__init__()
        self.zarr_dirs = zarr_dirs
        self.data_kinds = data_kinds
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.input_dims = input_dims
        self.input_overlap = input_overlap
        self.verify_fn = verify_fn
        self.augmentation = augmentation
        self.batch_size_gen = batch_size_gen
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.filter_thres = filter_thres
        self.random_state = random_state
        print('inside datamodule init')



    def construct_samples(self, zarr_paths, kinds):
        """

        """

        samples = {}
        for path, kind in zip(zarr_paths, kinds):
            sample_name = Path(path).stem
            if self.metadata[kind].input_mode == 'spectral':
                samples[sample_name] = dict(
                    pixels_path=path,
                    wavelist=list(self.metadata[kind].wavelist.values()),
                    bandwidth=list(self.metadata[kind].bandwidths.values()),
                    bandnames=list(self.metadata[kind].bandnames),
                    mean=list(self.metadata[kind].mean.values()),
                    std=list(self.metadata[kind].std.values()),
                    input_mode=str(self.metadata[kind].input_mode),
                    kernel_size=self.metadata[kind].get("kernel_size", None)
                )
            elif self.metadata[kind].input_mode == 'variable':
                samples[sample_name] = dict(
                    pixels_path=path,
                    meta_info=self.metadata[kind].meta_info,
                    language_embed=self.metadata[kind].language_embed,
                    input_mode=self.metadata[kind].input_mode,
                    kernel_size=self.metadata[kind].kernel_size
                )
        return samples

    def setup(self, stage: Literal["fit", "test", "predict"]| None = None) -> None:
        """Called by Lightning with stage='fit' | 'test' | 'predict'"""
        zarr_pathss = []
        path_kindss = []
        for zarr_dir, dir_kind in zip(self.zarr_dirs, self.data_kinds):
            zarr_paths = glob.glob(f"{zarr_dir}/*.zarr")
            path_kinds = [dir_kind for _ in zarr_paths]
            zarr_pathss.extend(zarr_paths)
            path_kindss.extend(path_kinds)
        if stage == "fit":
            train_path, val_path, train_kind, val_kind = train_test_split(zarr_pathss,
                                                                          path_kindss,
                                                                          stratify=path_kindss,
                                                                          test_size=(1 - self.split_ratio),
                                                                          shuffle=True,
                                                                          random_state=self.random_state
                                                                          )
            train_samples = self.construct_samples(train_path, train_kind)
            val_samples = self.construct_samples(val_path, val_kind)
            self.train_ds = CopernicusFMDataset(train_samples,
                                                input_dims=self.input_dims,
                                                input_overlap=self.input_overlap,
                                                mode="train",
                                                verify_fn=self.verify_fn,
                                                batch_size_gen=self.batch_size_gen,
                                                augmentation=self.augmentation,
                                                filter_thres=self.filter_thres
                                                )
            self.val_ds = CopernicusFMDataset(val_samples,
                                                input_dims=self.input_dims,
                                                input_overlap=self.input_overlap,
                                                mode="val",
                                                verify_fn=self.verify_fn,
                                                batch_size_gen=self.batch_size_gen,
                                                filter_thres=self.filter_thres
                                                )

        if stage == "test":
            test_samples = self.construct_samples(zarr_pathss, path_kindss)
            self.test_ds = CopernicusFMDataset(test_samples,
                                               input_dims=self.input_dims,
                                               input_overlap=self.input_overlap,
                                               mode="test",
                                               verify_fn=self.verify_fn,
                                               batch_size_gen=self.batch_size_gen,
                                               filter_thres=self.filter_thres
                                               )

        if stage == "predict":
            pred_samples = self.construct_samples(zarr_pathss, path_kindss)
            print(zarr_pathss)
            self.pred_ds = CopernicusFMDataset(pred_samples,
                                                input_dims=self.input_dims,
                                                input_overlap=self.input_overlap,
                                                verify_fn=self.verify_fn,
                                                mode="predict",
                                                batch_size_gen=self.batch_size_gen,
                                                filter_thres=self.filter_thres
                                                )

    def train_dataloader(self):
        print('DataLoader called')
        return DataLoader(
            dataset= self.train_ds,
            batch_size=None,
            num_workers=self.num_workers
        )
    def val_dataloader(self):
        return DataLoader(
            dataset= self.val_ds,
            batch_size=None,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_ds,
            batch_size=None,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset= self.pred_ds,
            batch_size=None,
            num_workers=self.num_workers
        )


