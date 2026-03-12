import glob
import math
from functools import partial
from itertools import batched
from pathlib import Path
from typing import Callable, Literal, Union

import lightning as L
import rioxarray
import torch
import xarray as xr
import yaml
import zarr
from box import Box
from kornia.augmentation import Normalize
from sklearn.model_selection import train_test_split
from torch.utils.data import (DataLoader, Dataset, IterableDataset,
                              get_worker_info)
from modulargeofm.datamodules.shared import create_batch_generator, filter_x, filter_y


class CopernicusFMIterableDataset(IterableDataset):
    """
    Iterable PyTorch dataset for handling Copernicus Foundation Model (FM) data stored in Zarr format.
    This dataset dynamically reads and batches satellite data from multiple modalities, applying
    optional normalization, filtering, and augmentations. It is designed for streaming large-scale
    Copernicus data efficiently across multiple worker processes.

    Parameters
    ----------
    samples : dict
        Dictionary describing each dataset sample. Keys are sample names, and values are dictionaries
        containing paths and metadata such as:
        - ``pixels_path`` (str): Path to the Zarr dataset.
        - ``wavelist`` (list): List of wavelengths or channels.
        - ``bandwidth`` (list): List of bandwidths per channel.
        - ``mean`` (list): Channel means for normalization.
        - ``std`` (list): Channel standard deviations for normalization.
        - ``bandnames`` (list): Band identifiers in the dataset.
        - ``language_embed`` (optional): Embedding vectors for semantic context.
        - ``input_mode`` (str): Input mode (e.g., "spectral" or "variable").
        - ``kernel_size`` (optional): Spatial kernel size for processing.
    input_dims : dict
        Spatial dimensions of input patches, e.g. ``{"x": 256, "y": 256}``.
    input_overlap : dict
        Overlap between adjacent patches, e.g. ``{"x": 16, "y": 16}``.
    mode : {"train", "val", "test", "predict"}
        Operational mode of the dataset.
    augmentation : callable, optional
        Augmentation function applied to input tensors (used in training/validation modes).
    verify_fn : {"basic"} or callable, default='basic'
        Function used to verify the quality of a patch. If 'basic', checks for invalid values.
    batch_size_gen : int, default=1
        Number of patches per generated batch.
    time_dim : str, default='time'
        Dimension name representing temporal axis in the dataset.
    filter_thres : float, default=0.05
        Maximum allowed fraction of invalid (NaN or zero) values before filtering out a patch.

    Yields
    ------
    dict
        A dictionary containing:
        - ``x`` (torch.Tensor): Normalized input tensor batch.
        - ``y`` (list): Target tensor or coordinates (depending on mode).
        - ``meta_info`` (torch.Tensor): Metadata tensor containing `[lon, lat, time, area]`.
        - ``wave_list`` (list): List of wavelength bands.
        - ``bandwidth`` (list): List of corresponding bandwidths.
        - ``language_embed`` (optional): Language embeddings, if available.
        - ``input_mode`` (str): Type of input representation.
        - ``kernel_size`` (torch.Tensor or None): Kernel size for processing.

    Notes
    -----
    This class supports distributed iteration using PyTorch `DataLoader` with multiple workers.
    Each worker receives a unique subset of the dataset's temporal slices to avoid overlap.

    Examples
    --------
    >>> dataset = CopernicusFMIterableDataset(
    ...     samples=my_samples,
    ...     input_dims={'x': 128, 'y': 128},
    ...     input_overlap={'x': 8, 'y': 8},
    ...     mode='train',
    ...     augmentation=None
    ... )
    """
    def __init__(self,
                 samples,
                 input_dims,
                 input_overlap,
                 mode: Literal["train", "val", "test", "predict"],
                 augmentation=None,
                 verify_x_fn ='default',
                 verify_y_fn ='default',
                 batch_size=1,
                 time_dim='time',
                 filter_x_thres=0.05,
                 filter_y_thres=0.05
                 ) -> None:

        self.samples = samples
        self.input_dims = input_dims
        self.input_overlap = input_overlap or {}
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.time_dim = time_dim
        self.mode = mode
        self.filter_x_thres = filter_x_thres
        self.filter_y_thres = filter_y_thres

        if verify_x_fn == 'default':
            self.verify_x_fn = partial(filter_x, threshold=self.filter_x_thres)
        elif callable(verify_x_fn):
            self.verify_x_fn = verify_x_fn # TODO: further rework on custom function
        else:
            raise ValueError(f"Invalid verify_fn: {verify_x_fn}")
        
        if verify_y_fn == 'default':
            self.verify_y_fn = partial(filter_y, threshold=self.filter_y_thres)
        elif callable(verify_y_fn):
            self.verify_y_fn = verify_y_fn # TODO: further rework on custom function
        else:
            raise ValueError(f"Invalid verify_fn: {verify_y_fn}")


    def __iter__(self):
        """ 
        Iterate over dataset patches and yield processed batches. 
        
        The method handles worker-specific partitioning, filtering, normalization, and batching of patches. 
        Remaining partial batches are also yielded at the end. 
        
        Yields 
        ------ 
        dict 
            Dictionary containing processed batch data and metadata. 
        """
        # handling splitting logic for workers cleverly in the case of different num of samples and time dimension.

        # get num of dataset modalities
        keys = list(self.samples.keys())

        # get time dimension of each
        sample_time_list = []

        for key in keys:
            modality = self.samples.get(key)
            xr_dataset = xr.open_zarr(modality.get('pixels_path'), decode_coords="all", chunks=None)
            if self.time_dim in xr_dataset.dims:
                num_times = len(xr_dataset[self.time_dim])
            else:
                num_times = 1
            sample_time_list.extend((key, xr_dataset, t) for t in range(num_times))
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
            sample_key, xr_dataset, time = sample_time_list[global_index]
            modality = self.samples.get(sample_key)
            wavelist = torch.tensor(modality.get('wavelist', None))
            bandwidth = torch.tensor(modality.get('bandwidth', None))
            mean = torch.tensor(modality.get('mean'), dtype=torch.float32)
            std = torch.tensor(modality.get('std'), dtype=torch.float32)
            bandnames = modality.get('bandnames', None)
            language_embed = modality.get('language_embed', None)
            input_mode = modality.get('input_mode')
            kernel_size = torch.tensor(modality['kernel_size']) if modality.get('kernel_size') is not None else 16 # folllow the default from copernicusfm
            self.normalise = Normalize(mean=mean, std=std)

            if self.time_dim in xr_dataset.dims:
                subset = xr_dataset[bandnames].isel({self.time_dim: time})
            else:
                if time > 0: 
                    raise Exception('num of times must be 0')
                subset = xr_dataset.expand_dims(dim={self.time_dim:1}, axis=-1)[bandnames]

            resolution = abs(subset.rio.resolution()[0]) # in meters
            area = self.input_dims['x'] * self.input_dims['y'] * resolution/(1000**2) # in km2
            area = torch.tensor(area)  

            # batch generator
            batch_gen = iter(create_batch_generator(subset, self.input_dims, self.input_overlap))

            for batch in batched(batch_gen, self.batch_size):
                batch_tensor = []
                xcenter_batch_tensor = []
                ycenter_batch_tensor = []
                xcoord_batch = []
                ycoord_batch = []
                spatial_ref_batch = []
                
                for patch_ds in batch:
                    # print('patch ds', patch_ds)
                    patch_np = patch_ds.to_array(dim='pixels').values
                    x_coords = patch_ds['x']
                    y_coords = patch_ds['y']
                    spatial_ref = patch_ds.rio.crs.to_epsg()

                    xcenter_patch_np = x_coords.mean().values
                    ycenter_patch_np = y_coords.mean().values

                    x_coords = x_coords.values
                    y_coords = y_coords.values

                    patch_tensor = torch.from_numpy(patch_np).float().squeeze()
                    xcenter_patch_tensor = torch.from_numpy(xcenter_patch_np).float()
                    ycenter_patch_tensor = torch.from_numpy(ycenter_patch_np).float()
                    
                    batch_tensor.append(patch_tensor)
                    xcenter_batch_tensor.append(xcenter_patch_tensor)
                    ycenter_batch_tensor.append(ycenter_patch_tensor)
                    xcoord_batch.append(x_coords)
                    ycoord_batch.append(y_coords)
                    spatial_ref_batch.append(spatial_ref)
                batch_tensor = torch.stack(batch_tensor, dim=0)
                xcenter_batch_tensor = torch.stack(xcenter_batch_tensor, dim=0)
                ycenter_batch_tensor = torch.stack(ycenter_batch_tensor, dim=0)

                if self.augmentation and self.mode!= 'predict':
                    batch_tensor = self.augmentation(batch_tensor)

                if self.mode != 'predict':
                    # filter each patch in the batch
                    x_batch_test = batch_tensor[:, 1:, :, :] # take only the actual data
                    y_batch_test = batch_tensor[:, 0,  :, :] # label
                    x_valid_mask = self.verify_x_fn(x_batch_test)
                    y_valid_mask = self.verify_y_fn(y_batch_test)
                    valid_mask = x_valid_mask & y_valid_mask
                    batch_tensor = batch_tensor[valid_mask]
                    if batch_tensor.shape[0] == 0:
                        continue
                    xcenter_batch_tensor = xcenter_batch_tensor[valid_mask]
                    ycenter_batch_tensor = ycenter_batch_tensor[valid_mask]
                    batch_x = batch_tensor[:, 1:, :, :] # input data
                    batch_y = batch_tensor[:, 0,  :, :] # label
                else:
                    batch_x = batch_tensor[:, :, :, :]
                    batch_y = [ycoord_batch, xcoord_batch, spatial_ref_batch]

                area_tensor = torch.full((batch_tensor.shape[0],), area)
                time_tensor = torch.full((batch_tensor.shape[0],), torch.nan) #TODO: later, consider time
                meta_info = torch.stack([xcenter_batch_tensor, ycenter_batch_tensor, time_tensor, area_tensor], dim=1)
                batch_x = self.normalise(batch_x).clamp(min=-1.0, max=1.0)
                batch_x = torch.nan_to_num(batch_x, nan=0.0)
                yield dict( x=batch_x, # [B, C, H, W]
                            y=batch_y, # [B, H, W]
                            meta_info=meta_info, # [B, 4]
                            wave_list=wavelist, # [C]
                            bandwidth=bandwidth, # [C]
                            language_embed=language_embed,
                            input_mode=input_mode, 
                            kernel_size=kernel_size
                            )
            subset.close()
            xr_dataset.close()
            del batch_iter, xr_dataset, subset
            gc.collect()

class CopernicusFMDataset(Dataset):
    def __init__(self, chip_zarr_dir, transform=None):
        self.chip_zarr_dir = chip_zarr_dir
        self.transform = transform
        self.index = []
        self.zarr_paths = glob.glob(f"{self.chip_zarr_dir}/*.zarr")
        for path_id, zarr_path in enumerate(self.zarr_paths):
            z = zarr.open(zarr_path, mode='r')
            n = z['images'].shape[0]
            self.index.extend([(path_id, i) for i in range(n)])    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        path_id, i = self.index[idx]
        z = zarr.open(self.zarr_paths[path_id], mode='r')
        image = torch.from_numpy(z['images'][i])
        label = torch.from_numpy(z['labels'][i])
        meta_info = torch.from_numpy(z['meta_info'][i])
        attrs = z.attrs
        wavelist = torch.tensor(attrs['wavelist'])
        bandwidth = torch.tensor(attrs['bandwidth'])
        # language_embed= [None] #if attrs['language_embed'] == None else attrs['language_embed'] #TODO rework for this variable
        input_mode = attrs['input_mode']
        kernel_size = attrs['kernel_size'] if attrs['kernel_size'] is not None else 16 # folllow the default from copernicusfm
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return dict(x=image, 
                    y=label, 
                    meta_info=meta_info, # [4]
                    wave_list=wavelist, # [C]
                    bandwidth=bandwidth, # [C]
                    language_embed="None",
                    input_mode=input_mode, 
                    kernel_size=kernel_size # [B]
                    )

        
    
class CopernicusFMIterableDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for managing Copernicus Foundation Model datasets.
    Handles dataset construction, train/validation/test splits, and DataLoader creation.
    Integrates metadata parsing from YAML and dynamic Zarr dataset discovery.

    Parameters
    ----------
    zarr_dirs : list of str
        Directories containing `.zarr` datasets.
    data_kinds : list of str
        Data type identifiers corresponding to each directory.
    metadata_path : str
        Path to YAML metadata describing available datasets and channels.
    input_dims : dict
        Spatial dimensions of input patches.
    input_overlap : dict, default={'x': 0, 'y': 0}
        Overlap size between adjacent patches.
    verify_fn : {"basic"} or callable, default='basic'
        Function for validating patch quality.
    augmentation : callable or str, optional
        Optional data augmentation function.
    batch_size_gen : int, default=1
        Number of patches per batch in generated data.
    num_workers : int, default=4
        Number of parallel workers for data loading.
    split_ratio : float, default=0.8
        Fraction of data used for training (remaining for validation).
    filter_thres : float, default=0.05
        Threshold for filtering invalid patches.
    random_state : int, default=46
        Random seed for deterministic splitting.

    Attributes
    ----------
    train_ds : CopernicusFMIterableDataset
        Training dataset instance.
    val_ds : CopernicusFMIterableDataset
        Validation dataset instance.
    test_ds : CopernicusFMIterableDataset
        Testing dataset instance.
    pred_ds : CopernicusFMIterableDataset
        Prediction dataset instance.

    Methods
    -------
    construct_samples(zarr_paths, kinds)
        Build sample dictionaries with metadata and paths.
    setup(stage)
        Initialize datasets depending on stage ("fit", "test", "predict").
    train_dataloader()
        Return DataLoader for training.
    val_dataloader()
        Return DataLoader for validation.
    test_dataloader()
        Return DataLoader for testing.
    predict_dataloader()
        Return DataLoader for prediction.

    Examples
    --------
    >>> datamodule = CopernicusFMDataModule(
    ...     zarr_dirs=["/data/zarr/train"],
    ...     data_kinds=["spectral"],
    ...     metadata_path="metadata.yaml",
    ...     input_dims={"x": 128, "y": 128}
    ... )
    >>> datamodule.setup("fit")
    >>> train_loader = datamodule.train_dataloader()
    >>> for batch in train_loader:
    ...     print(batch['x'].shape)
    """
    def __init__(self,
                 zarr_dirs: list,
                 data_kinds: list,
                 metadata_path: str,
                 input_dims: dict,
                 input_overlap: dict={'x': 0, 'y': 0},
                 verify_fn: Union[Callable, str]='default',
                 augmentation: Union[Callable, str]=None,
                 batch_size: int=1,
                 num_workers: int=4,
                 prefetch_factor: int = 8,
                 split_ratio: float=0.8,
                 filter_x_thres: float=0.05,
                 filter_y_thres: float=0.05,
                 random_state: int=46):
        super().__init__()
        self.zarr_dirs = zarr_dirs
        self.data_kinds = data_kinds
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.input_dims = input_dims
        self.input_overlap = input_overlap
        self.verify_fn = verify_fn
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.split_ratio = split_ratio
        self.filter_x_thres = filter_x_thres
        self.filter_y_thres = filter_y_thres
        self.random_state = random_state

    def construct_samples(self, zarr_paths, kinds):
        """ 
        Construct structured sample dictionaries for each dataset path. 
        
        Parameters 
        ---------- 
        zarr_paths : list of str 
            Paths to Zarr datasets. 
        kinds : list of str 
            Corresponding dataset kinds. 
            
        Returns
        ------- 
        dict 
            Dictionary mapping sample names to configuration dictionaries. 
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
        """ 
        Prepare datasets for training, validation, testing, or prediction.
         
        Parameters 
        ---------- 
        stage : {"fit", "test", "predict"}, optional 
            Stage for which to set up datasets. Default is None. 
        
        Notes 
        -----
        - During ``fit`` stage, datasets are split according to ``split_ratio``. 
        - During ``test`` or ``predict`` stages, all datasets are loaded without splitting. 
        """
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
            self.train_ds = CopernicusFMIterableDataset(train_samples,
                                                        input_dims=self.input_dims,
                                                        input_overlap=self.input_overlap,
                                                        mode="train",
                                                        verify_x_fn=self.verify_fn,
                                                        batch_size=self.batch_size,
                                                        augmentation=self.augmentation,
                                                        filter_x_thres=self.filter_x_thres,
                                                        filter_y_thres=self.filter_y_thres
                                                        )
            self.val_ds = CopernicusFMIterableDataset(val_samples,
                                                      input_dims=self.input_dims,
                                                      input_overlap=self.input_overlap,
                                                      mode="val",
                                                      verify_x_fn=self.verify_fn,
                                                      batch_size=self.batch_size,
                                                      filter_x_thres=self.filter_x_thres,
                                                      filter_y_thres=self.filter_y_thres
                                                      )

        if stage == "test":
            test_samples = self.construct_samples(zarr_pathss, path_kindss)
            self.test_ds = CopernicusFMIterableDataset(test_samples,
                                                       input_dims=self.input_dims,
                                                       input_overlap=self.input_overlap,
                                                       mode="test",
                                                       verify_x_fn=self.verify_fn,
                                                       batch_size=self.batch_size,
                                                       filter_x_thres=self.filter_x_thres,
                                                       filter_y_thres=self.filter_y_thres
                                                       )

        if stage == "predict":
            pred_samples = self.construct_samples(zarr_pathss, path_kindss)
            self.pred_ds = CopernicusFMIterableDataset(pred_samples,
                                                       input_dims=self.input_dims,
                                                       input_overlap=self.input_overlap,
                                                       verify_x_fn=self.verify_fn,
                                                       mode="predict",
                                                       batch_size=self.batch_size,
                                                       filter_x_thres=self.filter_x_thres,
                                                       filter_y_thres=self.filter_y_thres
                                                       )

    def train_dataloader(self):
        """Return PyTorch DataLoader for training data."""
        return DataLoader(
            dataset= self.train_ds,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor
        )
    def val_dataloader(self):
        """Return PyTorch DataLoader for validation data."""
        return DataLoader(
            dataset= self.val_ds,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        """Return PyTorch DataLoader for test data."""
        return DataLoader(
            dataset=self.test_ds,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor
        )

    def predict_dataloader(self):
        """Return PyTorch DataLoader for prediction data."""
        return DataLoader(
            dataset= self.pred_ds,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor
        )