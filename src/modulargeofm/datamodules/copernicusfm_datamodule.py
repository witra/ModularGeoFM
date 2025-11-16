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
import rioxarray 
from box import Box
from kornia.augmentation import Normalize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from xbatcher import BatchGenerator
from typing import Union, Callable


class CopernicusFMDataset(IterableDataset):
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
    >>> dataset = CopernicusFMDataset(
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
        """
        Create a batch generator from an xarray subset. 
        
        Parameters 
        ---------- 
        subset : xarray.Dataset 
            The dataset subset representing a specific temporal slice. 
        
        Returns 
        ------- 
        xbatcher.BatchGenerator 
            Iterator yielding patches of the dataset subset.
        """
        return BatchGenerator(
            subset,
            input_dims=self.input_dims,
            input_overlap=self.input_overlap)

    def basic_filter(self, patch, threshold=0.05):
        """ 
        Basic filtering function for rejecting invalid patches. 
        
        Checks if the fraction of zeros or NaN values in the input tensor exceeds a defined threshold. 
        
        Parameters 
        ---------- 
            patch : torch.Tensor Input patch tensor. 
            threshold : float, default=0.05 Maximum allowed fraction of invalid elements. 
        
        Returns 
        ------- 
        bool 
            ``True`` if patch is valid, ``False`` otherwise. 
        """

        total_elements = patch.numel()  # total number of elements
        # guard against empty patches
        if total_elements == 0:
            # treat an empty patch as invalid (reject)
            return False
        invalid_mask = (patch == 0) | torch.isnan(patch)
        invalid_fraction = invalid_mask.sum().float() / total_elements
        return invalid_fraction < threshold

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
        
        device = torch.device("cpu") # "cuda" if torch.cuda.is_available() else
        
        # get num of dataset modalities
        keys = list(self.samples.keys())

        # get time dimension of each
        sample_time_list = []

        for key in keys:
            modality = self.samples.get(key)
            xr_dataset = xr.open_zarr(modality.get('pixels_path'), chunks=None)
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
            wavelist = modality.get('wavelist', None)
            bandwidth = modality.get('bandwidth', None)
            mean = torch.tensor(modality.get('mean'), dtype=torch.float32, device=device)#.view(1, -1, 1, 1)
            std = torch.tensor(modality.get('std'), dtype=torch.float32, device=device)#.view(1, -1, 1, 1)
            bandnames = modality.get('bandnames', None)
            language_embed = modality.get('language_embed', None)
            input_mode = modality.get('input_mode')
            kernel_size = torch.tensor(modality['kernel_size']) if modality.get('kernel_size') is not None else None
            self.normalise = Normalize(mean=mean, std=std)

            if self.mode == 'predict':
                bandnames = bandnames[1:]
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
            batch_gen = iter(self.create_batch_generator(subset))

            # # Collect and yield batches
            batch_y, meta_infos = [], []

            batch_x = torch.empty((self.batch_size_gen, 
                                   len(wavelist),
                                   self.input_dims['x'], 
                                   self.input_dims['y']),
                                   dtype=torch.float32, device=device)

            for patch_ds in batch_gen:
                try:
                    patch_np = patch_ds.to_array(dim='pixels').values.squeeze()
                    patch_tensor = torch.from_numpy(patch_np).float().to(device)
                    
                    # verify per patch
                    status = False
                    if self.mode != 'predict':
                        patch_test = patch_tensor[1:, :, :] # take only the actual data
                        status = self.verify_fn(patch_test) # T/F
                    if self.mode == 'predict':
                        status = True
                    if status:
                        x_c = torch.tensor(patch_ds['x'].values.mean(), dtype=torch.float32)
                        y_c = torch.tensor(patch_ds['y'].values.mean(), dtype=torch.float32)
                        time = torch.tensor(np.nan)
                        meta_info = x_c, y_c, time, area,
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
                        batch_x[len(batch_y)] = pixels_x  # <--- write directly into preallocated batch tensor
                        batch_y.append(y)
                        meta_infos.append(meta_info)
                    if len(batch_y) == self.batch_size_gen:
                        batch_x = self.normalise(batch_x).clamp(min=-1.0, max=1.0)#.squeeze()
                        batch_x = torch.nan_to_num(batch_x, nan=-1.0)
                        yield dict( x=batch_x.clone(),
                                    y=batch_y.copy(),
                                    meta_info=torch.stack([torch.stack(metas) for metas in meta_infos]),
                                    wave_list=wavelist,
                                    bandwidth=bandwidth,
                                    language_embed=language_embed,
                                    input_mode=input_mode,
                                    kernel_size=kernel_size)
                        batch_y, meta_infos = [], []
                        batch_x = torch.empty((self.batch_size_gen, 
                                            len(wavelist),
                                            self.input_dims['x'], 
                                            self.input_dims['y']),
                                            dtype=torch.float32, device=device)

                except Exception as e:
                    # print(f'skipping problematic patch: {e}')
                    continue

            # Yield any remaining partial batch if the batchgen has been exhausted
            if len(batch_y)>0:
                batch_x_partial = batch_x[:len(batch_y)]
                batch_x_partial = self.normalise(batch_x_partial).clamp(min=-1.0, max=1.0)
                batch_x_partial = torch.nan_to_num(batch_x_partial, nan=-1.0)
                yield dict( x=batch_x_partial,
                            y=batch_y,
                            meta_info=torch.stack([torch.stack(metas) for metas in meta_infos]),
                            wave_list=wavelist,
                            bandwidth=bandwidth,
                            language_embed=language_embed,
                            input_mode=input_mode,
                            kernel_size=kernel_size)

class CopernicusFMDataModule(L.LightningDataModule):
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
    train_ds : CopernicusFMDataset
        Training dataset instance.
    val_ds : CopernicusFMDataset
        Validation dataset instance.
    test_ds : CopernicusFMDataset
        Testing dataset instance.
    pred_ds : CopernicusFMDataset
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
                 verify_fn: Union[Callable, str]='basic',
                 augmentation: Union[Callable, str]=None,
                 batch_size_gen: int=1,
                 num_workers: int=4,
                 prefetch_factor: int = 8,
                 split_ratio: float=0.8,
                 filter_thres: float=0.05,
                 random_state: int=46):
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
        self.prefetch_factor = prefetch_factor
        self.split_ratio = split_ratio
        self.filter_thres = filter_thres
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
            self.pred_ds = CopernicusFMDataset(pred_samples,
                                                input_dims=self.input_dims,
                                                input_overlap=self.input_overlap,
                                                verify_fn=self.verify_fn,
                                                mode="predict",
                                                batch_size_gen=self.batch_size_gen,
                                                filter_thres=self.filter_thres
                                                )

    def train_dataloader(self):
        """Return PyTorch DataLoader for training data."""
        return DataLoader(
            dataset= self.train_ds,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor
        )
    def val_dataloader(self):
        """Return PyTorch DataLoader for validation data."""
        return DataLoader(
            dataset= self.val_ds,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        """Return PyTorch DataLoader for test data."""
        return DataLoader(
            dataset=self.test_ds,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor
        )

    def predict_dataloader(self):
        """Return PyTorch DataLoader for prediction data."""
        return DataLoader(
            dataset= self.pred_ds,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor
        )


