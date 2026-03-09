from functools import partial
import glob, gc
import torch
import math
import rioxarray
import xarray as xr
import zarr
from operator import call
from box import Box
import yaml
import lightning as L
from pathlib import Path
from typing import Callable, Literal, Union
from kornia.augmentation import Normalize
from itertools import batched
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from sklearn.model_selection import train_test_split
from modulargeofm.datamodules.shared import create_batch_generator, filter_x, filter_y

class ClayIterableDataset(IterableDataset):
    """
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
    return datacube where:       
        datacube["pixels"],  # [B C H W]
        datacube["time"],  # [B 2]
        datacube["latlon"],  # [B 2]
        datacube["gsd"],  # 1
        datacube["waves"],  # [N]
    """

    def __init__(self,
                 samples,
                 input_dims,
                 input_overlap,
                 mode: Literal['train', 'val', 'test', 'predict'],
                 augmentation=None,
                 verify_x_fn='default',
                 verify_y_fn='default',
                 batch_size=1,
                 time_dim='time',
                 filter_x_thres=0.05,
                 filter_y_thres=0.05  
            
    ) -> None:
        super().__init__()
        self.samples = samples
        self.input_dims = input_dims
        self.input_overlap = input_overlap
        self.mode = mode
        self.augmentation = augmentation
        self.verify_x_fn = verify_x_fn
        self.verify_y_fn = verify_y_fn
        self.batch_size = batch_size
        self.time_dim = time_dim
        self.filter_x_thres = filter_x_thres
        self.filter_y_thres = filter_y_thres

        if verify_x_fn == 'default':
            self.verify_x_fn = partial(filter_x, threshold=self.filter_x_thres)
        elif callable(verify_x_fn):
            self.verify_x_fn = verify_x_fn
        else:
            raise ValueError("veriffy_x_fn should be 'default' or callable function")
        
        if verify_y_fn == 'default':
            self.verify_y_fn = partial(filter_y, threshold=self.filter_y_thres)
        elif callable(verify_y_fn):
            self.verify_y_fn = verify_y_fn
        else:
            raise ValueError("veriffy_y_fn should be 'default' or callable function")
    
    def __iter__(self):
        """
        """
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
            sample_time_list.extend((key, modality.get('pixels_path'), t) for t in range(num_times))
            xr_dataset.close()
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
            sample_key, pixels_path, time = sample_time_list[global_index]
            xr_dataset = xr.open_zarr(pixels_path, decode_coords="all", chunks=None)
            modality = self.samples.get(sample_key)
            wavelist = torch.tensor(modality.get('wavelist', None))
            mean = torch.tensor(modality.get('mean'), dtype=torch.float32)
            std = torch.tensor(modality.get('std'), dtype=torch.float32)
            bandnames = modality.get('bandnames', None)
            self.normalise = Normalize(mean=mean, std=std)

            if self.mode == 'predict':
                bandnames = bandnames[1:]
            if self.time_dim in xr_dataset.dims:
                subset = xr_dataset[bandnames].isel({self.time_dim: time})
            else:
                if time > 0: 
                    raise Exception('num of times must be 0')
                subset = xr_dataset.expand_dims(dim={self.time_dim:1}, axis=-1)[bandnames]
            
            gsd = torch.tensor(abs(subset.rio.resolution()[0])) # in meters

            # batch generator
            batch_gen = create_batch_generator(subset, self.input_dims, self.input_overlap)
            batch_iter = iter(batch_gen)
            for batch in batched(batch_iter, self.batch_size):
                batch_tensor = []
                xmin_batch_tensor = []
                xmax_batch_tensor = []
                ymin_batch_tensor = []
                ymax_batch_tensor = []
                xcoord_batch = []
                ycoord_batch = []
                spatial_ref_batch = []
                
                for patch_ds in batch:
                    patch_np = patch_ds.to_array(dim='pixels').values
                    x_coords = patch_ds['x']
                    y_coords = patch_ds['y']
                    spatial_ref = patch_ds.rio.crs.to_epsg()

                    xmin_patch_np = x_coords.min().values
                    xmax_patch_np = x_coords.max().values
                    ymin_patch_np = y_coords.min().values
                    ymax_patch_np = y_coords.max().values

                    x_coords = x_coords.values
                    y_coords = y_coords.values

                    patch_tensor = torch.from_numpy(patch_np).float().squeeze()
                    xmin_patch_tensor = torch.from_numpy(xmin_patch_np).float()
                    xmax_patch_tensor = torch.from_numpy(xmax_patch_np).float()
                    ymin_patch_tensor = torch.from_numpy(ymin_patch_np).float()
                    ymax_patch_tensor = torch.from_numpy(ymax_patch_np).float()
                    
                    batch_tensor.append(patch_tensor)

                    xmin_batch_tensor.append(xmin_patch_tensor)
                    xmax_batch_tensor.append(xmax_patch_tensor)
                    ymin_batch_tensor.append(ymin_patch_tensor)
                    ymax_batch_tensor.append(ymax_patch_tensor)
                    xcoord_batch.append(x_coords)
                    ycoord_batch.append(y_coords)
                    spatial_ref_batch.append(spatial_ref)
                batch_tensor = torch.stack(batch_tensor, dim=0)
                xmin_batch_tensor = torch.stack(xmin_batch_tensor, dim=0)
                xmax_batch_tensor = torch.stack(xmax_batch_tensor, dim=0)
                ymin_batch_tensor = torch.stack(ymin_batch_tensor, dim=0)
                ymax_batch_tensor = torch.stack(ymax_batch_tensor, dim=0)
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
                    xmin_batch_tensor = xmin_batch_tensor[valid_mask]
                    xmax_batch_tensor = xmax_batch_tensor[valid_mask]
                    ymin_batch_tensor = ymin_batch_tensor[valid_mask]
                    ymax_batch_tensor = ymax_batch_tensor[valid_mask]
                    batch_y = batch_tensor[:, 0,  :, :] # label
                else:
                    batch_y = [ycoord_batch, xcoord_batch, spatial_ref_batch]

                time_tensor = torch.full((batch_tensor.shape[0], 4), 0) #TODO: later, consider time
                latlon_tensor = torch.stack([ymin_batch_tensor,
                                             xmin_batch_tensor,
                                             ymax_batch_tensor,
                                             xmax_batch_tensor], 
                                             dim=1)
                batch_x = batch_tensor[:, 1:, :, :] # input data
                batch_x = self.normalise(batch_x).clamp(min=-1.0, max=1.0)
                batch_x = torch.nan_to_num(batch_x, nan=0.0)
                yield  dict(pixels=batch_x,  # [B C H W]
                           y=batch_y,  # [B, H, W]
                           time=time_tensor, # [B 4]
                           latlon=latlon_tensor,  # [B 4]
                           gsd=gsd, # 1
                           waves=wavelist,  # [N]
                            )
            subset.close()
            xr_dataset.close()
            del batch_iter, xr_dataset, subset
            gc.collect()
                
class ClayDataset(Dataset):
    def __init__(self, chip_zarr_dir, augment=None, num_augment=1):
        self.chip_zarr_dir = chip_zarr_dir
        self.augment = augment
        self.num_augment = num_augment
        self.index = []
        self.zarr_paths = glob.glob(f"{self.chip_zarr_dir}/*.zarr")
        for path_id, zarr_path in enumerate(self.zarr_paths):
            z = zarr.open(zarr_path, mode='r')
            n = z['pixels'].shape[0]
            self.index.extend([(path_id, i) for i in range(n)])    
    def __len__(self):
        if self.augment:
            return len(self.index) * self.num_augment
        else:
            return len(self.index)
    
    def __getitem__(self, idx):
        idx = idx // self.num_augment  if self.augment else idx
        path_id, i = self.index[idx]
        z = zarr.open(self.zarr_paths[path_id], mode='r')
        pixel = torch.from_numpy(z['pixels'][i])
        label = torch.from_numpy(z['labels'][i])
        time = torch.from_numpy(z['time'][i])
        latlon = torch.from_numpy(z['latlon'][i])
    
        attrs = z.attrs
        waves = torch.tensor(attrs['waves'])
        gsd = torch.tensor(attrs['gsd'])
        if self.augment:
            pixel = pixel[None]
            label = label[None, None]
            pixel, label = self.augment(pixel, label)
            pixel = pixel[0]
            label = label[0][0]
            
        return dict(pixels=pixel,  # [C H W]
                    y=label,  # [H, W]
                    time=time, # [4]
                    latlon=latlon,  # [4]
                    gsd=gsd, # 1
                    waves=waves,  # [N]
            )
    

class ClayIterableDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for managing Clay Foundation Model datasets.
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
            samples[sample_name] = dict(
                                        pixels_path=path,
                                        wavelist=list(self.metadata[kind].wavelist.values()),
                                        bandnames=list(self.metadata[kind].bandnames),
                                        mean=list(self.metadata[kind].mean.values()),
                                        std=list(self.metadata[kind].std.values())
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
            self.train_ds = ClayIterableDataset(train_samples,
                                                input_dims=self.input_dims,
                                                input_overlap=self.input_overlap,
                                                mode="train",
                                                verify_x_fn=self.verify_fn,
                                                batch_size=self.batch_size,
                                                augmentation=self.augmentation,
                                                filter_x_thres=self.filter_x_thres,
                                                filter_y_thres=self.filter_y_thres  
                                                )
            self.val_ds = ClayIterableDataset(val_samples,
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
            self.test_ds = ClayIterableDataset(test_samples,
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
            self.pred_ds = ClayIterableDataset(pred_samples,
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