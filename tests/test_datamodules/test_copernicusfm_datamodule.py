import os
import pytest
import torch
import yaml
import types
import numpy as np
import xarray as xr
import zarr
from unittest.mock import MagicMock, patch, mock_open
from box import Box
from modulargeofm.datamodules.copernicusfm_datamodule import CopernicusFMIterableDataset, CopernicusFMDataset, CopernicusFMIterableDataModule 

# ----------------------------------------
# CopernicusFMIterableDataset
# ----------------------------------------
class FakeXRArray:
    def __init__(self, values):
        self.values = values
    def mean(self):
        return self

class FakeXRDataset:
    def __init__(self, data, bandnames, num_times=1):
        self.data = data
        self.bandnames = bandnames
        self.dims = {"time":num_times} 
    def __getitem__(self, keys):
        return self
    
    def __len__(self):
        return self.dims["time"]

    def isel(self, *args, **kwargs):
        return self

    def expand_dims(self, **kwargs):
        return self

    def to_array(self, dim=None):
        return types.SimpleNamespace(values=self._data)

    @property
    def rio(self):
        return self

    def resolution(self):
        return (10.0, 10.0)

    @property
    def crs(self):
        return types.SimpleNamespace(to_epsg=lambda: 4326)

    def __getattr__(self, name):
        if name in ["x", "y"]:
            return FakeXRArray(np.array([0.0, 1.0]))
        raise AttributeError
class FakeNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)

def dummy_accepted_batched_data():
    C, H, W =  4, 16, 16  # channels, height, width
    pixels = np.random.rand(C, H, W).astype(np.float32)
    x = np.arange(W)
    y = np.arange(H)
    band = np.arange(C)
    # batch = np.arange(B)
    ds = xr.Dataset(
        data_vars={
            "pixels": (( "band", "y", "x"), pixels)
        },
        coords={
            "x": ("x", x),
            "y": ("y", y),
            "band": ("band", band),
        }
    )
    ds = ds.rio.write_crs("EPSG:4326")
    return ds

def dummy_reject_batched_data():
    C, H, W =  4, 16, 16  # channels, height, width
    pixels = np.zeros((C, H, W)).astype(np.float32)
    x = np.arange(W)
    y = np.arange(H)
    band = np.arange(C)
    # batch = np.arange(B)
    ds = xr.Dataset(
        data_vars={
            "pixels": (( "band", "y", "x"), pixels)
        },
        coords={
            "x": ("x", x),
            "y": ("y", y),
            "band": ("band", band),
        }
    )
    ds = ds.rio.write_crs("EPSG:4326")
    return ds
 
def fake_batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

@pytest.fixture
def dummy_samples():
    return {
        "sample1": dict(
            pixels_path='dummy_path.zarr',
            wavelist=[1, 2, 3],
            bandwidth=[10, 10, 10],
            mean=[1, 1, 1],
            std=[0.5, 0.5, 0.5],
            bandnames=['B1', 'B2', 'B3'],
            input_mode="spectral",
            kernel_size=3,
        )
    }
@pytest.fixture
def dummy_pred_samples():
    return {
        "sample1": dict(
            pixels_path='dummy_path.zarr',
            wavelist=[1, 2, 3, 4],
            bandwidth=[10, 10, 10, 10],
            mean=[1, 1, 1, 1],
            std=[0.5, 0.5, 0.5, 0.5],
            bandnames=['B1', 'B2', 'B3', 'B4'],
            input_mode="spectral",
            kernel_size=3,
        )
    }
@pytest.fixture(params=[('train', 1),
                        ('train', 4), 
                        ('val', 1),
                        ('val', 4),
                        ('test', 1),
                        ('test', 4), 
                        ('predict', 1),
                        ('predict', 4)
                        ])
def ds(dummy_samples, dummy_pred_samples, request):
    mode, batch_size = request.param
    ds = CopernicusFMIterableDataset(
        samples=dummy_samples if mode != 'predict' else dummy_pred_samples,
        input_dims={'x': 16, 'y': 16},
        input_overlap={'x': 0, 'y': 0},
        mode=mode,
        verify_x_fn='default',
        verify_y_fn='default',
        batch_size=batch_size,
        augmentation=None,
        filter_x_thres=0.01,
        filter_y_thres=0.001
    )
    return ds , mode

@pytest.fixture(params=['None', 'multi'])
def worker_info(request):
    if request.param == 'None':
        return None
    mock_info = MagicMock()
    mock_info.id = 0
    mock_info.num_workers = 2
    return mock_info

def test_dataset_init_basic(ds, request):
    """Ensure attributes are set correctly."""
    ds, mode = ds
    sample_1 = ds.samples['sample1']
    assert os.path.basename(sample_1['pixels_path']).split('.')[1] == 'zarr'
    assert ds.mode == mode
    assert callable(ds.verify_x_fn)
    assert callable(ds.verify_y_fn)
    assert isinstance(ds.input_dims, dict)
    assert isinstance(ds.input_overlap, dict)
    assert isinstance(ds.filter_x_thres, float)
    assert isinstance(ds.filter_y_thres, float)

def test_invalid_verify_fn_raises(dummy_samples):
    """Invalid verify_fn must raise ValueError."""
    with pytest.raises(ValueError):
        CopernicusFMIterableDataset(
            samples=dummy_samples,
            input_dims={"x": 16, "y": 16},
            input_overlap=None,
            mode="train",
            verify_x_fn="unknown",
            verify_y_fn="unknown"
        )

class FakeBatchGenerator:
    def __init__(self, subset, input_dims, input_overlap):
        self.subset = subset
        self.input_dims = input_dims
        self.input_overlap = input_overlap

    def __iter__(self):
        dummy_data = [dummy_accepted_batched_data(), dummy_reject_batched_data()]
        for i in range(len(dummy_data)):
            yield dummy_data[i]
    
def test_iter_yields_valid_batches(monkeypatch, ds):
    monkeypatch.setattr("modulargeofm.datamodules.copernicusfm_datamodule.xr.open_zarr",
                        lambda *a, **k: FakeXRDataset(
                            data=np.random.rand(2, 4, 4).astype(np.float32),  # [bands, H, W]
                            bandnames=["label", "b1"],
                        ))
    monkeypatch.setattr("modulargeofm.datamodules.copernicusfm_datamodule.create_batch_generator",
                        FakeBatchGenerator
                        )
    monkeypatch.setattr("modulargeofm.datamodules.copernicusfm_datamodule.Normalize", FakeNormalize)
    monkeypatch.setattr("modulargeofm.datamodules.copernicusfm_datamodule.batched", fake_batched)
    monkeypatch.setattr("modulargeofm.datamodules.copernicusfm_datamodule.get_worker_info", lambda: None)
    
    ds, mode = ds
    batch_size = ds.batch_size
    num_channels = 3 if mode != 'predict' else 4

    output = next(iter(ds))
    assert isinstance(output, dict) 

    # test the returned keys and types
    assert "x" in output
    assert "y" in output
    assert "meta_info" in output
    assert "wave_list" in output
    assert "bandwidth" in output
    assert "language_embed" in output
    assert "input_mode" in output
    assert "kernel_size" in output
    assert isinstance(output["x"], torch.Tensor)
    assert isinstance(output["y"], list) if mode=='predict' else isinstance(output["y"], torch.Tensor)

    # test the the filtered batch size
    assert len(output["x"]) == 2 if mode == 'predict' and batch_size > 1 else 1  # filter patch in different modes
    assert len(output["y"][0]) == 2 if mode == 'predict' and batch_size > 1 else 1  # filter patch in different modes
    
    # test the shape 
    assert output["x"][0].size() == (num_channels, ds.input_dims['y'], ds.input_dims['x']) if mode == 'predict' else (num_channels-1, ds.input_dims['y'], ds.input_dims['x'])
    assert len(output["y"]) == 3 if mode == 'predict' else output["y"][0].size() == (ds.input_dims['y'], ds.input_dims['x'])
    assert output["y"][0].size() == (ds.input_dims['y'], ds.input_dims['x']) if mode != 'predict' else True
     
# ----------------------------------------
# CopernicusFMDataset
# ----------------------------------------
def create_test_zarr(path, n=3, c=4, h=8, w=8):
    root = zarr.open(path, mode="w")

    root.create_array(
        "images",
        data=np.random.rand(n, c, h, w).astype(np.float32)
    )
    root.create_array(
        "labels",
        data=np.random.randint(0, 2, size=(n, h, w)).astype(np.int64)
    )
    root.create_array(
        "meta_info",
        data=np.random.rand(n, 4).astype(np.float32)
    )

    root.attrs["wavelist"] = list(range(c))
    root.attrs["bandwidth"] = [10.0] * c
    root.attrs["input_mode"] = "spectral"
    root.attrs["kernel_size"] = None

def test_len_multiple_zarrs(tmp_path):
    zarr1 = tmp_path / "a.zarr"
    zarr2 = tmp_path / "b.zarr"

    create_test_zarr(zarr1, n=2)
    create_test_zarr(zarr2, n=3)

    ds = CopernicusFMDataset(str(tmp_path))

    assert len(ds) == 5

def test_getitem_returns_expected_dict(tmp_path):
    zarr_path = tmp_path / "test.zarr"
    create_test_zarr(zarr_path, n=1)

    ds = CopernicusFMDataset(str(tmp_path))
    sample = ds[0]

    assert isinstance(sample, dict)

    required_keys = {
        "x", "y", "meta_info",
        "wave_list", "bandwidth",
        "language_embed", "input_mode", "kernel_size"
    }
    assert required_keys.issubset(sample.keys())

    assert isinstance(sample["x"], torch.Tensor)
    assert isinstance(sample["y"], torch.Tensor)
    assert isinstance(sample["meta_info"], torch.Tensor)
    assert isinstance(sample["wave_list"], torch.Tensor)
    assert isinstance(sample["bandwidth"], torch.Tensor)
    assert isinstance(sample["kernel_size"], int)

    assert sample["x"].ndim == 3  # [C, H, W]
    assert sample["y"].ndim == 2  # [H, W]
    assert sample["meta_info"].shape == (4,)

def test_kernel_size_default(tmp_path):
    zarr_path = tmp_path / "test.zarr"
    create_test_zarr(zarr_path, n=1)

    ds = CopernicusFMDataset(str(tmp_path))
    sample = ds[0]

    assert sample["kernel_size"] == 16

def test_transform_applied(tmp_path):
    zarr_path = tmp_path / "test.zarr"
    create_test_zarr(zarr_path, n=1)

    def augment(x, y):
        return x + 1, y+1

    ds = CopernicusFMDataset(str(tmp_path), augment=augment)
    sample = ds[0]

    assert torch.all(sample["x"] >= 1)
    assert torch.all(sample["y"] >= 1)

def test_indexing_across_files(tmp_path):
    create_test_zarr(tmp_path / "a.zarr", n=2)
    create_test_zarr(tmp_path / "b.zarr", n=2)

    ds = CopernicusFMDataset(str(tmp_path))

    samples = [ds[i] for i in range(len(ds))]

    assert len(samples) == 4
    for s in samples:
        assert "x" in s




# ----------------------------------------
# CopernicusFMIterableDataModule
# ----------------------------------------
# Sample metadata to mock yaml content
META = {
    "kind1": { "input_mode": "spectral", 
               "wavelist": {"wl": 1},
               "bandwidths": {"b": 1},
                "bandnames": ["b"],
                  "mean": {"b": 0}, 
                  "std": {"b": 1}, 
                  "kernel_size": 3},

    "kind2": {"input_mode": "variable", 
              "meta_info": {"info": 1}, 
              "language_embed": [0],
              "kernel_size": 5}
}

@pytest.fixture()
def datamodule():
    yaml_content = yaml.dump(META)
    with patch("builtins.open", mock_open(read_data=yaml_content)):
        dm =  CopernicusFMIterableDataModule(
            zarr_dirs=['./dummy_zarr_dir1/', './dummy_zarr_dir2/'],
            data_kinds=['S2_xz', 'DEM_yz'],
            metadata_path='./dummy_metadata.yaml',
            input_dims={'x': 64, 'y': 64},
            input_overlap={'x': 16, 'y': 16},
            verify_fn='basic',
            augmentation=None,
            batch_size=2,
            num_workers=0,
            filter_x_thres=0.01,
        )
    return dm
         
         
@patch("builtins.open", new_callable=mock_open, read_data=yaml.dump(META))
def test_copernicusfm_datamodule_init(mock_open_func):
    """Test datamodule initialization."""
    
    metadata_path = './dummy_metadata.yaml'
    dm =  CopernicusFMIterableDataModule(
        zarr_dirs=['./dummy_zarr_dir1/', './dummy_zarr_dir2/',],
        data_kinds=['S2_xz', 'DEM_yz'],
        metadata_path=metadata_path,
        input_dims={'x': 64, 'y': 64},
        input_overlap={'x': 16, 'y': 16},
        verify_fn='basic',
        augmentation=None,
        batch_size=2,
        num_workers=0,
        filter_x_thres=0.01,
        filter_y_thres=0.001
        )

    # ---------- Assertions ---------
    mock_open_func.assert_called_once_with(metadata_path)  
    assert isinstance(dm.zarr_dirs, list) and all(isinstance(p, str) for p in dm.zarr_dirs)
    assert isinstance(dm.data_kinds, list) and all(isinstance(k, str) for k in dm.data_kinds)
    assert isinstance(dm.metadata, Box) and dm.metadata.kind1.kernel_size == 3
    assert isinstance(dm.input_dims, dict) and dm.input_dims == {'x': 64, 'y': 64}
    assert isinstance(dm.input_overlap, dict) and dm.input_overlap == {'x': 16, 'y': 16}
    assert callable(dm.verify_fn) or dm.verify_fn == 'basic'
    assert callable(dm.augmentation) or dm.augmentation is None
    assert isinstance(dm.batch_size, int) and dm.batch_size == 2
    assert isinstance(dm.num_workers, int) and dm.num_workers == 0
    assert isinstance(dm.split_ratio, float) and dm.split_ratio == 0.8
    assert isinstance(dm.filter_x_thres, float) and dm.filter_x_thres == 0.01
    assert isinstance(dm.filter_y_thres, float) and dm.filter_y_thres == 0.001

def test_construct_samples(datamodule):
    zarr_paths = ['dummy1.zaar', 'dummy3.zaar', 'dummyZ.zaar']
    kinds = ['kind1', 'kind1', 'kind2']
    samples = datamodule.construct_samples(zarr_paths, kinds)

    assert isinstance(samples, dict)
    assert all(isinstance(samples[key], dict) for key in samples.keys())
    assert list(samples['dummy1'].keys()) == ['pixels_path', 'wavelist', 'bandwidth', 'bandnames', 'mean', 'std', 'input_mode', 'kernel_size']
    assert list(samples['dummyZ'].keys()) == ['pixels_path', 'meta_info', 'language_embed', 'input_mode', 'kernel_size']

@patch("modulargeofm.datamodules.copernicusfm_datamodule.train_test_split")
@patch("modulargeofm.datamodules.copernicusfm_datamodule.CopernicusFMIterableDataModule.construct_samples")
@patch("modulargeofm.datamodules.copernicusfm_datamodule.glob.glob")
@patch("modulargeofm.datamodules.copernicusfm_datamodule.CopernicusFMIterableDataset")
@pytest.mark.parametrize("stage", ["fit", "test", "predict"])
def test_setup_stages(mock_dataset, 
                      mock_glob, 
                      mock_construct_samples, 
                      mock_traintest_split, 
                      stage, 
                      datamodule
                      ):
    mock_dataset.return_value =  "CopernicusFM_IterableDataset"
    mock_glob.return_value = MagicMock()
    mock_traintest_split.return_value = ('train_path', 'val_path', 'train_kind', 'val_kind')
    mock_construct_samples.return_value = MagicMock()

    datamodule.setup(stage=stage)
    if stage=='fit': 
        print('prin train ds', datamodule.train_ds)

    attrs = {"fit": ["train_ds", "val_ds"], "test": ["test_ds"], "predict": ["pred_ds"]}
    assert all(hasattr(datamodule, attr) for attr in attrs[stage]) 
    assert all(getattr(datamodule, attr) == "CopernicusFM_IterableDataset" for attr in attrs[stage])  



