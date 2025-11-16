from importlib import metadata
import os
import pytest
import torch
import yaml
from unittest.mock import MagicMock, patch, mock_open
from box import Box
from modulargeofm.datamodules.copernicusfm_datamodule import CopernicusFMDataset, CopernicusFMDataModule 

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
@pytest.fixture(params=[('train', 1),
                        ('train', 4), 
                        ('val', 1),
                        ('val', 4),
                        ('test', 1),
                        ('test', 4), 
                        ('predict', 1),
                        ('predict', 4)
                        ])
def ds(dummy_samples, request):
    mode, batch_size = request.param
    ds = CopernicusFMDataset(
        samples=dummy_samples,
        input_dims={'x': 16, 'y': 16},
        input_overlap={'x': 0, 'y': 0},
        mode=mode,
        verify_fn='basic',
        batch_size_gen=batch_size,
        augmentation=None,
        filter_thres=0.05
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
    assert callable(ds.verify_fn)
    assert isinstance(ds.input_dims, dict)
    assert isinstance(ds.input_overlap, dict)
    assert isinstance(ds.filter_thres, float)

def test_invalid_verify_fn_raises(dummy_samples):
    """Invalid verify_fn must raise ValueError."""
    with pytest.raises(ValueError):
        CopernicusFMDataset(
            samples=dummy_samples,
            input_dims={"x": 16, "y": 16},
            input_overlap=None,
            mode="train",
            verify_fn="unknown"
        )

def test_basic_filter_threshold(ds):
    """Patch with zeros and NaNs should be filtered out properly."""
    ds, _ = ds
    patch = torch.tensor([[0.0, float("nan")], [1.0, 1.0]])
    assert not ds.basic_filter(patch, threshold=0.2)
    assert ds.basic_filter(torch.ones((2,2)), threshold=0.2)
    assert not ds.basic_filter(torch.tensor([]), threshold=0.2)

@patch("modulargeofm.datamodules.copernicusfm_datamodule.BatchGenerator")
def test_create_batch_generator(mock_bg, ds):
    """Check that create_batch_generator calls BatchGenerator correctly."""
    ds, _ = ds
    dummy_subset = MagicMock()
    ds.create_batch_generator(dummy_subset)
    mock_bg.assert_called_once_with(dummy_subset,
                                    input_dims=ds.input_dims,
                                    input_overlap=ds.input_overlap)
    

@patch("modulargeofm.datamodules.copernicusfm_datamodule.Normalize")
@patch("modulargeofm.datamodules.copernicusfm_datamodule.CopernicusFMDataset.create_batch_generator")
@patch("modulargeofm.datamodules.copernicusfm_datamodule.xr.open_zarr")
@patch("modulargeofm.datamodules.copernicusfm_datamodule.get_worker_info")
def test_iter_yields_batches(mock_get_worker, 
                             mock_xr_open, 
                             mock_batchgen, 
                             mock_normalize,
                             ds, 
                             worker_info):
    """Test the __iter__ method yields expected dict structure."""
    ds, mode = ds
    coord_x = list(range(ds.input_dims['x']))
    coord_y = list(range(ds.input_dims['y']))
    num_channels = 3
    batch_size = ds.batch_size_gen
    
    # mock workers
    mock_get_worker.return_value = worker_info

    # ------- create Mock xarray dataset ---------
    mock_xr = MagicMock()
    mock_xr.dims = ds.input_dims
    mock_xr.expand_dims.return_value = mock_xr
    mock_xr.__getitem__.return_value = mock_xr

    mock_rio = MagicMock()
    mock_rio.resolution.return_value = (10.0, 10.0)
    mock_xr.rio = mock_rio
    mock_xr_open.return_value = mock_xr


    # --- Create patches ---
    # accepted patch 
    accept_patch = MagicMock()
    accept_x_central = MagicMock()
    accept_y_central = MagicMock()
    accept_x_central.values.mean.return_value = ds.input_dims['x']/2
    accept_y_central.values.mean.return_value = ds.input_dims['y']/2
    accept_patch.rio.crs.to_epsg.return_value = 4326
    accept_patch.__getitem__.side_effect = lambda key: accept_x_central if key == 'x' else accept_y_central if key == 'y' else None

    accept_array = MagicMock()
    batch_size = ds.batch_size_gen
    accepted_array_predict = torch.randint(1,5, (num_channels,  ds.input_dims['y'], ds.input_dims['x'])).detach().cpu().numpy()
    accepted_array_non_predict = torch.randint(1,5, (num_channels+1,  ds.input_dims['y'], ds.input_dims['x'])).detach().cpu().numpy()
    accept_array.values.squeeze.return_value = accepted_array_predict if mode == 'predict' else accepted_array_non_predict
    accept_patch.to_array.return_value = accept_array

    accept_x_coords = MagicMock()
    accept_x_coords.values = coord_x

    accept_y_coords = MagicMock()
    accept_y_coords.values = coord_y

    accept_coords = MagicMock()
    accept_coords.__getitem__.side_effect = lambda key: accept_x_coords if key == 'x' else accept_y_coords if key == 'y' else None
    accept_patch.coords = accept_coords 


    # rejected patch 
    reject_patch = MagicMock()

    reject_x_central = MagicMock()
    reject_y_central = MagicMock()
    reject_x_central.values.mean.return_value = ds.input_dims['x']/2
    reject_y_central.values.mean.return_value = ds.input_dims['y']/2
    reject_patch.rio.crs.to_epsg.return_value = 4326
    reject_patch.__getitem__.side_effect = lambda key: reject_x_central if key == 'x' else reject_y_central if key == 'y' else None

    reject_array = MagicMock()
    null_array_predict_predict = torch.zeros(num_channels, ds.input_dims['y'], ds.input_dims['x']).detach().cpu().numpy()
    null_array_predict_non_predict = torch.zeros(num_channels+1, ds.input_dims['y'], ds.input_dims['x']).detach().cpu().numpy()
    reject_array.values.squeeze.return_value = null_array_predict_predict if mode == 'predict' else null_array_predict_non_predict
    reject_patch.to_array.return_value = reject_array

    reject_x_coords = MagicMock()
    reject_x_coords.values = coord_x

    reject_y_coords = MagicMock()
    reject_y_coords.values = coord_y
    
    reject_coords = MagicMock()
    reject_coords.__getitem__.side_effect = lambda key: reject_x_coords if key == 'x' else reject_y_coords if key == 'y' else None
    reject_patch.coords = reject_coords 
     
    # bad patch triggers Exception
    bad_patch = MagicMock()
    bad_patch.to_array.side_effect = Exception("Simulated patch error")
   
    # BatchGenerator yields: good patch first, then bad patch
    mock_batchgen.return_value = [accept_patch, reject_patch, bad_patch]

    # mock normalise
    mock_normalize_instance = MagicMock()
    mock_normalize_instance.side_effect = lambda x: x  # identity function
    mock_normalize.return_value = mock_normalize_instance

    # --- Act ---
    outputs = next(iter(ds))

    
    # --- Assertions and test on different mode ---
    
    # test the returned outputs
    assert isinstance(outputs, dict) 

    # test the returned keys and types
    assert "x" in outputs
    assert "y" in outputs
    assert "meta_info" in outputs
    assert "wave_list" in outputs
    assert "bandwidth" in outputs
    assert "language_embed" in outputs
    assert "input_mode" in outputs
    assert "kernel_size" in outputs
    assert isinstance(outputs["x"], torch.Tensor)
    assert isinstance(outputs["y"], list)

    # test the the filtered batch size
    assert len(outputs["x"]) == 2 if mode == 'predict' and batch_size > 1 else 1  # filter patch in different modes
    assert len(outputs["y"]) == 2 if mode == 'predict' and batch_size > 1 else 1  # filter patch in different modes
    
    # test the shape 
    assert outputs["x"][0].size() == (num_channels, ds.input_dims['y'], ds.input_dims['x']) if mode == 'predict' else (num_channels-1, ds.input_dims['y'], ds.input_dims['x'])
    assert len(outputs["y"][0]) == 3 if mode == 'predict' else True
    assert outputs["y"][0].size() == (ds.input_dims['y'], ds.input_dims['x']) if mode != 'predict' else True
     

# ================================= test data module =================================

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
        dm =  CopernicusFMDataModule(
            zarr_dirs=['./dummy_zarr_dir1/', './dummy_zarr_dir2/'],
            data_kinds=['S2_xz', 'DEM_yz'],
            metadata_path='./dummy_metadata.yaml',
            input_dims={'x': 64, 'y': 64},
            input_overlap={'x': 16, 'y': 16},
            verify_fn='basic',
            augmentation=None,
            batch_size_gen=2,
            num_workers=0,
            filter_thres=0.01,
        )
    return dm
         
         
@patch("builtins.open", new_callable=mock_open, read_data=yaml.dump(META))
def test_copernicusfm_datamodule_init(mock_open_func):
    """Test datamodule initialization."""
    
    metadata_path = './dummy_metadata.yaml'
    dm =  CopernicusFMDataModule(
        zarr_dirs=['./dummy_zarr_dir1/', './dummy_zarr_dir2/',],
        data_kinds=['S2_xz', 'DEM_yz'],
        metadata_path=metadata_path,
        input_dims={'x': 64, 'y': 64},
        input_overlap={'x': 16, 'y': 16},
        verify_fn='basic',
        augmentation=None,
        batch_size_gen=2,
        num_workers=0,
        filter_thres=0.01,
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
    assert isinstance(dm.batch_size_gen, int) and dm.batch_size_gen == 2
    assert isinstance(dm.num_workers, int) and dm.num_workers == 0
    assert isinstance(dm.split_ratio, float) and dm.split_ratio == 0.8
    assert isinstance(dm.filter_thres, float) and dm.filter_thres == 0.01

def test_construct_samples(datamodule):
    zarr_paths = ['dummy1.zaar', 'dummy3.zaar', 'dummyZ.zaar']
    kinds = ['kind1', 'kind1', 'kind2']
    samples = datamodule.construct_samples(zarr_paths, kinds)

    assert isinstance(samples, dict)
    assert all(isinstance(samples[key], dict) for key in samples.keys())
    assert list(samples['dummy1'].keys()) == ['pixels_path', 'wavelist', 'bandwidth', 'bandnames', 'mean', 'std', 'input_mode', 'kernel_size']
    assert list(samples['dummyZ'].keys()) == ['pixels_path', 'meta_info', 'language_embed', 'input_mode', 'kernel_size']

@patch("modulargeofm.datamodules.copernicusfm_datamodule.train_test_split")
@patch("modulargeofm.datamodules.copernicusfm_datamodule.CopernicusFMDataModule.construct_samples")
@patch("modulargeofm.datamodules.copernicusfm_datamodule.glob.glob")
@patch("modulargeofm.datamodules.copernicusfm_datamodule.CopernicusFMDataset")
@pytest.mark.parametrize("stage", ["fit", "test", "predict"])
def test_setup_stages(mock_dataset, 
                      mock_glob, 
                      mock_construct_samples, 
                      mock_traintest_split, 
                      stage, 
                      datamodule
                      ):
    mock_dataset.return_value =  "CopernicusFM_Dataset"
    mock_glob.return_value = MagicMock()
    mock_traintest_split.return_value = ('train_path', 'val_path', 'train_kind', 'val_kind')
    mock_construct_samples.return_value = MagicMock()

    datamodule.setup(stage=stage)

    attrs = {"fit": ["train_ds", "val_ds"], "test": ["test_ds"], "predict": ["pred_ds"]}
    assert all(hasattr(datamodule, attr) for attr in attrs[stage]) 
    assert all(getattr(datamodule, attr) == "CopernicusFM_Dataset" for attr in attrs[stage])  



