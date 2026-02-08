import numpy as np
import pytest
import torch
import xarray as xr
from unittest.mock import MagicMock, patch
from modulargeofm.datamodules.clay_datamodule import ClayIterableDataset
from modulargeofm.datamodules.shared import create_batch_generator

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
    ds = ClayIterableDataset(
        samples=dummy_samples,
        input_dims={'x': 16, 'y': 16},
        input_overlap={'x': 0, 'y': 0},
        mode=mode,
        verify_x_fn='default',
        verify_y_fn='default',
        batch_size=batch_size,
        augmentation=None,
        filter_thres=0.05
    )
    return ds , mode

@patch("modulargeofm.datamodules.shared.BatchGenerator")
def test_create_batch_generator(mock_bg, ds):
    """Check that create_batch_generator calls BatchGenerator correctly."""
    ds, _ = ds
    dummy_subset = MagicMock()
    create_batch_generator(dummy_subset, ds.input_dims, ds.input_overlap)
    mock_bg.assert_called_once_with(dummy_subset,
                                    input_dims=ds.input_dims,
                                    input_overlap=ds.input_overlap)
def test_default_xfilter_threshold(ds):
    """Patch with zeros and NaNs should be filtered out properly."""
    ds, _ = ds
    patch = torch.tensor([[0.0, float("nan")], [1.0, 1.0]])
    assert torch.equal(ds.verify_x_fn(patch, threshold=0.2), torch.tensor([False, True]))
    assert torch.equal(ds.verify_x_fn(torch.ones((2,2)), threshold=0.2), torch.tensor([True, True])) 

def test_default_yfilter_threshold(ds):
    """Patch with zeros and NaNs should be filtered out properly."""
    ds, _ = ds
    patch = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    assert torch.equal(ds.verify_y_fn(patch), torch.tensor([False, True]))
    assert torch.equal(ds.verify_y_fn(torch.ones((2,2))), torch.tensor([False, False])) 