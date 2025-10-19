from math import e
import pytest
import torch
from modulargeofm.backbones.decoders.heads import SegmentationHead

@pytest.fixture
def seghead():
    return SegmentationHead(in_channels=12, out_channels=1, upscale_factor=2)

def test_init_segmentation_head(seghead):
    assert isinstance(seghead.conv, torch.nn.Conv2d)
    assert seghead.conv.in_channels == 12
    assert seghead.conv.out_channels == 1
    assert seghead.upscale_factor == 2
def test_forward_segmentation_head(seghead):
    batch_size = 2
    height, width = 16, 16
    input_tensor = torch.randn(batch_size, 12, height, width)

    output = seghead(input_tensor)

    expected_height = height * seghead.upscale_factor
    expected_width = width * seghead.upscale_factor

    print(expected_height, expected_width)
    assert output.shape == (batch_size, 1, expected_height, expected_width)