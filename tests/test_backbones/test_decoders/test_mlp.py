from math import e
import pytest
import torch
from modulargeofm.backbones.decoders.mlp import SimpleMLPDecoder, ChannelLayerNorm

@pytest.mark.parametrize("norm_layer", ["BatchNorm2d", "LayerNorm"])
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
@pytest.mark.parametrize("out_channels", [8, 24])
def test_simple_mlp_decoder_forward(out_channels, norm_layer, residual, dropout, capsys):
    """Test SimpleMLPDecoder forward pass, residual, and shape."""

    # Configuration
    batch_size = 2
    feature_channels = [8, 8, 8]
    H, W = 4, 4
    features = [torch.randn(batch_size, c, H, W) for c in feature_channels]

    decoder = SimpleMLPDecoder(
        embed_dims=feature_channels,
        out_channels=out_channels,
        num_layers=2,
        activation="ReLU",
        dropout=dropout,
        norm_layer=norm_layer,
        residual=residual
    )

    # Case 1: Correct input
    output = decoder(features)
    assert output.shape == (batch_size, out_channels, H, W)
    assert isinstance(output, torch.Tensor)

    # Case 2: Residual connection applied if conditions met
    if residual and sum(feature_channels) == out_channels:
        # Output should differ from simple concatenation but keep the same shape
        concat = torch.cat(features, dim=1)
        # Residual adds concat to mlp output; they should not be identical
        assert not torch.allclose(output, concat), "Residual not applied correctly"

    # Case 3: Input length mismatch
    with pytest.raises(ValueError):
        decoder(features[:-1])  # Remove one feature to trigger error
    
    captured = capsys.readouterr()
    print("Captured print:", captured.out)  # display in pytest output



@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [4, 8])
@pytest.mark.parametrize("H", [2, 4])
@pytest.mark.parametrize("W", [2, 4])
def test_channel_layer_norm_shape_and_mean(batch_size, channels, H, W):
    """
    Test ChannelLayerNorm preserves input shape and normalizes channels per pixel.
    """
    x = torch.randn(batch_size, channels, H, W)
    norm = ChannelLayerNorm(channels)
    y = norm(x)

    # Check shape is unchanged
    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"

    # Check mean and std per pixel across channels
    # y: (B, C, H, W) → mean/std over C dimension
    mean = y.mean(dim=1)
    std = y.std(dim=1, unbiased=False)

    # Since LayerNorm standardizes channels, mean ~0, std ~1
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-3), f"Mean not zero: {mean}"
    assert torch.allclose(std, torch.ones_like(std), atol=1e-3), f"Std not one: {std}"