import pytest
from unittest.mock import MagicMock
import torch
import modulargeofm.backbones.models.copernicusfm_models as copernicus_mlp
from modulargeofm.backbones.models.copernicusfm_models import CopernicusMLP


@pytest.fixture
def dummy_encoder():
    encoder = MagicMock()
    encoder.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    encoder.return_value = (None, [torch.randn(1, 384, 4, 4)])
    encoder.load_state_dict = MagicMock(return_value="mocked_load_result")
    return encoder



@pytest.fixture
def dummy_decoder():
    decoder = MagicMock()
    decoder.return_value = torch.randn(1, 8, 4, 4)
    return decoder


@pytest.fixture
def dummy_head():
    head = MagicMock()
    head.return_value = torch.randn(1, 3, 16, 16)
    return head


@pytest.fixture
def dummy_encoder_config():
    return {"intermediate_indices": [0, 1, 2]}


@pytest.fixture
def dummy_decoder_config():
    return {"out_channels": 8}


@pytest.fixture
def monkeypatched_env(monkeypatch, dummy_encoder, dummy_decoder, dummy_head):
    """Patch module-level dependencies for isolated CopernicusMLP tests."""
    monkeypatch.setitem(
        copernicus_mlp.__dict__,
        "encoder_dict",
        {
            "vit_small_patch16": lambda **kwargs: dummy_encoder,
            "vit_base_patch16": lambda **kwargs: dummy_encoder,
            "vit_large_patch16": lambda **kwargs: dummy_encoder,
            "vit_huge_patch14": lambda **kwargs: dummy_encoder,
        },
    )
    monkeypatch.setitem(copernicus_mlp.__dict__, "SimpleMLPDecoder", lambda **kwargs: dummy_decoder)
    monkeypatch.setitem(copernicus_mlp.__dict__, "SegmentationHead", lambda **kwargs: dummy_head)

@pytest.mark.parametrize(
    "encoder_name,expected_patch,expected_dim",
    [
        ("vit_small_patch16", 16, 384),
        ("vit_base_patch16", 16, 768),
        ("vit_large_patch16", 16, 1024),
        ("vit_huge_patch14", 16, 1280),
    ],
)
def test_init_valid_encoders(monkeypatched_env, encoder_name, expected_patch, expected_dim, dummy_encoder_config, dummy_decoder_config):
    """Test that valid encoder names create model with correct fields."""
    model = CopernicusMLP(
        encoder_name=encoder_name,
        encoder_config=dummy_encoder_config,
        decoder_config=dummy_decoder_config,
        num_class=3,
    )

    # check attributes
    assert model.patch_size == expected_patch
    assert model.embed_dim == expected_dim
    assert isinstance(model.encoder, MagicMock)
    assert isinstance(model.decoder, MagicMock)
    assert isinstance(model.head, MagicMock)


def test_init_invalid_encoder(monkeypatched_env, dummy_encoder_config, dummy_decoder_config):
    """Test invalid encoder raises ValueError."""
    with pytest.raises(ValueError, match="is not handled yet"):
        CopernicusMLP(
            encoder_name="vit_unknown",
            encoder_config=dummy_encoder_config,
            decoder_config=dummy_decoder_config,
            num_class=3,
        )


def test_freeze_encoder(monkeypatched_env, dummy_encoder_config, dummy_decoder_config):
    """Ensure freeze_encoder sets requires_grad=False."""
    model = CopernicusMLP(
        encoder_name="vit_small_patch16",
        encoder_config=dummy_encoder_config,
        decoder_config=dummy_decoder_config,
        num_class=3,
    )
    for param in model.encoder.parameters():
        assert param.requires_grad is False

def test_load_encoder_weights(monkeypatched_env, dummy_encoder_config, dummy_decoder_config, monkeypatch):
    """Test loading pretrained weights works with and without 'model' key."""
    model = CopernicusMLP(
        encoder_name="vit_small_patch16",
        encoder_config=dummy_encoder_config,
        decoder_config=dummy_decoder_config,
        num_class=3,
    )

    fake_state_dict = {"weight": torch.randn(3, 3)}
    mock_load_state_dict = MagicMock(return_value="mocked_load_result")
    model.encoder.load_state_dict = mock_load_state_dict

    # Patch torch.load to return fake checkpoint
    monkeypatch.setattr("torch.load", lambda *args, **kwargs: {"model": fake_state_dict})

    result = model.load_encoder_weights("dummy_path")
    assert result == "mocked_load_result"
    mock_load_state_dict.assert_called_once_with(fake_state_dict, strict=False)

def test_forward_pass(monkeypatched_env, dummy_encoder_config, dummy_decoder_config):
    """Test that forward returns a tensor of expected shape."""
    model = CopernicusMLP(
        encoder_name="vit_small_patch16",
        encoder_config=dummy_encoder_config,
        decoder_config=dummy_decoder_config,
        num_class=3,
    )

    data_dict = {"x": torch.randn(1, 3, 32, 32)}
    output = model.forward(data_dict)
    assert isinstance(output, torch.Tensor)
