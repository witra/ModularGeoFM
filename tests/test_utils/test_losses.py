import pytest
import torch
from unittest.mock import patch, MagicMock
from modulargeofm.utils.losses import tversky_loss, boundary_loss, boundary_iou_loss, CombinedSegLoss

def manual_tversky(TP, FP, FN, alpha=0.7, beta=0.3, eps=1e-6):
    return 1. - ((TP) / (TP + alpha * FP + beta * FN + eps))

@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_tversky_index_shape_dtype(reduction):
    """Check output type and shape for different reductions."""
    B, H, W = 2, 4, 4
    pred = torch.randn(B, 1, H, W)
    target = torch.randint(0, 2, (B, H, W))
    out = tversky_loss(pred, target, reduction=reduction)

    assert torch.is_tensor(out)
    assert out.dtype == torch.float32
    if reduction == "none":
        assert out.shape == (B,)
    else:
        assert out.ndim == 0

def test_tversky_index_perfect_prediction():
    """Perfect match → Tversky ≈ 1."""
    B, H, W = 1, 4, 4
    pred = torch.ones(B, 1, H, W) * 10
    target = torch.ones(B, H, W)
    loss = tversky_loss(pred, target)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)

def test_tversky_index_all_wrong():
    """Completely wrong prediction → low Tversky value."""
    B, H, W = 1, 4, 4
    pred = torch.ones(B, 1, H, W) * -10  
    target = torch.ones(B, H, W)
    loss = tversky_loss(pred, target)
    # All false negatives: TP=0, FP=0, FN=16
    expected = manual_tversky(TP=0, FP=0, FN=16)
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-3)

def test_tversky_index_half_overlap():
    """Half correct predictions → test numerical correctness."""
    B, H, W = 1, 4, 4
    pred = torch.zeros(B, 1, H, W)
    pred[:, :, :2, :] = 10  # top half predicted positive
    pred[:, :, 2:, :] = -10  # bottom half predicted positive
    target = torch.zeros(B, H, W)
    target[:, :2, :] = 1  # top half positive (TP=8)
    target[:, 2:, :] = 1  # bottom half missed (FN=8)
    expected = manual_tversky(8, 0, 8)
    loss = tversky_loss(pred, target)
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)

def test_tversky_index_with_mask():
    """Masked areas should be ignored in calculation."""
    B, H, W = 1, 4, 4
    pred = torch.ones(B, 1, H, W) * 10
    target = torch.ones(B, H, W)
    mask = torch.zeros(B, H, W)
    loss = tversky_loss(pred, target, mask=mask)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-4)

def test_tversky_index_multiclass_correctness():
    """Check multiclass case gives plausible score."""
    B, C, H, W = 1, 3, 6, 6
    pred = torch.ones(B, C, H, W) * -10
    pred[:, 0, :, :2] = 10
    pred[:, 1, :, 2:4] = 10
    pred[:, 2, :, 4:] = 10
    target = torch.zeros(B, H, W, dtype=torch.long)
    target[:, :, :2] = 0
    target[:, :, 2:4] = 1
    target[:, :, 4:] = 2
    loss = tversky_loss(pred, target, num_classes=C, reduction="mean")
    assert loss <= 0.001  # expect high overlap

def make_mock_distances(B, C, H, W, value_fg=1.0, value_bg=2.0):
    """Create controlled distance maps for foreground and background."""
    dist_fg = torch.ones(B, C, H, W) * value_fg       # foreground distance = 1
    dist_bg = torch.ones(B, C, H, W) * value_bg       # background distance = 2
    return dist_fg, dist_bg


@patch("modulargeofm.utils.losses.l1_distance_transform")
def test_boundary_loss_mock_numerical_correctness(mock_dist):
    """
    Test the core math:
       loss = α * (p * d_fg / sum(d_fg))
            + β * ((1-p) * d_bg / sum(d_bg))
    using fully controlled distance maps.
    """

    B, C, H, W = 1, 1, 4, 4

    # Mock distance maps returned in sequence:
    #   dist_fg = l1_distance_transform(target)
    #   dist_bg = l1_distance_transform(1 - target)
    dist_fg, dist_bg = make_mock_distances(B, C, H, W)
    mock_dist.side_effect = [dist_fg, dist_bg]

    # Prediction: confidently foreground everywhere
    pred_logits = torch.ones(B, C, H, W) * 10
    p = torch.ones(B, C, H, W)  # after sigmoid

    # Target: foreground everywhere
    target = torch.ones(B, H, W).long()

    # Manual math:
    # FG distances = 1, BG distances = 2 everywhere
    # weighted_fg = sum(p * d_fg) = 16 * 1 = 16
    # weighted_bg = sum((1-p) * d_bg) = 0
    # normalizer_fg = sum(d_fg) = 16 * 1 = 16
    # normalizer_bg = sum(d_bg) = 16 * 2 = 32
    # loss = α*(16/16) + β*(0/32) = α
    alpha, beta = 0.5, 0.5
    expected = alpha

    out = boundary_loss(pred_logits, target, alpha=alpha, beta=beta)
    assert torch.isclose(out, torch.tensor(expected), atol=1e-6)


# -----------------------------------------------------------
# MASK test with mock distances
# -----------------------------------------------------------
@patch("modulargeofm.utils.losses.l1_distance_transform")
def test_boundary_loss_mask_with_mock(mock_dist):
    B, C, H, W = 1, 1, 4, 4

    dist_fg, dist_bg = make_mock_distances(B, C, H, W, value_fg=1.0, value_bg=3.0)
    mock_dist.side_effect = [dist_fg, dist_bg]

    pred_logits = torch.ones(B, C, H, W) * -10  # predict background everywhere
    target = torch.ones(B, H, W).long()         # true foreground

    # Mask excludes everything → weighted numerators become zero
    mask = torch.zeros(B, H, W)
    out = boundary_loss(pred_logits, target, mask=mask)

    # Because all terms are masked, weighted_fg = weighted_bg = 0
    # => loss = α*0/eps + β*0/eps → approx 0
    assert out < 1e-3


@patch("modulargeofm.utils.losses.l1_distance_transform")
def test_boundary_loss_multiclass_mock(mock_dist):
    B, C, H, W = 1, 3, 4, 4

    dist_fg, dist_bg = make_mock_distances(B, C, H, W)
    mock_dist.side_effect = [dist_fg, dist_bg]

    pred_logits = torch.zeros(B, C, H, W)
    pred_logits[:, 1, :, :] = 10  # Predict class 1 everywhere
    target = torch.ones(B, H, W, dtype=torch.long)  # Class 1 everywhere

    out = boundary_loss(pred_logits, target, num_classes=C)
    # distance weights uniform → confident correct prediction → small loss
    assert out <= 0.5
@pytest.mark.parametrize(
    "pred,target,num_classes,min_expected,max_expected,desc",
    [
        # --- Binary segmentation cases (8x8) ---
        (
            torch.tensor([[[
                 10, 10, 10, 10,-10,-10,-10,-10,
                 10, 10, 10, 10,-10,-10,-10,-10,
                 10, 10, 10, 10,-10,-10,-10,-10,
                 10, 10, 10, 10,-10,-10,-10,-10,
                -10,-10,-10,-10,-10,-10,-10,-10,
                -10,-10,-10,-10,-10,-10,-10,-10,
                -10,-10,-10,-10,-10,-10,-10,-10,
                -10,-10,-10,-10,-10,-10,-10,-10
            ]]]).reshape(1,1,8,8).float(),
            torch.tensor([[[
                1,1,1,1,0,0,0,0,
                1,1,1,1,0,0,0,0,
                1,1,1,1,0,0,0,0,
                1,1,1,1,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0
            ]]]).reshape(1,8,8).float(),
            1, 0, 0.1, "Perfect binary match 8x8"
        ),
        (
            torch.tensor([[[
                 10, 10, 10, 10,-10,-10,-10,-10,
                 10, 10, 10,-10,-10,-10,-10,-10,
                 10, 10, 10,-10,-10,-10,-10,-10,
                 10, 10, 10,-10,-10,-10,-10,-10,
                -10,-10,-10,-10,-10,-10,-10,-10,
                -10,-10,-10,-10,-10,-10,-10,-10,
                -10,-10,-10,-10,-10,-10,-10,-10,
                -10,-10,-10,-10,-10,-10,-10,-10
            ]]]).reshape(1,1,8,8).float(),
            torch.tensor([[[
                1,1,1,1,0,0,0,0,
                1,1,1,1,0,0,0,0,
                1,1,1,1,0,0,0,0,
                1,1,1,1,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0
            ]]]).reshape(1,8,8).float(),
            1, 0.2, 0.5, "Partial binary boundary 8x8"
        ),
        (
            torch.tensor([[[
                 10, 10, 10, 10,-10,-10,-10,-10,
                 10, 10, 10, 10,-10,-10,-10,-10,
                 10, 10, 10, 10,-10,-10,-10,-10,
                 10, 10, 10, 10,-10,-10,-10,-10,
                -10,-10,-10,-10,-10,-10,-10,-10,
                -10,-10,-10,-10,-10,-10,-10,-10,
                -10,-10,-10,-10,-10,-10,-10,-10,
                -10,-10,-10,-10,-10,-10,-10,-10
            ]]]).reshape(1,1,8,8).float(),
            torch.tensor([[[
                0,0,0,0,0,0,1,1,
                0,0,0,0,0,0,1,1,
                0,0,0,0,0,0,1,1,
                0,0,0,0,0,0,1,1,
                0,0,0,0,0,0,1,1,
                0,0,0,0,0,0,1,1,
                1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1
            ]]]).reshape(1,8,8).float(),
            1, 0.99, 1, "No overlap 8x8"
        ),

        # --- Multiclass segmentation case (8x8) ---
        (
            torch.tensor([[
                # Class 0
                [[ 10, 10, 10, 10,-10,-10,-10,-10],
                 [ 10, 10, 10, 10,-10,-10,-10,-10],
                 [ 10, 10, 10, 10,-10,-10,-10,-10],
                 [ 10, 10, 10, 10,-10,-10,-10,-10],
                 [-10,-10,-10,-10,-10,-10,-10,-10],
                 [-10,-10,-10,-10,-10,-10,-10,-10],
                 [-10,-10,-10,-10,-10,-10,-10,-10],
                 [-10,-10,-10,-10,-10,-10,-10,-10]],
                # Class 1
                [[-10,-10,-10,-10, 10, 10, 10, 10],
                 [-10,-10,-10,-10, 10, 10, 10, 10],
                 [-10,-10,-10,-10, 10, 10, 10, 10],
                 [-10,-10,-10,-10, 10, 10, 10, 10],
                 [ 10, 10, 10, 10,-10,-10,-10,-10],
                 [ 10, 10, 10, 10,-10,-10,-10,-10],
                 [ 10, 10, 10, 10,-10,-10,-10,-10],
                 [ 10, 10, 10, 10,-10,-10,-10,-10]],
                # Class 2 (background)
                [[-10,-10,-10,-10,-10,-10,-10,-10],
                 [-10,-10,-10,-10,-10,-10,-10,-10],
                 [-10,-10,-10,-10,-10,-10,-10,-10],
                 [-10,-10,-10,-10,-10,-10,-10,-10],
                 [-10,-10,-10,-10, 10, 10, 10, 10],
                 [-10,-10,-10,-10, 10, 10, 10, 10],
                 [-10,-10,-10,-10, 10, 10, 10, 10],
                 [-10,-10,-10,-10, 10, 10, 10, 10]]
            ]]).float(),
            torch.tensor([[
                [0,0,0,0,1,1,1,1],
                [0,0,0,0,1,1,1,1],
                [0,0,0,0,1,1,1,1],
                [0,0,0,0,1,1,1,1],
                [1,1,1,1,0,0,0,2],
                [1,1,1,1,0,0,0,2],
                [1,1,1,1,0,0,0,2],
                [1,1,1,1,0,0,0,2],
            ]]).float(),
            3, 0.0, 1, "Multiclass 8x8 partial overlap"
        ),
    ]
)
def test_boundary_iou_large(pred, target, num_classes, min_expected, max_expected, desc):
    """Tests boundary_iou on larger masks so morphological ops produce nonzero boundaries."""
    mask = torch.tensor([[[
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0
            ]]]).reshape(1,8,8).bool()
    value_mean = boundary_iou_loss(pred, target, num_classes=num_classes, reduction='mean')
    value_sum = boundary_iou_loss(pred, target, num_classes=num_classes, reduction='sum')
    value_tensor = boundary_iou_loss(pred, target, num_classes=num_classes, reduction='None')
    value_mask = boundary_iou_loss(pred, target, num_classes=num_classes, mask=mask, reduction='mean')
    assert torch.is_tensor(value_mean)
    assert torch.is_tensor(value_sum)
    assert torch.is_tensor(value_tensor)
    assert value_mean.ndim == 0
    assert value_sum.ndim == 0
    assert value_tensor.ndim == 1 # metric per class
    assert min_expected <= value_mean.item() <= max_expected, f"{desc}: got {value_mean.item():.3f}"
    assert min_expected * pred.shape[1] <= value_sum.item() <= max_expected * pred.shape[1], f"{desc}: got {value_sum.item():.3f}"
    assert value_mask.item() == 1.0, f"{desc}: got {value_mask.item():.3f}"


@patch("modulargeofm.utils.losses.to_one_hot", return_value=torch.tensor([[[1., 0.]]]))
@patch("modulargeofm.utils.losses.boundary_iou_loss", return_value=torch.tensor(3.0))
@patch("modulargeofm.utils.losses.tversky_loss", return_value=torch.tensor(2.0))
@patch("torch.nn.functional.binary_cross_entropy_with_logits", return_value=torch.tensor(10.0))
def test_combined_loss_weighted_sum(mock_bce, mock_tversky, mock_boundary, mock_onehot):

    loss_fn = CombinedSegLoss(
        num_classes=1,
        ce_weight=0.4,
        tversky_weight=0.3,
        boundary_weight=0.3,
    )

    pred = torch.randn(1, 1, 1, 2)
    target = torch.tensor([[0, 1]])

    out = loss_fn(pred, target)

    # Expected:
    #   ce = 10
    #   tversky = 2
    #   boundary = 3
    # loss = 0.4*10 + 0.3*2 + 0.3*3 = 4 + 0.6 + 0.9 = 5.5
    expected = 5.5
    assert torch.isclose(out, torch.tensor(expected), atol=1e-6)

@patch("modulargeofm.utils.losses.to_one_hot", return_value=torch.tensor([[[1., 0.]]]))
@patch("modulargeofm.utils.losses.tversky_loss", return_value=torch.tensor(0.0))
@patch("modulargeofm.utils.losses.boundary_iou_loss", return_value=torch.tensor(0.0))
def test_combined_loss_mask_normalization(mock_boundary, mock_tversky, mock_onehot):

    loss_fn = CombinedSegLoss(num_classes=1, ce_weight=1.0, tversky_weight=0.0, boundary_weight=0.0)

    # Force CE to output elementwise tensor so mask affects it
    def fake_bce(pred_logits, target_onehot, reduction='none'):
        return torch.tensor([[[4.0, 6.0]]])  # CE per-pixel
    
    with patch("torch.nn.functional.binary_cross_entropy_with_logits", fake_bce):

        pred = torch.randn(1, 1, 1, 2)
        target = torch.tensor([[0, 1]])
        mask = torch.tensor([[0.0, 1.0]])  # only second pixel is valid

        out = loss_fn(pred, target, mask=mask)

        # Expected masked CE:
        # numerator = 6.0
        # denom = 1
        # CE = 6
        assert torch.isclose(out, torch.tensor(6.0), atol=1e-6)

@patch("modulargeofm.utils.losses.to_one_hot", return_value=torch.randn(1, 3, 2, 2))
@patch("modulargeofm.utils.losses.tversky_loss", return_value=torch.tensor(1.0))
@patch("modulargeofm.utils.losses.boundary_iou_loss", return_value=torch.tensor(1.0))
def test_combined_loss_multiclass_uses_cross_entropy(mock_boundary, mock_tversky, mock_onehot):
    # fake cross_entropy: return per-pixel value 5
    def fake_ce(logits, target, reduction='none'):
        return torch.ones(1, 2, 2) * 5.0

    with patch("torch.nn.functional.cross_entropy", fake_ce):
        loss_fn = CombinedSegLoss(num_classes=3, ce_weight=1.0, tversky_weight=0.0, boundary_weight=0.0)

        pred = torch.randn(1, 3, 2, 2)
        target = torch.randint(0, 3, (1, 2, 2))

        out = loss_fn(pred, target)

        # reduction='mean' → mean of all 4 pixels → 5
        assert torch.isclose(out, torch.tensor(5.0), atol=1e-6)
