import pytest
import torch
import pytest
from modulargeofm.utils.metrics import tversky_index, dice_coefficient, boundary_iou, SegmentationMetrics

## --- Helper for expected Tversky calculation (manual reference) ---
def manual_tversky(TP, FP, FN, alpha=0.7, beta=0.3, eps=1e-6):
    return (TP + eps) / (TP + alpha * FP + beta * FN + eps)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_tversky_index_shape_dtype(reduction):
    """Check output type and shape for different reductions."""
    B, H, W = 2, 4, 4
    pred = torch.randn(B, 1, H, W)
    target = torch.randint(0, 2, (B, H, W))
    out = tversky_index(pred, target, reduction=reduction)

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
    score = tversky_index(pred, target)
    assert torch.isclose(score, torch.tensor(1.0), atol=1e-4)


def test_tversky_index_all_wrong():
    """Completely wrong prediction → low Tversky value."""
    B, H, W = 1, 4, 4
    pred = torch.ones(B, 1, H, W) * -10  # all zeros after sigmoid>0.5 threshold
    target = torch.ones(B, H, W)
    score = tversky_index(pred, target)
    # All false negatives: TP=0, FP=0, FN=16
    expected = manual_tversky(TP=0, FP=0, FN=16)
    assert torch.isclose(score, torch.tensor(expected), atol=1e-6)


def test_tversky_index_half_overlap():
    """Half correct predictions → test numerical correctness."""
    B, H, W = 1, 4, 4
    pred = torch.zeros(B, 1, H, W)
    pred[:, :, :2, :] = 10  # top half predicted positive
    target = torch.zeros(B, H, W)
    target[:, :2, :] = 1  # top half positive (TP=8)
    target[:, 2:, :] = 1  # bottom half missed (FN=8)
    # TP=8, FP=0, FN=8
    expected = manual_tversky(8, 0, 8)
    score = tversky_index(pred, target)
    assert torch.isclose(score, torch.tensor(expected), atol=1e-6)


def test_tversky_index_with_mask():
    """Masked areas should be ignored in calculation."""
    B, H, W = 1, 4, 4
    pred = torch.ones(B, 1, H, W) * 10
    target = torch.ones(B, H, W)
    mask = torch.zeros(B, H, W)
    score = tversky_index(pred, target, mask=mask)
    # Mask zeros everything → TP=FP=FN=0 → returns 1.0 (by definition with eps)
    assert torch.isclose(score, torch.tensor(1.0), atol=1e-4)


def test_tversky_index_multiclass_correctness():
    """Check multiclass case gives plausible score."""
    B, C, H, W = 1, 3, 4, 4
    pred = torch.zeros(B, C, H, W)
    pred[:, 0, :, :2] = 10
    pred[:, 1, :, 2:] = 10
    target = torch.zeros(B, H, W, dtype=torch.long)
    target[:, :, 2:] = 1
    score = tversky_index(pred, target, num_classes=C, reduction="mean")
    assert 0.8 <= score <= 1.0  # expect high overlap

def test_dice_correctness_and_equivalence():
    B, H, W = 1, 4, 4
    # Perfect match → 1.0
    pred = torch.ones(B, 1, H, W) * 10
    target = torch.ones(B, H, W)
    assert torch.isclose(dice_coefficient(pred, target), torch.tensor(1.0), atol=1e-4)

    # Half overlap: TP=8, FP=0, FN=8 → Dice = 16/(16+8)=0.6667
    pred = torch.zeros(B, 1, H, W); pred[:, :, :2, :] = 10
    target = torch.ones(B, H, W)
    dice = dice_coefficient(pred, target)
    tversky_half = tversky_index(pred, target, alpha=0.5, beta=0.5)
    assert torch.isclose(dice, torch.tensor(0.6667), atol=1e-3)
    assert torch.isclose(dice, tversky_half, atol=1e-8)

    # All wrong → 0.0
    pred = torch.ones(B, 1, H, W) * -10
    target = torch.ones(B, H, W)
    assert torch.isclose(dice_coefficient(pred, target), torch.tensor(0.0), atol=1e-4)


def test_dice_multiclass_behavior():
    B, C, H, W = 1, 3, 4, 4
    pred = torch.zeros(B, C, H, W)
    pred[:, 0, :, :2] = 10; pred[:, 1, :, 2:] = 10
    target = torch.zeros(B, H, W, dtype=torch.long); target[:, :, 2:] = 1
    score = dice_coefficient(pred, target, num_classes=C)
    assert 0.8 <= score <= 1.0

@pytest.mark.parametrize(
    "pred,target,num_classes,min_expected,max_expected,desc",
    [
        # --- Binary segmentation cases (8x8) ---
        (
            torch.tensor([[[
                1,1,1,1,0,0,0,0,
                1,1,1,1,0,0,0,0,
                1,1,1,1,0,0,0,0,
                1,1,1,1,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0
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
            1, 0.99, 1.0, "Perfect binary match 8x8"
        ),
        (
            torch.tensor([[[
                1,1,1,1,0,0,0,0,
                1,1,1,0,0,0,0,0,
                1,1,1,0,0,0,0,0,
                1,1,1,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0
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
            1, 0.4, 0.8, "Partial binary boundary 8x8"
        ),
        (
            torch.tensor([[[
                1,1,1,1,0,0,0,0,
                1,1,1,1,0,0,0,0,
                1,1,1,1,0,0,0,0,
                1,1,1,1,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0
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
            1, 0.0, 0.1, "No overlap 8x8"
        ),

        # --- Multiclass segmentation case (8x8) ---
        (
            torch.tensor([[
                # Class 0
                [[1,1,1,1,0,0,0,0],
                 [1,1,1,1,0,0,0,0],
                 [1,1,1,1,0,0,0,0],
                 [1,1,1,1,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0]],
                # Class 1
                [[0,0,0,0,1,1,1,1],
                 [0,0,0,0,1,1,1,1],
                 [0,0,0,0,1,1,1,1],
                 [0,0,0,0,1,1,1,1],
                 [1,1,1,1,0,0,0,0],
                 [1,1,1,1,0,0,0,0],
                 [1,1,1,1,0,0,0,0],
                 [1,1,1,1,0,0,0,0]],
                # Class 2 (background)
                [[0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,1,1,1,1],
                 [0,0,0,0,1,1,1,1],
                 [0,0,0,0,1,1,1,1],
                 [0,0,0,0,1,1,1,1]]
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
    value_mean = boundary_iou(pred, target, num_classes=num_classes, reduction='mean')
    value_sum = boundary_iou(pred, target, num_classes=num_classes, reduction='sum')
    value_tensor = boundary_iou(pred, target, num_classes=num_classes, reduction='None')
    value_mask = boundary_iou(pred, target, num_classes=num_classes, mask=mask, reduction='mean')
    assert torch.is_tensor(value_mean)
    assert torch.is_tensor(value_sum)
    assert torch.is_tensor(value_tensor)
    assert value_mean.ndim == 0
    assert value_sum.ndim == 0
    assert value_tensor.ndim == 1 # metric per class
    assert min_expected <= value_mean.item() <= max_expected, f"{desc}: got {value_mean.item():.3f}"
    assert min_expected * pred.shape[1] <= value_sum.item() <= max_expected * pred.shape[1], f"{desc}: got {value_sum.item():.3f}"
    assert value_mask.item() == 0.0, f"{desc}: got {value_mask.item():.3f}"

@pytest.mark.parametrize("num_classes", [1, 3])
def test_segmentation_metrics_basic(num_classes):
    batch_size = 2
    height = 4
    width = 4

    # Random logits and integer targets
    logits = torch.rand(batch_size, num_classes, height, width)
    if num_classes == 1:
        targets = torch.randint(0, 2, (batch_size, height, width))
    else:
        targets = torch.randint(0, num_classes, (batch_size, height, width))

    # Optional mask (test masked case)
    mask = torch.randint(0, 2, (batch_size, height, width))

    metric = SegmentationMetrics(num_classes=num_classes)

    # --- update with first batch ---
    metric.update(logits, targets, mask=mask)

    # Compute metrics
    out = metric.compute()
    assert isinstance(out, dict)
    assert "tversky_index" in out
    assert "dice_coefficient" in out
    assert "boundary_iou" in out
    for k, v in out.items():
        assert torch.is_tensor(v)
        assert v.dim() == 0  # scalar

    # --- update with second batch ---
    metric.update(logits, targets, mask=mask)
    out2 = metric.compute()
    # Values should have changed (accumulated)
    for k in out.keys():
        assert out2[k] >= out[k]

    # --- test reset ---
    metric.reset()
    out_reset = metric.compute()
    for v in out_reset.values():
        assert v == 0.0