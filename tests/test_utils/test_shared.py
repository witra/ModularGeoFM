import pytest
import torch
from modulargeofm.utils.shared import to_one_hot, boundary_map

@pytest.mark.parametrize(
        "x, num_classes, expected_shape",
        [
        # --- Multi-class ---
        (torch.tensor([[[0, 1], [2, 1]]]), 3, (1, 3, 2, 2)),

        # --- Binary case: num_classes=1 ---
        (torch.tensor([[[0, 1], [1, 0]]]), 1, (1, 1, 2, 2)),

        # --- Already has channel dimension (simulated by squeeze/unsqueeze) ---
        (torch.tensor([[[0, 1], [1, 0]]]), 1, (1, 1, 2, 2)),

        # --- Larger batch, multi-class ---
        (torch.randint(0, 4, (5, 4, 4)), 4, (5, 4, 4, 4))
    ],
)
def test_to_one_hot(x, num_classes, expected_shape):
    
    out = to_one_hot(x, num_classes)
    assert out.dtype == torch.float32
    assert out.shape == expected_shape

def test_boundary_map_expected_output():
    """Verify numerical correctness on a simple shape."""
    mask = torch.tensor(
        [[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,0],
           [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,1],
           [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,1],
           [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,1],
           [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,0],]]],
        dtype=torch.float32,
    )
    result = boundary_map(mask, kernel_size=3).round()  # boundaries should be crisp

    # Expected boundary pattern (1-pixel border)
    expected = torch.tensor(
        [[[[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0,0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1],
           [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0,0],
           [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0,0],
           [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,0],]]],
        dtype=torch.float32,
    )

    print('results \n', result)
    assert torch.allclose(result, expected, atol=1e-3)

    
@pytest.mark.parametrize(
    "mask,kernel_size,expected_nonzero",
    [
        # Empty mask → no boundary
        (torch.zeros(1, 1, 8, 8), 3, False),

        # Full mask → no boundary
        (torch.ones(1, 1, 8, 8), 3, False),

        # Binary square → has boundary
        ((torch.zeros(1, 1, 8, 8).scatter_(2, torch.arange(2, 6).view(1, 1, -1, 1).repeat(1, 1, 1, 4), 1.0)), 3, True),

        # Soft gradient mask → has boundary
        (torch.linspace(0, 1, 64).view(1, 1, 8, 8), 3, True),
    ],
)
def test_boundary_map_basic(mask, kernel_size, expected_nonzero):
    result = boundary_map(mask.float(), kernel_size=kernel_size)

    # Check shape and type
    assert result.shape == mask.shape
    assert result.dtype == torch.float32
    assert torch.all((result >= 0) & (result <= 1))

    # Boundary presence check
    nonzero = result.sum() > 0
    assert nonzero == expected_nonzero


def test_batch_and_determinism():
    """Check batch/multi-channel handling and deterministic output."""
    mask = torch.zeros(2, 3, 8, 8)
    mask[:, :, 3:5, 3:5] = 1.0

    out1 = boundary_map(mask, kernel_size=3)
    out2 = boundary_map(mask.clone(), kernel_size=3)

    assert out1.shape == mask.shape
    assert torch.allclose(out1, out2, atol=1e-6)
    assert out1.sum() > 0