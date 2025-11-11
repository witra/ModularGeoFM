import torch
import torch.nn.functional as F
import kornia.morphology as morph

def to_one_hot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert integer class labels to one-hot encoding.

    Parameters
    ----------
    x : torch.Tensor
        Integer tensor of shape [B, H, W] representing class labels.
    num_classes : int
        Number of classes. For binary segmentation, set num_classes=1.

    Returns
    -------
    torch.Tensor
        One-hot encoded tensor of shape [B, C, H, W], where C=num_classes.
        For binary case (num_classes=1), returns a float tensor with shape
        [B, 1, H, W].
    
    Notes
    -----
    - Input tensor should contain integer class labels in [0, num_classes-1].
    - Values are converted to float.
    """
    if num_classes != 1:
         return F.one_hot(x.long(), num_classes).permute(0, 3, 1, 2).float()
    if x.dim() == 3:  # [B,H,W] # Binary casae
         x = x.unsqueeze(1)
    return x.float()


def boundary_map(mask, kernel_size=3):
     """
    Compute a boundary map from a binary segmentation mask using morphological gradient.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask tensor of shape [B, C, H, W], where 1 indicates foreground.
    kernel_size : int, default=3
        Size of the square structuring element used for dilation and erosion.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as `mask` representing the boundary map.
        Boundary pixels have value 1, non-boundary pixels 0.

    Notes
    -----
    - Uses Kornia morphology operations (dilation - erosion).
    - Fully vectorized for batch and channel dimensions.
    - The output is clamped between 0 and 1.
    """
     kernel = torch.ones(kernel_size, kernel_size, device=mask.device)
     dilated = morph.dilation(mask, kernel)
     erosion = morph.erosion(mask, kernel)
     return (dilated - erosion).clamp(min=0, max=1)