import torch
from modulargeofm.utils.shared import to_one_hot, boundary_map
from torchmetrics import Metric

def tversky_index(pred_logits, target, mask=None, num_classes=1, reduction='mean', alpha=0.7, beta=0.3, eps=1e-6):
    """
    Compute the Tversky Index for binary or multiclass segmentation.

    Parameters
    ----------
    pred_logits : torch.Tensor
        Logits output from the model, shape [B, C, H, W] for multiclass or [B, 1, H, W] for binary.
    target : torch.Tensor
        Ground truth labels, shape [B, H, W].
    mask : torch.Tensor, optional
        Optional mask to select regions for metric computation, shape [B, H, W] or [B, 1, H, W].
    num_classes : int, default=1
        Number of classes. 1 for binary segmentation.
    reduction : {'mean', 'sum', None}, default='mean'
        Specifies the reduction to apply over the batch:
        - 'mean' : average over batch,
        - 'sum' : sum over batch,
        - None : return per-sample metric.
    alpha : float, default=0.7
        Weight for false positives in Tversky calculation.
    beta : float, default=0.3
        Weight for false negatives in Tversky calculation.
    eps : float, default=1e-6
        Small value to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Tversky Index. Scalar if reduction is 'mean' or 'sum', else per-sample tensor.
    """
    p = torch.sigmoid(pred_logits) if num_classes == 1 else torch.softmax(pred_logits, dim=1)
    p = (p > 0.5).float()
    target = to_one_hot(target, num_classes)

    if mask is not None and mask.ndim < target.ndim:
        mask = mask.unsqueeze(1)  # [B,1,H,W]

    ones = torch.ones_like(p)

    if mask is not None:
        TP = (p * target * mask).sum(dim=[2,3])
        FP = (p * (ones - target) * mask).sum(dim=[2,3])
        FN = ((ones - p) * target * mask).sum(dim=[2,3])
    else:
        TP = (p * target).sum(dim=[2,3])
        FP = (p * (ones - target)).sum(dim=[2,3])
        FN = ((ones - p) * target).sum(dim=[2,3])

    tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    ti = tversky.mean(dim=1)  # average across classes

    if reduction == 'mean':
        return ti.mean()
    elif reduction == 'sum':
        return ti.sum()
    else:
        return ti  # per-sample
    
def dice_coefficient(pred_logits, target, num_classes=1, reduction='mean', eps=1e-6):
    """
    Compute Dice coefficient for binary or multiclass segmentation.

    Dice is equivalent to Tversky index with alpha=0.5 and beta=0.5.

    Parameters
    ----------
    pred_logits : torch.Tensor
        Logits output from the model.
    target : torch.Tensor
        Ground truth labels.
    num_classes : int, default=1
        Number of classes.
    reduction : {'mean', 'sum', None}, default='mean'
        Specifies reduction over batch.
    eps : float, default=1e-6
        Small value to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Dice coefficient. Scalar if reduction is 'mean' or 'sum', else per-sample tensor.
    """
    return tversky_index(pred_logits, target, num_classes=num_classes, alpha=0.5, beta=0.5,  reduction=reduction, eps=eps)


def boundary_iou(pred_logits, target, mask=None, num_classes=1, kernel_size=3, reduction='mean', eps=1e-6):
    """
    Compute Boundary Intersection-over-Union (IoU) for segmentation masks.

    The boundary is computed using a morphological gradient: dilation - erosion.

    Parameters
    ----------
    pred_logits : torch.Tensor
        Logits from the model, shape [B, C, H, W] for multiclass or [B, 1, H, W] for binary.
    target : torch.Tensor
        Ground truth labels, shape [B, H, W].
    mask : torch.Tensor, optional
        Optional mask to select regions for boundary IoU computation, shape [B, H, W] or [B,1,H,W].
    num_classes : int, default=1
        Number of classes.
    kernel_size : int, default=3
        Size of the structuring element for boundary extraction.
    reduction : {'mean', 'sum', None}, default='mean'
        Reduction over classes:
        - 'mean' : average across classes,
        - 'sum' : sum across classes,
        - None : return per-class tensor.
    eps : float, default=1e-6
        Small value to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Boundary IoU. Scalar if reduction='mean' or 'sum', else per-class tensor of shape [C].
    """
    p = torch.sigmoid(pred_logits) if num_classes == 1 else torch.softmax(pred_logits, dim=1)
    p = (p > 0.5).float()
    target = to_one_hot(target, num_classes).float()
    
    pred_boundary = boundary_map(p, kernel_size=kernel_size)
    target_boundary = boundary_map(target, kernel_size=kernel_size)

    if mask is not None:
        if mask.ndim < target.ndim:
            mask = mask.unsqueeze(1)  # [B,1,H,W]
        pred_boundary = pred_boundary * mask
        target_boundary = target_boundary * mask
  
    intersection = (pred_boundary * target_boundary).sum(dim=[0, 2, 3])
    union = ((pred_boundary + target_boundary)>0).float().sum(dim=[0, 2, 3])
    iou = (intersection) / (union + eps) # per class metric [C]
    if reduction == 'mean':
        return iou.mean()
    elif reduction == 'sum':
        return iou.sum()
    else:
        return iou

class SegmentationMetrics(Metric):
    """
    TorchMetrics Metric class to compute epoch-wise segmentation metrics:
    Tversky Index, Dice Coefficient, and Boundary IoU.

    This class accumulates metrics per batch and computes the mean at the end of the epoch.
    Supports optional masking to evaluate only specific regions of the input.

    Parameters
    ----------
    num_classes : int, default=1
        Number of segmentation classes.
    tversky_alpha : float, default=0.7
        Alpha parameter for Tversky index (controls false positive weight).
    tversky_beta : float, default=0.3
        Beta parameter for Tversky index (controls false negative weight).

    Attributes
    ----------
    tversky_sum : torch.Tensor
        Accumulated sum of Tversky scores.
    dice_sum : torch.Tensor
        Accumulated sum of Dice scores.
    biou_sum : torch.Tensor
        Accumulated sum of Boundary IoUs.
    n_samples : torch.Tensor
        Number of batches accumulated.

    Methods
    -------
    update(pred_logits, target, mask=None)
        Update internal state with metrics from a batch.
    compute()
        Compute mean metrics over all accumulated batches.
    """
    full_state_update = False

    def __init__(self, num_classes=1, tversky_alpha=0.7, tversky_beta=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        self.add_state("tversky_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("dice_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("biou_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0.), dist_reduce_fx="sum")
    
    def update(self, pred_logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        Update metric state with a new batch.

        Parameters
        ----------
        pred_logits : torch.Tensor
            Model logits for the batch.
        target : torch.Tensor
            Ground truth labels for the batch.
        mask : torch.Tensor, optional
            Optional mask to select pixels/regions for metric computation.
        """
        tversky = tversky_index(pred_logits, target, mask=mask, alpha=self.tversky_alpha,
                                beta=self.tversky_beta, num_classes=self.num_classes, reduction='mean')
        dice = dice_coefficient(pred_logits, target, num_classes=self.num_classes, reduction='mean')
        biou = boundary_iou(pred_logits, target, mask=mask, num_classes=self.num_classes, reduction='mean')

        self.tversky_sum += tversky
        self.dice_sum += dice
        self.biou_sum += biou
        self.n_samples += 1
    
    def compute(self):
        """
        Compute mean metrics over all accumulated batches.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'tversky_index' : mean Tversky Index
            - 'dice_coefficient' : mean Dice Coefficient
            - 'boundary_iou' : mean Boundary IoU
        """
        if self.n_samples == 0:
            return {
                "tversky_index": torch.tensor(0., device=self.tversky_sum.device),
                "dice_coefficient": torch.tensor(0., device=self.dice_sum.device),
                "boundary_iou": torch.tensor(0., device=self.biou_sum.device),
            }
        return {
            "tversky_index": self.tversky_sum / self.n_samples,
            "dice_coefficient": self.dice_sum / self.n_samples,
            "boundary_iou": self.biou_sum / self.n_samples,
        }
