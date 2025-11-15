import torch
import torch.nn.functional as F
from distmap import l1_distance_transform
from modulargeofm.utils.shared import to_one_hot, boundary_map

def tversky_loss(pred_logits, target, mask=None, alpha=0.7, beta=0.3, eps=1e-6, reduction='mean', num_classes=1):
    """
    Compute the Tversky loss for binary or multiclass segmentation.

    The Tversky index is a generalization of Dice and Jaccard, defined as:
        TI = TP / (TP + α·FP + β·FN)
    The loss is: 1 − mean(TI).

    Args:
        pred_logits (Tensor): Raw model logits of shape (B, C, H, W).
        target (Tensor): Ground-truth labels of shape (B, H, W) with class indices.
        mask (Tensor, optional): Spatial mask of shape (B, 1, H, W) or (B, H, W).
            Elements outside the mask do not contribute to the loss.
        alpha (float): Weight for false positives (FP).
        beta (float): Weight for false negatives (FN).
        eps (float): Numerical stability constant.
        reduction (str): One of {'mean', 'sum', 'none'} specifying output reduction.
        num_classes (int): Number of segmentation classes.

    Returns:
        Tensor: The Tversky loss. A scalar if reduced, or per-sample values otherwise.

    Notes:
        • For num_classes = 1, sigmoid activation is used; otherwise softmax.
        • Target is internally converted to one-hot format.
        • Supports optional spatial masking for partial supervision.
    """
    p = torch.sigmoid(pred_logits) if num_classes == 1 else torch.softmax(pred_logits, dim=1)
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

    tversky = (TP) / (TP + alpha * FP + beta * FN + eps)
    loss = 1. - tversky.mean(dim=1)  # average across classes

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # per-sample

def boundary_loss(pred_logits, target, mask=None, alpha=0.5, beta=0.5,  eps=1e-16, num_classes=1):
    """
    Compute the boundary loss using foreground and background distance maps.

    Boundary loss penalizes prediction errors proportionally to their
    distance from ground‐truth boundaries:

        L = 1 - (alpha * (Σ(p * D_fg) / Σ(D_fg))
                + beta  * (Σ((1 - p) * D_bg) / Σ(D_bg)))

    This encourages accurate boundary localization and is especially
    effective for highly unbalanced segmentation problems.

    Parameters
    ----------
    pred_logits : torch.Tensor
        Raw logits [B,C,H,W]. Sigmoid (binary) or softmax (multiclass)
        is applied internally.
    target : torch.Tensor
        Ground‐truth labels [B,H,W].
    mask : torch.Tensor, optional
        Optional mask [B,1,H,W] to restrict the loss computation.
    alpha : float
        Weight for foreground distances.
    beta : float
        Weight for background distances.
    eps : float
        Numerical stability constant.
    num_classes : int
        Number of classes.

    References
    ----------
    Kervadec et al., *"Boundary loss for highly unbalanced segmentation"*,
    Medical Image Analysis, 2021. (original MIDL 2019 paper)

    Returns
    -------
    torch.Tensor
        Scalar boundary loss.
    """

    p = torch.sigmoid(pred_logits) if num_classes == 1 else torch.softmax(pred_logits, dim=1)
    target = to_one_hot(target, num_classes)

    if mask is not None and mask.ndim < p.ndim:
        mask = mask.unsqueeze(1)

    ones = torch.ones_like(target)
    distmap_fg  = l1_distance_transform(target, ndim=2) # distance from foreground to boundary
    distmap_bg = l1_distance_transform(ones - target, ndim=2)  # distance from background to boundary
   
    if mask is not None:
        weighted_dist_fg = (p * distmap_fg * mask).sum(dim=[2,3])
        weighted_dist_bg = ((ones-p) * distmap_bg * mask).sum(dim=[2,3])
        normalizer_fg = (distmap_fg * mask).sum(dim=[2,3]) + eps
        normalizer_bg = (distmap_bg * mask).sum(dim=[2,3]) + eps
    else:
        weighted_dist_fg = (p * distmap_fg).sum(dim=[2, 3])
        weighted_dist_bg = ((ones-p) * distmap_bg).sum(dim=[2, 3])
        normalizer_fg = distmap_fg.sum(dim=[2, 3]) + eps
        normalizer_bg = distmap_bg.sum(dim=[2, 3]) + eps

    return 1-(alpha * weighted_dist_fg / normalizer_fg + beta * weighted_dist_bg / normalizer_bg).mean()

def boundary_iou_loss(pred_logits, target, mask=None, num_classes=1, kernel_size=3, reduction='mean', eps=1e-6):
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
    biou = (intersection) / (union + eps) # per class metric [C]
    loss = 1. - biou

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

class CombinedSegLoss:
    """
    Combined segmentation loss integrating cross-entropy, Tversky, and boundary losses.

    This loss function is designed for semantic segmentation tasks. It combines
    region-based (cross-entropy or BCE), shape-sensitive (Tversky), and boundary-aware
    (boundary IoU) losses to improve both overall overlap and boundary accuracy.

    The combined loss is computed as:

        Loss = w_ce * CE(pred, target)
             + w_tversky * TverskyLoss(pred, target)
             + w_boundary * BoundaryIoULoss(pred, target)

    where weights are specified by the user.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes. Use 1 for binary segmentation.
    ce_weight : float
        Weight of the cross-entropy/BCE loss term.
    tversky_weight : float
        Weight of the Tversky loss term.
    boundary_weight : float
        Weight of the boundary loss term.
    tversky_alpha : float
        Alpha parameter for Tversky loss (FP weight).
    tversky_beta : float
        Beta parameter for Tversky loss (FN weight).
    boundary_alpha : float
        Alpha parameter for foreground term in boundary loss.
    boundary_beta : float
        Beta parameter for background term in boundary loss.

    Methods
    -------
    __call__(pred_logits, target, mask=None, eps=1e-6)
        Compute the combined loss for a batch of predictions and targets.

    Example
    -------
    >>> loss_fn = CombinedSegLoss(num_classes=3, ce_weight=0.4, tversky_weight=0.3, boundary_weight=0.3)
    >>> loss = loss_fn(pred_logits, target)
    """
    def __init__(self,
                 num_classes=1,
                 ce_weight=0.4,
                 tversky_weight=0.3,
                 boundary_weight=0.3,
                 tversky_alpha=0.7,
                 tversky_beta=0.3,
                 boundary_alpha=0.5,
                 boundary_beta=0.5):
        self.num_classes = num_classes
        self.ce_w = ce_weight
        self.tversky_w = tversky_weight
        self.boundary_w = boundary_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.boundary_alpha = boundary_alpha
        self.boundary_beta = boundary_beta

    def __call__(self, pred_logits, target, mask=None, eps=1e-6):
        target_one_hot = to_one_hot(target, self.num_classes)

        if mask is not None and mask.ndim < pred_logits.ndim:
            mask = mask.unsqueeze(1)

        # BCE / CE
        if self.num_classes == 1:
            ce_elementwise = F.binary_cross_entropy_with_logits(pred_logits, target_one_hot, reduction='none')
        else:
            ce_elementwise = F.cross_entropy(pred_logits, target, reduction='none').unsqueeze(1)

        if mask is not None:
            ce = (ce_elementwise * mask).sum() / (mask.sum() + eps)
        else:
            ce = ce_elementwise.mean()

        # Tversky
        tversky = tversky_loss(pred_logits,
                               target,
                               mask=mask,
                               alpha=self.tversky_alpha,
                               beta=self.tversky_beta,
                               eps=eps,
                               reduction='mean',
                               num_classes=self.num_classes)

        # boundary loss
        # boundary = boundary_loss(pred_logits,
        #                           target,
        #                           mask=mask,
        #                           alpha=self.boundary_alpha,
        #                           beta=self.boundary_beta,
        #                           eps=eps,
        #                           num_classes=self.num_classes
        #                               )
        boundary = boundary_iou_loss(pred_logits, 
                                     target, 
                                     mask=mask, 
                                     num_classes=self.num_classes, 
                                     kernel_size=3, 
                                     reduction='mean', 
                                     eps=1e-6)
        return self.ce_w * ce + self.tversky_w * tversky + self.boundary_w * boundary

