import torch
import torch.nn.functional as F
from distmap import l1_distance_transform


def to_one_hot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer labels [B,H,W] to one-hot [B,C,H,W]."""
    if num_classes == 1:  # binary case
        if x.dim() == 3:  # [B,H,W]
            x = x.unsqueeze(1)
        return x.float()
    else:  # multiclass
        return F.one_hot(x.long(), num_classes).permute(0, 3, 1, 2).float()

def tversky_loss(pred_logits, target, mask=None, alpha=0.7, beta=0.3, eps=1e-6, reduction='mean', num_classes=1):
    if num_classes == 1:
        p = torch.sigmoid(pred_logits)
        target = to_one_hot(target, num_classes)
    else:
        p = torch.softmax(pred_logits, dim=1)
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
    loss = 1. - tversky.mean(dim=1)  # average across classes

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # per-sample

def boundary_loss(pred_logits, target, mask=None, alpha=0.5, beta=0.5,  eps=1e-16, num_classes=1):
    if num_classes == 1:
        p = torch.sigmoid(pred_logits)
        target = to_one_hot(target, num_classes)
    else:
        p = torch.softmax(pred_logits, dim=1)
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

    loss = (alpha * weighted_dist_fg / normalizer_fg + beta * weighted_dist_bg / normalizer_bg).mean()
    return loss


class CombinedSegLoss:
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
        boundary = boundary_loss(pred_logits,
                                  target,
                                  mask=mask,
                                  alpha=self.boundary_alpha,
                                  beta=self.boundary_beta,
                                  eps=eps,
                                  num_classes=self.num_classes
                                      )
        # print('check loss loss', ce, tversky, boundary)
        loss = self.ce_w * ce + self.tversky_w * tversky + self.boundary_w * boundary
        # print('total loss', loss)
        return loss

