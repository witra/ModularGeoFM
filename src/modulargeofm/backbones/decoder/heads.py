import torch.nn.functional as F
import torch
from torch import nn

class SegmentationHead(nn.Module):
    """
    Segmentation Head for binary or multi-class segmentation.

    This module takes the feature map output of a decoder and maps it to
    segmentation logits. Optionally, it upsamples the logits to match the
    desired output resolution.

    Args:
        in_channels (int): Number of channels in the input feature map
                           (should match decoder output channels).
        out_channels (int): Number of output channels.
                            - 1 for binary segmentation
                            - N for multi-class segmentation
        upscale_factor (int, optional): Upsampling factor to scale logits
                                        back to the original image size.
                                        Defaults to 1 (no upsampling).

    Input:
        x (torch.Tensor): Feature map of shape (B, in_channels, H, W)
            - B: Batch size
            - in_channels: Channels from decoder output
            - H, W: Height and width of feature map

    Output:
        torch.Tensor: Segmentation logits of shape (B, out_channels, H_out, W_out)
            - H_out, W_out = H * upscale_factor, W * upscale_factor
            - If `upscale_factor=1`, H_out=H and W_out=W.
            - For binary segmentation, apply `torch.sigmoid` at inference.
            - For multi-class segmentation, apply `torch.softmax` over channel dim.
    """

    def __init__(self, in_channels: int, out_channels: int = 1, upscale_factor: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SegmentationHead.

        Args:
            x (torch.Tensor): Input feature map of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Segmentation logits of shape
                          (B, out_channels, H_out, W_out).
        """
        x = self.conv(x)  # project to segmentation logits
        if self.upscale_factor > 1:
            x = F.interpolate(
                x,
                scale_factor=self.upscale_factor,
                mode='bilinear',
                align_corners=False
            )
        return x