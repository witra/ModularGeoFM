import torch
from torch import nn

class SimpleMLPDecoder(nn.Module):
    """
    Lightweight per-pixel MLP decoder for fusing transformer features.

    This decoder concatenates multiple feature maps from a transformer encoder 
    along the channel dimension, then applies a sequence of 1x1 convolutions 
    (per-pixel MLP) with optional normalization, activation, and dropout. 
    Spatial resolution is preserved.

    Parameters
    ----------
    embed_dims : list[int]
        List of channel dimensions for each encoder feature map.
    out_channels : int, optional (default=256)
        Number of channels in the output feature map.
    num_layers : int, optional (default=2)
        Number of MLP layers (Conv + Norm + Activation) to apply.
    activation : str, optional (default="ReLU")
        Activation function name (must be a valid nn.Module, e.g., "ReLU", "GELU").
    dropout : float, optional (default=0.0)
        Dropout probability applied after each activation.
    norm_layer : str, optional (default="BatchNorm2d")
        Normalization layer name. Supports "BatchNorm2d" or "LayerNorm".
    residual : bool, optional (default=True)
        If True, adds a residual connection when input channels match output channels.

    Attributes
    ----------
    n_inputs : int
        Number of input feature maps expected (length of embed_dims).
    mlp : nn.Sequential
        Sequence of 1x1 convolution, normalization, activation, and optional dropout layers.

    Inputs
    ------
    features : list[torch.Tensor]
        List of encoder feature maps, each of shape (B, C_i, H, W).
        All feature maps must have the same spatial dimensions (H, W).

    Outputs
    -------
    torch.Tensor
        Fused feature map of shape (B, out_channels, H, W).

    Notes
    -----
    - Supports optional channel-wise LayerNorm via ChannelLayerNorm for 4D CNN tensors.
    - Residual connection is applied only if the concatenated input channels 
      match out_channels.
    - Useful for fusing multi-level transformer features before task-specific heads.

    Example
    -------
    >>> features = [torch.randn(2, 256, 32, 32) for _ in range(4)]
    >>> decoder = SimpleMLPDecoder(
    ...     embed_dims=[256, 256, 256, 256],
    ...     out_channels=256,
    ...     num_layers=2,
    ...     activation="ReLU",
    ...     dropout=0.1,
    ...     norm_layer="LayerNorm",
    ...     residual=True
    ... )
    >>> output = decoder(features)
    >>> output.shape
    torch.Size([2, 256, 32, 32])
    """

    def __init__(
        self,
        embed_dims: list[int],
        out_channels: int = 256,
        num_layers: int = 2,
        activation: str = "ReLU",
        dropout: float = 0.0,
        norm_layer: str = "BatchNorm2d",
        residual: bool = True
    ):
        super().__init__()
        self.n_inputs = len(embed_dims)
        in_channels = sum(embed_dims)  # concat features along channels

        act = getattr(nn, activation)

        layers = []
        for i in range(num_layers):
            # Choose normalization layer
            if norm_layer == "LayerNorm":
                norm = ChannelLayerNorm(out_channels)
            else:
                norm = getattr(nn, norm_layer)(out_channels)

            layers.extend((
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                norm,
                act()
            ))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

        self.mlp = nn.Sequential(*layers)
        self.residual = residual and (in_channels == out_channels)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the SimpleMLPDecoder.

        Args:
            features (list[torch.Tensor]): List of encoder feature maps.
                                           Each of shape (B, C_i, H, W).

        Returns:
            torch.Tensor: Fused feature map of shape (B, out_channels, H, W).
        """
        if len(features) != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} features, got {len(features)}.")

        x = torch.cat(features, dim=1)  # concatenate along channel
        out = self.mlp(x)

        # Optional residual: only when channel dims match
        if self.residual:
            out = out + x
        return out
    
class ChannelLayerNorm(nn.Module):
    """
    LayerNorm over the channel dimension for 4D feature maps (B, C, H, W).

    This wrapper allows nn.LayerNorm to normalize CNN feature maps by permuting
    the tensor to (B, H, W, C), applying LayerNorm over channels, then permuting back.

    Parameters
    ----------
    num_channels : int
        Number of channels to normalize.
    eps : float, optional (default=1e-5)
        Small value added for numerical stability.

    Inputs
    ------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W).

    Outputs
    -------
    torch.Tensor
        Normalized tensor of the same shape as input.

    Notes
    -----
    - Useful when batch size is small or batch-independent normalization is desired.
    - Maintains spatial dimensions while normalizing channels.
    """
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x: (B, C, H, W) → permute to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # permute back to (B, C, H, W)
        return x.permute(0, 3, 1, 2)