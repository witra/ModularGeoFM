import torch
from torch import nn

class SimpleMLPDecoder(nn.Module):
    """
    Lightweight MLP-style decoder for transformer features of the same spatial resolution.

    This decoder fuses multiple feature maps from a transformer encoder by
    concatenating them along the channel dimension and applying one or more
    1x1 convolution layers (MLP per pixel). It preserves the spatial resolution
    of the input features.

    Args:
        embed_dims (list[int]): List of channel dimensions for each feature map
                                from the encoder. The length of this list
                                determines the number of feature maps expected.
        out_channels (int, optional): Number of channels in the decoder output
                                      feature map. Defaults to 256.
        num_layers (int, optional): Number of MLP layers (1x1 Conv + Activation)
                                    to apply after concatenation. Defaults to 2.
        activation (str, optional): Activation function to use between layers.
                                    Must be a valid nn.Module string, e.g., "ReLU",
                                    "GELU". Defaults to "ReLU".
        dropout (float, optional): Dropout probability applied after each activation.
                                   Defaults to 0.0 (no dropout).

    Attributes:
        n_inputs (int): Number of feature maps expected (len(embed_dims)).
        mlp (nn.Sequential): The sequence of 1x1 convolution, activation,
                             and optional dropout layers used for fusion.

    Input:
        features (list[torch.Tensor]): List of feature maps from the encoder.
                                       Each tensor should have shape (B, C_i, H, W),
                                       where C_i is the corresponding channel
                                       from embed_dims. All feature maps must have
                                       the same spatial dimensions H and W.

    Output:
        torch.Tensor: Fused feature map of shape (B, out_channels, H, W).

    Example:
        >>> features = [torch.randn(2, 256, 32, 32) for _ in range(4)]
        >>> decoder = SimpleMLPDecoder(embed_dims=[256, 256, 256, 256],
        ...                            out_channels=256,
        ...                            num_layers=2,
        ...                            activation="ReLU",
        ...                            dropout=0.1)
        >>> output = decoder(features)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])

    Notes:
        - All input feature maps must have the same spatial resolution.
        - The decoder performs **per-pixel MLP** via 1x1 convolutions.
        - Useful when encoder outputs multiple same-resolution features (e.g., transformer blocks).
        - Can be combined with a SegmentationHead to produce final masks.
    """

    def __init__(
        self,
        embed_dims: list[int],
        out_channels: int = 256,
        num_layers: int = 2,
        activation: str = "ReLU",
        dropout: float = 0.0
    ):
        super().__init__()
        self.n_inputs = len(embed_dims)
        in_channels = sum(embed_dims)  # concat features along channels

        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels,
                                    out_channels, kernel_size=1))
            layers.append(getattr(nn, activation)())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

        self.mlp = nn.Sequential(*layers)

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
        return self.mlp(x)