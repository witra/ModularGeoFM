import torch
from torch import nn
from modulargeofm.backbones.encoder.copernicusfm_encoder import vit_small_patch16, vit_base_patch16, vit_large_patch16, vit_huge_patch14
from modulargeofm.backbones.decoders.mlp import SimpleMLPDecoder
from modulargeofm.backbones.decoders.heads import SegmentationHead

encoder_dict = {'vit_small_patch16': vit_small_patch16,
                'vit_base_patch16': vit_base_patch16,
                'vit_large_patch16': vit_large_patch16,
                'vit_huge_patch14': vit_huge_patch14}

class CopernicusMLP(nn.Module):
    """
    Copernicus-style MLP segmentation model wrapper.

    This class builds a segmentation model that combines a pretrained ViT encoder,
    a simple MLP decoder that consumes intermediate encoder features, and a
    segmentation head that produces upsampled class logits. It is intended as a
    convenience wrapper for experiments and pipelines that follow the Copernicus
    feature-mapping design: extract multi-scale features from a ViT, merge them
    with a lightweight decoder, and predict segmentation maps at a resolution
    determined by the encoder patch size.

        Name of the Vision Transformer encoder architecture. Supported values in
        the current implementation include: 'vit_small_patch16', 'vit_base_patch16',
        'vit_large_patch16', 'vit_huge_patch14'. Unsupported names raise ValueError.
        Keyword arguments passed to the encoder constructor. Must include the key
        ``intermediate_indices`` (an iterable) which determines how many intermediate
        feature maps the encoder will return; this length controls the number of
        inputs the decoder receives.
        Keyword arguments passed to the MLP decoder constructor. Must include
        ``out_channels`` which configures the number of channels consumed by the
        SegmentationHead.
        Number of output segmentation classes. This sets the final head's number
        of output channels (logits).

    Attributes
    encoder : torch.nn.Module
        The instantiated pretrained Vision Transformer encoder.
    decoder : torch.nn.Module
        The MLP decoder that consumes a list of intermediate feature tensors and
        returns a fused feature map.
    head : torch.nn.Module
        Segmentation head that maps decoder output to class logits and upsamples
        predictions by the encoder patch size.
    patch_size : int
        Patch spatial size used by the encoder (height and width stride of the
        encoder). The segmentation head upsamples outputs by this factor to recover
        image-space resolution.
    embed_dim : int
        Embedding dimensionality inferred from the selected encoder variant.
    frozen : bool
        Indicates whether encoder parameters have been frozen (requires_grad=False).

    Forward
        Run a forward pass through encoder, decoder and segmentation head.

        data_dict : dict
            Dictionary of inputs forwarded to the encoder. Typical keys include
            ``'x'`` (image tensor of shape (B, C, H, W)) and any encoder-specific
            extra fields (for example ``meta_info``, ``wave_list``, ``bandwidth``,
            ``language_embed``, ``input_mode``, ``kernel_size``), depending on the
            chosen encoder implementation.

        Returns
        torch.Tensor
            Class logits tensor of shape (B, num_class, H_out, W_out). H_out and
            W_out correspond to the input spatial dimensions divided by the encoder
            patch size (i.e., the head upsamples decoder output by `patch_size`).

        Load pretrained weights into the encoder. The method will attempt to
        accept checkpoints that store the model under the key ``'model'``; if that
        key is absent the checkpoint dict itself is used. Returns the result of
        ``encoder.load_state_dict(..., strict=False)``.
        Set ``requires_grad = False`` for all encoder parameters to prevent updates
        during training.

    Raises
    ------
    ValueError
        If an unsupported ``encoder_name`` is provided or if required keys are
        missing from ``encoder_config`` or ``decoder_config``.

    - The class automatically selects a sensible ``patch_size`` and ``embed_dim`` for
      a small set of ViT variants. If you add new encoder types, update the mapping
      to ensure correct values for these fields.
    - The decoder construction relies on ``encoder_config['intermediate_indices']``
      to determine how many intermediate embeddings it receives.
    - The SegmentationHead upsamples by ``patch_size``; ensure this matches the
      encoder's patching behavior to obtain correctly scaled outputs.
    """
    def __init__(self,
                 encoder_name,
                 encoder_config,
                 decoder_config,
                 num_class):
        super().__init__()
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        if encoder_name == 'vit_small_patch16':
            self.patch_size = 16
            self.embed_dim = 384
        elif encoder_name == 'vit_base_patch16':
            self.patch_size = 16
            self.embed_dim = 768
        elif encoder_name == 'vit_large_patch16':
            self.patch_size = 16
            self.embed_dim = 1024
        elif encoder_name == 'vit_huge_patch14':
            self.patch_size = 16
            self.embed_dim = 1280
        else:
            raise ValueError(f"Encoder {encoder_name} is not handled yet")
        self.encoder = encoder_dict[encoder_name](**self.encoder_config)
        self.freeze_encoder()
        self.decoder = SimpleMLPDecoder(embed_dims=[self.embed_dim for _ in self.encoder_config['intermediate_indices']],
                                        **self.decoder_config
                                        )
        self.head = SegmentationHead(
            in_channels=self.decoder_config['out_channels'],
            out_channels=num_class,
            upscale_factor=self.patch_size
        )

    def forward(self, data_dict):
        """
        data_dict = dict(x=batch_data,
                        meta_info=meta_infos,
                        wave_list=wave_list,
                        bandwidth=bandwidth,
                        language_embed=language_embed,
                        input_mode=input_mode,
                        kernel_size=kernel_size)
        """

        _, intermediates = self.encoder(**data_dict)
        decoder_out = self.decoder(intermediates)
        return self.head(decoder_out)

    def load_encoder_weights(self, ckpt_path, map_location='cpu'):
        """Load pretrained weights for encoder"""
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        return self.encoder.load_state_dict(state_dict, strict=False)
    def freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False



