import torch
from torch import nn
from einops import rearrange, repeat
from modulargeofm.backbones.encoder.clay_encoder import clay_encoder_base, clay_encoder_small, clay_encoder_large, ClayTransformerEncoder
from modulargeofm.backbones.decoders.mlp import SimpleMLPDecoder
from modulargeofm.backbones.decoders.heads import SegmentationHead

encoder_dict = {"clay_encoder_base": clay_encoder_base,
                "clay_encoder_small": clay_encoder_small,
                "clay_encoder_large": clay_encoder_large,}
    
class ClayMLPSegmentor(nn.Module):
    """
    Clay model with MLP decoder for segmentation tasks.
    """
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 num_class):
        super().__init__()
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.encoder = ClayTransformerEncoder(**self.encoder_config)
        self.decoder = SimpleMLPDecoder(embed_dims=[self.encoder_config['dim']], 
                                        **self.decoder_config)
        self.head = SegmentationHead(
            in_channels=self.decoder_config['out_channels'],
            out_channels=num_class,
            upscale_factor=self.encoder_config['patch_size'])

    def forward(self, datacube):
        """
        datacube["pixels"],  # [B C H W]
        datacube["time"],  # [B 4]
        datacube["latlon"],  # [B 4]
        datacube["gsd"],  # 1
        datacube["waves"],  # [N]
        """
        B, C_in, H_in, W_in = datacube["pixels"].shape
        
        patches = self.encoder(datacube) # [B L D]
        # Reshape embeddings to [B, D, H', W']
        H_patches = H_in // self.encoder_config['patch_size']
        W_patches = W_in // self.encoder_config['patch_size']
        x = rearrange(patches, "B (H W) D -> B D H W", H=H_patches, W=W_patches)
        x = self.decoder([x])
        return self.head(x)
    def load_encoder_weights(self, ckpt_path, map_location='cpu'):
        """Load pretrained weights for encoder"""
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        return self.encoder.load_state_dict(state_dict, strict=False)
    def freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False


