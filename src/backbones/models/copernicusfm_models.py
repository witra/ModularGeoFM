import torch
from torch import nn
from functools import partial
from ..encoder.copernicusfm_encoder import vit_small_patch16, vit_base_patch16, vit_large_patch16, vit_huge_patch14
from ..decoder.satmae_decoder import SatMAEHeadViT

class CopernicusFMSatMae(nn.Module):
    def __init__(self,
                 image_size,
                 in_channels,
                 encoder_ckpt_path,
                 encoder='vit_small_patch16',
                 decoder='SatMAEHeadViT',
                 map_location = 'cpu',
                 **kwargs):
        self.image_size = image_size,
        self.in_channels = in_channels
        self.encoder_ckpt_path = encoder_ckpt_path
        self.map_location = map_location
        if encoder == 'vit_small_patch16':
            self.encoder = vit_small_patch16(**kwargs)
            self.patch_size = 16
            self.embed_dim = 384
        elif encoder == 'vit_base_patch16':
            self.encoder = vit_base_patch16(**kwargs)
            self.patch_size = 16
            self.embed_dim = 768
        elif encoder == 'vit_large_patch16':
            self.encoder = vit_large_patch16(**kwargs)
            self.patch_size = 16
            self.embed_dim = 1024
        elif encoder == 'vit_huge_patch14':
            self.encoder == vit_huge_patch14(**kwargs)
            self.patch_size = 14
            self.embed_dim = 1280
        else:
            raise Exception(f'Encoder {encoder} is not handled yet')

        if image_size % self.patch_size != 0:
            raise ValueError(f"Image size {image_size} must be divisible by patch size {self.patch_size}")

        self.num_patches: int = image_size // self.patch_size

        if decoder == 'SatMAEHeadViT':
            self.decoder = SatMAEHeadViT(embed_dim=self.embed_dim,
                                         patch_size=self.patch_size,
                                         num_patches=self.num_patches,
                                         in_chans=self.in_channels,
                                         **kwargs)
        self.load_encoder_weights(self.encoder_ckpt_path, self.map_location)
        self.freeze_encoder()

    def load_encoder_weights(self, encoder_ckpt_path, map_location='cpu'):
        """Load pretrained weights for encoder"""
        checkpoint = torch.load(encoder_ckpt_path, map_location=map_location)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        self.encoder.load_state_dict(state_dict, strict=False)
    
    def freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x, meta_info, wave_list, bandwidth, language_embed, input_mode, kernel_size=None):
        _, fx= self.encoder(x, meta_info, wave_list, bandwidth, language_embed, input_mode, kernel_size=None) # x is logit, fx is embeddings
        img = self.decoder(fx)
        return img


def copernicus_fm_sat_mae_base(encoder='vit_small_patch16',
                               decoder='SatMAEHeadViT',
                               **kwargs):
    return CopernicusFMSatMae(encoder=encoder,
                              decoder=decoder,
                              **kwargs)



