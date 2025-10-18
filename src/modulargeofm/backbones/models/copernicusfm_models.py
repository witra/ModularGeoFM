import torch
from torch import nn
from ..encoder.copernicusfm_encoder import vit_small_patch16, vit_base_patch16, vit_large_patch16, vit_huge_patch14
from ..decoder.mlp import SimpleMLPDecoder
from ..decoder.heads import SegmentationHead

encoder_dict = {'vit_small_patch16': vit_small_patch16,
                'vit_base_patch16': vit_base_patch16,
                'vit_large_patch16': vit_large_patch16,
                'vit_huge_patch14': vit_huge_patch14}

class CopernicusMLP(nn.Module):
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
            raise Exception(f"Encoder {encoder_name} is not handled yet")
        self.encoder = encoder_dict[encoder_name](**self.encoder_config)
        self.freeze_encoder()
        self.decoder = SimpleMLPDecoder(embed_dims=[self.embed_dim for i in self.encoder_config['intermediate_indices']],
                                        **self.decoder_config
                                        )
        self.head = SegmentationHead(
            in_channels=self.decoder_config['out_channels'],
            out_channels=num_class,
            upscale_factor=self.patch_size
        )

    def forward(self, data_dict):
        """
        data = dict(x=batch_data,
                    meta_info=meta_infos,
                   wave_list=wave_list,
                   bandwidth=bandwidth,
                   language_embed=language_embed,
                   input_mode=input_mode,
                   kernel_size=kernel_size)
        """

        features, intermediates = self.encoder(**data_dict)
        decoder_out = self.decoder(intermediates)
        logit = self.head(decoder_out)
        return logit

    def load_encoder_weights(self, ckpt_path, map_location='cpu'):
        """Load pretrained weights for encoder"""
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        return msg

    def freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False



