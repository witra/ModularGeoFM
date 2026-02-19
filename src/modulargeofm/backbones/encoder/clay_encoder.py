# This file contains code adapted from the Clay Foundation project:
# https://github.com/Clay-foundation/model
# Licensed under the Apache License 2.0.
#
# Included classes:
# - `ClayTransformerEncoder()`: insipred from `claymodel/finetune/segment/factory.py`
#   Source: https://github.com/Clay-foundation/model/blob/main/claymodel/finetune/segment/factory.py
#
# Modifications:
# - implemented SegmentEncoder to ClayTransformerEncoder
# - Adjusted import paths and minor formatting
#
# You may obtain a copy of the Apache License 2.0 at:
# http://www.apache.org/licenses/LICENSE-2.0


import os, re
import torch
from einops import repeat
from torch import nn
from claymodel.model import Encoder as ClayEncoder

torch.set_float32_matmul_precision("medium")
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"

class SegmentEncoder(ClayEncoder):
    """
    Encoder class for segmentation tasks

    """

    def __init__(  # noqa: PLR0913
            self,
            mask_ratio,
            patch_size,
            shuffle,
            dim,
            depth,
            heads,
            dim_head,
            mlp_ratio,
            ckpt_path=None,
    ):
        super().__init__(
            mask_ratio,
            patch_size,
            shuffle,
            dim,
            depth,
            heads,
            dim_head,
            mlp_ratio,
        )

        # Set device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Load model from checkpoint if provided
        self.load_from_ckpt(ckpt_path)

    def load_from_ckpt(self, ckpt_path):
        """
        Load the model's state from a checkpoint file.

        Args:
            ckpt_path (str): The path to the checkpoint file.
        """
        if ckpt_path:
            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt.get("state_dict")

            # Prepare new state dict with the desired subset and naming
            new_state_dict = {
                re.sub(r"^model\.encoder\.", "", name): param
                for name, param in state_dict.items()
                if name.startswith("model.encoder")
            }

            # Load the modified state dict into the model
            model_state_dict = self.state_dict()
            for name, param in new_state_dict.items():
                if (
                        name in model_state_dict
                        and param.size() == model_state_dict[name].size()
                ):
                    model_state_dict[name].copy_(param)
                else:
                    print(f"No matching parameter for {name} with size {param.size()}")

            # Freeze the loaded parameters
            for name, param in self.named_parameters():
                if name in new_state_dict:
                    param.requires_grad = False

    def forward(self, datacube):
        """
        Forward pass of the SegmentEncoder.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            list: A list of feature maps extracted from the datacube.
        """
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )

        B, C, H, W = cube.shape


        # Patchify and create embeddings per patch
        patches, waves_encoded = self.to_patch_embed(cube, waves)  # [B L D]
        patches = self.add_encodings(patches, time, latlon, gsd)  # [B L D]

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        patches = torch.cat((cls_tokens, patches), dim=1)  # [B (1 + L) D]

        patches = self.transformer(patches) # [B L+1 D]
        # patches = patches[:, 1:, :]  # [B L D]
        return patches



class ClayTransformerEncoder(ClayEncoder):
    """
    Core transformer-based encoder derived from the Clay foundation model.

    It serves as a reusable feature extractor for downstream tasks 
    like segmentation, and regression.
    
    Reference: https://github.com/Clay-foundation/model/blob/main/claymodel/finetune/segment/factory.py
    """
    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        patch_size,
        shuffle,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        ckpt_path=None,
    ):
        super().__init__(
            mask_ratio,
            patch_size,
            shuffle,
            dim,
            depth,
            heads,
            dim_head,
            mlp_ratio,
        )

        # Set device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Load model from checkpoint if provided
        self.load_from_ckpt(ckpt_path)
    
    def load_from_ckpt(self, ckpt_path):
        """
        Load the model's state from a checkpoint file.

        Args:
            ckpt_path (str): The path to the checkpoint file.
        
        """
        if ckpt_path:
            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt.get("state_dict")

            # Prepare new state dict with the desired subset and naming
            new_state_dict = {
                re.sub(r"^model\.encoder\.", "", name): param
                for name, param in state_dict.items()
                if name.startswith("model.encoder")
            }

            # Load the modified state dict into the model
            model_state_dict = self.state_dict()
            for name, param in new_state_dict.items():
                if (
                    name in model_state_dict
                    and param.size() == model_state_dict[name].size()
                ):
                    model_state_dict[name].copy_(param)
                else:
                    print(f"No matching parameter for {name} with size {param.size()}")

            # Freeze the loaded parameters
            for name, param in self.named_parameters():
                if name in new_state_dict:
                    param.requires_grad = False

    def forward(self, datacube):
        """
        Forward pass of the SegmentEncoder.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            list: A list of feature maps extracted from the datacube.
        
        """
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )

        B, C, H, W = cube.shape

        # Patchify and create embeddings per patch
        patches, waves_encoded = self.to_patch_embed(cube, waves)  # [B L D]
        patches = self.add_encodings(patches, time, latlon, gsd)  # [B L D]

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        patches = torch.cat((cls_tokens, patches), dim=1)  # [B (1 + L) D]

        patches = self.transformer(patches)
        patches = patches[:, 1:, :]  # [B L D] , D = dim

        return patches
def clay_encoder_small(**kwargs):
    args = {
        # ENCODER
        "dim": 384,
        "depth": 6,
        "heads": 6,
        "dim_head": 64,
        "mlp_ratio": 2,
    }
    args.update(kwargs)
    return ClayEncoder(**args)

def clay_encoder_base(**kwargs):
    args = {
        # ENCODER
        "dim": 768,
        "depth": 12,
        "heads": 12,
        "dim_head": 64,
        "mlp_ratio": 4,
    }
    args.update(kwargs)
    return ClayEncoder(**args)


def clay_encoder_large(**kwargs):
    args = {
        # ENCODER
        "dim": 1024,
        "depth": 24,
        "heads": 16,
        "dim_head": 64,
        "mlp_ratio": 4,
    }
    args.update(kwargs)
    return ClayEncoder(**args)