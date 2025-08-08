from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
import yaml
from box import Box
from einops import rearrange, reduce
from torch import nn, optim

from src.backbones.decoder.satmae_decoder import SatMAEHeadViT as Decoder
from src.backbones.encoder.clay_encoder import SegmentEncoder

torch.set_float32_matmul_precision("medium")
"""
ref: https://github.com/Clay-foundation/model/blob/main/
"""
class ClaySatMae(nn.Module):
    def __init__(
            self,
            # encoder
            mask_ratio,
            patch_size,
            shuffle,
            dim,
            depth_encoder ,
            heads,
            dim_head,
            mlp_ratio,
            image_size=256,

            # decoder
            out_channels = 1024, # transformer's dim
            depth_decoder = 1,
            in_chans = 10, # S2

            metadata_path: str = "configs/metadata.yaml",
            norm_pix_loss = False


    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.dim = dim
        self.depth = depth_encoder
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_ratio = mlp_ratio
        self.image_size = image_size
        self.out_channels = out_channels
        self.num_patches = int(self.image_size/self.patch_size)
        self.depth = depth_decoder
        self.in_chans = in_chans
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.norm_pix_loss = norm_pix_loss
        self.encoder = SegmentEncoder(
            mask_ratio=self.mask_ratio,
            patch_size=self.patch_size,
            shuffle=self.shuffle,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            mlp_ratio=self.mlp_ratio,
        )

        self.decoder = Decoder(
            embed_dim=self.dim,
            out_channels=self.out_channels,
            bias=True,
            num_heads=self.heads,
            mlp_ratio=self.mlp_ratio,
            patch_size=self.patch_size,
            num_patches=self.num_patches,
            depth = self.depth,
            in_chans = self.in_chans
        )

    def forward(self, datacube):

        waves = list(self.metadata.bands.wavelength.values())
        gsd = torch.tensor(10.0)
        datacube['waves'] = torch.tensor(waves)
        datacube['gsd'] = gsd


        # get the embedding produced by encoder
        embeddings = self.encoder(datacube)

        # # take the encoded patch token
        # embeddings = embeddings[:, 1:, :]

        # reconstruct the pixels
        pixels = self.decoder(embeddings)

        # calculate the loss
        reconstruction_loss = self.per_pixel_loss(
            datacube["pixels"], pixels
        )
        return {"pixels": pixels, "loss": reconstruction_loss}

    def per_pixel_loss(self, cube, pixels):
        """
        cube: [B C H W]
        pixels: [B L (C P P)]
        masked_matrix: [B L], 0 is unmasked, 1 is masked
        """
        cube_patches = rearrange(
            cube,
            "B C (h p1) (w p2) -> B (h w) (C p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )  # [B L (C P P)]

        pixels_patches = rearrange(
            pixels,
            "B C (h p1) (w p2) -> B (h w) (C p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )  # [B L (C P P)]

        if self.norm_pix_loss:
            mean = cube_patches.mean(dim=-1, keepdim=True)
            var = cube_patches.var(dim=-1, keepdim=True)
            cube_patches = (cube_patches - mean) / (var + 1e-6) ** 0.5

        loss = F.l1_loss(cube_patches, pixels_patches, reduction="none")  # loss per pixel
        loss = reduce(loss, "B L D -> B L", reduction="mean")  # loss per patch

        print(loss.sum(), loss.count_nonzero(), loss.shape)
        print('current loss: ', loss.sum() / loss.count_nonzero())
        loss = loss.sum()/loss.count_nonzero()  # loss on masked patches only

        return loss

class ClayModule(L.LightningModule):
    def __init__(
            self,
            ckpt_path,
            metadata_path: str = "configs/metadata.yaml",
            norm_pix_loss = False,
            lr=1e-5,
            wd=0.05,
            b1=0.9,
            b2=0.95,
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.metadata_path = metadata_path
        self.norm_pix_loss = norm_pix_loss
        self.lr = lr
        self.wd = wd
        self.b1 = b1
        self.b2 = b2
        self.model =  ClaySatMae(
            mask_ratio=0.75,
            patch_size=8,
            shuffle=True,
            dim=1024,
            depth_encoder=1,
            heads=16,
            dim_head=64,
            mlp_ratio=4,

            # decoder
            out_channels=1024,
            depth_decoder=1,

            metadata_path =  self.metadata_path,
            norm_pix_loss = self.norm_pix_loss

        )

    def forward(self, datacube):
        """
        Forward pass through the segmentation model.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The segmentation logits.
        """
        return self.model(datacube)


    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx) -> Any:
        """
        Test step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx) -> Any:
        """
        Test step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "predict")

    def shared_step(self, batch, batch_idx: int, phase: str):
        forward = self.forward(batch)
        loss = forward["loss"]
        self.log(
            name=f"{phase}/loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return forward


    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and scheduler
            configuration.
        """
        optimizer = optim.AdamW(
            [
                param
                for name, param in self.model.named_parameters()
                if param.requires_grad
            ],
            lr=self.lr,
            weight_decay=self.wd,
            betas=(self.b1, self.b2),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,
            T_mult=1,
            eta_min=self.lr * 100,
            last_epoch=-1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


