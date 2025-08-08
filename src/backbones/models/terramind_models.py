from torch import nn, optim
from terratorch.registry import BACKBONE_REGISTRY, TERRATORCH_BACKBONE_REGISTRY, TERRATORCH_DECODER_REGISTRY
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
import lightning as L
from typing import Any

class TerraSatMae(nn.Module):
    def __init__(
            self,
            backbone_name: str = "terramind_v1_base"
    ):
        super(TerraSatMae, self).__init__()
        self.backbone_name = backbone_name
        self.encoder = BACKBONE_REGISTRY.build(
            self.backbone_name,
            modalities=["S2L1C"],
            pretrained=True)

    def forward(self, datacube):
        embeddings = self.encoder(datacube['pixels'])
        pixels = self.decoder(embeddings)

        # calculate the loss
        reconstruction_loss = self.per_pixel_loss(datacube['pixels'], pixels)

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
        loss = loss.sum() / loss.count_nonzero()  # loss on masked patches only

        return loss


class TerramindModule(L.LightningModule):
    def __init__(
            self,
            lr=1e-5,
            wd=0.05,
            b1=0.9,
            b2=0.95,
    ):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.b1 = b1
        self.b2 = b2
        self.model =  TerraSatMae()

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