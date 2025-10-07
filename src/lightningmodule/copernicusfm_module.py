import lightning as L
import torch

from ..backbones.models.copernicusfm_models import CopernicusMLP
from ..utils.losses import CombinedSegLoss

class CopernicusMLPModule(L.LightningModule):
    def __init__(self,
                 encoder_name,
                 encoder_config,
                 decoder_config,
                 num_classes,
                 yhat_post_fn=None
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.model = CopernicusMLP(
            encoder_name,
            encoder_config,
            decoder_config,
            num_classes)
        self.loss_fn = CombinedSegLoss(num_classes=num_classes,
                                       ce_weight=0.4,
                                       tversky_weight=0.3,
                                       boundary_weight=0.3)
        self.yhat_post_fn = yhat_post_fn

    def forward(self, data_dict):
        return self.model(data_dict)

    def training_step(self, batch):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch):
        return self.shared_step(batch, 'val')

    def test_step(self, batch):
        return self.shared_step(batch, 'test')

    def predict_step(self, batch):
        coords = batch['y']

        batch_on_device = dict(x=batch['x'].to(self.device),
                               meta_info=batch['meta_info'].to(self.device),
                               wave_list=batch['wave_list'].to(self.device),
                               bandwidth=batch['bandwidth'].to(self.device),
                               language_embed=batch['language_embed'].to(self.device) if batch['language_embed'] is not None else None,
                               input_mode=batch['input_mode'],
                               kernel_size=batch['kernel_size'].to(self.device) if batch['kernel_size'] is not None else None)
        pred_logits = self.model(batch_on_device)
        pred_logits = pred_logits.detach().cpu()
        if self.num_classes==1:
            prob = torch.sigmoid(pred_logits)
            pred = torch.where(prob>0.5, 1, 0)
        else:
            prob = torch.softmax(pred_logits, dim=1)
            pred = torch.argmax(prob, dim=1)

        return {'coords': coords, 'pred': pred, 'prob': prob}

    def configure_optimizers(self):
       optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
       return optimizer

    def shared_step(self, batch, mode):
        y_hat = torch.stack(batch["y"], dim=0).to(self.device)
        if self.yhat_post_fn: # adjust y_hat size according to task type e.g., binary segmentation
            y_hat = self.yhat_post_fn(y_hat)
        mask_x = torch.isfinite(batch['x']).all(dim=1)
        mask_yhat = torch.isfinite(y_hat)
        mask = mask_x & mask_yhat

        batch_on_device = dict(x=batch['x'].to(self.device),
                               meta_info=batch['meta_info'].to(self.device),
                               wave_list=batch['wave_list'].to(self.device),
                               bandwidth=batch['bandwidth'].to(self.device),
                               language_embed=batch['language_embed'].to(self.device) if batch['language_embed'] is not None else None,
                               input_mode=batch['input_mode'],
                               kernel_size=batch['kernel_size'].to(self.device) if batch['kernel_size'] is not None else None)
        pred_logits = self.model(batch_on_device)
        loss = self.loss_fn(pred_logits, y_hat, mask=mask, eps=1e-6)

        # Log and check for NaNs
        if torch.isnan(loss):
            raise ValueError(f"NaN loss detected!, {loss}")

        self.log(name=f"{mode}_loss",
                value=loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True)
        return loss


    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # skip automatic transfer
        return batch


