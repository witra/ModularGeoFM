import lightning as L
import torch

from modulargeofm.backbones.models.copernicusfm_models import CopernicusMLP
from modulargeofm.utils.losses import CombinedSegLoss
from modulargeofm.utils.metrics import SegmentationMetrics

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
                                       ce_weight=0.2,
                                       tversky_weight=0.3,
                                       boundary_weight=0.5)
        self.val_metrics_fn = SegmentationMetrics(num_classes=num_classes, tversky_alpha=0.7, tversky_beta=0.3)
        self.test_metrics_fn = SegmentationMetrics(num_classes=num_classes, tversky_alpha=0.7, tversky_beta=0.3)
        self.yhat_post_fn = yhat_post_fn

    def forward(self, data_dict):
        return self.model(data_dict)

    def training_step(self, batch):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch):
        return self.shared_step(batch, 'val')

    def test_step(self, batch):
        return self.shared_step(batch, 'test')

    def on_fit_start(self):
        self.to(dtype=torch.float32)

    def predict_step(self, batch):
        coords = batch['y']
        batch.pop('y')
        pred_logits = self.model(batch)
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
        y_hat = batch["y"] #torch.stack(batch["y"], dim=0).to(self.device)
        batch['wave_list'] = batch['wave_list'][0] if batch['wave_list'].ndim > 1 else batch['wave_list']
        batch['bandwidth'] = batch['bandwidth'][0] if batch['bandwidth'].ndim > 1 else batch['bandwidth']
        batch['kernel_size'] = batch['kernel_size'] if isinstance(batch['kernel_size'], int) else batch['kernel_size'][0].item() 
        batch['input_mode'] = batch['input_mode'][0] if isinstance( batch['input_mode'], list) else batch['input_mode']
        batch.pop('y')
        if self.yhat_post_fn: # adjust y_hat size according to task type e.g., binary segmentation
            y_hat = self.yhat_post_fn(y_hat)
        mask_x = torch.isfinite(batch['x']).all(dim=1).unsqueeze(1) # (B, 1, H, W)
        mask_yhat = torch.isfinite(y_hat) # (B, 1, H, W), not one-hot code yet
        mask = mask_x & mask_yhat
        pred_logits = self.model(batch)
        total_loss, ce_loss, tversky_loss, boundary_loss = self.loss_fn(pred_logits, y_hat, mask=mask, eps=1e-6)

        # Log and check for NaNs
        if torch.isnan(total_loss):
            raise ValueError(f"NaN loss detected!, {total_loss}")
        if mode == 'val':
            self.val_metrics_fn.update(pred_logits, y_hat, mask=mask)
        
        if mode == 'test':
            self.test_metrics_fn.update(pred_logits, y_hat, mask=mask)

        self.log(name=f"{mode}_losstotal",
                batch_size=len(batch['x']),
                value=total_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True)
        self.log(name=f"{mode}_lossce",
                batch_size=len(batch['x']),
                value=ce_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True)
        self.log(name=f"{mode}_losstversky",
                batch_size=len(batch['x']),
                value=tversky_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True)
        self.log(name=f"{mode}_lossboundary",
                batch_size=len(batch['x']),
                value=boundary_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True)
        return total_loss
    
    def on_validation_epoch_end(self):
        metrics = self.val_metrics_fn.compute()
        self.log("val_tversky", metrics["tversky_index"], prog_bar=True)
        self.log("val_dice", metrics["dice_coefficient"], prog_bar=True)
        self.log("val_biou", metrics["boundary_iou"], prog_bar=True)
        self.val_metrics_fn.reset()  
    
    def on_test_epoch_end(self):
        metrics = self.val_metrics_fn.compute()
        self.log("test_tversky", metrics["tversky_index"])
        self.log("test_dice", metrics["dice_coefficient"])
        self.log("test_biou", metrics["boundary_iou"])
        self.val_metrics_fn.reset()


    

