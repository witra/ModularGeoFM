
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from modulargeofm.datamodules.copernicusfm_datamodule import CopernicusFMDataset
from modulargeofm.lightningmodules.copernicusfm_module import CopernicusMLPModule
from modulargeofm.utils.callbacks import GPUMonitor

if __name__ == '__main__':
    ckpt_path = '../../weights/CopernicusFM_ViT_base_varlang_e100.pth'
    encoder_config = dict(
        img_size = 224,
        drop_rate = 0.0,
        wv_planes = 128,
        num_classes = 0,
        global_pool = True,
        loc_option = 'cartesian',
        return_intermediate= True,
        intermediate_indices = [0,])


    decoder_config = dict(
            out_channels = 256,
            num_layers = 2,
            activation = "ReLU",
            dropout = 0.0,
            norm_layer="LayerNorm"
    )

    wandb_logger = WandbLogger(project="Test_ModularGeoF_CNRS_LCCon_mapDataset")
    module = CopernicusMLPModule(
        encoder_name = 'vit_base_patch16',
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        num_classes=6, 
        yhat_post_fn=lambda y_hat: torch.unsqueeze(y_hat, dim=1)
    )

    module.model.load_encoder_weights(ckpt_path=ckpt_path, map_location='cpu')
    for param in module.model.parameters():
        param.requires_grad = True
    
    # set datamodule
    zarr_dir = '../../dataset/AI4LCC_CNRS_multisen/chips_for_training'
    dataset = CopernicusFMDataset(chip_zarr_dir=zarr_dir)
    loader = DataLoader(
                dataset= dataset,
                batch_size=64,
                num_workers=3,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=1)

    checkpoint_cb = ModelCheckpoint( 
        dirpath="./checkpointsCopernicusMLP/",        
        filename="CopernicusMLP_{epoch:02d}-{val_losstotal:.2f}",
        save_top_k=1,                 
        monitor="val_losstotal",            
        mode="min"
    )
    gpu_monitor = GPUMonitor(interval=0.1, output_csv="GPUlog_on_MapDataset.csv")

    trainer = L.Trainer(accelerator='gpu',
                        max_epochs=3,
                        precision=32,  
                        gradient_clip_val=1.0,       
                        detect_anomaly=True,         
                        log_every_n_steps=10,
                        callbacks=[checkpoint_cb, gpu_monitor],
                        logger=wandb_logger)
    training = trainer.fit(model=module, train_dataloaders=loader, val_dataloaders=loader)
