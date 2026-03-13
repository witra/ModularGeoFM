import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from modulargeofm.datamodules.copernicusfm_datamodule import CopernicusFMIterableDataModule
from modulargeofm.lightningmodules.copernicusfm_module import CopernicusMLPModule
from modulargeofm.utils.callbacks import GPUMonitor


if __name__ == '__main__':
    # set the dataset path
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
            num_layers = 1,
            activation = "ReLU",
            dropout = 0.7,
            norm_layer="LayerNorm"
    )

    wandb_logger = WandbLogger(project="Test_ModularGeoF_CNRS_LCC_on_iterableDataset")
    module = CopernicusMLPModule(
        encoder_name = 'vit_base_patch16',
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        num_classes=6, 
        yhat_post_fn=lambda y_hat: torch.unsqueeze(y_hat, dim=1)
    )

    module.model.load_encoder_weights(ckpt_path=ckpt_path,  map_location='cpu')
    for param in module.model.parameters():
        param.requires_grad = True

    # set datamodule
    metadata_path = '../../configs/metadata.yaml'
    zarr_dirs = ['../../dataset/AI4LCC_CNRS_multisen/data_for_training/sentinel2_reduced2020_withlabelclass1', ]
    zarr_kinds = ['s2-CNRS',]
    datamodule = CopernicusFMIterableDataModule(
        zarr_dirs=zarr_dirs,
        data_kinds=zarr_kinds,
        metadata_path=metadata_path,
        input_dims={'x': 224, 'y': 224, 'time':1},
        input_overlap={'x': 112, 'y': 112},
        batch_size=64,
        num_workers=3,
        prefetch_factor=1,
        split_ratio=0.95,
        filter_x_thres=0.05 # pass only data with nan and inf less than 5%
    )
    datamodule.setup(stage='fit')

    checkpoint_cb = ModelCheckpoint( 
    dirpath="./checkpointsCopernicusMLP_CNRS/",        
    filename="CopernicusMLP_CNRS{epoch:02d}-{val_losstotal:.2f}",
    save_top_k=1,                 
    monitor="val_losstotal",           
    mode="min"
    )
    gpu_monitor = GPUMonitor(interval=0.1, output_csv="./GPUlog_on_IterableDataset.csv")
    loader = datamodule.train_dataloader()
    trainer = L.Trainer(accelerator='gpu',
                        max_epochs=3,
                        precision=32,  
                        gradient_clip_val=1.0,       
                        detect_anomaly=True,         
                        log_every_n_steps=10,
                        callbacks=[checkpoint_cb, gpu_monitor],
                        logger=wandb_logger)
    training = trainer.fit(model=module, train_dataloaders=loader, val_dataloaders=loader)