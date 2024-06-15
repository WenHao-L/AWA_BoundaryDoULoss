# create a standard UNet

from monai.networks.nets import UNet
from monai.networks.nets import SwinUNETR, UNETR
from prostate158.aw_unet import AW_UNet
from prostate158.wso_unet import WSO_UNet
from prostate158.aw_swinunetr import AW_SwinUNETR
from prostate158.aw_unetr import AW_UNETR


def get_model(config: dict):
    if config.model_name == "unet":
        if config.use_aw:
            return AW_UNet(
                spatial_dims=config.ndim,
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                channels=config.model.channels,
                strides=config.model.strides,
                num_res_units=config.model.num_res_units,
                act=config.model.act,
                norm=config.model.norm,
                dropout=config.model.dropout,
                use_checkpoint=config.use_checkpoint,
                )
        elif config.use_wso:
            return WSO_UNet(
                spatial_dims=config.ndim,
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                channels=config.model.channels,
                strides=config.model.strides,
                num_res_units=config.model.num_res_units,
                act=config.model.act,
                norm=config.model.norm,
                dropout=config.model.dropout,
                act_window=config.model.act_window,
                upbound_window=config.model.upbound_window,
                window_init=config.model.window_init
                )
        else:
            return UNet(
                spatial_dims=config.ndim,
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                channels=config.model.channels,
                strides=config.model.strides,
                num_res_units=config.model.num_res_units,
                act=config.model.act,
                norm=config.model.norm,
                dropout=config.model.dropout,
                )
    elif config.model_name == "swinunetr":
        if config.use_aw:
            return AW_SwinUNETR(
                img_size=(96,96,96), 
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                feature_size=48,
                use_checkpoint=config.use_checkpoint
                )
        else:
            return SwinUNETR(
                img_size=(96,96,96), 
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                feature_size=48,
                use_checkpoint=config.use_checkpoint
                )
    elif config.model_name == "unetr":
        if config.use_aw:
            return AW_UNETR(
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                img_size=(96,96,96),
                feature_size=32,
                norm_name='batch'
                )
        else:
            return UNETR(
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                img_size=(96,96,96),
                feature_size=32,
                norm_name='batch'
                )
    else:
        raise NotImplemented