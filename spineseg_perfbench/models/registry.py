from __future__ import annotations

from monai.networks.nets import SegResNet, UNet


def build_model(name: str, in_channels: int = 1, out_channels: int = 26, smoke: bool = False):
    name = name.lower()
    if name == "segresnet":
        if smoke:
            return SegResNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                init_filters=8,
                blocks_down=(1, 1),
                blocks_up=(1,),
                dropout_prob=0.0,
            )
        return SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=16,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.0,
        )
    if name == "unet":
        if smoke:
            return UNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=(4, 8, 16),
                strides=(2, 2),
                num_res_units=1,
            )
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    raise ValueError(f"Unknown model: {name}")
