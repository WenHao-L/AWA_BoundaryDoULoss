import os
import torch

from glob import glob
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Spacingd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    KeepLargestConnectedComponentd,
    SaveImaged
)
from monai.networks.layers.factories import Act, Norm
from prostate158.aw_unet import AW_UNet


def main(tempdir):

    images = sorted(glob(os.path.join(tempdir, "*_0000.nii.gz")))
    files = [{"img": img} for img in images]

    # define pre transforms
    pre_transforms = Compose(
        [
            LoadImaged(keys="img"),
            EnsureChannelFirstd(keys="img"),
            Orientationd(keys="img", axcodes="RAS"),
            Spacingd(keys="img", pixdim=[0.5, 0.5, 0.5]),
            ScaleIntensityd(keys="img", channel_wise=True),
        ]
    )
    # define dataset and dataloader
    dataset = Dataset(data=files, transform=pre_transforms)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    # define post transforms
    post_transforms = Compose(
        [
            AsDiscreted(
                keys="pred", 
                argmax=True, 
                # to_onehot=3, 
                num_classes=3
            ),
            KeepLargestConnectedComponentd(
                keys="pred", 
                applied_labels=list(range(1, 3))
            ),
            Invertd(
                keys="pred",  # invert the `pred` data field, also support multiple fields
                transform=pre_transforms,
                orig_keys="img",  # get the previously applied pre_transforms information on the `img` data field,
                # then invert `pred` based on this information. we can use same info
                # for multiple fields, also support different orig_keys for different fields
                nearest_interp=True,  # don't change the interpolation mode to "nearest" when inverting transforms
                # to ensure a smooth output, then execute `AsDiscreted` transform
                to_tensor=True,  # convert to PyTorch Tensor after inverting
            ),
            SaveImaged(keys="pred", output_dir="/path/to/picai_anatomy_mask", output_postfix="seg", resample=False),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AW_UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=[16, 32, 64, 128, 256, 512],
        strides=[2, 2, 2, 2, 2],
        num_res_units=4,
        act=Act.PRELU,
        norm=Norm.BATCH,
        dropout=0.15,
        use_checkpoint=False
    ).to(device)
    net.load_state_dict(torch.load("/path/to/model.pt"))

    net.eval()
    with torch.no_grad():
        for d in dataloader:
            images = d["img"].to(device)
            # define sliding window size and batch size for windows inference
            d["pred"] = sliding_window_inference(inputs=images, roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5, predictor=net)
            # decollate the batch data into a list of dictionaries, then execute postprocessing transforms
            d = [post_transforms(i) for i in decollate_batch(d)]


if __name__ == "__main__":
    tempdir = "/path/to/PICAI/output/nnUNet_raw_data/Task2201_picai_baseline/imagesTr"
    main(tempdir)