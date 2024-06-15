from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    Spacingd,
    Invertd,
    ScaleIntensityd
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import torch
import matplotlib.pyplot as plt
import os
import glob

from aw_unet import AW_UNet

use_aw = False
for i in range(0, 5, 1):
    print("==========================================================================")
    print("Fold:", i)
    print("==========================================================================")
    result_dir = "/result_path/hepatic_vessels_aw_unet" if use_aw else "/result_path/hepatic_vessels_unet"
    result_dir = os.path.join(result_dir, "fold_" + str(i))
    os.makedirs(result_dir, exist_ok=True)
    root_dir = "/MSD_path/"
    data_dir = os.path.join(root_dir, "Task08_HepaticVessel")
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

    fold_size = len(data_dicts) // 5
    start_index = i * fold_size
    end_index = start_index + fold_size

    if i == 4:
        val_files = data_dicts[start_index:]
        train_files = data_dicts[:start_index]
    else:
        val_files = data_dicts[start_index:end_index]
        train_files = data_dicts[:start_index] + data_dicts[end_index:]

    set_determinism(seed=0)
    device = torch.device("cuda:0")

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 0.75), mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=2,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 0.75), mode=("bilinear", "nearest")),
        ]
    )

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    # train_ds = Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    if use_aw:
        model = AW_UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)
    else:
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)

    # loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    max_epochs = 600
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=3)])
    post_label = Compose([AsDiscrete(to_onehot=3)])

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (96, 96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(result_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")

    with open(os.path.join(result_dir, "result.txt"), 'a+') as file:
        file.write(f"best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}\n")
    with open(os.path.join(result_dir, "loss.txt"), 'a+') as file:
        for loss_value in epoch_loss_values:
            file.write(str(loss_value) + '\n')
    with open(os.path.join(result_dir, "dice.txt"), 'a+') as file:
        for dice_value in metric_values:
            file.write(str(dice_value) + '\n')


    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig(os.path.join(result_dir, 'training_plot.png'))
    # plt.show()


    # ======================================================================
    # Metric on original image spacing
    # ======================================================================
    device = torch.device("cpu")
    val_org_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 0.75), mode="bilinear"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0, channel_wise=True),
            CropForegroundd(keys=["image"], source_key="image"),
        ]
    )

    val_org_ds = Dataset(data=val_files, transform=val_org_transforms)
    val_org_loader = DataLoader(val_org_ds, batch_size=1, num_workers=4)

    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=val_org_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
                device=device,
            ),
            AsDiscreted(keys="pred", argmax=True, to_onehot=3),
            AsDiscreted(keys="label", to_onehot=3),
        ]
    )

    if use_aw:
        model = AW_UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)
    else:
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)

    model.load_state_dict(torch.load(os.path.join(result_dir, "best_metric_model.pth")))
    model.eval()

    with torch.no_grad():
        for val_data in val_org_loader:
            val_inputs = val_data["image"].to(device)
            roi_size = (96, 96, 96)
            sw_batch_size = 4
            val_data["pred"] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)

        # aggregate the final mean dice result
        metric_org = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

    print("Metric on original image spacing: ", metric_org)
    with open(os.path.join(result_dir, "result.txt"), 'a+') as file:
        file.write(f"Metric on original image spacing: {metric_org:.4f}\n")