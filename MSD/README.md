# Evaluation the adaptive window adjustment (AWA) module on the MSD dataset
we employed the 3D U-Net model as a baseline on the Task02_Heart, Task03_Liver, Task08_HepaticVessel, and Task09_Spleen tasks within the MSD dataset. We then compared the performance with and without the AWA module to assess its effectiveness. The evaluation metric is the Dice Similarity Coefficient (DSC). 

The experimental results are shown in the following table:

|      model        |            dataset             |   fold1   |   fold2   |   fold3   |   fold4   |   fold5   |   mean    |
| :---------------: | :---------------------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|      unet3D       |    MSD_Task02_Heart (MRI)     |   92.53   | **90.57** |   90.27   |   89.92   | **92.85** |   91.23   |
| (ours) unet3D+AWA |    MSD_Task02_Heart (MRI)     |   92.53   |   90.54   | **90.44** | **91.18** |   92.59   | **91.46** |
|      unet3D       |     MSD_Task03_Liver (CT)     |   76.61   |   72.10   |   72.31   |   72.11   |   75.89   |   73.80   |
| (ours) unet3D+AWA |     MSD_Task03_Liver (CT)     | **80.31** | **75.40** | **75.95** | **73.86** | **79.80** | **77.06** |
|      unet3D       | MSD_Task08_HepaticVessel (CT) |   54.53   |   56.32   |   55.26   |   56.73   |   57.43   |   56.05   |
| (ours) unet3D+AWA | MSD_Task08_HepaticVessel (CT) | **57.03** | **57.65** | **57.76** | **60.16** | **58.35** | **58.19** |
|      unet3D       |    MSD_Task09_Spleen (CT)     |   95.12   |   93.26   |   94.30   |   90.47   |   94.64   |   93.56   |
| (ours) unet3D+AWA |    MSD_Task09_Spleen (CT)     | **96.04** | **95.33** | **95.04** | **93.69** | **95.71** | **95.16** |

# Quick Start
## Dataset
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)

Download the dataset and change the `root_dir` in `[Task]_train.py` to the dataset path and `result_dir` to the result path.

## Training
1. MSD_Task02_Heart (MRI)
```
python heart_train.py
```
2. MSD_Task03_Liver (CT)
```
python liver_train.py
```
3. MSD_Task08_HepaticVessel (CT)
```
python hepatic_vessels_train.py
```
4. MSD_Task09_Spleen (CT)
```
python spleen_train.py
```

# Reference
The baseline model (3D U-Net) and experimental code refer to [https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb)