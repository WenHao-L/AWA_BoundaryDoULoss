# Manuscript Data

## 1. Dataset
- [PI-CAI_image](https://pi-cai.grand-challenge.org/DATA/)
- [PI-CAI_label](https://github.com/DIAGNijmegen/picai_labels)
- [Prostate158_train_data](https://zenodo.org/records/6481141)
- [Prostate158_test_data](https://zenodo.org/records/6592345)
- [ProstateX](https://github.com/rcuocolo/PROSTATEx_masks)
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)

## 2. Experimental results
### 2.1 Evaluation on the Prostate158+ProstateX dataset for prostate anatomy segmentation
|                 Model                  |       DSC(%)       | HD  |    ASD     |
| :-----------------------------------: | :----------------: | :-----: | :---------------: |
|(baseline) 3D U-Net+DiceCELoss | 81.27 | 19.61 | 2.25 |
|3D U-Net+3DBoundaryDoULoss | 81.33 | 19.56 | 2.20 |
|3D U-Net+AWA+DiceCELoss | 81.38 | 19.33 | 2.21 |
|(ours) 3D U-Net+AWA+3DBoundaryDoULoss | 81.47 | 19.17 | 2.18 |

### 2.2 Evaluation on the PI-CAI dataset for prostate lesion segmentation
|                 Model                  |       Fold1       | Fold2  |    Fold3     |    Fold4     |    Fold5     |    Mean     |
| :-----------------------------------: | :----------------: | :-----: | :---------------: | :----------------: | :----------------: | :----------------: |
|(baseline) ITUNet+FocalLoss| 46.99| 43.34| 40.80| 52.52| 49.01| 46.53|
|ITUNet+2DBoundaryDoULoss| 45.51| 44.03| 45.93| 55.16| 54.50| 49.03|
|ITUNet+AWA+FocalLoss| 47.86| 42.74| 46.30| 53.78| 52.13| 48.56|
|ITUNet+CZPZ+FocalLoss| 49.22| 46.44| 46.31| 53.78| 50.51| 49.25|
|(ours) ITUNet+AWA+CZPZ+2DBoundaryDoULoss| 49.50| 49.69| 48.46| 57.90| 53.05| 51.72|

### 2.3 Comparison between AWA and WSO
Anatomy Segmentation :
|                 Model                  |       DSC(%)       | HD  |    ASD     |
| :-----------------------------------: | :----------------: | :-----: | :---------------: |
|3D U-Net | 81.27 | 19.61 | 2.25 |
|3D U-Net+WSO(ReLU) | 80.64 | 19.96 | 2.33 |
|3D U-Net+WSO(Sigmoid) | 81.06 | 19.59 | 2.26 |
|3D U-Net+AWA | 81.38 | 19.33 | 2.21 |

Lesion Segmentation :
|                 Model                  |       Fold1       | Fold2  |    Fold3     |    Fold4     |    Fold5     |    Mean     |
| :-----------------------------------: | :----------------: | :-----: | :---------------: | :----------------: | :----------------: | :----------------: |
|ITUNet| 46.99| 43.34| 40.80| 52.52| 49.01| 46.53|
|ITUNet+WSO(ReLU)| 44.59 | 43.13 | 38.12 | 53.01 | 49.09 | 45.59|
|ITUNet+WSO(Sigmoid)| 45.34 | 42.36 | 40.09 | 54.10 | 49.59 | 46.30|
|ITUNet+AWA| 47.86 | 42.74 | 46.30 | 53.78 | 52.13 | 48.56|

### 2.4 Parameter count and FLOPs
|                 Input Shape                  |       Model       |  Params  |    FlOPs     |
| :-----------------------------------: | :----------------: | :-----: | :---------------: |
|1×1×96×96×96| WSO | 2 | 0.88M |
|1×1×96×96×96| AWA | 0.03M | 0.12G |
|1×1×96×96×96| 3D U-Net | 38.17M | 15.88G |
|1×3×384×384| WSO | 6 | 0.44M |
|1×3×384×384| AWA | 0.10M | 0.31G |
|1×3×384×384| ITUnet | 18.13M | 32.67G |

### 2.5 Applicability of the AWA module in SwinUNETR and UNETR
|                 Dataset                  |       Model       | DSC(%)  |
| :-----------------------------------: | :----------------: | :-----: |
|PI-CAI (Fold 1)|(2D)SwinUNETR|48.73|
|PI-CAI (Fold 1)|(2D)SwinUNETR+AWA|48.99|
|PI-CAI (Fold 1)|(2D)UNETR|38.59|
|PI-CAI (Fold 1)|(2D)UNETR+AWA|41.83|
|Prostate158+ProstateX|(3D)SwinUNETR|80.57|
|Prostate158+ProstateX|(3D)SwinUNETR+AWA|80.61|
|Prostate158+ProstateX|(3D)UNETR|77.95|
|Prostate158+ProstateX|(3D)UNETR+AWA|78.34|

### 2.6 Applicability of the AWA module on the MSD dataset
|                 Dataset                  |       Model       | Fold1  |Fold2  |Fold3  |Fold4  |Fold5  |Mean  |
| :-----------------------------------: | :----------------: | :-----: |:-----: |:-----: |:-----: |:-----: |:-----: |
|Task02_Heart (MRI)|3D U-Net|92.53| 90.57| 90.27| 89.92| 92.85| 91.23|
|Task02_Heart (MRI)|3D U-Net+AWA|92.53| 90.54| 90.44| 91.18| 92.59| 91.46|
|Task03_Liver (CT)|3D U-Net|76.61| 72.10| 72.31| 72.11| 75.89| 73.80|
|Task03_Liver (CT)|3D U-Net+AWA|80.31| 75.40| 75.95| 73.86| 79.80| 77.06|
|Task08_HepaticVessel (CT)|3D U-Net|54.53| 56.32| 55.26| 56.73| 57.43| 56.05|
|Task08_HepaticVessel (CT)|3D U-Net+AWA|57.03| 57.65| 57.76| 60.16| 58.35| 58.19|
|Task09_Spleen (CT)|3D U-Net|95.12| 93.26| 94.30| 90.47| 94.64| 93.56|
|Task09_Spleen (CT)|3D U-Net+AWA|96.04| 95.33| 95.04| 93.69| 95.71| 95.16|

### 2.7 Comparison Boundary DoU Loss with other loss functions
Anatomy Segmentation :
|                 Model         | Loss         |       DSC(%)       | HD  |    ASD     |
| :-----------------------------------: | :----------------: | :-----: | :---------------: |:---------------: |
|3D U-Net | DiceCELoss|81.27 | 19.61 | 2.25 |
|3D U-Net| TverskyLoss| 77.04 |29.12 |9.60 |
|3D U-Net| BoundaryLoss |81.47 |19.17 |2.18|
|3D U-Net| BoundaryDoULoss|77.98 |22.23 |4.34 |

Lesion Segmentation :
|                 Model       | Loss           |       Fold1       | Fold2  |    Fold3     |    Fold4     |    Fold5     |    Mean     |
| :-----------------------------------: | :----------------: | :-----: | :---------------: | :----------------: | :----------------: | :----------------: | :----------------: |
|ITUNet| FocalLoss|46.99| 43.34| 40.80| 52.52| 49.01| 46.53|
|ITUNet| TverskyLoss|42.84| 40.41| 38.37| 53.09| 47.82| 44.51|
|ITUNet| BoundaryLoss|46.90| 42.01| 41.73| 54.93| 51.17| 47.35|
|ITUNet| BoundaryDoULoss|45.51| 44.03| 45.93| 55.16| 54.50| 49.03|

### 2.8. Hyperparameter perturbation experiments

Comparison experiments of our proposed method
(ITUNet+AWA+CZPZ+2DBoundaryDoULoss) using different hyperparameters
(Optimizer and Learning Rate Schedule) for prostate lesion segmentation on the
PI-CAI dataset (Fold1). The evaluation metric is the Dice Similarity Coefficient
(DSC).

The combination of the Adam optimizer and the CosineAnnealingLR
learning rate schedule achieved better result than the result reported in
the manuscript, indicating that more detailed parameter tuning can further improve
the accuracy of the model.

|                 Model                  |       Dataset       | Optimizer  |    Learing Rate Schedule     |   DSC   |
| :-----------------------------------: | :----------------: | :-----: | :---------------: | :-------: |
| ITUNet+FocalLoss | PI-CAI(Fold 1) |  Adam   |      PolyLR       |   46.99	  |
| ITUNet+AWA+CZPZ+BoundaryDoULoss | PI-CAI(Fold 1) |  Adam   |      PolyLR       |   49.50  |
|    ITUNet+AWA+CZPZ+BoundaryDoULoss     | PI-CAI(Fold 1) |   SGD   |      PolyLR       |   47.95   |
|    ITUNet+AWA+CZPZ+BoundaryDoULoss     | PI-CAI(Fold 1) | Adagrad |      PolyLR       |   47.79   |
|    ITUNet+AWA+CZPZ+BoundaryDoULoss     | PI-CAI(Fold 1) |  Adam   | ReduceLROnPlateau |   50.76   |
|    ITUNet+AWA+CZPZ+BoundaryDoULoss     | PI-CAI(Fold 1) |  Adam   | CosineAnnealingLR | **51.03** |