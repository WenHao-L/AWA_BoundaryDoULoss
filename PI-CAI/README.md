# Evaluation on PI-CAI dataset for prostate lesion segmentation

# Quick Start
## Dataset
- [PI-CAI_image](https://pi-cai.grand-challenge.org/DATA/)
- [PI-CAI_label](https://github.com/DIAGNijmegen/picai_labels)

## Training

### 1. Data preprocessing
We provide scripts for processing the picai dataset and processing anatomical masks:

- [preprocess.py](./preprocess/preprocess.py)

- [preprocess_anatomy_masks](./preprocess/preprocess_anatomy_masks.py)


### 2. Training parameters
Modify the [config.py](./segmentation/config.py) file.
```python
MODEL = "itunet_d24"  # "itunet_d24", "swinunetr_d24", "unetr_d24", "unet_d24"
LOSS = 'FocalLoss' # FocalLoss, BoundaryDoULoss, TverskyLoss, BoundaryLoss
CHANNELS = 3  # 3: T2WI, ADC, DWI;   5: T2WI, ADC, DWI, CZ, PZ
USE_AW = False  # use the AWA module
USE_WSO = False  # use the WSO module
WSO_ACT_WINDOW = "relu"  # "relu" or "sigmoid"
```


### 3. Training
```
python supervised_segmentation.py
```