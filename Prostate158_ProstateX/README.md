# Evaluation on Prostate158+ProstateX dataset for prostate anatomy segmentation

# Quick Start
## Dataset
- [Prostate158_train_data](https://zenodo.org/records/6481141)
- [Prostate158_test_data](https://zenodo.org/records/6592345)
- [ProstateX_data](https://www.cancerimagingarchive.net/collection/prostatex/)
- [ProstateX_label](https://github.com/rcuocolo/PROSTATEx_masks)

## Training

### 1. Data preprocessing
We provide a [notebook](./preprocessing.ipynb) that converts and partitions the dataset.

### 2. Training parameters
Modify the [anatomy.yaml](./anatomy.yaml) file.
```yaml
model_name: "unet"  # "unet", "swinunetr" or "unetr"
use_aw: False  # use the AWA module
use_wso: False  # use the WSO module
data: # data path
  data_dir: /path/to/prostate158_PROSTATEx_dataset
  train_csv: /path/to/prostate158/dataset/prostate_PROSTATEx_train.csv
  valid_csv: /path/to/prostate158/dataset/prostate_PROSTATEx_vaild.csv
  test_csv: /path/to/prostate158/dataset/prostate_PROSTATEx_test.csv
  cache_dir: /path/to/monai-cache
loss:  # loss fuction
  DiceCELoss:
    include_background: False
    softmax: True
    to_onehot_y: True
  # BoundaryDoULoss:
  #   n_classes: 3
  # BoundaryLoss:
  #   n_classes: 3
  # TverskyLoss:
  #   include_background: False
  #   softmax: True
  #   to_onehot_y: True
```

### 3. Training
```
python train.py
```

### 4. Infer PI-CAI dataset anatomy masks
```
python picai_anatomy_infer.py
```