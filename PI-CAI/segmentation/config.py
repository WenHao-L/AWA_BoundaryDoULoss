import glob
import os

from segmentation.utils import get_weight_path

MODEL = "itunet_d24"  # "itunet_d24", "swinunetr_d24", "unetr_d24", "unet_d24"
LOSS = 'FocalLoss' # FocalLoss, BoundaryDoULoss, TverskyLoss, BoundaryLoss
CHANNELS = 3  # 3: T2WI, ADC, DWI;   5: T2WI, ADC, DWI, CZ, PZ
USE_AW = False
USE_WSO = False
WSO_ACT_WINDOW = "relu"  # "relu" or "sigmoid"
WSO_UPBOUND_WINDOW = 1.
WSO_WINDOW_INIT = None
USE_CHECKPOINT = False

TRANSFORMER_DEPTH = 24
if USE_AW:
    VERSION = MODEL + '_aw'
elif USE_WSO:
    VERSION = MODEL + '_wso_'+str(WSO_ACT_WINDOW)+"_up"+str(WSO_UPBOUND_WINDOW)
    VERSION = VERSION+"_init" if WSO_WINDOW_INIT is not None else VERSION
else:
    VERSION = MODEL
VERSION = VERSION+"_ch"+str(CHANNELS) if CHANNELS == 5 else VERSION
VERSION = VERSION+"_dou" if LOSS == 'BoundaryDoULoss' else VERSION
VERSION = VERSION+"_tk" if LOSS == 'TverskyLoss' else VERSION
VERSION = VERSION+"_bou" if LOSS == 'BoundaryLoss' else VERSION

# VERSION = VERSION + "_CosineAnnealingLR"  # "_CosineAnnealingLR" or "_ReduceLROnPlateau"
# VERSION = VERSION + "_Adagrad"  # "_SGD" or "_Adagrad"

PHASE = 'seg'
NUM_CLASSES = 2 if 'seg' in PHASE else 3

DEVICE = '0,1'
# True if use internal pre-trained model
# Must be True when pre-training and inference
PRE_TRAINED = False
# True if use resume model
CKPT_POINT = False

FOLD_NUM = 5
# [1-FOLD_NUM]
CURRENT_FOLD = 5
GPU_NUM = len(DEVICE.split(','))

#--------------------------------- mode and data path setting
PATH_DIR = './dataset/segdata/data_2d'
PATH_LIST = glob.glob(os.path.join(PATH_DIR,'*.hdf5'))
PATH_AP = './dataset/segdata/data_3d'
AP_LIST = glob.glob(os.path.join(PATH_AP,'*.hdf5'))
#--------------------------------- 

CKPT_PATH = './ckpt/{}/{}/fold{}'.format(PHASE,VERSION,str(CURRENT_FOLD))
WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

#you could set it for your device
INIT_TRAINER = {
  'channels' : CHANNELS,
  'num_classes':NUM_CLASSES, 
  'n_epoch':150,
  'batch_size':24,  
  'num_workers':4,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'ckpt_point':CKPT_POINT,
  'weight_path':WEIGHT_PATH,
  'use_fp16':False,
  'transformer_depth': TRANSFORMER_DEPTH,
  'use_aw': USE_AW,
  'use_checkpoint': USE_CHECKPOINT,
  'use_wso': USE_WSO,
  'act_window': WSO_ACT_WINDOW,
  'upbound_window': WSO_UPBOUND_WINDOW,
  'window_init': WSO_WINDOW_INIT,
  'model': MODEL
 }
#---------------------------------

SETUP_TRAINER = {
  'tz_2d_dir': '/home/data2/wan/itunet_dataset/output/segmentation/segdata/tz_2d',
  'tz_3d_dir': '/home/data2/wan/itunet_dataset/output/segmentation/segdata/tz_3d',
  'pz_2d_dir': '/home/data2/wan/itunet_dataset/output/segmentation/segdata/pz_2d',
  'pz_3d_dir': '/home/data2/wan/itunet_dataset/output/segmentation/segdata/pz_3d',
  'output_dir':'./ckpt/{}/{}'.format(PHASE,VERSION),
  'log_dir':'./log/{}/{}'.format(PHASE,VERSION),
  'phase':PHASE,
  'loss': LOSS,
  }
