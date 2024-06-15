# %%
import os
import h5py
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


# %%
# Separate resampled peripheral band (pz) and central band (tz) masks
pz_tz_dir = "/path/to/picai_anatomy_mask"
tz_dir = "/path/to/picai_tz_mask"
pz_dir = "/path/to/picai_pz_mask"
os.makedirs(pz_dir, exist_ok=True)
os.makedirs(tz_dir, exist_ok=True)


data_list = os.listdir(pz_tz_dir)
for data in tqdm(data_list):
    pz_tz_path = os.path.join(pz_tz_dir, data, data+'_seg.nii.gz')

    data_id = data.split('_0000')[0]
    pz_tz_mask = sitk.ReadImage(pz_tz_path)
    pz_tz_array = sitk.GetArrayFromImage(pz_tz_mask)
    labels = np.unique(pz_tz_array)

    for label in labels:
        if label == 1:
            tz_array = (pz_tz_array == label).astype(np.uint8)
            tz_mask = sitk.GetImageFromArray(tz_array)
            tz_mask.CopyInformation(tz_mask)
            output_path = os.path.join(tz_dir, data_id+".nii.gz")
            sitk.WriteImage(tz_mask, output_path)
        elif label ==2:
            pz_array = (pz_tz_array == label).astype(np.uint8)
            pz_mask = sitk.GetImageFromArray(pz_array)
            pz_mask.CopyInformation(pz_mask)
            output_path = os.path.join(pz_dir, data_id+".nii.gz")
            sitk.WriteImage(pz_mask, output_path)
        else:
            continue


# %%
# Convert pz and tz masks into 2D slices
def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()

def store_images_labels_2d(save_path, patient_id, labels, key):

    for i in range(labels.shape[0]):
        lab = labels[i,:,:]

        hdf5_file = h5py.File(os.path.join(save_path, '%s_%d.hdf5' % (patient_id, i)), 'w')
        hdf5_file.create_dataset(key, data=lab.astype(np.uint8))
        hdf5_file.close()

def make_segdata(mask_dir, lesions_dir, output_dir,pz_tz='pz'):

    os.makedirs(output_dir, exist_ok=True)

    data_dir_2d = os.path.join(output_dir,pz_tz+'_2d')
    os.makedirs(data_dir_2d, exist_ok=True)
    data_dir_3d = os.path.join(output_dir,pz_tz+'_3d')
    os.makedirs(data_dir_3d, exist_ok=True)

    count = 0

    pathlist = os.listdir(lesions_dir)
    pathlist = sorted(list(set(pathlist)))

    for path in tqdm(pathlist):
        lesions_mask = sitk.ReadImage(os.path.join(lesions_dir, path))
        lesions_array = sitk.GetArrayFromImage(lesions_mask).astype(np.uint8)
        if np.max(lesions_array) == 0:
            continue
        
        mask = sitk.ReadImage(os.path.join(mask_dir, path))
        array = sitk.GetArrayFromImage(mask).astype(np.uint8)

        hdf5_path = os.path.join(data_dir_3d, str(count) + '.hdf5')

        save_as_hdf5(array, hdf5_path, pz_tz)

        store_images_labels_2d(data_dir_2d, count, array, pz_tz)

        count += 1

    print(count)

lesions_dir = '/path/to/PICAI/output/nnUNet_raw_data/Task2201_picai_baseline/labelsTr'
tz_mask_dir = '/path/to/picai_tz_mask'
pz_mask_dir = '/path/to/picai_pz_mask'
output_dir = '/path/to/PICAI/output/segmentation/segdata'

make_segdata(tz_mask_dir, lesions_dir, output_dir, pz_tz='tz')
make_segdata(pz_mask_dir, lesions_dir, output_dir, pz_tz='pz')

