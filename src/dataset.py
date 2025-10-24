import tensorflow as tf
import nrrd
import numpy as np

# Load data automatically (too lazy to manually write out each path)
# TODO: integrate preprocessing?
def load_hanseg():
    nrrd_ct_data = [] #1024x1024 images with varying depth
    nrrd_mri_data = [] #512x512 images with varying depth
    nrrd_segment_data = [] #1024x1024 images with varying depth

    for i in range(1, 43):
        case_num = f"{i:02d}"
        filepath = f"../HaN-Seg/set_1/case_{case_num}/case_{case_num}_IMG_CT.nrrd"
        data, header = nrrd.read(filepath, index_order='C')
        nrrd_ct_data.append(data)

    for i in range (1, 42):
        case_num = f"{i:02d}"
        filepath = f"../HaN-Seg/set_1/case_{case_num}/case_{case_num}_IMG_MR_T1.nrrd"
        data, header = nrrd.read(filepath, index_order='C')
        nrrd_mri_data.append(data)


    for i in range (1, 42):
        case_num = f"{i:02d}"
        filepath = f"../HaN-Seg/set_1/case_{case_num}/case_{case_num}_OAR_Bone_Mandible.seg.nrrd"
        data, header = nrrd.read(filepath, index_order='C')
        nrrd_segment_data.append(data)

    return nrrd_ct_data, nrrd_mri_data, nrrd_segment_data

# Creates a dataset using tensorflow for CNN
def tf_load():
    nrrd_ct_data, nrrd_mri_data, nrrd_segment_data = load_hanseg()

    nrrd_ct_data = np.expand_dims(nrrd_ct_data, axis=-1)
    nrrd_mri_data = np.expand_dims(nrrd_mri_data, axis=-1)
    nrrd_segment_data = np.expand_dims(nrrd_segment_data, axis=-1)

    dataset = tf.data.Dataset.from_tensor_slices((nrrd_ct_data, nrrd_segment_data))
    # batch & shuffle?

    return dataset
