import nibabel as nib
import numpy as np


def load_mri(file):
    img = nib.load(file)
    volume = img.get_fdata()
    return volume


def normalize(volume):
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return volume


def get_middle_slice(volume):
    slice_index = volume.shape[2] // 2
    return volume[:, :, slice_index]
