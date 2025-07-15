import logging
import os
from scipy.ndimage import zoom
import SimpleITK as sitk
from typing import List
import numpy as np


def bounds_per_dimension(array):
    return map(lambda e: range(e.min(), e.max() + 1), np.where(array != 0))


def zero_trim_ndarray(array):
    return array[np.ix_(*bounds_per_dimension(array))]


def resize(subject: tuple, output_shape: List[int], output_dir: str, shrink: bool):
    study = subject[1]

    print('Running the resize Tool\n')
    logging.info("resize tool started")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    label_orig = None
    label_direct = None
    modality_list = ['FLAIR','Label', 'T1', 'T1ce', 'T2']
    modules = list(study.values())[0]
    for modality in modality_list:
        if modules[modality]:
            if not os.path.exists(modules[modality]):
                logging.error("resize tool - file was not found " + modules[modality])
                print("resize tool - Error: file was not found " + modules[modality])
                continue

            if not modules[modality].endswith('.nii') and not modules[modality].endswith('.nii.gz'):
                logging.error(f"resize tool - file is not of type nifty {modules[modality]}")
                print(f"resize tool - Error: file is not of type nifty {modules[modality]}")
                continue

            out_path = os.path.join(output_dir, os.path.basename(modules[modality]))
            if shrink and modules[modality].endswith('.nii'):
                out_path = out_path.replace(".nii", ".nii.gz")

            img_itk = sitk.ReadImage(modules[modality])
            origin = label_orig if label_orig else img_itk.GetOrigin()
            direction = label_direct if label_direct else img_itk.GetDirection()
            if modality == '-FLAIR':
                label_orig = origin
                label_direct = direction

            array = sitk.GetArrayFromImage(img_itk)
            array = zoom(array, (output_shape[2] / array.shape[0],
                                 output_shape[1] / array.shape[1],
                                 output_shape[0] / array.shape[2]), order=1)

            img_itk_new = sitk.GetImageFromArray(array)
            img_itk_new.SetOrigin(origin)
            img_itk_new.SetDirection(direction)
            sitk.WriteImage(img_itk_new, out_path)

    logging.info("resize tool ended")
