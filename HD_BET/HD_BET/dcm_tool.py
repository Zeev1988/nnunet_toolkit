import os
import logging
import dicom2nifti
from pathlib import Path


def dcm2nii(subject: tuple, output_dir: str, shrink: bool, modalities: list):
    patient, study = subject
    ext = '.nii.gz' if shrink else '.nii'

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    modules = list(study.values())[0]
    print('Converting dcm series to nifti format\n')
    for modality in modalities:
        if modules[modality]:
            if not os.path.isdir(modules[modality]):
                continue

            logging.info("Dicom2Nifty - converting subject dicom to nifty started")

            out_file = os.path.join(output_dir, f'{modality}{ext}')
            dicom2nifti.dicom_series_to_nifti(original_dicom_directory=modules[modality],
                                              output_file=out_file,
                                              reorient_nifti=False)

            modules[modality] = out_file
    modules['is_dcm'] = False

    for f in os.listdir(output_dir):
        if f.endswith(".json"):
            os.remove(os.path.join(output_dir, f))

    logging.info("Dicom2Nifty - converting subject dicom to nifty ended")



