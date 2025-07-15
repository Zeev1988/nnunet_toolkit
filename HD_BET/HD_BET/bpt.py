import pandas as pd
import os
import torch
import gc
from tqdm import tqdm

from HD_BET.HD_BET import dcm_tool, registration
from HD_BET.HD_BET.checkpoint_download import maybe_download_parameters
from HD_BET.HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict
from data_utils import df2data, rename_copy_files, generate_data, data2csv
from params import ToolKitParams

BET_SCANS_DIR = 'BET_SCANS'
REG_SCANS_DIR = 'REG_SCANS'
RAW_SCANS_DIR = 'RAW_SCANS'

class BrainPreProcessingTool:

    def __init__(self, params: ToolKitParams):
        self.params = params


    def preprocess(self):
        """
        This is the main function for brain pre processing.
        """
        save_dcm_output = self.params.dcm_save_out
        save_reg_output = self.params.perform_reg and self.params.reg_save_out
        save_bet_output = self.params.perform_bet and self.params.bet_save_out
        df = pd.read_csv(self.params.csv_path, dtype=str).fillna('')
        data, is_train = df2data(df, self.params)
        reg_summary = []
        bet_summary = []

        bet_out_dir = os.path.join(self.params.out_path, BET_SCANS_DIR)
        reg_out_dir = os.path.join(self.params.out_path, REG_SCANS_DIR)
        raw_out_dir = os.path.join(self.params.out_path, RAW_SCANS_DIR)

        bet_dir = bet_out_dir
        reg_dir = reg_out_dir if save_reg_output else bet_dir
        raw_dir = raw_out_dir if save_dcm_output else reg_dir

        maybe_download_parameters()
        predictor = get_hdbet_predictor(
            use_tta=True,
            device=torch.device(0)
        )
        total_subjects = sum(len(studies) for _, studies in data.items())
        processed = 0

        for subject_name, studies in tqdm(data.items(), desc=f'Running brain pre processing'):
            for study in studies:
                try:
                    print(f'\nRunning the pre processing tool for patient: {subject_name}, '
                          f'series: {str(list(study.keys())[0])}\n')
                    subject = (subject_name, study)
                    study_id = str(list(study.keys())[0])

                    yield {
                        "subject_name": subject_name,
                        "study": study_id,
                        "progress": f"{(processed / total_subjects) * 100:.1f}%",
                        "status": "processing"
                    }

                    if not self.params.overwrite and (os.path.exists(os.path.join(raw_dir, subject_name, study_id)) or
                                                      os.path.exists(os.path.join(reg_dir, subject_name, study_id)) or
                                                      os.path.exists(os.path.join(bet_dir, subject_name, study_id))):
                        processed += 1
                        continue

                    # convert dicom files to .nii/.nii.gz format
                    out_dir = os.path.join(raw_dir, subject_name, study_id)
                    dcm_tool.dcm2nii(subject, out_dir, self.params.shrink_output, self.params.modalities)
                    rename_copy_files(subject, out_dir, self.params.modalities)

                    # co-register images
                    if self.params.perform_reg:
                        out_dir = os.path.join(reg_dir, subject_name, study_id)
                        registration.register(subject, self.params.elastix_exe, self.params.elastix_params, out_dir,
                                              self.params.modalities, self.params.reg_fixed_module, self.params.label_name,
                                              self.params.shrink_output)
                        if save_reg_output:
                            reg_summary.append(generate_data(subject, self.params))

                    # BET
                    if self.params.perform_bet:
                        study = subject[1]
                        label_file = list(study.values())[0][self.params.label_name]
                        os.rename(label_file, label_file.replace(".nii.gz", ""))
                        out_dir = os.path.join(bet_dir, subject_name, study_id)
                        hdbet_predict(out_dir, out_dir, predictor)
                        os.rename(label_file.replace(".nii.gz",""), label_file)
                    bet_summary.append(generate_data(subject, self.params))

                    processed += 1
                    yield {
                        "subject_name": subject_name,
                        "study": study_id,
                        "status": "completed"
                    }

                except Exception as e:
                    processed += 1
                    yield {
                        "subject_name": subject_name,
                        "status": "failed",
                        "error": str(e)
                    }

            gc.collect()

        if save_reg_output:
            data2csv(reg_summary, reg_dir, is_train)
        if save_bet_output:
            data2csv(bet_summary, bet_dir, is_train)
        yield {
            "status": "finished",
            "last_output_directory": bet_dir,
        }

