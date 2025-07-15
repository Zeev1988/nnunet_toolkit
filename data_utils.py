from typing import Tuple, Dict, List, Optional

import pandas as pd
import os
import shutil

from params import ToolKitParams


def df2data(df: pd.DataFrame, params: ToolKitParams) -> Tuple[Dict, Optional[List]]:
    data = {}
    is_train = df['is_train'].to_list() if 'is_train' in df else None
    for _, row in df.iterrows():
        modules = {params.t1_name: row[params.t1_name],
                   params.t1ce_name: row[params.t1ce_name],
                   params.t2_name: row[params.t2_name],
                   params.flair_name: row[params.flair_name],
                   params.label_name: row[params.label_name]}
        if row[params.subject_col_name] not in data:
            data[row[params.subject_col_name]] = []
        data[row[params.subject_col_name]].append({row[params.study_col_name]: modules})
    return data, is_train


def generate_data(subject: tuple, params: ToolKitParams) -> dict:
    study = subject[1]
    values = list(study.values())[0]
    return {params.subject_col_name: subject[0],
            params.study_col_name: list(study.keys())[0],
            params.t1_name: values[params.t1_name],
            params.t1ce_name: values[params.t1ce_name],
            params.t2_name: values[params.t2_name],
            params.flair_name: values[params.flair_name],
            params.label_name: values[params.label_name]}


def data2csv(data: list, output_dir: str, is_train=Optional[List]):
    df = pd.DataFrame(data)
    if is_train and len(is_train) == len(df):
        df['is_train'] = is_train
    df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)


def rename_copy_files(subject: tuple, output_dir: str, modalities: list):
    study = subject[1]
    modules = list(study.values())[0]
    for modality in modalities:
        if modules[modality]:
            in_dir = os.path.dirname(modules[modality])
            ext = '.nii.gz' if modules[modality].endswith('.gz') else '.nii'

            if in_dir != output_dir:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                shutil.copy(modules[modality], output_dir)

            new_file_name = os.path.join(output_dir, f'{modality}{ext}')
            os.rename(os.path.join(output_dir, os.path.basename(modules[modality])), new_file_name)
            modules[modality] = new_file_name
