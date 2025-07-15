import json
import os
import pickle
import json
import re
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

from params import ToolKitParams


def create_json_file_from_dir(task_dir: str, patient_mapping: dict, num_folds: int = 5):
    labelsTr_dir = os.path.join(task_dir, "labelsTr")
    all_cases = []

    # Get all case IDs from labelsTr directory
    for file in os.listdir(labelsTr_dir):
        if file.endswith('.nii.gz'):
            case_id = os.path.splitext(os.path.splitext(file)[0])[0]
            all_cases.append(case_id)

    # Extract patient groups for stratification
    patient_groups = [patient_mapping.get(case_id, case_id) for case_id in all_cases]

    # Use GroupKFold to ensure patients stay together in same fold
    splits = []
    gkf = GroupKFold(n_splits=num_folds)

    # Convert to numpy array for sklearn
    all_cases_array = np.array(all_cases)

    for train_idx, val_idx in gkf.split(all_cases_array, groups=patient_groups):
        train_cases = all_cases_array[train_idx].tolist()
        val_cases = all_cases_array[val_idx].tolist()
        splits.append({'train': train_cases, 'val': val_cases})

    # Save splits to JSON file
    with open(os.path.join(task_dir, "splits_final.json"), 'w') as f:
        json.dump(splits, f, indent=2)


def create_pickle_file_from_dir(task_dir: str, patient_mapping: dict, num_folds: int = 5):
    labelsTr_dir = os.path.join(task_dir, "labelsTr")
    all_cases = []

    # Get all case IDs from labelsTr directory
    for file in os.listdir(labelsTr_dir):
        if file.endswith('.nii.gz'):
            case_id = os.path.splitext(os.path.splitext(file)[0])[0]
            all_cases.append(case_id)

    # Extract patient groups for stratification
    patient_groups = [patient_mapping.get(case_id, case_id) for case_id in all_cases]

    # Use GroupKFold to ensure patients stay together in same fold
    splits = []
    gkf = GroupKFold(n_splits=num_folds)

    # Convert to numpy array for sklearn
    all_cases_array = np.array(all_cases)

    for train_idx, val_idx in gkf.split(all_cases_array, groups=patient_groups):
        train_cases = all_cases_array[train_idx].tolist()
        val_cases = all_cases_array[val_idx].tolist()
        splits.append({'train': train_cases, 'val': val_cases})

    # Save splits to pickle file
    with open(os.path.join(task_dir, "splits_final.pkl"), 'wb') as f:
        pickle.dump(splits, f)


def create_dataset_json(params: ToolKitParams, task_dir: str):
    train_files = [(f"./imagesTr/{p}", f"./labelsTr/{p}")
             for p in os.listdir(os.path.join(task_dir, 'labelsTr')) if p.endswith('.nii.gz')]
    test_files = [(f"./imagesTs/{p}")
             for p in os.listdir(os.path.join(task_dir, 'labelsTs')) if p.endswith('.nii.gz')]
    modality = {str(int(id)): mod for mod, id in params.modality_ids.items()}
    dataset = {
        "description": "nnunet_toolkit",
        "labels": params.label_ids,
        "licence": "",
        "channel_names": modality,
        "name": "nnunet_toolkit",
        "numTraining": len(train_files),
        "numTest": len(test_files),
        "training": [{"image": f[0], "label": f[1]} for f in train_files],
        "test": test_files,
        "file_ending": ".nii.gz"
    }
    with open(os.path.join(task_dir, 'dataset.json'), "w") as outfile:
        json.dump(dataset, outfile)


def create_task_folders(task_path: str):
    labels_dir = 'labelsTr'
    img_dir = 'imagesTr'
    pred_dir = 'predictTs'
    test_img = 'imagesTs'
    test_lbl = 'labelsTs'
    if os.path.exists(task_path):
        raise FileExistsError(f"Directory already exists: {task_path}")

    os.makedirs(task_path)
    os.makedirs(os.path.join(task_path, labels_dir))
    os.makedirs(os.path.join(task_path, img_dir))
    os.makedirs(os.path.join(task_path, pred_dir))
    os.makedirs(os.path.join(task_path, test_img))
    os.makedirs(os.path.join(task_path, test_lbl))


def ichilov_data_to_nnunet_format(csv_path: str, params: ToolKitParams, task_dir: str):
    ext = '.nii.gz'
    create_task_folders(task_dir)

    df = pd.read_csv(csv_path)
    patient_mapping = {}

    for index, row in df.iterrows():
        patient_name = row[params.subject_col_name]
        case_id = f'NNUNET_{index:03}'
        patient_mapping[case_id] = re.sub(r'[0-9]+', '', patient_name)

        out_dir = os.path.join(task_dir, "imagesTr") if row['is_train'] else os.path.join(task_dir, "imagesTs")
        for mod in params.modalities:
            file_name = f'{case_id}_{params.modality_ids[mod]}{ext}'
            shutil.copy(row[mod], os.path.join(out_dir, file_name))
        if params.label_name in row:
            out_dir = os.path.join(task_dir, "labelsTr") if row['is_train'] else os.path.join(task_dir, "labelsTs")
            shutil.copy(row[params.label_name],
                        os.path.join(out_dir, f'{case_id}{ext}'))

    create_dataset_json(params, task_dir)
    create_json_file_from_dir(task_dir, patient_mapping)
