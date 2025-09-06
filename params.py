import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ToolKitStage(Enum):
    TRAINING = 'training'
    INFERENCE = 'inference'
    PREPROCESSING = 'preprocessing'


@dataclass
class ToolKitParams:
    ## general params ##
    task: int
    subject_col_name: str = 'Name'
    study_col_name: str = 'Study'
    t1_name: str = 'T1'
    t2_name: str = 'T2'
    t1ce_name: str = 'T1ce'
    flair_name: str = 'FLAIR'
    label_name: str = 'Label'
    modalities: list = field(default_factory=lambda: ['T1', 'T1ce', 'T2', 'FLAIR', 'Label'])
    modality_ids: dict = field(default_factory=lambda: {'FLAIR': '0000', 'T1': '0001', 'T1ce': '0002', 'T2': '0003'})
    label_ids: dict = field(default_factory=lambda: {})

    ## bpt params ##
    csv_path: str = None
    out_path: str = None
    shrink_output: bool = True
    dcm_save_out: bool = False
    reg_save_out: bool = False
    bet_save_out: bool = True
    elastix_exe: str = r"D:\users\Yuval\BET_ZEEV\elastix-5.0.1-win64/elastix.exe"
    elastix_params: str = r"D:\users\Yuval\BET_ZEEV\elastix-5.0.1-win64/Parameters_Rigid.txt"
    bet_device: int = 0
    perform_reg: bool = True
    perform_bet: bool = True
    perform_resize: bool = False
    overwrite: bool = False
    reg_fixed_module: str = 'T1ce'

    ## training params ##
    configuration: str = '3d_fullres'
    optimizer: str = 'SGD'
    loss_function: str = 'DiceCE'
    folds: Optional[list] = None
    max_epochs: int = 50
    do_preprocessing_training: bool = True
    do_transfer_lr: bool = False

    @staticmethod
    def load_from_json(dir_path: str) -> 'ToolKitParams':
        """Load ToolKitParams from a JSON file."""
        file_path = os.path.join(dir_path, "nnunet_tollkit_params.json")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return ToolKitParams(**data)
        except:
            return None

    def save_to_json(self, dir_path: str) -> None:
        """Save ToolKitParams to a JSON file."""
        file_path = os.path.join(dir_path, "nnunet_tollkit_params.json")
        with open(file_path, 'w') as f:
            # Convert the dataclass instance to a dictionary and save to JSON
            json.dump(self.__dict__, f, indent=4)

    @staticmethod
    def update(dir_path: str, params: 'ToolKitParams', stage: ToolKitStage) -> 'ToolKitParams':
        loaded_params = ToolKitParams.load_from_json(dir_path)
        if loaded_params is not None:
            if stage == ToolKitStage.PREPROCESSING:
                loaded_params.subject_col_name = params.subject_col_name
                loaded_params.study_col_name = params.study_col_name
                loaded_params.t1_name = params.t1_name
                loaded_params.t2_name = params.t2_name
                loaded_params.t1ce_name = params.t1ce_name
                loaded_params.flair_name = params.flair_name
                loaded_params.label_name = params.label_name
                loaded_params.modalities = params.modalities
                loaded_params.modality_ids = params.modality_ids
                loaded_params.label_ids = params.label_ids
                loaded_params.shrink_output = params.shrink_output
                loaded_params.dcm_save_out = params.dcm_save_out
                loaded_params.reg_save_out = params.reg_save_out
                loaded_params.bet_save_out = params.bet_save_out
                loaded_params.elastix_exe = params.elastix_exe
                loaded_params.elastix_params = params.elastix_params
                loaded_params.bet_device = params.bet_device
                loaded_params.perform_reg = params.perform_reg
                loaded_params.perform_bet = params.perform_bet
                loaded_params.perform_resize = params.perform_resize
                loaded_params.overwrite = params.overwrite
                loaded_params.reg_fixed_module = params.reg_fixed_module
            elif stage == ToolKitStage.TRAINING:
                loaded_params.configuration = params.configuration
                loaded_params.optimizer = params.optimizer
                loaded_params.loss_function = params.loss_function
                loaded_params.folds = params.folds
                loaded_params.max_epochs = params.max_epochs
            loaded_params.save_to_json(dir_path)
            return loaded_params
        params.csv_path = ''
        params.save_to_json(dir_path)
        return params