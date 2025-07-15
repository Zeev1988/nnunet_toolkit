from dataclasses import dataclass, field
from typing import Optional


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
    reg_fixed_module: str = t1ce_name

    ## training params ##
    configuration: str = '3d_fullres'
    optimizer: str = 'SGD'
    loss_function: str = 'DiceCE'
    folds: Optional[list] = None
    max_epochs: int = 50
    do_preprocessing_training: bool = True
    do_transfer_lr: bool = False
