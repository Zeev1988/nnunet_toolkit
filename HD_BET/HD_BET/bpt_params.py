from dataclasses import dataclass


@dataclass
class BptParams:
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