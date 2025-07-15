# Welcome to the nnU-Net Toolkit!

## Instalation
- in conda terminal:
 - git clone https://github.com/Zeev1988/nnunet_toolkit.git
 - cd nnunet_toolkit
 - conda env create -f enviroment.yml

## Running:
- in conda terminal - set up enviroment before running:
 - conda activate conda_nnunet_toolkit
 - export nnUNet_raw="/path/to/your/nnunet_raw"
 - export nnUNet_preprocessed="/path/to/your/nnunet_preprocessed"
 - export nnUNet_preprocessed="/path/to/your/nnunet_preprocessed"
 - export nnUNet_results="/path/to/your/nnunet_results"
 - export CUDA_VISIBLE_DEVICES=0 (or any other gpu you want)
- run:
 - streamlit run ./gui.py







 
Additional information:
- [Learning from sparse annotations (scribbles, slices)](documentation/ignore_label.md)
- [Region-based training](documentation/region_based_training.md)
- [Manual data splits](documentation/manual_data_splits.md)
- [Pretraining and finetuning](documentation/pretraining_and_finetuning.md)
- [Intensity Normalization in nnU-Net](documentation/explanation_normalization.md)
- [Manually editing nnU-Net configurations](documentation/explanation_plans_files.md)
- [Extending nnU-Net](documentation/extending_nnunet.md)
- [What is different in V2?](documentation/changelog.md)

# Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).
