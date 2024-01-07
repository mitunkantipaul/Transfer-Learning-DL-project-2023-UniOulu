# Project Documentation

## Folder Structure

- [database](database): Database-related files. 

- [datasets](datasets): Datasets used in the project.
  - [Mini-ImageNet](datasets/Mini-ImageNet): Mini-ImageNet dataset of 100 classes.
    - [train_train](datasets/Mini-ImageNet/train_train): Train subset of 64 class.
    - [train_val](datasets/Mini-ImageNet/train_val): Validation subset of 64 class.
    - [train_test](datasets/Mini-ImageNet/train_test): Test subset of 64 class.
    - [val](datasets/Mini-ImageNet/val): Val subset(not used).
    - [test](datasets/Mini-ImageNet/test): Test subset(not used).
  
  - [EuroSAT_RGB_100](datasets/EuroSAT_RGB_100): EuroSAT dataset of _**(5 class * 20 images) = 100**_ images 
      - Five class folders

  - [EuroSAT_PR](datasets/EuroSAT_PR): EuroSAT_RGB_100 dataset splitted into _**[25,25,50:Train,Val,Test]**_ ratio.
      - [train](datasets/EuroSAT_PR/train): Train subset.
      - [val](datasets/EuroSAT_PR/val): Validation subset.
      - [test](datasets/EuroSAT_PR/test): Test subset.
        

- [logs](logs): Log files of hyper-parameter tuning.

- [models](models): Model files in **_.pth_** format

- [plots](plots): Plots and visualizations.
  - [not pretrained](plots/not%20pretrained): Plots for models without pretraining.
  - [Pretrained](plots/Pretrained): Plots for pretrained models.

- [src](src): Source code files.
  - [configs.py](src/configs.py): Plots for pretrained models.
  - [utils.py](src/utils.py): Plots for pretrained models.
  - [download_dataset.py](src/download_dataset.py): Plots for pretrained models.
  - [data_loader.py](src/data_loader.py): Plots for pretrained models.
  - [models.py](src/models.py): Plots for pretrained models.
  - [logger.py](src/logger.py): Plots for pretrained models.
  - [train.py](src/train.py): Plots for pretrained models.
  - [hyp_tune_miniimagenet.py](src/hyp_tune_miniimagenet.py): Plots for pretrained models.
  - [hyp_tune_eurosat.py](src/hyp_tune_eurosat.py): Plots for pretrained models.
  
src\data_loader.py
src\download_dataset.py
src\hyp_tune_eurosat.py
src\hyp_tune_miniimagenet.py
src\logger.py
src\models.py
src\train.py
C:\DL\deep learning project\src\transforms.py
src\utils.py

## Documentation Structure

- [Project Documentation](#project-documentation)
  - [Folder Structure](#folder-structure)
  - [Documentation Structure](#documentation-structure)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## Overview

Provide a brief overview of your project, its purpose, and goals.

## Installation

Include instructions on how to set up the project, install dependencies, and activate the virtual environment.

## Usage

Explain how to use your project, including any specific commands or configurations.

## Contributing

Describe how others can contribute to your project, guidelines for submitting issues or pull requests.

## License

Specify the project's license information.
