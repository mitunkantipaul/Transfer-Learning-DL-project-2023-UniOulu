# Transfer Learning: Deep Learning Final Project 2023 | Universiy of Oulu
Authors: Mitun Kanti Paul, Akash Sinha Bappy, Tanvir Shuvo

## Project Structure

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
  - [configs.py](src/configs.py): cofigurations of project.
  - [utils.py](src/utils.py): Utility functions.
  - [download_dataset.py](src/download_dataset.py): Data download functions.
  - [data_loader.py](src/data_loader.py): Dataloader functions.
  - [models.py](src/models.py): Models definition.
  - [logger.py](src/logger.py): Defining logger.
  - [train.py](src/train.py): train and evaluation functions.
  - [hyp_tune_miniimagenet.py](src/hyp_tune_miniimagenet.py): Objective function Definintion for Mini-ImageNet.
  - [hyp_tune_eurosat.py](src/hyp_tune_eurosat.py): Objective function Definintion for EuroSAT.

- [run_tuning_miniimagenet.py](run_tuning_miniimagenet.py): Run hyper-parameter tuning on Mini-ImageNet.
- [run_tuning_eurosat.py](run_tuning_eurosat.py): Run hyper-parameter tuning on EuroSAT.
- [train_miniImagenet.py](train_miniImagenet.py): Run trainig and testing on Mini-ImageNet.
- [training_EuroSAT.py](training_EuroSAT.py): Run trainig and testing on EuroSAT.
- [requirements.txt](requirements.txt): pip install requirements.txt.
- [readme.md](readme.md): ReadMe file.



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


## License

Specify the project's license information.
