# Transfer Learning: Deep Learning Final Project 2023 | Universiy of Oulu
Authors: Mitun Kanti Paul, Akash Sinha Bappy, Tanvir Shuvo

## Transfer learning-based image classification

**Motivation:** In some application domains, we cannot get a large amount of data, which makes it difficult or even impossible to train the deep learning models from scratch. One common approach to address this problem is transfer learning. Researchers attempt to address this problem with transfer learning.

* Understand what is transfer learning, and why transfer learning can help to address this problem. 
* **Goal:** Improve the performance in remote sensing application with small dataset via transfer learning.


# Table of Contents

- [Transfer Learning: Deep Learning Final Project 2023 | Universiy of Oulu](#transfer-learning-deep-learning-final-project-2023--universiy-of-oulu)
  - [Transfer learning-based image classification](#transfer-learning-based-image-classification)
- [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Overview](#overview)
  - [Installation](#installation)
  - [License](#license)


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





## Overview

This project centers on applying transfer learning to enhance the efficacy of deep learning models for remote sensing applications, specifically addressing challenges posed by limited datasets. The primary datasets utilized are [Mini-ImageNet](https://lyy.mpi-inf.mpg.de/mtl/download/), a subset designed for few-shot learning, and [EuroSat-RGB](https://github.com/phelber/EuroSAT), a benchmark dataset for land use classification. The **ResNet-50** architecture is chosen as the primary model for Mini-ImageNet training, with preliminary exploration involving **ResNet-18** and **VGG16**. Key steps involve _data preparation_, _preprocessing with augmentation techniques_, _model selection_, _hyperparameter optimization using Optuna_, and _iterative training processes for both Mini-ImageNet and EuroSat_. The project aims to demonstrate the potential of transfer learning in overcoming data scarcity issues in remote sensing, providing valuable insights and outcomes for real-world applications.

## Installation

The installation is pretty much straight forward. all of the dependencies are included in requirement.txt in the root directory. So running the following command in terminal will install all the necessary dependencies. 

```
pip3 install -r requirement.txt
```
while running the traning the dataset will be downloaded from a dropbox link automatically. So no setup necessary for dataset whatsoever. 

As we used optuna for hypermeter auto optimizations, a range of the parameters can be set in the `run_tuning_miniimagenet.py` and `run_tuning_eurosat.py`. All the input variables are written in capital cases to find them easily. 

To train the model with a specific hyperparametere configaration, the variable values can be set in  `train_miniImagenet.py` and `training_EuroSAT.py`

to run any script, navigate to the root directory and run like the following manner. 
```
python run_tuning_miniimagenet.py
```
the code structure is in modular form so the funnctions and classes which are reused in multiple scripts are written separatelty and kept in **src** folder. 



## License

This project is licensed under the [MIT License](./LICENSE).
