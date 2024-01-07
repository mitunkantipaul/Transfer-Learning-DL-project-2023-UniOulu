import os
import torch
import optuna
import logging
import torch.nn as nn
import torch.optim as optim
from src.download_dataset import download_dataset
from src.data_loader import load_data
from src.models import get_model, get_optimizer
from src.train import train, eval
from src.hyperparameter_tuning import objective, my_callback
from src.configs import EURO_SAT_URL, EURO_SAT_DIR, EURO_SAT_100_DIR, DATASETS
from src.utils import shuffle_and_split, selct_images_from_classes
from src.configs import ROOT, EURO_SAT_100_DIR, EURO_SAT_100_PRO_DIR
from src.hyp_tune_eurosat import tuning


# Root directory
ROOT = os.getcwd()
# EuroSAT path
EuroSAT_path = os.path.join(ROOT, DATASETS, EURO_SAT_DIR)
EuroSAT_100_path = os.path.join(ROOT, DATASETS, EURO_SAT_100_DIR)
EuroSAT_100_processed = os.path.join(ROOT, DATASETS, EURO_SAT_100_PRO_DIR)


def download_EuroSAT():
    # Download Dataset
    print('Downloading EuroSAT.....')
    download_dataset(root=ROOT, url=EURO_SAT_URL, zip_file_name=EURO_SAT_DIR)


def EuroSAT_100():
    # making 100 imge dataset from 5 classes
    print('Making 100 image EuroSAT Dataset.....')
    selct_images_from_classes(
        input_directory=EuroSAT_path,
        output_directory=EuroSAT_100_path,
        selct_classes=5,
        selct_images_per_class=20,
        seed=42
    )


def EuroSAT_processed():
    print('Splitting Dataset.....')
    shuffle_and_split(
        input_directory=EuroSAT_100_path,
        output_directory=EuroSAT_100_processed,
        train_ratio=.25,
        val_ratio=.25,
        test_ratio=0.5,
        seed=42
    )



if __name__ == '__main__':
    # download_EuroSAT()
    # EuroSAT_100()
    # EuroSAT_processed()
    tuning()
