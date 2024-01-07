import os 
import copy
import torch
import torchvision
import optuna
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from src.download_dataset import download_dataset
from src.data_loader import load_data, EuroSATDataset
from src.models import get_model, get_optimizer
from src.train import train, eval
from src.hyp_tune_miniimagenet import objective
from src.utils import save_model, plot_metrics, shuffle_and_split
from src.transforms import  train_transform

# Set a random seed for PyTorch
torch.manual_seed(42)

# Define pretranind model path
pretraing_weight_file = 'resnet50_E_70_Acc_0.713645875453949.pth'

# Trial 4: lr = 0.006816680046354532, optimizer = Adam, batch_size = 64, dropout_rate = 0.2396375372357702, weight_decay = 4.2773090763216474e-05, momentum = 0.5828054012581577, beta1 = 0.8380169232606872, beta2 = 0.8223679226043014

MODEL = 'resnet50' # 'vgg16', 
CRITERION = nn.CrossEntropyLoss()

# Hyper Parameters
NUM_CLASSES = 5  # For Mini-ImageNet
NUM_EPOCHS = 40
LEARNING_RATE = 0.006816680046354532
OPTIMIZER = 'adam' # sgd ,
BATCH_SIZE = 64
DROP_OUT = 0.8396375372357702
WEIGHT_DECAY = 4.2773090763216474e-05
MOMENTUM = 0.5828054012581577
BETA1 = 0.8380169232606872
BETA2 = 0.8223679226043014


NUM_FINE_TUNE_RUNS = 5 

OPTIMIZER_ARGS = {
                "optimizer_name": OPTIMIZER,
                "lr": LEARNING_RATE,
                "momentum" : MOMENTUM,
                "weight_decay" : WEIGHT_DECAY,
                "beta1" : BETA1,
                "beta2" : BETA2,
            }


# Root directory
ROOT = os.getcwd()

# EuroSAT path
EuroSAT_path = os.path.join(ROOT, 'datasets', 'EuroSAT_RGB_100')


def training_EuroSAT():

    # Load the EuroSAT dataset
    dataset = EuroSATDataset(root=EuroSAT_path, transform=transforms.ToTensor())

    # Create a DataLoader for the entire dataset
    # full_data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # # Load EuroSAT
    # print('Loading EuroSAT.....')
    # train_loader, valid_loader, test_loader = load_data(EuroSAT_path, batch_size=BATCH_SIZE)

    # Get pretrained Model
    pretrained_model = get_model(
                    model_name=MODEL, 
                    num_classes=NUM_CLASSES, 
                    pretrained=True, 
                    pretrained_path=f'./models/{pretraing_weight_file}',
                    dropout_rate=DROP_OUT
                    )

    print('Please wait patiently, it may take some time...')

    average_accuracy = 0.0

    for run in range(NUM_FINE_TUNE_RUNS):
        print(f"\nRun {run + 1}/{NUM_FINE_TUNE_RUNS}")
        # Split the dataset into training (25 images) and validation (75 images)
        train_size = 25
        val_size = 75
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_dataset.dataset.transform = train_transform
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Clone the original model for each run
        model = copy.deepcopy(pretrained_model)

        trained_network, train_acc, val_acc, metrics = train(
                                                    model=model, 
                                                    train_loader=train_loader, 
                                                    valid_loader=val_loader, 
                                                    epoches=NUM_EPOCHS, 
                                                    criterion=CRITERION, 
                                                    optimizer_args=OPTIMIZER_ARGS
                                                    )

        average_accuracy += val_acc

        # Save the fine-tuned model for each run if needed
        save_model(model=model, file_path=f"./models/EuroSAT_fine_tuned_model_run_{run}.pth")

        # plotting 
        plot = plot_metrics(
            train_accuracies=metrics['train_acc'], 
            val_accuracies=metrics['val_acc'], 
            train_losses=metrics['train_loss'] 
            )
        # Saving plot
        plot.savefig(f'./plots/EuroSAT_fine_tuned_model_run_{run}.png')

    average_accuracy /= NUM_FINE_TUNE_RUNS
    print(f"\nAverage validation Accuracy over {NUM_FINE_TUNE_RUNS} runs: {average_accuracy}") 




if __name__ == '__main__':
    training_EuroSAT()