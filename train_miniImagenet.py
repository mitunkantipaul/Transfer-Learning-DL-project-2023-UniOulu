import os
import torch
import optuna
import numpy as np
import torch.nn as nn
import torch.optim as optim
from src.download_dataset import download_MiniImageNet
from src.data_loader import load_data
from src.models import get_model, get_optimizer
from src.train import train, eval
from src.hyp_tune_miniimagenet import objective
from src.utils import save_model, plot_metrics

# Set a random seed for PyTorch
torch.manual_seed(42)

# Hyper Parameters
NUM_CLASSES = 64  # For Mini-ImageNet
NUM_EPOCHS = 100
BATCH_SIZE = 64
OPTIMIZER = 'rmsprop'  # sgd ,
LEARNING_RATE = 0.0000424124242507866
MOMENTUM = 0.484914052750008
WEIGHT_DECAY = 0.0000850961896707493
BETA1 = 0.922352109535754
BETA2 = 0.863857451280227
MODEL = 'resnet50'  # 'vgg16',
CRITERION = nn.CrossEntropyLoss()
# regulariztion
DROP_OUT = 0.435479123335354

OPTIMIZER_ARGS = {
    "optimizer_name": OPTIMIZER,
    "lr": LEARNING_RATE,
    "momentum": MOMENTUM,
    "weight_decay": WEIGHT_DECAY,
    "beta1": BETA1,
    "beta2": BETA2,
}

# Root directory
ROOT = os.getcwd()

# MiniImageNet URL
MiniImageNet_Url = 'https://www.dropbox.com/s/a2a0bll17f5dvhr/Mini-ImageNet.zip?dl=1'

# MiniImageNet path
MiniImageNet_path = os.path.join(ROOT, "datasets", "Mini-ImageNet")


def training_MiniImageNet():
    # Download Dataset
    print('Downloading MiniImageNet.....')
    download_MiniImageNet(root=ROOT, url=MiniImageNet_Url)

    # Load MiniImageNet
    print('Loading MiniImageNet.....')
    splits = {
        'train': 'train_train',
        'val': 'train_val',
        'test': 'train_test'
    }
    train_loader, valid_loader, test_loader = load_data(MiniImageNet_path, batch_size=BATCH_SIZE, splits=splits)

    # Train Model
    model = get_model(
        model_name=MODEL,
        num_classes=NUM_CLASSES,
        pretrained=False,
        dropout_rate=DROP_OUT
    )

    print('Please wait patiently, it may take some time...')

    trained_network, train_acc, val_acc, metrics = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epoches=NUM_EPOCHS,
        criterion=CRITERION,
        optimizer_args=OPTIMIZER_ARGS
    )

    # Test Model
    acc_test = eval(
        model=trained_network,
        data_loader=test_loader
    )
    print('accuracy on testing data: %f' % acc_test)

    # Save Model
    save_model(model=trained_network, file_path=f'./models/{MODEL}_E_{NUM_EPOCHS}_Acc_{acc_test}.pth')
    # print(np.array(metrics['train_acc']))
    # plotting 
    plot = plot_metrics(
        train_accuracies=metrics['train_acc'],
        val_accuracies=metrics['val_acc'],
        train_losses=metrics['train_loss']
    )
    # Saving plot
    plot.savefig(f'./plots/{MODEL}_E_{NUM_EPOCHS}_Acc_{acc_test}.png')


if __name__ == '__main__':
    training_MiniImageNet()
