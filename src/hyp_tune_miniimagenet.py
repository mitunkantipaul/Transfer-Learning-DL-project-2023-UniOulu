import os
import time
import torch
import optuna
import logging
import torch.nn as nn
from .data_loader import load_data
from .models import get_model, get_optimizer
from .train import train, eval

# Get current working directory
cwd = os.getcwd()
# Configure logging to write to a specific file
log_file_path = os.path.join(cwd, 'logs', 'logfile.log')  # Specify your path and file name
logging.basicConfig(level=logging.INFO, 
                    filename=log_file_path, 
                    filemode='a',  # 'a' for append, 'w' for overwrite
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Logging configuration
logger = logging.getLogger(__name__)


MiniImageNet_path = os.path.join(cwd, 'datasets', 'Mini-ImageNet')

# Callback function
def my_callback(study, trial):
    if study.best_trial == trial:
        logger.info(f"New best trial: {trial.number} with value: {trial.value}")

def objective(trial):
    # Capture the start time
    start_time = time.time()

    # Define hyperparameters using Optuna's trial object
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    optimizer_name = trial.suggest_categorical('optimizer', ['RMSprop'])  #['Adam', 'RMSprop', 'SGD']
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    criterion = nn.CrossEntropyLoss()
    num_classes = 64
    num_epochs = 10
    

    # Uniform parameters for momentum, beta1, and beta2
    momentum = trial.suggest_float('momentum', 0.1, 1.0)
    beta1 = trial.suggest_float('beta1', 0.8, 0.999)
    beta2 = trial.suggest_float('beta2', 0.8, 0.999)

    optimizer_args = {
        "optimizer_name": optimizer_name,
        "lr": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "beta1": beta1,
        "beta2": beta2,
    }
    
    # Model setup
    model_name = 'resnet18'
    model = get_model(
                    model_name=model_name, 
                    num_classes=num_classes, 
                    pretrained=False,  
                    dropout_rate=dropout_rate
                    )

    # Log the hyperparameters for each trial
    logger.info(f"Trial {trial.number}: lr = {lr}, optimizer = {optimizer_name}, batch_size = {batch_size}, dropout_rate = {dropout_rate}, weight_decay = {weight_decay}, momentum = {momentum}, beta1 = {beta1}, beta2 = {beta2}")


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print (device)
    # Load MiniImageNet
    print('Loading MiniImageNet.....')
    train_loader, valid_loader, test_loader = load_data(data_path=MiniImageNet_path, batch_size=batch_size)

    # Train Model
    print('[Training]:Please wait patiently, it may take some time...')
    trained_network, train_acc, val_acc, _ = train(
                                                model=model, 
                                                train_loader=train_loader, 
                                                valid_loader=valid_loader, 
                                                epoches=num_epochs, 
                                                criterion=criterion, 
                                                optimizer_args=optimizer_args
                                                )

    print("Testing Model.....")
    # Test Model
    test_acc = eval(
                    model=trained_network, 
                    data_loader=test_loader
                    )
    print('accuracy on testing data: %f' % test_acc)

    # Capture the end time
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    # Log other accurcies
    logger.info(f"Trial {trial.number}: train_acc = {train_acc}, val_acc = {val_acc}, test_acc = {test_acc}, elapsed_time = {elapsed_time}")


    # Print the elapsed time
    print(f"Elapsed time of trial {trial.number}: {elapsed_time} seconds")

    return  val_acc


def tuner_MiniImageNet():
    # Database URL
    db_path = os.path.join(cwd, 'database', 'tuning_study_1.db')

    # Create or load a study
    study = optuna.create_study(
                                study_name="study_1", 
                                direction='maximize', 
                                storage=f'sqlite:///{db_path}', 
                                load_if_exists=True)

    # Start or resume optimization
    study.optimize(objective, n_trials=100, callbacks=[my_callback])

    print("Best hyperparameters: ", study.best_trial.params)
    logger.info(f"Best hyperparameters: {study.best_trial.params}")

