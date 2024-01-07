import os
import time
import torch
import optuna
import logging
import torch.nn as nn
from .data_loader import load_data
from .models import get_model, get_optimizer
from .train import train, eval
from .configs import ROOT, DATASETS, EURO_SAT_100_PRO_DIR
from .logger import get_logger


# Log file name
log_file_name = 'eurosat_tuning_study_2'
# file_mode='a',  # 'a' for append, 'w' for overwrite

# # Get current working directory
# cwd = os.getcwd()
# # Configure logging to write to a specific file
# log_file_path = os.path.join(cwd, 'logs', f'{log_file_name}.log')  # Specify your path and file name
# logging.basicConfig(level=logging.INFO, 
#                     filename=log_file_path, 
#                     filemode=file_mode,  # 'a' for append, 'w' for overwrite
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# Logging configuration
logger = get_logger(
    logger_name=log_file_name,
    log_file=f'./logs/{log_file_name}.log',
    log_level=logging.INFO
)




# Callback function
def my_callback(study, trial):
    if study.best_trial == trial:
        logger.info(f"New best trial: {trial.number} with value: {trial.value}")

def objective(trial):
    # Capture the start time
    start_time = time.time()

    # Define pretranind model path
    pretraing_weight_file = 'resnet50_E_70_Acc_0.713645875453949.pth'

    # Define hyperparameters using Optuna's trial object
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam'])  #['Adam', 'RMSprop', 'SGD']
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.9)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    criterion = nn.CrossEntropyLoss()
    num_classes = 5
    num_epochs = 20
    

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
    model_name = 'resnet50'
    model = get_model(
                    model_name=model_name, 
                    num_classes=num_classes, 
                    pretrained=True,  
                    pretrained_path=f'./models/{pretraing_weight_file}',
                    dropout_rate=dropout_rate
                    )

    # Log the hyperparameters for each trial
    logger.info(f"Trial {trial.number}: lr = {lr}, optimizer = {optimizer_name}, batch_size = {batch_size}, dropout_rate = {dropout_rate}, weight_decay = {weight_decay}, momentum = {momentum}, beta1 = {beta1}, beta2 = {beta2}")


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print (device)

    EuroSAT_path = os.path.join(ROOT, DATASETS, EURO_SAT_100_PRO_DIR)
    # Load EuroSAT
    print('Loading EuroSAT.....')
    train_loader, valid_loader, test_loader = load_data(data_path=EuroSAT_path, batch_size=batch_size)

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


def tuning():
    study_name="EuroSAT_study_2"

    # Database URL
    db_path = os.path.join(ROOT, 'database', f'{study_name}.db')

    # Create or load a study
    study = optuna.create_study(
                                study_name=study_name, 
                                direction='maximize', 
                                storage=f'sqlite:///{db_path}', 
                                load_if_exists=True)

    # Start or resume optimization
    study.optimize(objective, n_trials=100, callbacks=[my_callback])

    print("Best hyperparameters: ", study.best_trial.params)
    logger.info(f"Best hyperparameters: {study.best_trial.params}")

# if __name__ == '__main__':
#     main()