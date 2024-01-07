import os
import torch
import zipfile
import requests
import shutil
import tarfile
import random
import matplotlib.pyplot as plt

def check_folder_exists(folder_path):
    """
    Check if a folder exists.
    :param folder_path: Path to the folder.
    :return: True if exists, False otherwise.
    """
    return os.path.exists(folder_path) and os.path.isdir(folder_path)

def download_zip_file(url, destination):
    """
    Download a ZIP file from a URL.
    :param url: URL of the ZIP file to download.
    :param destination: Local path to save the downloaded file.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded file to {destination}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def extract_zip_file(file_path, extract_to_path):
    """
    Extract a ZIP file to a specified directory.
    :param file_path: Path to the ZIP file.
    :param extract_to_path: Path where to extract the files.
    """
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    print(f"Extracted files to {extract_to_path}")

def delete_file(file_path):
    """
    Delete a file.

    Parameters:
    - file_path (str): Path to the file to be deleted.

    Returns:
    - None
    """
    try:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def delete_directory(directory_path):
    """
    Delete a directory and its contents.

    Parameters:
    - directory_path (str): Path to the directory to be deleted.

    Returns:
    - None
    """
    try:
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' and its contents deleted successfully.")
    except FileNotFoundError:
        print(f"Directory '{directory_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def shuffle_and_split(input_directory, output_directory, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    """
    Shuffle and split images into train, validation, and test sets.

    Parameters:
    - input_directory (str): Path to the input directory containing class subdirectories.
    - output_directory (str): Path to the output directory where the split datasets will be saved.
    - train_ratio (float): Ratio of images to include in the training set (default is 0.8).
    - val_ratio (float): Ratio of images to include in the validation set (default is 0.1).
    - test_ratio (float): Ratio of images to include in the test set (default is 0.1).
    - seed (int): Seed for the randomization (optional).

    Returns:
    - None
    """
    if seed is not None:
        random.seed(0)

    for class_folder in os.listdir(input_directory):
        class_path = os.path.join(input_directory, class_folder)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            random.shuffle(images)

            total_images = len(images)
            train_split = int(train_ratio * total_images)
            val_split = int(val_ratio * total_images)

            train_images = images[:train_split]
            val_images = images[train_split:train_split + val_split]
            test_images = images[train_split + val_split:]

            for split, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
                split_path = os.path.join(output_directory, split, class_folder)
                os.makedirs(split_path, exist_ok=True)

                for image in split_images:
                    source_path = os.path.join(class_path, image)
                    destination_path = os.path.join(split_path, image)
                    shutil.copyfile(source_path, destination_path)

def selct_images_from_classes(input_directory, output_directory, selct_classes=5, selct_images_per_class=20, seed=None):
    """
    Select specific number of images from random specific number of claases .

    Parameters:
    - input_directory (str): Path to the input directory containing class subdirectories.
    - output_directory (str): Path to the output directory where the split datasets will be saved.
    - selct_classes (int): Ratio of images to include in the training set (default is 0.8).
    - selct_images_per_class (int): Ratio of images to include in the validation set (default is 0.1).
    - seed (int): Seed for the randomization (optional).

    Returns:
    - None
    """
    if seed is not None:
        random.seed(0)
    classes = os.listdir(input_directory)
    classes = random.sample(classes, selct_classes)
    for class_folder in classes:
        class_path = os.path.join(input_directory, class_folder)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            random.shuffle(images)
            images = random.sample(images, selct_images_per_class)

            for image in images:
                source_path = os.path.join(class_path, image)
                destination_dir = destination_path = os.path.join(output_directory, class_folder)
                destination_path = os.path.join(output_directory, class_folder, image)
                os.makedirs(destination_dir, exist_ok=True)
                shutil.copyfile(source_path, destination_path)
    print(f'Dataset is created:{output_directory}')


def save_model(model, file_path):
    """
    Save PyTorch model to a .pth file.

    Parameters:
    - model: PyTorch model to be saved.
    - file_path: File path where the model will be saved.
    """
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses=None):

    # Move tensors to CPU before plotting
    train_accuracies = train_accuracies.cpu().numpy() if torch.is_tensor(train_accuracies) and train_accuracies.is_cuda else train_accuracies
    val_accuracies = val_accuracies.cpu().numpy() if torch.is_tensor(val_accuracies) and val_accuracies.is_cuda else val_accuracies
    train_losses = train_losses.cpu().numpy() if torch.is_tensor(train_losses) and train_losses.is_cuda else train_losses
    if val_losses:
        val_losses = val_losses.cpu().numpy() if torch.is_tensor(val_losses) and val_losses.is_cuda else val_losses

    # Plotting
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    return plt