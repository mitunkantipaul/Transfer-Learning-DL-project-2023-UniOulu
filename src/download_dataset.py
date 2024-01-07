import os
import tarfile 
from .utils import check_folder_exists, extract_zip_file, download_zip_file, delete_file

def download_MiniImageNet(root, url):

    MiniImageNet_file_path = os.path.join(root, 'datasets', 'mini-imagenet')
    MiniImageNet_zip = os.path.join(root, "mini-imagenet.zip")

    if not check_folder_exists(MiniImageNet_file_path):
        download_zip_file(url, MiniImageNet_zip)
        extract_MiniImageNet(root)
        delete_file(MiniImageNet_zip)
    else:
        print("Mini ImageNet directory already exists.")

def extract_MiniImageNet(root):
    
    MiniImageNet_zip = os.path.join(root, "mini-imagenet.zip")
    MiniImageNet_extract_to_path = os.path.join(root, 'datasets')
    MiniImageNet_file_path = os.path.join(root, 'datasets', 'mini-imagenet')

    # Check and extract Mini ImageNet
    if not check_folder_exists(MiniImageNet_file_path):
        print("Extracting Mini ImageNet...")
        extract_zip_file(MiniImageNet_zip, MiniImageNet_extract_to_path)

        # tar_files = os.listdir(MiniImageNet_file_path)
        # for file in tar_files:
        #     with tarfile.open(os.path.join(MiniImageNet_file_path, file), 'r') as tar:
        #         tar.extractall(path=os.path.join(MiniImageNet_file_path))
        #         delete_file(os.path.join(MiniImageNet_file_path, file))
    else:
        print("Mini ImageNet directory already exists.")



def download_dataset(root, url, zip_file_name):

    dataset_file_path = os.path.join(root, 'datasets', zip_file_name)
    dataset_zip = os.path.join(root, f"{zip_file_name}.zip")

    if not check_folder_exists(dataset_file_path):
        download_zip_file(url, dataset_zip)
        extract_dataset(root, zip_file_name)
        delete_file(dataset_zip)
    else:
        print(f"{zip_file_name} directory already exists.")

def extract_dataset(root, zip_file_name):
    
    dataset_extract_to_path = os.path.join(root, 'datasets')
    dataset_file_path = os.path.join(root, 'datasets', zip_file_name)
    dataset_zip = os.path.join(root, f"{zip_file_name}.zip")

    # Check and extract Dataset
    if not check_folder_exists(dataset_file_path):
        print(f"Extracting {zip_file_name}...")
        extract_zip_file(dataset_zip, dataset_extract_to_path)

        # tar_files = os.listdir(dataset_file_path)
        # for file in tar_files:
        #     with tarfile.open(os.path.join(dataset_file_path, file), 'r') as tar:
        #         tar.extractall(path=os.path.join(dataset_file_path))
        #         delete_file(os.path.join(dataset_file_path, file))
    else:
        print(f"{zip_file_name} directory already exists.")