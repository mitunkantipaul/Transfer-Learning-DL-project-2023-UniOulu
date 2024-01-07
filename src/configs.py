import os

"""
The pahts and directories is defined here
"""
ROOT = os.getcwd()
# PATHS
DATASETS_PATH = f'./datasets/'
LOGS_PATH = f'./logs/'
DATBASE_PATH = f'./database/'
PLOTS_PATH = f'./plots/'

# DIRECTORIES
MINI_IMAGE_NET_DIR = 'Mini-ImageNet'
EURO_SAT_DIR = 'EuroSAT_RGB'         # Original EuroSAT data
EURO_SAT_100_DIR = 'EuroSAT_RGB_100' # EuroSAT 100 image directory unsplitted
EURO_SAT_100_PRO_DIR = 'EuroSAT_PR'  # EuroSAT 100 image directory splitted 

# URLS
# MINI_IMAGE_NET_URL = ''
EURO_SAT_URL = 'https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1'

# print(os.listdir(DATASETS_PATH))