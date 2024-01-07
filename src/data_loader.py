import os
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class EuroSATDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # Assuming that EuroSAT dataset is organized as subfolders by class
        self.dataset = ImageFolder(root=self.root, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Retrieve the image and its label using the ImageFolder dataset
        img, label = self.dataset[index]

        return img, label


def load_data(data_path, batch_size, splits=None):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        # Previous augmentations:
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomVerticalFlip(),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomAutocontrast(p=0.2),
        # New advanced augmentations:
        transforms.RandomInvert(p=0.2),  # Randomly inverts the colors of the image
        transforms.RandomSolarize(threshold=128, p=0.2),  # Randomly applies solarization
        transforms.RandomPosterize(bits=2, p=0.2),  # Reduces the number of bits for each color channel
        transforms.RandomEqualize(p=0.2),  # Applies histogram equalization
        # Standard normalization:
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    if splits:
        train_split = splits['train']
        val_split = splits['val']
        test_split = splits['test']

    else:
        train_split = 'train'
        val_split = 'val'
        test_split = 'test'

    train_set = torchvision.datasets.ImageFolder(root=os.path.join(data_path, train_split), transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)

    val_set = torchvision.datasets.ImageFolder(root=os.path.join(data_path, val_split), transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=3)

    test_set = torchvision.datasets.ImageFolder(root=os.path.join(data_path, test_split), transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=3)

    return train_loader, val_loader, test_loader
