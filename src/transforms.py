import torch
from torchvision.transforms import transforms


"""
Transforms are defined here
"""

# staderd_normalizations = 

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